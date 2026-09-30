"""Final dual-branch VRWKV model for hyperspectral change detection."""
from __future__ import annotations

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from .hi_rwkv import HiRWKV


class CrossSpatialVRWKV(nn.Module):
    """T1→T2 interleaving along rows and columns with shared VRWKV weights."""

    def __init__(self, input_channels=128, output_channels=64, depth=3):
        super().__init__()
        self.vrwkv = HiRWKV(
            in_channels=input_channels, hidden_dim=output_channels, depth=depth)
        self.horizontal_fusion = nn.Conv2d(
            2 * output_channels, output_channels, 1, bias=False)
        self.vertical_fusion = nn.Conv2d(
            2 * output_channels, output_channels, 1, bias=False)

    def _horizontal(self, first, second):
        sequence = torch.stack((first, second), dim=4).reshape(
            first.shape[0], first.shape[1], first.shape[2],
            2 * first.shape[3])
        encoded = self.vrwkv(sequence)
        restored = torch.cat(
            (encoded[:, :, :, 0::2], encoded[:, :, :, 1::2]), dim=1)
        return self.horizontal_fusion(restored)

    def _vertical(self, first, second):
        first_t = first.transpose(2, 3).contiguous()
        second_t = second.transpose(2, 3).contiguous()
        sequence = torch.stack((first_t, second_t), dim=4).reshape(
            first_t.shape[0], first_t.shape[1], first_t.shape[2],
            2 * first_t.shape[3])
        encoded = self.vrwkv(sequence)
        restored = torch.cat(
            (encoded[:, :, :, 0::2], encoded[:, :, :, 1::2]), dim=1)
        return self.vertical_fusion(restored).transpose(2, 3).contiguous()

    def forward(self, first, second):
        return 0.5 * (self._horizontal(first, second)
                      + self._vertical(first, second))


class ChangeResidualGate(nn.Module):
    """Inject aligned absolute-change evidence after spatial aggregation."""

    def __init__(self, input_channels=128, output_channels=64):
        super().__init__()
        self.refine = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 1, bias=False),
            nn.Conv2d(output_channels, output_channels, 3, padding=1,
                      groups=output_channels, bias=False),
            nn.BatchNorm2d(output_channels), nn.SiLU(inplace=True))
        self.gate = nn.Conv2d(2 * output_channels, output_channels, 1)
        self.scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, spatial, first, second):
        evidence = self.refine((second - first).abs())
        gate = torch.sigmoid(self.gate(torch.cat((spatial, evidence), dim=1)))
        return spatial + self.scale * gate * evidence


class AdaptiveSpectralBranch(nn.Module):
    """Adaptive three-scale comparison of aligned centre spectra."""

    def __init__(self, bands, width=16, projection_length=32,
                 representation_dim=64, dropout=0.05):
        super().__init__()

        def encoder(kernel, dilation=1):
            padding = dilation * (kernel // 2)
            return nn.Sequential(
                nn.Conv1d(2, width, kernel, padding=padding,
                          dilation=dilation, bias=False),
                nn.BatchNorm1d(width), nn.SiLU(inplace=True),
                nn.Conv1d(width, width, 1, bias=False),
                nn.BatchNorm1d(width), nn.SiLU(inplace=True))

        self.local = encoder(3)
        self.region = encoder(7)
        self.long_range = encoder(3, dilation=4)
        self.scale_selector = nn.Sequential(
            nn.Linear(3 * width, width), nn.SiLU(inplace=True),
            nn.Linear(width, 3))
        self.spectral_projection = nn.Linear(bands, projection_length)
        self.feature_projection = nn.Sequential(
            nn.Linear(width * projection_length, representation_dim),
            nn.SiLU(inplace=True), nn.Dropout(dropout))
        self.classifier = nn.Linear(representation_dim, 2)

    def forward(self, first, second):
        evidence = torch.stack(
            ((second - first).abs(), 0.5 * (first + second)), dim=1)
        scales = (self.local(evidence), self.region(evidence),
                  self.long_range(evidence))
        summary = torch.cat([value.mean(-1) for value in scales], dim=1)
        weight = torch.softmax(self.scale_selector(summary), dim=1)
        relation = sum(weight[:, i, None, None] * value
                       for i, value in enumerate(scales))
        relation = self.spectral_projection(relation)
        feature = self.feature_projection(relation.flatten(1))
        return self.classifier(feature)


class VRWKVChangeDetector(nn.Module):
    """Final spatial-context and centre-spectrum dual-branch network."""

    def __init__(self, bands, patch_size=7, spectral_weight=0.5,
                 share_spatial_projection=True):
        super().__init__()
        self.bands = bands
        self.patch_size = patch_size
        self.spectral_weight = float(spectral_weight)
        self.first_projection = nn.Conv2d(bands, 128, 1)
        self.second_projection = (None if share_spatial_projection
                                  else copy.deepcopy(self.first_projection))
        self.spatial_encoder = CrossSpatialVRWKV(128, 64, depth=3)
        self.change_gate = ChangeResidualGate(128, 64)
        self.spatial_classifier = nn.Sequential(
            nn.BatchNorm1d(64 * patch_size * patch_size),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(0.05),
            nn.Linear(64 * patch_size * patch_size, 2))
        self.spectral_encoder = AdaptiveSpectralBranch(bands)

    def forward(self, date1, date2):
        batch, side = date1.shape[0], self.patch_size
        first = date1.reshape(batch, self.bands, side, side)
        second = date2.reshape(batch, self.bands, side, side)
        first_spatial = self.first_projection(first)
        second_spatial = (self.first_projection(second)
                          if self.second_projection is None
                          else self.second_projection(second))
        spatial = self.spatial_encoder(first_spatial, second_spatial)
        spatial = self.change_gate(spatial, first_spatial, second_spatial)
        spatial_logits = self.spatial_classifier(spatial.flatten(1))

        center = side // 2
        spectral_logits = self.spectral_encoder(
            first[:, :, center, center], second[:, :, center, center])
        spatial_probability = F.softmax(spatial_logits, dim=1)
        spectral_probability = F.softmax(spectral_logits, dim=1)
        probability = ((1.0 - self.spectral_weight) * spatial_probability
                       + self.spectral_weight * spectral_probability)
        fused_logits = probability.clamp_min(1e-7).log()
        return {"logits": fused_logits, "spatial_logits": spatial_logits,
                "spectral_logits": spectral_logits}
