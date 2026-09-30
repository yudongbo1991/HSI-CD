"""Final fixed-fusion VRWKV network for hyperspectral change detection."""
from __future__ import annotations

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba

from .hi_rwkv import HiRWKV


class HierarchicalGroupedMamba(nn.Module):
    """Bidirectional intra-group and inter-group spectral modelling."""

    def __init__(self, channels=128, groups=16, d_state=16, expand=2,
                 intra_state=None, inter_state=None):
        super().__init__()
        if channels % groups:
            raise ValueError("channels must be divisible by groups")
        self.groups, self.width = groups, channels // groups
        intra_state = d_state if intra_state is None else intra_state
        inter_state = d_state if inter_state is None else inter_state
        self.intra = Mamba(d_model=groups, d_state=intra_state, expand=expand,
                           dt_rank="auto")
        self.inter = Mamba(d_model=self.width, d_state=inter_state, expand=expand,
                           dt_rank="auto")
        self.norm = nn.LayerNorm(self.width)

    @staticmethod
    def _bidirectional(module, sequence):
        reverse = torch.flip(module(torch.flip(sequence, (1,))), (1,))
        return 0.5 * (module(sequence) + reverse)

    def forward(self, x):
        b, c, h, w = x.shape
        grouped = x.permute(0, 2, 3, 1).reshape(-1, self.groups, self.width)
        intra = self._bidirectional(
            self.intra, grouped.transpose(1, 2).contiguous())
        encoded = self._bidirectional(
            self.inter, self.norm(intra.transpose(1, 2)))
        return encoded.reshape(b, h, w, c).permute(0, 3, 1, 2).contiguous()


class SobelEvidence(nn.Module):
    """Parameter-free edge evidence shared by both dates."""

    def __init__(self):
        super().__init__()
        self.register_buffer("kx", torch.tensor(
            [[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]])[None, None])
        self.register_buffer("ky", torch.tensor(
            [[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]])[None, None])

    def magnitude(self, x):
        b, c, h, w = x.shape
        flat = x.reshape(b * c, 1, h, w)
        gx = F.conv2d(flat, self.kx, padding=1)
        gy = F.conv2d(flat, self.ky, padding=1)
        return (gx.square() + gy.square() + 1e-8).sqrt().reshape(
            b, c, h, w).mean(1, keepdim=True)

    def forward(self, first, second):
        edge = torch.maximum(self.magnitude(first), self.magnitude(second))
        return torch.sigmoid(
            edge / (edge.mean((2, 3), keepdim=True) + 1e-8) - 1.0)


class LightweightChannelAttention(nn.Module):
    """ECA-style channel reweighting with one shared local 1-D kernel."""

    def __init__(self, kernel_size=3):
        super().__init__()
        if kernel_size < 1 or kernel_size % 2 == 0:
            raise ValueError("channel attention kernel must be positive and odd")
        self.conv = nn.Conv1d(
            1, 1, kernel_size, padding=kernel_size // 2, bias=False)
        # 2*sigmoid(0)=1: start as an exact identity map, then learn only
        # evidence-supported channel deviations under the 1% label budget.
        nn.init.zeros_(self.conv.weight)

    def forward(self, x):
        descriptor = x.mean((2, 3), keepdim=False).unsqueeze(1)
        weight = (2.0 * torch.sigmoid(self.conv(descriptor))).squeeze(1)
        return x * weight[:, :, None, None]


class CrossSpatialVRWKV(nn.Module):
    """Shared VRWKV over both axes and both temporal interleaving orders."""

    def __init__(self, input_channels=128, output_channels=64, depth=2):
        super().__init__()
        if depth < 1:
            raise ValueError("VRWKV depth must be positive")
        rwkv_spec = [{"type": "rwkv", "num_blocks": [1]}]
        for _ in range(depth - 1):
            rwkv_spec.extend((
                {"type": "pool", "mode": "max", "k": 3, "s": 1, "p": 1},
                {"type": "act", "name": "silu"},
                {"type": "rwkv", "num_blocks": [1]}))
        self.vrwkv = HiRWKV(
            in_channels=input_channels, hidden_dim=output_channels, num_classes=10,
            group_num=1, rwkv_spec=rwkv_spec, use_channal=False)
        self.horizontal_fusion = nn.Conv2d(
            2 * output_channels, output_channels, 1, bias=False)
        self.vertical_fusion = nn.Conv2d(
            2 * output_channels, output_channels, 1, bias=False)
        # Kept as explicit constants for the training-entry compatibility.
        self.four_cross = True
        self.horizontal_bidirectional = False
        self.horizontal_forward_only = False
        self.horizontal_vertical_forward = False
        self.learn_axis_balance = False

    def _ordered_cross(self, first, second):
        horizontal = torch.stack((first, second), dim=4).reshape(
            first.shape[0], first.shape[1], first.shape[2], 2 * first.shape[3])
        vertical = torch.stack((first, second), dim=3).reshape(
            first.shape[0], first.shape[1], 2 * first.shape[2], first.shape[3])
        horizontal = self.vrwkv(horizontal)
        vertical = self.vrwkv(vertical)
        horizontal = self.horizontal_fusion(torch.cat((
            horizontal[:, :, :, 0::2], horizontal[:, :, :, 1::2]), dim=1))
        vertical = self.vertical_fusion(torch.cat((
            vertical[:, :, 0::2, :], vertical[:, :, 1::2, :]), dim=1))
        return horizontal, vertical

    def _ordered_horizontal(self, first, second):
        """Temporal interleaving along width, without a vertical VRWKV pass."""
        horizontal = torch.stack((first, second), dim=4).reshape(
            first.shape[0], first.shape[1], first.shape[2],
            2 * first.shape[3])
        horizontal = self.vrwkv(horizontal)
        return self.horizontal_fusion(torch.cat((
            horizontal[:, :, :, 0::2], horizontal[:, :, :, 1::2]), dim=1))

    def _ordered_column_vertical(self, first, second):
        """T1/T2 pixel interleaving scanned column-first, then restore H,W."""
        first_t = first.transpose(2, 3).contiguous()
        second_t = second.transpose(2, 3).contiguous()
        vertical = torch.stack((first_t, second_t), dim=4).reshape(
            first_t.shape[0], first_t.shape[1], first_t.shape[2],
            2 * first_t.shape[3])
        vertical = self.vrwkv(vertical)
        vertical = self.vertical_fusion(torch.cat((
            vertical[:, :, :, 0::2], vertical[:, :, :, 1::2]), dim=1))
        return vertical.transpose(2, 3).contiguous()

    def forward(self, first, second):
        if self.horizontal_vertical_forward:
            horizontal = self._ordered_horizontal(first, second)
            vertical = self._ordered_column_vertical(first, second)
            return 0.5 * (horizontal + vertical)
        if self.horizontal_forward_only:
            return self._ordered_horizontal(first, second)
        if self.horizontal_bidirectional:
            h12 = self._ordered_horizontal(first, second)
            h21 = self._ordered_horizontal(second, first)
            return 0.5 * (h12 + h21)
        h12, v12 = self._ordered_cross(first, second)
        h21, v21 = self._ordered_cross(second, first)
        return 0.25 * (h12 + v12 + h21 + v21)


class ChangeAwareResidual(nn.Module):
    """Inject gated absolute-change evidence after spatial VRWKV."""

    def __init__(self, input_channels=128, output_channels=64,
                 mode="gated"):
        super().__init__()
        if mode not in {"gated", "direct"}:
            raise ValueError("change residual mode must be gated or direct")
        self.mode = mode
        self.refine = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 1, bias=False),
            nn.Conv2d(output_channels, output_channels, 3, padding=1,
                      groups=output_channels, bias=False),
            nn.BatchNorm2d(output_channels), nn.SiLU(inplace=True))
        self.gate = nn.Conv2d(2 * output_channels, output_channels, 1)
        self.scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, spatial, first, second):
        evidence = self.refine((second - first).abs())
        if self.mode == "direct":
            return spatial + self.scale * evidence
        gate = torch.sigmoid(self.gate(torch.cat((spatial, evidence), dim=1)))
        return spatial + self.scale * gate * evidence


class GroupedPixelSpectralBranch(nn.Module):
    """Symmetric centre-spectrum encoder with grouped/global aggregation."""

    def __init__(self, bands, groups=8, width=16, representation_dim=64,
                 kernel_size=3, disable_grouping=False,
                 disable_global_pooling=False,
                 disable_global_component=False,
                 include_ungrouped_component=False,
                 remove_shared_encoder=False, remove_mean=False,
                 raw_grouping=False):
        super().__init__()
        self.groups = groups
        self.disable_grouping = bool(disable_grouping)
        self.disable_global_pooling = bool(disable_global_pooling)
        self.disable_global_component = bool(disable_global_component)
        self.include_ungrouped_component = bool(include_ungrouped_component)
        self.remove_shared_encoder = bool(remove_shared_encoder)
        self.remove_mean = bool(remove_mean)
        self.raw_grouping = bool(raw_grouping)
        padding = kernel_size // 2
        if self.remove_shared_encoder:
            self.shared_encoder = None
            interaction_input = 1 if self.remove_mean else 2
        else:
            self.shared_encoder = nn.Sequential(
                nn.Conv1d(1, width, kernel_size, padding=padding, bias=False),
                nn.BatchNorm1d(width), nn.SiLU(inplace=True),
                nn.Conv1d(width, width, kernel_size, padding=padding, bias=False),
                nn.BatchNorm1d(width), nn.SiLU(inplace=True))
            interaction_input = 2 * width
        self.symmetric_interaction = nn.Sequential(
            nn.Conv1d(interaction_input, width, 3, padding=1, bias=False),
            nn.BatchNorm1d(width), nn.SiLU(inplace=True))
        if self.remove_shared_encoder:
            projection_input = width * (groups if self.raw_grouping else bands)
        elif self.disable_global_pooling:
            projection_input = width * bands
        elif self.include_ungrouped_component:
            projection_input = width * (bands + groups)
        elif self.disable_global_component:
            projection_input = width * groups
        elif self.disable_grouping:
            projection_input = width
        else:
            projection_input = width * (groups + 1)
        self.projection = nn.Sequential(
            nn.Linear(projection_input, representation_dim),
            nn.SiLU(inplace=True))
        self.classifier = nn.Linear(representation_dim, 2)

    def forward(self, first, second):
        if self.remove_shared_encoder:
            difference = (second - first).abs()[:, None, :]
            if self.remove_mean:
                interaction_input = difference
            else:
                mean = (0.5 * (first + second))[:, None, :]
                interaction_input = torch.cat((difference, mean), dim=1)
            relation = self.symmetric_interaction(interaction_input)
            if self.raw_grouping:
                aggregation = F.adaptive_avg_pool1d(
                    relation, self.groups).flatten(1)
            else:
                aggregation = relation.flatten(1)
            feature = self.projection(aggregation)
            return feature, self.classifier(feature)
        h1 = self.shared_encoder(first[:, None, :])
        h2 = self.shared_encoder(second[:, None, :])
        relation = self.symmetric_interaction(torch.cat(
            ((h2 - h1).abs(), 0.5 * (h1 + h2)), dim=1))
        global_feature = F.adaptive_avg_pool1d(relation, 1).flatten(1)
        if self.disable_global_pooling:
            aggregation = relation.flatten(1)
        elif self.include_ungrouped_component:
            grouped = F.adaptive_avg_pool1d(
                relation, self.groups).flatten(1)
            aggregation = torch.cat((relation.flatten(1), grouped), dim=1)
        elif self.disable_global_component:
            aggregation = F.adaptive_avg_pool1d(
                relation, self.groups).flatten(1)
        elif self.disable_grouping:
            aggregation = global_feature
        else:
            grouped = F.adaptive_avg_pool1d(relation, self.groups).flatten(1)
            aggregation = torch.cat((grouped, global_feature), dim=1)
        feature = self.projection(aggregation)
        return feature, self.classifier(feature)


class AdaptiveMultiScalePixelSpectralBranch(nn.Module):
    """Two-evidence, sample-adaptive multi-scale centre-spectrum branch."""

    def __init__(self, bands, width=16, representation_dim=64,
                 dropout=0.1, dense_long_kernel=False, linear_length=0):
        super().__init__()

        def scale_encoder(kernel_size, dilation=1):
            padding = dilation * (kernel_size // 2)
            return nn.Sequential(
                nn.Conv1d(2, width, kernel_size, padding=padding,
                          dilation=dilation, bias=False),
                nn.BatchNorm1d(width), nn.SiLU(inplace=True),
                nn.Conv1d(width, width, 1, bias=False),
                nn.BatchNorm1d(width), nn.SiLU(inplace=True))

        self.local_scale = scale_encoder(3)
        self.region_scale = scale_encoder(7)
        self.long_scale = (scale_encoder(9) if dense_long_kernel
                           else scale_encoder(3, dilation=4))
        self.scale_selector = nn.Sequential(
            nn.Linear(3 * width, width), nn.SiLU(inplace=True),
            nn.Linear(width, 3))
        self.linear_length = int(linear_length)
        self.spectral_axis_projection = (
            nn.Linear(bands, self.linear_length)
            if self.linear_length > 0 else nn.Identity())
        flattened_length = self.linear_length if self.linear_length > 0 else bands
        self.projection = nn.Sequential(
            nn.Linear(width * flattened_length, representation_dim),
            nn.SiLU(inplace=True), nn.Dropout(dropout))
        self.classifier = nn.Linear(representation_dim, 2)

    def forward(self, first, second):
        difference = (second - first).abs()[:, None, :]
        temporal_state = (0.5 * (first + second))[:, None, :]
        evidence = torch.cat((difference, temporal_state), dim=1)
        scales = (self.local_scale(evidence),
                  self.region_scale(evidence),
                  self.long_scale(evidence))
        summaries = torch.cat(
            [feature.mean(dim=-1) for feature in scales], dim=1)
        weights = torch.softmax(self.scale_selector(summaries), dim=1)
        relation = sum(weights[:, index, None, None] * feature
                       for index, feature in enumerate(scales))
        relation = self.spectral_axis_projection(relation)
        representation = self.projection(relation.flatten(1))
        return representation, self.classifier(representation)


class CleanVRWKVCD(nn.Module):
    """Final model: spatial-context VRWKV and centre-spectrum branches."""

    FINAL_VARIANT = "compact_signed_prototype_e2e_reliability"
    FINAL_RELIABILITY = "embedded_logistic_learned_adaptive"
    VARIANTS = {FINAL_VARIANT}

    def __init__(self, bands=154, patch_size=5, local_size=5,
                 variant=FINAL_VARIANT, reliability_mode=FINAL_RELIABILITY,
                 spectral_groups=16, pixel_spectral_groups=8,
                 pixel_spectral_width=16, pixel_spectral_dim=64,
                 pixel_spectral_kernel=3, pixel_fusion_alpha=0.5,
                 mamba_state=16, mamba_intra_state=None,
                 mamba_inter_state=None, mamba_expand=2, mamba_depth=1,
                 vrwkv_depth=2, classifier_dropout=0.1,
                 spatial_input_dim=128, spatial_feature_dim=64,
                 change_residual_mode="gated",
                 spatial_channel_attention="none",
                 spatial_classifier_mode="flatten",
                 disable_sobel_modulation=False,
                 disable_change_residual_gate=False,
                 disable_mamba_module=False,
                 disable_vrwkv_module=False,
                 disable_pixel_spectral_grouping=False,
                 spectral_branch_only=False,
                 raw_center_difference_only=False,
                 spatial_branch_only=False,
                 disable_pixel_global_pooling=False,
                 disable_pixel_global_component=False,
                 include_pixel_ungrouped_component=False,
                 pixel_spectral_remove_shared_encoder=False,
                 pixel_spectral_remove_mean=False,
                 pixel_spectral_raw_grouping=False,
                 pixel_spectral_adaptive_multiscale=False,
                 spatial_unshared_input_projection=False,
                 pixel_spectral_linear_length=0, **_):
        super().__init__()
        if variant != self.FINAL_VARIANT:
            raise ValueError("only the final architecture is available")
        if reliability_mode != self.FINAL_RELIABILITY:
            raise ValueError("only the final fixed-fusion configuration is available")
        if patch_size != local_size or patch_size < 3 or patch_size % 2 == 0:
            raise ValueError("the final model uses one odd-sized spatial scale")
        self.bands, self.patch_size = bands, patch_size
        self.classifier_dropout = float(classifier_dropout)
        if not 0 <= self.classifier_dropout < 1:
            raise ValueError("classifier dropout must be in [0, 1)")
        self.pixel_fusion_alpha = float(pixel_fusion_alpha)
        self.disable_sobel_modulation = bool(disable_sobel_modulation)
        self.disable_change_residual_gate = bool(disable_change_residual_gate)
        self.disable_mamba_module = bool(disable_mamba_module)
        self.disable_vrwkv_module = bool(disable_vrwkv_module)
        self.spectral_branch_only = bool(spectral_branch_only)
        self.raw_center_difference_only = bool(raw_center_difference_only)
        self.spatial_branch_only = bool(spatial_branch_only)
        if not 0 <= self.pixel_fusion_alpha <= 1:
            raise ValueError("fusion weight must be in [0, 1]")

        if spatial_input_dim <= 0 or spatial_feature_dim <= 0:
            raise ValueError("spatial dimensions must be positive")
        if spatial_channel_attention not in {"none", "pre", "post"}:
            raise ValueError("spatial channel attention must be none/pre/post")
        if spatial_classifier_mode not in {
                "flatten", "global_avg", "center", "center3",
                "gaussian"}:
            raise ValueError(
                "spatial classifier mode must be "
                "flatten/global_avg/center/center3/gaussian")
        self.spatial_channel_attention = spatial_channel_attention
        self.spatial_classifier_mode = spatial_classifier_mode
        self.input_projection = nn.Conv2d(bands, spatial_input_dim, 1)
        self.spatial_unshared_input_projection = bool(
            spatial_unshared_input_projection)
        self.second_input_projection = (
            copy.deepcopy(self.input_projection)
            if self.spatial_unshared_input_projection else None)
        self.pre_channel_attention = (
            LightweightChannelAttention()
            if spatial_channel_attention == "pre" else None)
        self.post_channel_attention = (
            LightweightChannelAttention()
            if spatial_channel_attention == "post" else None)
        self.spectral_model = HierarchicalGroupedMamba(
            spatial_input_dim, spectral_groups, mamba_state, mamba_expand,
            mamba_intra_state, mamba_inter_state)
        if mamba_depth < 1:
            raise ValueError("Mamba depth must be positive")
        self.spectral_extra = nn.ModuleList(
            HierarchicalGroupedMamba(
                spatial_input_dim, spectral_groups, mamba_state, mamba_expand,
                mamba_intra_state, mamba_inter_state)
            for _ in range(mamba_depth - 1))
        self.sobel = SobelEvidence()
        self.edge_scale = nn.Parameter(torch.zeros(()))
        self.detail_scale = None
        self.spatial_model = CrossSpatialVRWKV(
            spatial_input_dim, spatial_feature_dim, vrwkv_depth)
        # Dimension-only bridge for the no-VRWKV ablation. A pointwise linear
        # map adds no spatial context and therefore cannot replace VRWKV.
        self.no_vrwkv_projection = nn.Conv2d(
            spatial_input_dim, spatial_feature_dim, 1, bias=False)
        self.change_residual = ChangeAwareResidual(
            spatial_input_dim, spatial_feature_dim, change_residual_mode)
        if spatial_classifier_mode in {"flatten", "gaussian"}:
            classifier_dim = spatial_feature_dim * patch_size * patch_size
        elif spatial_classifier_mode == "center3":
            classifier_dim = spatial_feature_dim * 9
        else:
            classifier_dim = spatial_feature_dim
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(classifier_dim),
            nn.LeakyReLU(inplace=True),
            nn.Linear(classifier_dim, 2))
        axis = torch.arange(patch_size, dtype=torch.float32)
        axis = axis - (patch_size - 1) / 2.0
        yy, xx = torch.meshgrid(axis, axis, indexing="ij")
        sigma = patch_size / 2.0
        readout_weight = torch.exp(-(xx.square() + yy.square()) /
                                   (2.0 * sigma * sigma))
        readout_weight = readout_weight / readout_weight.mean()
        self.register_buffer("spatial_readout_weight",
                             readout_weight[None, None], persistent=True)
        pixel_spectral_dense_kernel9 = bool(
            _.get("pixel_spectral_dense_kernel9", False))
        if pixel_spectral_adaptive_multiscale:
            self.pixel_spectral_branch = AdaptiveMultiScalePixelSpectralBranch(
                bands, pixel_spectral_width, pixel_spectral_dim,
                classifier_dropout, pixel_spectral_dense_kernel9,
                pixel_spectral_linear_length)
        else:
            self.pixel_spectral_branch = GroupedPixelSpectralBranch(
                bands, pixel_spectral_groups, pixel_spectral_width,
                pixel_spectral_dim, pixel_spectral_kernel,
                disable_pixel_spectral_grouping,
                disable_pixel_global_pooling,
                disable_pixel_global_component,
                include_pixel_ungrouped_component,
                pixel_spectral_remove_shared_encoder,
                pixel_spectral_remove_mean,
                pixel_spectral_raw_grouping)
        self.raw_center_classifier = nn.Linear(bands, 2)

        self.single_pass = True
        self.pixel_fused_loss_weight = 1.0
        self.pixel_main_loss_weight = 1.0
        self.pixel_branch_loss_weight = 1.0
        self.pixel_reliability_loss_weight = 0.0
        self.disable_embedded_physics_routing = True

    def forward(self, date1, date2):
        batch = date1.shape[0]
        side = self.patch_size
        first = date1.reshape(batch, self.bands, side, side)
        second = date2.reshape(batch, self.bands, side, side)
        center = side // 2
        center1 = first[:, :, center, center]
        center2 = second[:, :, center, center]

        if self.spatial_branch_only:
            spectral_feature = spectral_logits = None
        elif self.raw_center_difference_only:
            spectral_feature = center2 - center1
            spectral_logits = self.raw_center_classifier(spectral_feature)
        else:
            spectral_feature, spectral_logits = self.pixel_spectral_branch(
                center1, center2)
        if self.spectral_branch_only:
            zeros = torch.zeros_like(spectral_logits)
            ones = spectral_logits.new_ones((batch,))
            return {
                "logits": spectral_logits, "fused_logits": spectral_logits,
                "main_logits": zeros,
                "pixel_spectral_logits": spectral_logits,
                "pixel_spectral_feature": spectral_feature,
                "pixel_fusion_alpha": ones, "feature": spectral_feature,
                "coarse": None, "context": None,
                "context_reliability": ones,
                "pixel_reliability_weights": None,
                "physical_context_evidence": None,
                "physics_logits": zeros, "learned_logits": zeros,
                "reliability_weights": torch.stack((
                    torch.zeros_like(ones), ones, torch.zeros_like(ones)), 1),
            }

        first = self.input_projection(first)
        second = (self.second_input_projection(second)
                  if self.second_input_projection is not None
                  else self.input_projection(second))
        if self.spatial_channel_attention == "pre":
            first = self.pre_channel_attention(first)
            second = self.pre_channel_attention(second)
        if not self.disable_mamba_module:
            first = self.spectral_model(first)
            second = self.spectral_model(second)
            for block in self.spectral_extra:
                first = first + block(first)
                second = second + block(second)
        if not self.disable_sobel_modulation:
            edge = self.sobel(first, second)
            gain = 1.0 + 0.25 * torch.tanh(self.edge_scale) * edge
            first, second = first * gain, second * gain
        if self.disable_vrwkv_module:
            fused = self.no_vrwkv_projection((second - first).abs())
        else:
            fused = self.spatial_model(first, second)
        if not self.disable_vrwkv_module and not self.disable_change_residual_gate:
            fused = self.change_residual(fused, first, second)
        if self.spatial_channel_attention == "post":
            fused = self.post_channel_attention(fused)
        feature = F.adaptive_avg_pool2d(fused, 1).flatten(1)
        if self.spatial_classifier_mode == "flatten":
            classifier_input = fused.flatten(1)
        elif self.spatial_classifier_mode == "gaussian":
            classifier_input = (
                fused * self.spatial_readout_weight).flatten(1)
        elif self.spatial_classifier_mode == "center3":
            radius = 1
            classifier_input = fused[:, :,
                                     side // 2 - radius:side // 2 + radius + 1,
                                     side // 2 - radius:side // 2 + radius + 1
                                     ].flatten(1)
        elif self.spatial_classifier_mode == "center":
            classifier_input = fused[:, :, side // 2, side // 2]
        else:
            classifier_input = feature
        classifier_feature = self.classifier[1](
            self.classifier[0](classifier_input))
        main_logits = self.classifier[2](F.dropout(
            classifier_feature, p=self.classifier_dropout,
            training=self.training))

        if self.spatial_branch_only:
            ones = main_logits.new_ones((batch,))
            zeros = torch.zeros_like(main_logits)
            return {
                "logits": main_logits, "fused_logits": main_logits,
                "main_logits": main_logits, "pixel_spectral_logits": None,
                "pixel_spectral_feature": None,
                "pixel_fusion_alpha": None, "feature": feature,
                "coarse": None, "context": None,
                "context_reliability": ones,
                "pixel_reliability_weights": None,
                "physical_context_evidence": None,
                "physics_logits": zeros, "learned_logits": zeros,
                "reliability_weights": torch.stack((
                    ones, torch.zeros_like(ones), torch.zeros_like(ones)), 1),
            }

        main_probability = F.softmax(main_logits, dim=1)
        spectral_probability = F.softmax(spectral_logits, dim=1)
        alpha = main_logits.new_full((batch,), self.pixel_fusion_alpha)
        probability = ((1.0 - alpha[:, None]) * main_probability
                       + alpha[:, None] * spectral_probability)
        logits = probability.clamp_min(1e-7).log()
        zeros = torch.zeros_like(main_logits)
        return {
            "logits": logits, "fused_logits": logits,
            "main_logits": main_logits,
            "pixel_spectral_logits": spectral_logits,
            "pixel_spectral_feature": spectral_feature,
            "pixel_fusion_alpha": alpha, "feature": feature,
            "coarse": None, "context": None,
            "context_reliability": torch.ones(batch, device=first.device),
            "pixel_reliability_weights": None,
            "physical_context_evidence": None,
            "physics_logits": zeros, "learned_logits": zeros,
            "reliability_weights": torch.cat((
                torch.ones_like(alpha[:, None]),
                torch.zeros_like(alpha[:, None]),
                torch.zeros_like(alpha[:, None])), dim=1),
        }
