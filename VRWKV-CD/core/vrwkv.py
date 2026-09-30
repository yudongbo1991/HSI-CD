"""Minimal spatial VRWKV used by the released dual-branch detector."""
from pathlib import Path
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load

_CUDA_DIR = Path(__file__).resolve().parent / "cuda_new"
_wkv_cuda = load(
    name="vrwkv_cd_bi_wkv",
    sources=[str(_CUDA_DIR / "bi_wkv.cpp"), str(_CUDA_DIR / "bi_wkv_kernel.cu")],
    verbose=False,
    extra_cuda_cflags=["--use_fast_math", "-O3", "-Xptxas=-O3",
                       "-DTmax=262144", "-allow-unsupported-compiler"],
)


class _WKV(torch.autograd.Function):
    @staticmethod
    def forward(ctx, decay, first, key, value):
        ctx.save_for_backward(decay, first, key, value)
        output = _wkv_cuda.bi_wkv_forward(
            decay.float().contiguous(), first.float().contiguous(),
            key.float().contiguous(), value.float().contiguous())
        return output.to(key.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        decay, first, key, value = ctx.saved_tensors
        gradients = _wkv_cuda.bi_wkv_backward(
            decay.float().contiguous(), first.float().contiguous(),
            key.float().contiguous(), value.float().contiguous(),
            grad_output.float().contiguous())
        return tuple(gradient.to(tensor.dtype) for gradient, tensor in
                     zip(gradients, (decay, first, key, value)))


def _sobel_magnitude(features):
    batch, channels, height, width = features.shape
    kernel_x = features.new_tensor(
        [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).view(1, 1, 3, 3)
    kernel_y = features.new_tensor(
        [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]).view(1, 1, 3, 3)
    flat = features.reshape(batch * channels, 1, height, width)
    grad_x = F.conv2d(flat, kernel_x, padding=1)
    grad_y = F.conv2d(flat, kernel_y, padding=1)
    magnitude = torch.sqrt(grad_x.square() + grad_y.square() + 1e-8)
    return magnitude.view(batch, channels, height, width).mean(1, keepdim=True)


class SpatialWKV(nn.Module):
    """The recurrent operator used by every retained VRWKV block."""
    def __init__(self, dimension, decay_min=0.15, decay_max=4.0):
        super().__init__()
        self.key = nn.Linear(dimension, dimension, bias=False)
        self.value = nn.Linear(dimension, dimension, bias=False)
        self.receptance = nn.Linear(dimension, dimension, bias=False)
        self.output = nn.Linear(dimension, dimension, bias=False)
        if not 0.0 < decay_min <= decay_max:
            raise ValueError("decay range must satisfy 0 < min <= max")
        pair_count = (dimension + 1) // 2
        magnitude = torch.exp(torch.linspace(
            math.log(decay_min), math.log(decay_max), pair_count))
        signed = torch.stack((-magnitude, magnitude), dim=1).reshape(-1)
        self.spatial_decay = nn.Parameter(signed[:dimension].clone())
        self.spatial_first = nn.Parameter(torch.ones(dimension))

    def forward(self, tokens, spatial_shape):
        batch, token_count, channels = tokens.shape
        height, width = spatial_shape
        key, value = self.key(tokens), self.value(tokens)
        receptance = torch.sigmoid(self.receptance(tokens))
        feature_map = tokens.transpose(1, 2).reshape(
            batch, channels, height, width)
        edge = torch.sigmoid(_sobel_magnitude(feature_map))
        receptance = receptance + edge.flatten(2).transpose(1, 2)
        context = _WKV.apply(self.spatial_decay / token_count,
                             self.spatial_first / token_count, key, value)
        return self.output(receptance * context)


class VRWKVBlock(nn.Module):
    def __init__(self, dimension, first_block=False):
        super().__init__()
        self.input_norm = nn.LayerNorm(dimension) if first_block else None
        self.context_norm = nn.LayerNorm(dimension)
        self.spatial_wkv = SpatialWKV(dimension)

    def forward(self, features):
        batch, channels, height, width = features.shape
        tokens = features.flatten(2).transpose(1, 2)
        if self.input_norm is not None:
            tokens = self.input_norm(tokens)
        tokens = tokens + self.spatial_wkv(
            self.context_norm(tokens), (height, width))
        return tokens.transpose(1, 2).reshape(batch, channels, height, width)


class HSI_RWKV(nn.Module):
    def __init__(self, dim=128, num_blocks=(1,), use_channal=False):
        super().__init__()
        if use_channal:
            raise ValueError("the released model contains spatial VRWKV only")
        self.blocks = nn.Sequential(*[
            VRWKVBlock(dim, first_block=(index == 0))
            for index in range(int(num_blocks[0]))
        ])

    def forward(self, features):
        return self.blocks(features)
