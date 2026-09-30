from typing import Sequence
import os
import math

import logging
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.utils.checkpoint as cp
#from mmcv.runner.base_module import BaseModule, ModuleList
from mmengine.model import BaseModule, ModuleList
from typing import Optional, Sequence
from .drop import DropPath
from einops import rearrange
logger = logging.getLogger(__name__)


from pathlib import Path
from torch.utils.cpp_extension import load
_CUDA_DIR = Path(__file__).resolve().parent / "cuda_new"
wkv_cuda = load(name="route11_weighted_multiscale_bi_wkv", sources=[str(_CUDA_DIR / "bi_wkv.cpp"), str(_CUDA_DIR / "bi_wkv_kernel.cu")],
                verbose=True, extra_cuda_cflags=[
                                    "--use_fast_math",
                                    "-O3",
                                    "-Xptxas=-O3",
                                    "-DTmax=262144",
                                    "-allow-unsupported-compiler"
                                ]
                )


class WKV(torch.autograd.Function):
    @staticmethod
    def forward(ctx, w, u, k, v):
        half_mode = (w.dtype == torch.half)
        bf_mode = (w.dtype == torch.bfloat16)
        ctx.save_for_backward(w, u, k, v)
        w = w.float().contiguous()
        u = u.float().contiguous()
        k = k.float().contiguous()
        v = v.float().contiguous()
        y = wkv_cuda.bi_wkv_forward(w, u, k, v)
        if half_mode:
            y = y.half()
        elif bf_mode:
            y = y.bfloat16()
        return y

    @staticmethod
    def backward(ctx, gy):
        w, u, k, v = ctx.saved_tensors
        half_mode = (w.dtype == torch.half)
        bf_mode = (w.dtype == torch.bfloat16)
        gw, gu, gk, gv = wkv_cuda.bi_wkv_backward(w.float().contiguous(),
                          u.float().contiguous(),
                          k.float().contiguous(),
                          v.float().contiguous(),
                          gy.float().contiguous())
        if half_mode:
            return (gw.half(), gu.half(), gk.half(), gv.half())
        elif bf_mode:
            return (gw.bfloat16(), gu.bfloat16(), gk.bfloat16(), gv.bfloat16())
        else:
            return (gw, gu, gk, gv)


def RUN_CUDA(w, u, k, v):
    return WKV.apply(w.cuda(), u.cuda(), k.cuda(), v.cuda())


def q_shift(input, shift_pixel=1, gamma=1/4, patch_resolution=None):
    assert gamma <= 1/4
    B, N, C = input.shape
    # import ipdb
    # ipdb.set_trace()
    input = input.transpose(1, 2).reshape(B, C, patch_resolution[0], patch_resolution[1])
    B, C, H, W = input.shape
    output = torch.zeros_like(input)
    output[:, 0:int(C*gamma), :, shift_pixel:W] = input[:, 0:int(C*gamma), :, 0:W-shift_pixel]
    output[:, int(C*gamma):int(C*gamma*2), :, 0:W-shift_pixel] = input[:, int(C*gamma):int(C*gamma*2), :, shift_pixel:W]
    output[:, int(C*gamma*2):int(C*gamma*3), shift_pixel:H, :] = input[:, int(C*gamma*2):int(C*gamma*3), 0:H-shift_pixel, :]
    output[:, int(C*gamma*3):int(C*gamma*4), 0:H-shift_pixel, :] = input[:, int(C*gamma*3):int(C*gamma*4), shift_pixel:H, :]
    output[:, int(C*gamma*4):, ...] = input[:, int(C*gamma*4):, ...]
    return output.flatten(2).transpose(1, 2)



def spatial_grad_mag_sobel(x_bchw: torch.Tensor) -> torch.Tensor:
    #print("zuoguo")
    B, C, H, W = x_bchw.shape
    # Sobel
    kx = torch.tensor([[-1, 0, 1],
                       [-2, 0, 2],
                       [-1, 0, 1]], dtype=x_bchw.dtype, device=x_bchw.device).view(1,1,3,3)
    ky = torch.tensor([[-1,-2,-1],
                       [ 0, 0, 0],
                       [ 1, 2, 1]], dtype=x_bchw.dtype, device=x_bchw.device).view(1,1,3,3)


    x_flat = x_bchw.reshape(B*C, 1, H, W)
    gx = F.conv2d(x_flat, kx, padding=1)  # [B*C,1,H,W]
    gy = F.conv2d(x_flat, ky, padding=1)
    # import ipdb; ipdb.set_trace()
    eps = 1e-8
    mag = torch.sqrt(gx * gx + gy * gy + eps).view(B, C, H, W)
    # mag = (gx + gy).view(B, C, H, W)
    mag = mag.mean(dim=1, keepdim=True)   # [B,1,H,W]
    return mag
def spatial_grad_gaussian(x_bchw: torch.Tensor, sigma=1.0) -> torch.Tensor:
    device = x_bchw.device
    dtype = x_bchw.dtype
    B, C, H, W = x_bchw.shape

    # 1. 构建高斯导数核
    kernel_size = int(6 * sigma + 1)
    if kernel_size % 2 == 0: kernel_size += 1
    range_vec = torch.arange(kernel_size, device=device, dtype=dtype) - kernel_size // 2

    # 高斯分布公式
    gauss = torch.exp(-range_vec**2 / (2 * sigma**2))
    # 高斯一阶导数公式
    gauss_grad = -range_vec * torch.exp(-range_vec**2 / (2 * sigma**2))

    # 分离卷积核：利用高斯核的可分离性提高效率
    # dx 方向：x方向求导，y方向平滑
    kernel_dx = (gauss_grad.view(-1, 1) @ gauss.view(1, -1)).view(1, 1, kernel_size, kernel_size)
    # dy 方向：y方向求导，x方向平滑
    kernel_dy = (gauss.view(-1, 1) @ gauss_grad.view(1, -1)).view(1, 1, kernel_size, kernel_size)

    # 归一化（防止数值爆炸）
    kernel_dx /= kernel_dx.abs().sum()
    kernel_dy /= kernel_dy.abs().sum()

    # 2. 计算梯度
    x_flat = x_bchw.reshape(B * C, 1, H, W)
    pad = kernel_size // 2
    gx = F.conv2d(x_flat, kernel_dx, padding=pad)
    gy = F.conv2d(x_flat, kernel_dy, padding=pad)

    # 3. 计算幅值并融合通道
    mag = torch.sqrt(gx**2 + gy**2 + 1e-8)
    mag = mag.view(B, C, H, W).mean(dim=1, keepdim=True)

    return mag



class VRWKV_SpatialMix(nn.Module):
    def __init__(self,
                 n_embd: int,
                 n_layer: int,
                 layer_id: int,
                 shift_mode: str = 'q_shift',
                 channel_gamma: float = 1/4,
                 shift_pixel = [1,2,4],
                 init_mode: str = 'fancy',
                 key_norm: bool = False,
                 with_cp: bool = False,
                 edge_gate: bool = True,
                 edge_gate_strength: float = 1.0):
        super().__init__()
        self.layer_id = layer_id
        self.n_layer = n_layer
        self.n_embd = n_embd
        self.with_cp = with_cp


        self.shift_pixel = shift_pixel
        if isinstance(shift_mode, str):
            self.shift_func = eval(shift_mode)
        else:
            self.shift_func = shift_mode
        self.channel_gamma = channel_gamma


        attn_sz = n_embd
        self.key        = nn.Linear(n_embd, attn_sz, bias=False)
        self.value      = nn.Linear(n_embd, attn_sz, bias=False)
        self.receptance = nn.Linear(n_embd, attn_sz, bias=False)
        self.output     = nn.Linear(attn_sz, n_embd, bias=False)

        self.key.scale_init        = 0
        self.receptance.scale_init = 0
        self.output.scale_init     = 0

        self.key_norm = nn.LayerNorm(n_embd) if key_norm else None


        self._init_weights(init_mode)


        self.edge_gate = edge_gate
        self.edge_gate_strength = float(edge_gate_strength)
        self.edge_gate_mode = os.environ.get('RWKV_EDGE_GATE_MODE', 'legacy')

    def _init_weights(self, init_mode: str):
        if init_mode == 'fancy':
            self.decay_parameterization = os.environ.get(
                'RWKV_DECAY_PARAM', 'direct')
            decay_init = os.environ.get('RWKV_DECAY_INIT', 'original')
            if decay_init in ('adaptive_bank', 'adaptive_residual_bank',
                              'edge_adaptive_bank',
                              'edge_adaptive_bank_detached',
                              'edge_signed_channel_bank'):
                # Three content-adaptive memory scales reuse the same K/V/U
                # and unchanged CUDA recurrence.  The gate is sample-specific,
                # overcoming the single decay vector shared by every scene.
                initial_banks = torch.tensor([0.5, 1.0, 2.0])
                raw_banks = torch.log(torch.expm1(initial_banks))
                self.spatial_decay_banks = nn.Parameter(
                    raw_banks[:, None].expand(-1, self.n_embd).clone())
                if decay_init not in ('edge_adaptive_bank',
                                      'edge_adaptive_bank_detached',
                                      'edge_signed_channel_bank'):
                    gate_count = (2 if decay_init == 'adaptive_residual_bank'
                                  else 3)
                    self.spatial_decay_gate = nn.Linear(
                        self.n_embd, gate_count, bias=True)
                    nn.init.zeros_(self.spatial_decay_gate.weight)
                # Begin close to the established medium-memory W=1 path while
                # keeping nonzero probability/gradient for short and long W.
                with torch.no_grad():
                    if decay_init == 'adaptive_bank':
                        self.spatial_decay_gate.bias.copy_(
                            torch.tensor([-2.0, 2.0, -2.0]))
                    elif decay_init == 'adaptive_residual_bank':
                        # The residual formulation is exactly the established
                        # W=1 path at initialization.  A token may subsequently
                        # add bounded short/long-memory corrections, avoiding
                        # early dilution of the reliable baseline.
                        self.spatial_decay_gate.bias.zero_()
                self.adaptive_decay_residual = (
                    decay_init == 'adaptive_residual_bank')
                self.edge_adaptive_decay = decay_init in (
                    'edge_adaptive_bank', 'edge_adaptive_bank_detached',
                    'edge_signed_channel_bank')
                self.edge_adaptive_detach = (
                    decay_init == 'edge_adaptive_bank_detached')
                if self.edge_adaptive_decay:
                    # One shared signed amplitude is far less likely to fit
                    # the 739 labeled coordinates than a free token MLP.  The
                    # zero initialization exactly recovers the W=1 baseline.
                    # Opt181 uses one signed scalar.  The refined mode uses a
                    # signed value per latent channel. Together with the
                    # per-token edge selector this factorises a T x C decay
                    # correction without making the recurrence itself
                    # unstable or fitting a free parameter at every pixel.
                    edge_scale_shape = ((self.n_embd,)
                        if decay_init == 'edge_signed_channel_bank' else ())
                    self.spatial_decay_edge_scale = nn.Parameter(
                        torch.zeros(edge_scale_shape))
            elif decay_init == 'global_residual':
                # A coherent global memory scale receives gradients summed
                # over all channels, while a bounded zero-initialized residual
                # permits only evidence-supported channel specialization.  The
                # effective vector still matches the unchanged CUDA API.
                initial_global = torch.tensor(1.0)
                self.spatial_decay_global = nn.Parameter(
                    torch.log(torch.expm1(initial_global)))
                self.spatial_decay_offset = nn.Parameter(
                    torch.zeros(self.n_embd))
                self.spatial_decay_residual_bound = float(os.environ.get(
                    'RWKV_DECAY_RESIDUAL_BOUND', '0.5'))
            elif decay_init == 'multiscale':
                # `forward` divides this by T, so these values describe total
                # sequence decay. Log spacing gives simultaneous long-, medium-
                # and short-memory channels without changing the CUDA kernel.
                decay_min = float(os.environ.get('RWKV_DECAY_MIN', '0.15'))
                decay_max = float(os.environ.get('RWKV_DECAY_MAX', '4.0'))
                if not (0.0 < decay_min <= decay_max):
                    raise ValueError('RWKV decay range must satisfy 0 < min <= max')
                total_decay = torch.exp(torch.linspace(
                    math.log(decay_min), math.log(decay_max), self.n_embd))
                if self.decay_parameterization == 'softplus':
                    total_decay = torch.log(torch.expm1(total_decay))
                self.spatial_decay = nn.Parameter(total_decay)
            elif decay_init == 'uniform':
                uniform_decay = float(os.environ.get('RWKV_DECAY_MIN', '2.0'))
                if uniform_decay <= 0.0:
                    raise ValueError('uniform RWKV decay must be positive')
                self.spatial_decay = nn.Parameter(
                    torch.full((self.n_embd,), uniform_decay))
            elif decay_init == 'signed_symmetric':
                # Directly exercise both directions of the original CUDA
                # parameter: positive w suppresses distant tokens whereas
                # negative w promotes them through exp(-w * distance).
                # Alternating signs avoids coupling one sign to q_shift's
                # contiguous directional channel partitions.  Magnitudes are
                # log-spaced so local, neutral-ish and strongly global modes
                # coexist from initialisation and remain fully trainable.
                decay_min = float(os.environ.get('RWKV_DECAY_MIN', '0.15'))
                decay_max = float(os.environ.get('RWKV_DECAY_MAX', '4.0'))
                if not (0.0 < decay_min <= decay_max):
                    raise ValueError('signed decay magnitudes require 0 < min <= max')
                pair_count = (self.n_embd + 1) // 2
                magnitudes = torch.exp(torch.linspace(
                    math.log(decay_min), math.log(decay_max), pair_count))
                signed = torch.stack((-magnitudes, magnitudes), dim=1).reshape(-1)
                self.spatial_decay = nn.Parameter(signed[:self.n_embd].clone())
            else:
                # Backward-compatible original path.
                self.spatial_decay = nn.Parameter(torch.ones(self.n_embd))
            self.spatial_first = nn.Parameter(torch.ones(self.n_embd))

            self.spatial_mix_k = nn.Parameter(torch.ones(1, 1, self.n_embd))
            self.spatial_mix_v = nn.Parameter(torch.ones(1, 1, self.n_embd))
            self.spatial_mix_r = nn.Parameter(torch.ones(1, 1, self.n_embd))
        else:
            raise NotImplementedError(f"init_mode={init_mode} not supported")

    def jit_func(self, x: torch.Tensor, patch_resolution):
        """
        x: [B, T, C], T=H*W
        return: sr (sigmoid(r)), k, v
        """

        xk = x
        xv = x
        xr = x

        k = self.key(xk)
        v = self.value(xv)
        r = self.receptance(xr)
        sr = torch.sigmoid(r)  # [B,T,C]
        return sr, k, v

    def forward(self, x: torch.Tensor, patch_resolution=None) -> torch.Tensor:
        """
        x: [B, T, C], T = H*W
        patch_resolution: (H, W)
        """
        B, T, C = x.shape
        assert patch_resolution is not None, "patch_resolution (H,W) is required"
        H, W = patch_resolution

        sr, k, v = self.jit_func(x, patch_resolution)  # sr: [B,T,C]

        if self.edge_gate:
            x_hw = x.transpose(1, 2).reshape(B, C, H, W)          # [B,C,H,W]
            edge = spatial_grad_mag_sobel(x_hw)                      # [B,1,H,W]
            #edge = spatial_grad_gaussian(x_hw)
            if self.edge_gate_mode == 'normalized_context':
                edge = (edge - edge.mean((2, 3), keepdim=True)) / (
                    edge.std((2, 3), keepdim=True, unbiased=False) + 1e-4)
                edge_g = torch.tanh(edge)
                edge_g = edge_g.reshape(B, 1, T).transpose(1, 2)
                # Edge evidence changes how strongly each token reads the
                # already-computed WKV context. It is zero-centred, bounded,
                # and requires no additional recurrent pass.
                sr = sr * (1.0 + 0.25 * self.edge_gate_strength * edge_g)
            elif self.edge_gate_mode != 'off':
                edge_g = torch.sigmoid(edge)  # [B,1,H,W]
                edge_g = edge_g.reshape(B, 1, T)
                sr = sr + edge_g.transpose(1, 2)


        if hasattr(self, 'spatial_decay_banks'):
            decay_banks = F.softplus(self.spatial_decay_banks)
            # One gate per spatial-temporal token.  Exact token-varying decay
            # inside the recurrence would require replacing the CUDA forward
            # and backward kernels; mixing several valid recurrent states gives
            # each pixel an adaptive effective memory without touching CUDA.
            bank_outputs = torch.stack([
                RUN_CUDA(bank / T, self.spatial_first / T, k, v)
                for bank in decay_banks
            ], dim=1)
            if self.edge_adaptive_decay:
                feature_map = x.transpose(1, 2).reshape(B, C, H, W)
                edge = spatial_grad_mag_sobel(feature_map).flatten(2).transpose(1, 2)
                edge = (edge - edge.mean(1, keepdim=True)) / (
                    edge.std(1, keepdim=True, unbiased=False) + 1e-4)
                if self.edge_adaptive_detach:
                    edge = edge.detach()
                selector = torch.tanh(edge)
                amplitude = 0.25 * torch.tanh(
                    self.spatial_decay_edge_scale)
                if amplitude.ndim == 1:
                    amplitude = amplitude.view(1, 1, -1)
                middle = bank_outputs[:, 1]
                short_weight = 0.5 * (1.0 + selector)
                long_weight = 1.0 - short_weight
                xwkv = (middle + amplitude * (
                    short_weight * (bank_outputs[:, 0] - middle)
                    + long_weight * (bank_outputs[:, 2] - middle)))
            elif self.adaptive_decay_residual:
                correction = 0.25 * torch.tanh(self.spatial_decay_gate(x))
                middle = bank_outputs[:, 1]
                xwkv = (middle
                        + correction[:, :, 0, None]
                        * (bank_outputs[:, 0] - middle)
                        + correction[:, :, 1, None]
                        * (bank_outputs[:, 2] - middle))
            else:
                bank_weight = F.softmax(
                    self.spatial_decay_gate(x), dim=2)
                xwkv = (bank_outputs
                        * bank_weight.permute(0, 2, 1)[:, :, :, None]).sum(dim=1)
        elif hasattr(self, 'spatial_decay_global'):
            shared_decay = F.softplus(self.spatial_decay_global)
            spatial_decay = (shared_decay
                + self.spatial_decay_residual_bound
                * torch.tanh(self.spatial_decay_offset)).clamp_min(0.05)
            xwkv = RUN_CUDA(
                spatial_decay / T, self.spatial_first / T, k, v)
        elif self.decay_parameterization == 'softplus':
            spatial_decay = F.softplus(self.spatial_decay)
            xwkv = RUN_CUDA(
                spatial_decay / T, self.spatial_first / T, k, v)
        else:
            spatial_decay = self.spatial_decay
            xwkv = RUN_CUDA(
                spatial_decay / T, self.spatial_first / T, k, v)
        #print(sr.shape)
        #print(xwkv.shape)
        #xwkv = xwkv.contiguous()
        #sr = sr.contiguous()
        #xwkv = sr[:, None, :] * xwkv
        xwkv = sr * xwkv  # [B,T,C]


        if self.key_norm is not None:
            xwkv = self.key_norm(xwkv)


        out = self.output(xwkv)  # [B,T,C]
        return out


class ReceptanceLinearTransform(BaseModule):
    def __init__(self, n_embd):
        super().__init__()
        self.n_embd = n_embd
        attn_sz = n_embd
        self.receptance = nn.Linear(n_embd, attn_sz, bias=False)
        self.output = nn.Linear(attn_sz, n_embd, bias=False)
        self.receptance.scale_init = 1
        self.output.scale_init = 1

    def forward(self, x):
        sr = torch.sigmoid(self.receptance(x))
        x = sr * x
        x = self.output(x)
        return x


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    def forward(self, x):
        # x: [..., dim]
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        x = x / rms
        return x * self.weight


class GroupLinear(nn.Module):
    def __init__(self, in_c: int, out_c: int, groups: int = 4, bias: bool = False):
        super().__init__()
        assert in_c % groups == 0 and out_c % groups == 0, \
            f"in_c={in_c}, out_c={out_c}, groups={groups} not divisible"
        self.groups = groups
        self.weight = nn.Parameter(torch.randn(groups, out_c // groups, in_c // groups))
        self.bias = nn.Parameter(torch.zeros(groups, out_c // groups)) if bias else None
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,T,C]
        B, T, C = x.shape
        G = self.groups
        Ci = C // G
        xg = x.view(B, T, G, Ci)                               # [B,T,G,Ci]
        yg = torch.einsum('btgc,goc->btgo', xg, self.weight)   # [B,T,G,Co]
        if self.bias is not None:
            yg = yg + self.bias
        y = yg.reshape(B, T, -1)                               # [B,T,out_c]
        return y


class BandDropout(nn.Module):
    def __init__(self, p: float = 0.0, contiguous: bool = False, min_block: int = 4):
        super().__init__()
        self.p = float(p)
        self.contiguous = contiguous
        self.min_block = int(min_block)
    def forward(self, x):
        # x: [B,T,C]
        if not self.training or self.p <= 0.0:
            return x
        B, T, C = x.shape
        device = x.device
        mask = torch.ones(C, device=device, dtype=x.dtype)
        drop_count = int(C * self.p)
        if drop_count <= 0:
            return x
        if self.contiguous:
            width = max(self.min_block, drop_count)
            width = min(width, C)
            start = torch.randint(0, C - width + 1, (1,), device=device).item()
            mask[start:start+width] = 0.0
        else:
            idx = torch.randperm(C, device=device)[:drop_count]
            mask[idx] = 0.0
        mask = mask.view(1, 1, C) # [1,1,C]
        return x * mask


def channel_shuffle(x: torch.Tensor, groups: int) -> torch.Tensor:
    # x: [B,T,C]
    if groups <= 1:
        return x
    B, T, C = x.shape
    assert C % groups == 0, f"C={C} not divisible by groups={groups}"
    x = x.view(B, T, groups, C // groups)           # [B,T,G,Cg]
    x = x.transpose(2, 3).contiguous()               # [B,T,Cg,G]
    x = x.view(B, T, C)                              # [B,T,C]
    return x


class VRWKV_ChannelMix(nn.Module):

    def __init__(self,
                 n_embd: int,
                 n_layer: int,
                 layer_id: int,
                 shift_mode: str = 'q_shift',
                 channel_gamma: float = 1/4,
                 shift_pixel = [0],
                 hidden_rate: int = 4,
                 init_mode: str = 'fancy',
                 wavelength: Optional[Sequence[float]] = None,
                 whiten: bool = True,
                 eca_kernel: int = 1,
                 gate_temp: float = 1.0,
                 key_norm=None):
        super().__init__()
        self.layer_id = layer_id
        self.n_layer = n_layer
        self.n_embd = n_embd

        self.shift_pixel = shift_pixel
        self.channel_gamma = channel_gamma
        if isinstance(shift_mode, str):
            self.shift_func = eval(shift_mode)
        else:
            self.shift_func = shift_mode

        self.has_wavelength = (wavelength is not None)
        if self.has_wavelength:
            wl = torch.tensor(wavelength, dtype=torch.float32) if not torch.is_tensor(wavelength) \
                 else wavelength.float()
            wl = wl.view(-1)
            if wl.numel() > 1:
                wl_n = (wl - wl.min()) / (wl.max() - wl.min() + 1e-8)  # [0,1]
                wl_n = wl_n * 2.0 - 1.0                                # [-1,1]
            else:
                wl_n = wl.clone()
            if wl_n.numel() != n_embd:
                wl_n = self._interp_1d_to_len(wl_n, n_embd)            # [C]
            self.register_buffer('wavelength_norm', wl_n.view(1, 1, n_embd))  # [1,1,C]
            self.wl_scale = nn.Parameter(torch.tensor(1.0))
        else:
            self.band_embed = nn.Parameter(torch.zeros(1, 1, n_embd))

        self.whiten = nn.Linear(n_embd, n_embd, bias=False) if whiten else None
        if self.whiten is not None:
            with torch.no_grad():
                nn.init.eye_(self.whiten.weight)

        self.eca_kernel = int(eca_kernel) if eca_kernel % 2 == 1 else int(eca_kernel) + 1
        self.eca = nn.Conv1d(1, 1, kernel_size=self.eca_kernel,
                             padding=self.eca_kernel // 2, bias=False)

        hidden_sz = hidden_rate * n_embd
        self.key   = nn.Linear(n_embd, hidden_sz, bias=False)
        self.value = nn.Linear(hidden_sz, n_embd, bias=False)
        self.value.scale_init = 0

        self.receptance = nn.Linear(n_embd, n_embd, bias=False)
        self.receptance.scale_init = 0
        self.gate_temp = float(gate_temp)

        self._init_weights(init_mode)

    # ---------- utils ----------
    @staticmethod
    def _interp_1d_to_len(vec: torch.Tensor, target_len: int) -> torch.Tensor:
        L = vec.numel()
        if L == target_len:
            return vec
        src = vec.view(1, 1, L)
        out = F.interpolate(src, size=target_len, mode='linear', align_corners=True)
        return out.view(-1)

    def _apply_band_embed(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,T,C]
        if self.has_wavelength:
            return x + self.wl_scale * self.wavelength_norm
        else:
            return x + self.band_embed

    def _init_weights(self, init_mode: str):
        if init_mode == 'fancy':
            with torch.no_grad():
                self.spatial_mix_k = nn.Parameter(torch.ones([1, 1, self.n_embd]))
                self.spatial_mix_r = nn.Parameter(torch.ones([1, 1, self.n_embd]))
        else:
            raise NotImplementedError(f"init_mode={init_mode} not supported")

    def regularization_loss(self) -> torch.Tensor:
        if self.whiten is None:
            return torch.tensor(0., device=next(self.parameters()).device)
        W = self.whiten.weight  # [C, C]
        I = torch.eye(W.shape[0], device=W.device, dtype=W.dtype)
        reg = torch.norm(W.t() @ W - I, p='fro')  # Frobenius 范数
        return reg

    # ---------- forward ----------
    def forward(self, x: torch.Tensor, patch_resolution=None):
        """
        x: [B,T,C], T=H*W
        """
        B, T, C = x.shape

        x = self._apply_band_embed(x)

        if self.whiten is not None:
            x = self.whiten(x)  # [B,T,C]

        x = x  # [B,T,C]

        xk = x
        xr = x
        if isinstance(self.shift_pixel, (list, tuple)):
            for sp in self.shift_pixel:
                if sp > 0 and self.shift_func is not None:
                    xx = self.shift_func(x, sp, self.channel_gamma, patch_resolution)  # [B,T,C]
                    xk = xk + xx * (1.0 - self.spatial_mix_k)
                    xr = xr + xx * (1.0 - self.spatial_mix_r)

        k  = self.key(xk)                 # [B,T,hidden]
        k  = torch.square(F.relu(k))      # ReLU^2
        kv = self.value(k)                # [B,T,C]

        gate = torch.sigmoid(self.receptance(xr) / max(self.gate_temp, 1e-6))  # [B,T,C]
        out  = gate * kv
        return out


class Block(BaseModule):
    def __init__(self, n_embd, n_layer, layer_id, shift_mode='q_shift',
                 channel_gamma=1/4, shift_pixel=1, drop_path=0., hidden_rate=4,
                 init_mode='fancy', init_values=None, post_norm=False, key_norm=False,
                 with_cp=False,use_channal=False):
        super().__init__()
        self.layer_id = layer_id
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.ln3 = nn.LayerNorm(n_embd)
        self.ln4 = nn.LayerNorm(n_embd)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        if self.layer_id == 0:
            self.ln0 = nn.LayerNorm(n_embd) # nn.LayerNorm(n_embd)
        self.att = VRWKV_SpatialMix(n_embd, n_layer, layer_id, shift_mode,
                                   channel_gamma, [0], init_mode,
                                   key_norm=key_norm)
        # self.att = ReceptanceLinearTransform(n_embd)
        self.ffn = VRWKV_ChannelMix(n_embd, n_layer, layer_id, shift_mode,
                                   channel_gamma, [0], hidden_rate,
                                   init_mode, key_norm=key_norm)
        self.layer_scale = (init_values is not None)
        self.post_norm = post_norm
        # if self.layer_scale:
        self.gamma1 = nn.Parameter(torch.ones((n_embd)), requires_grad=True)
        self.gamma2 = nn.Parameter(torch.ones((n_embd)), requires_grad=True)
        self.with_cp = with_cp
        self.use_channal = use_channal

    def forward(self, x, patch_resolution=None):
        b, c, h, w = x.shape
        patch_resolution = (h, w)

        def _inner_forward(x):
            if self.layer_id == 0:
                x = rearrange(x, 'b c h w -> b (h w) c')
                x = self.ln0(x)
                x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
            x = rearrange(x, 'b c h w -> b (h w) c')
            if self.use_channal:
                x_cm = x + self.att(self.ln1(x), patch_resolution)
                x_sm = x + self.ffn(self.ln2(x + x_cm), patch_resolution)
                # x = self.gamma1 * self.ln3(x_sm) + self.gamma2 * self.ln4(x_cm)
                x = rearrange(x_sm, 'b (h w) c -> b c h w', h=h, w=w)
            else:
                x_cm = x + self.att(self.ln1(x), patch_resolution)
                #x_sm = x + self.ffn(self.ln2(x + x_cm), patch_resolution)
                # x = self.gamma1 * self.ln3(x_sm) + self.gamma2 * self.ln4(x_cm)
                x = rearrange(x_cm, 'b (h w) c -> b c h w', h=h, w=w)
            return x
        if self.with_cp and x.requires_grad:
            x = cp.checkpoint(_inner_forward, x)
        else:
            x = _inner_forward(x)
        return x

class HSI_RWKV(nn.Module):
    def __init__(self,
        dim = 128,
        num_blocks = [1],
        use_channal = False
    ):
        super(HSI_RWKV, self).__init__()
        self.encoder_level1 = nn.Sequential(*[Block(n_embd=dim, n_layer=num_blocks[0], layer_id=i,use_channal=use_channal) for i in range(num_blocks[0])])
        # self.relu = nn.ReLU()
    def forward(self, inp_img):
        out_enc_level1 = self.encoder_level1(inp_img)
        # x = rearrange(inp_img, 'b c h w -> b c w h')
        # out_enc_level1 = self.encoder_level1(x).permute(0, 1, 3, 2) + out_enc_level1
        return out_enc_level1
