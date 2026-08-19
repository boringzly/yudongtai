#!/usr/bin/env python3

# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


import torch
import torch.nn as nn
import math
from timm.models.registry import register_model
import torch.nn.functional as F
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
from einops import rearrange, repeat
from functools import lru_cache
from typing import Optional, Callable, Any
from timm.models.layers import DropPath, trunc_normal_



from hilbertcurve.hilbertcurve import HilbertCurve


# # --- 1. 生成扫描索引函数们 ---

def snake_scan_indices(h, w, device):
    idxs = []
    for i in range(h):
        row = list(range(w)) if i % 2 == 0 else list(range(w - 1, -1, -1))
        idxs.extend([(i, j) for j in row])
    return torch.tensor(idxs, device=device)

def diagonal_scan_indicesd(h, w, device):
    rows = torch.arange(h, device=device).unsqueeze(1)   # shape (h,1)，表示行坐标
    cols = torch.arange(w, device=device).unsqueeze(0)   # shape (1,w)，表示列坐标
    diag_sums = rows + cols                             # shape (h,w)，对角线编号矩阵

    row_indices = rows.expand(h, w).reshape(-1)         # 所有像素的行索引，长度 h*w
    col_indices = cols.expand(h, w).reshape(-1)         # 所有像素的列索引，长度 h*w
    diag_sums_flat = diag_sums.reshape(-1)              # 每个像素所在的对角线编号

    sorted_indices = torch.argsort(diag_sums_flat)      # 根据对角线编号排序

    sorted_row_indices = row_indices[sorted_indices]    # 排序后的行索引
    sorted_col_indices = col_indices[sorted_indices]    # 排序后的列索引

    return torch.stack([sorted_row_indices, sorted_col_indices], dim=1)


def zorder_scan_indices(h, w, device):
    """
    生成 Morton Z-order (Z曲线扫描) 的索引 (h*w, 2)
    向量化版本，避免 Python for 循环
    """
    def part1by1(n):
        n &= 0xFFFF
        n = (n | (n << 8)) & 0x00FF00FF
        n = (n | (n << 4)) & 0x0F0F0F0F
        n = (n | (n << 2)) & 0x33333333
        n = (n | (n << 1)) & 0x55555555
        return n

    def morton(x, y):
        return part1by1(x) | (part1by1(y) << 1)

    # 生成所有行列索引 (向量化)
    rows = np.repeat(np.arange(h, dtype=np.int32), w)
    cols = np.tile(np.arange(w, dtype=np.int32), h)

    # 计算 Morton 编码
    morton_codes = morton(rows, cols)

    # 按 Morton 编码排序
    order = np.argsort(morton_codes)
    coords = np.stack([rows[order], cols[order]], axis=1)

    return torch.tensor(coords, device=device)

def zigzag_scan_indices(h, w, device):
    """
    生成 Zigzag 扫描顺序的索引 (h*w, 2)
    """
    order = []
    for s in range(h + w - 1):
        line = []
        for i in range(max(0, s - w + 1), min(h, s + 1)):
            j = s - i
            if j < w:
                line.append((i, j))
        # 奇偶对角线方向不同
        if s % 2 == 0:
            order.extend(line[::-1])
        else:
            order.extend(line)
    
    return torch.tensor(order, device=device)

def alternate_snake_scan_indices(h, w, device):
    idxs = []
    for j in range(w):
        col = list(range(h)) if j % 2 == 0 else list(range(h - 1, -1, -1))
        idxs.extend([(i, j) for i in col])
    return torch.tensor(idxs, device=device)

def hilbert_scan_indices(h, w, device):
    n_bits = max(h, w).bit_length()
    hilbert = HilbertCurve(n_bits, 2)
    coords = []
    for d in range(h * w * 2):  # 加倍防止截断
        x, y = hilbert.point_from_distance(d)
        if x < h and y < w:
            coords.append((x, y))
        if len(coords) >= h * w:
            break
    return torch.tensor(coords, device=device)


# --- 2. 扫描应用函数 ---

def apply_scan_order(x, idxs, W):
    B, C, L = x.shape
    x_flat = x.contiguous()  # (B, C, H*W)
    linear_idxs = idxs[:, 0] * W + idxs[:, 1]
    reordered = x_flat[:, :, linear_idxs]  # (B, C, N)
    return reordered  # (B, N, C)


# --- 3. 向量化逆变换函数 ---

def inverse_scan_order_vectorized(x_seq, idxs, H, W):
    B, N, C = x_seq.shape
    device = x_seq.device

    linear_idx = idxs[:, 0] * W + idxs[:, 1]  # (N,)
    x_img_flat = torch.zeros(B, N, C, dtype=x_seq.dtype, device=device)

    linear_idx_expanded = linear_idx.view(1, N, 1).expand(B, N, C)
    x_img_flat.scatter_(dim=1, index=linear_idx_expanded, src=x_seq)

    return x_img_flat


class LayerNorm2d(nn.LayerNorm):
    def forward(self, x: torch.Tensor):
        x = x.permute(0, 2, 3, 1)
        x = nn.functional.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        x = x.permute(0, 3, 1, 2)
        return x


class MambaVision(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=4,
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        conv_bias=True,
        bias=False,
        use_fast_path=True, 
        layer_idx=None,
        device=None,
        dtype=None,
        scan_type=None,
        crossscan_version="v1",
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.scan_type = scan_type
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.use_fast_path = use_fast_path
        self.layer_idx = layer_idx
        self.in_proj = nn.Linear(self.d_model, self.d_inner, bias=bias, **factory_kwargs)    
        self.x_proj = nn.Linear(
            self.d_inner//2, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
        )
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner//2, bias=True, **factory_kwargs)
        dt_init_std = self.dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError
        dt = torch.exp(
            torch.rand(self.d_inner//2, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        self.dt_proj.bias._no_reinit = True
        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=self.d_inner//2,
        ).contiguous()
        A_log = torch.log(A)
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True
        self.D = nn.Parameter(torch.ones(self.d_inner//2, device=device))
        self.D._no_weight_decay = True
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.conv1d_x = nn.Conv1d(
            in_channels=self.d_inner//2,
            out_channels=self.d_inner//2,
            bias=conv_bias//2,
            kernel_size=d_conv,
            groups=self.d_inner//2,
            **factory_kwargs,
        )
        self.conv1d_z = nn.Conv1d(
            in_channels=self.d_inner//2,
            out_channels=self.d_inner//2,
            bias=conv_bias//2,
            kernel_size=d_conv,
            groups=self.d_inner//2,
            **factory_kwargs,
        )

    def forward(self, hidden_states, mask):
        """
        hidden_states: (B, L, D)
        Returns: same shape as hidden_states
        """
        bs, seqlen, chn = hidden_states.shape
        H, W = int(seqlen**0.5), int(seqlen**0.5)
        xz = self.in_proj(hidden_states)
        
        # 使用新的扫描顺序
        if self.scan_type != None: # use new scan mode
            idxs = scan_funcs[self.scan_type](H, W, hidden_states.device)
            xz = apply_scan_order(xz.transpose(1,2), idxs, W).transpose(1,2)        # 扫描
        xz = rearrange(xz, "b l d -> b d l")
        x, z = xz.chunk(2, dim=1)  #将张量按通道划分成两块
        A = -torch.exp(self.A_log.float())
        x = F.silu(F.conv1d(input=x, weight=self.conv1d_x.weight, bias=self.conv1d_x.bias, padding='same', groups=self.d_inner//2))
        z = F.silu(F.conv1d(input=z, weight=self.conv1d_z.weight, bias=self.conv1d_z.bias, padding='same', groups=self.d_inner//2))
        x_dbl = self.x_proj(rearrange(x, "b d l -> (b l) d"))
        dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = rearrange(self.dt_proj(dt), "(b l) d -> b d l", l=seqlen)
        B = rearrange(B, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
        C = rearrange(C, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
        y = selective_scan_fn(x, 
                              dt, 
                              A, 
                              B, 
                              C, 
                              self.D.float(), 
                              z=None, 
                              delta_bias=self.dt_proj.bias.float(), 
                              delta_softplus=True, 
                              return_last_state=None)
        
        y = torch.cat([y, z], dim=1)
        y = rearrange(y, "b d l -> b l d")
        if self.scan_type != None:
            y = inverse_scan_order_vectorized(y, idxs, H, W)  # 逆变换
        out = self.out_proj(y)
        return out
    

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.,channels_first=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        Linear = Linear2d if channels_first else nn.Linear
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class gMlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.,channels_first=False):
        super().__init__()
        self.channel_first = channels_first
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        Linear = Linear2d if channels_first else nn.Linear
        self.fc1 = Linear(in_features, 2 * hidden_features)
        self.act = act_layer()
        self.fc2 = Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor):
        x = self.fc1(x)
        x, z = x.chunk(2, dim=(1 if self.channel_first else -1))
        x = self.fc2(x * self.act(z))
        x = self.drop(x)
        return x

    
class VSSBlockMultiScan(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 0,
        drop_path: float = 0,
        norm_layer: nn.Module = nn.LayerNorm,
        channel_first=False,
        # =============================
        ssm_d_state: int = 16,
        ssm_ratio=2.0,
        ssm_dt_rank: Any = "auto",
        ssm_act_layer=nn.SiLU,
        ssm_conv: int = 3,
        ssm_conv_bias=True,
        ssm_drop_rate: float = 0,
        ssm_init="v0",
        forward_type="v2",
        scan_type=None,
        crossscan_version="v1",
        # =============================
        mlp_ratio=4.0,
        mlp_act_layer=nn.GELU,
        mlp_drop_rate: float = 0.0,
        gmlp=False,
        # =============================
        use_checkpoint: bool = False,
        post_norm: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.ssm_branch = ssm_ratio > 0
        self.mlp_branch = mlp_ratio > 0
        self.use_checkpoint = use_checkpoint
        self.post_norm = post_norm
        if self.ssm_branch:
            self.norm = nn.LayerNorm(hidden_dim)
            self.op = MambaVision(hidden_dim, scan_type=scan_type, crossscan_version=crossscan_version)
        
        self.drop_path = DropPath(drop_path)
        
        if self.mlp_branch:
            _MLP = Mlp if not gmlp else gMlp
            self.norm2 = nn.LayerNorm(hidden_dim)
            mlp_hidden_dim = int(hidden_dim * mlp_ratio)
            self.mlp = _MLP(in_features=hidden_dim, hidden_features=mlp_hidden_dim, act_layer=mlp_act_layer, drop=mlp_drop_rate, channels_first=channel_first)

    def _forward(self, input: torch.Tensor, mask: torch.Tensor, crossscan_version: str):
        if self.ssm_branch:
            if self.post_norm:
                x = input + self.drop_path(self.norm(self.op(input, mask)))
            else:
                x = input + self.drop_path(self.op(self.norm(input), mask))
        if self.mlp_branch:
            if self.post_norm:
                x = x + self.drop_path(self.norm2(self.mlp(x))) # FFN
            else:
                x = x + self.drop_path(self.mlp(self.norm2(x))) # FFN
        return x

    def forward(self, input: torch.Tensor, mask: torch.Tensor, crossscan_version: str='v1'):
        if self.use_checkpoint:
            return checkpoint.checkpoint(self._forward, input)
        else:
            return self._forward(input, mask, crossscan_version)


class Attention(nn.Module):

    def __init__(
            self,
            dim,
            num_heads=8,
            qkv_bias=False,
            qk_norm=False,
            attn_drop=0.,
            proj_drop=0.,
            norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = True

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.fused_attn:
            x = F.scaled_dot_product_attention(
             q, k, v,
                dropout_p=self.attn_drop.p,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
