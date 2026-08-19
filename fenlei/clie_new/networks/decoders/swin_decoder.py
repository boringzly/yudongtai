import torch.nn as nn
import torch
import torch.nn.functional as F
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import numpy as np
from networks.backbone.transformers import SwinBlock
import einops


class PatchExpanding(nn.Module):
    """ Patch Expanding Layer
    Args:
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """
    def __init__(self, dim, out_dim, norm_layer=nn.LayerNorm, ratio=2, factor=2, n_divs=1):
        assert dim % n_divs == 0, f'dim [{dim}] is not divisible by n_divs [{n_divs}]'
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.ratio = ratio
        self.factor = factor
        Linear = nn.Linear
        self.expansion = Linear(dim, self.factor * self.ratio * out_dim, bias=False, **{'n_divs': n_divs for n in [n_divs] if n > 1})
        self.norm = norm_layer(out_dim*self.factor*self.ratio)
        # self.concat = Concatenate(dim=-1, n_divs=n_divs)
        self.H, self.W = None, None

    def forward(self, x):  # , H, W):
        """ Forward function.
        Args:
            x: Input feature, tensor size (B, H*W, C).
            H, W: Spatial resolution of the input feature.
        """
        H, W = self.H, self.W

        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        x = self.expansion(x)
        x = self.norm(x)

        # padding
        # pad_input = (H % self.ratio == 1) or (W % self.ratio == 1)
        # if pad_input:
        #     x = F.pad(x, (0, 0, 0, W % self.ratio, 0, H % self.ratio))
        x = x.view(B, H*self.ratio, W*self.ratio, self.out_dim//self.ratio * self.factor)

        x = einops.rearrange(x, 'b h w (p2 p1 c) -> b (h p1) (w p2) c', p1=self.ratio, p2=self.ratio)

        x = x.view(B, -1, self.out_dim//self.ratio * self.factor)  # B H*2*W*2 C/2

        return x


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


class DecodeLayerUnet(nn.Module):
    """ A basic Swin Transformer layer for one stage.
    Args:
        dim (int): Number of feature channels
        depth (int): Depths of this stage.
        num_heads (int): Number of attention head.
        window_size (int): Local window size. Default: 7.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
    """

    def __init__(self,
                 dim,
                 pre_dim,
                 depth,
                 num_heads,
                 window_size=8,
                 mlp_ratio=4,
                 qkv_bias=False,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 norm_layer=nn.LayerNorm,
                 upsample=None,
                 use_checkpoint=False,
                 n_divs=1,
                 merge_type='concat'):
        super().__init__()
        assert merge_type in ['concat', 'add']

        assert dim % n_divs == 0, f'dim [{dim}] is not divisible by n_divs [{n_divs}]'
        self.window_size = window_size
        self.shift_size = window_size // 2
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # patch merging layer
        self.upsample = upsample
        if self.upsample is not None:
            self.upsample = upsample(dim=pre_dim, out_dim=dim, norm_layer=norm_layer, n_divs=n_divs)

        self.merge_type = merge_type
        if self.merge_type == 'concat':
            self.concat = torch.cat
            Linear = nn.Linear
            self.fc = Linear(dim*2, dim, **{'n_divs': n_divs for n in [n_divs] if n > 1})
            # self.merge = nn.Sequential(concat, fc)

        # print(depth)
        # build blocks
        self.blocks = nn.ModuleList([
            # SwinTransformerBlock(
            #     dim=dim,
            #     num_heads=num_heads,
            #     window_size=window_size,
            #     shift_size=0 if (i % 2 == 0) else self.shift_size,
            #     mlp_ratio=mlp_ratio,
            #     qkv_bias=qkv_bias,
            #     qk_scale=qk_scale,
            #     drop=drop,
            #     attn_drop=attn_drop,
            #     drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
            #     norm_layer=norm_layer,
            #     n_divs=n_divs,
            #     use_checkpoint=use_checkpoint
            # )
            SwinBlock(
                embed_dims=dim,
                num_heads=num_heads,
                feedforward_channels=mlp_ratio*dim,
                window_size=window_size,
                shift=False if i % 2 == 0 else True,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop_rate=drop,
                attn_drop_rate=attn_drop,
                drop_path_rate=drop_path[i] if isinstance(drop_path, list) else drop_path,
                act_cfg={'type': 'gelu', 'opts': {}},
                norm_cfg={'type': 'layernorm', 'opts': {"data_format": "channels_last"}},
            )
            for i in range(depth)])
        # self.H, self.W, self.H1, self.W1 = None, None, None, None

    def forward(self, x, y):
        """ Forward function.
        Args:
            x: Input feature, tensor size (B, H*W, C).
            y: Input feature, tensor size (B, H1*W1, C).
        """
        if len(x.shape) == 4:
            H, W = (x.shape[2], x.shape[3])
            x = x.flatten(2).transpose(2,1)
        else:
            H, W = (int(x.shape[1]**0.5), int(x.shape[1]**0.5))
        
        H1, W1 = (y.shape[2], y.shape[3])
        y = y.flatten(2).transpose(2,1)
        B, _, _ = x.shape
        # print(H, W, H1, W1)
        if self.upsample is not None:
            self.upsample.H, self.upsample.W = H, W
            # x = self.upsample(x, H, W)
            x = self.upsample(x)
            H, W = H * 2, W * 2

        if H1 != H or W1 != W:
            x = x.view(B, H, W, -1)
            x = x[:, :H1, :W1, :].contiguous()
            x = x.view(B, H1 * W1, -1)
            H, W = H1, W1

        # calculate attention mask for SW-MSA
        Hp = int(np.ceil(H / self.window_size)) * self.window_size
        Wp = int(np.ceil(W / self.window_size)) * self.window_size
        img_mask = torch.zeros((1, Hp, Wp, 1), device=x.device)  # 1 Hp Wp 1
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

        if self.merge_type == 'concat':
            x = self.concat([x, y], 2)
            x = self.fc(x)
        else:
            x = x + y
        for i, blk in enumerate(self.blocks):
            blk.H, blk.W = H, W
            x = blk(x, (H, W))
        self.H, self.W = H, W
        return x

class Head(nn.Module):
    def __init__(self, decode_dim=96, patch_size=4, out_chans=None, n_divs=1):
        super().__init__()
        assert decode_dim % n_divs == 0, f'dim [{decode_dim}] is not divisible by n_divs [{n_divs}]'
        self.expand = PatchExpanding(decode_dim, decode_dim, ratio=patch_size, factor=patch_size, n_divs=n_divs)
        self.final = nn.Linear(in_features=decode_dim, out_features=out_chans if out_chans else decode_dim)
        self.out_chans = out_chans if out_chans else decode_dim
        self.patch_size = patch_size

    def forward(self, x, target_size):  # , H, W):
        B, H, W = (x.shape[0], int(x.shape[1]**0.5), int(x.shape[1]**0.5))
        H1, W1 = target_size
        self.expand.H, self.expand.W = H, W
        # x = self.expand(x, H, W)
        x = self.expand(x)
        H, W = H * self.patch_size, W * self.patch_size
        if H1 != H or W1 != W:
            x = x.view(B, H, W, -1)
            x = x[:, :H1, :W1, :].contiguous()
            x = x.view(B, H1 * W1, -1)
        # print(x.shape)
        x = self.final(x)
        return x.view(B, H1, W1, -1).permute(0, 3, 1, 2)


class SwinDecoder(nn.Module):
    def __init__(self, in_channels, num_class, dropout_ratio=0.1, **kwargs):
        super(SwinDecoder, self).__init__()
        self.num_class = num_class
        num_heads = (4, 4, 4, 4)
        depths = (2, 2, 2, 2)
        decode_depths = (2, 2, 2, 2)
        drop_path_rate=0.1
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        self.layers_up = nn.ModuleList([])
        for i_layer in range(len(in_channels)):
            drop_path = dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])]
            self.layers_up.append(DecodeLayerUnet(
                dim=in_channels[i_layer],
                pre_dim=in_channels[i_layer-1],
                depth=decode_depths[i_layer],  # 1,  #depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=8,
                mlp_ratio=4,
                qkv_bias=True,
                qk_scale=None,
                drop=0,
                attn_drop=0,
                drop_path=drop_path,
                norm_layer=nn.LayerNorm,
                upsample=PatchExpanding if i_layer != 0 else None,
                use_checkpoint=False,
                n_divs=1,
                merge_type='concat'
            ))

        # self.head = Head(decode_dim=in_channels[-1], patch_size=4, out_chans=num_class, n_divs=1)
        self.head = nn.Sequential(
                nn.Conv2d(in_channels[-1], 16, kernel_size=1, stride=1, padding=0),
                nn.ReLU(inplace=True),
                torch.nn.Conv2d(16, num_class, kernel_size=3, stride=1, padding=1),
            )

    def forward(self, x, features, target_size):
        for i in range(len(self.layers_up)):
            x = self.layers_up[i](x, features[len(self.layers_up)-i-1])
        B, H, W = (x.shape[0], int(x.shape[1]**0.5), int(x.shape[1]**0.5))
        x = x.view(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        x = F.interpolate(x, size=target_size, mode="bilinear", align_corners=True)
        x = self.head(x)

        return x

__all__ = [
    "SwinDecoder",
]