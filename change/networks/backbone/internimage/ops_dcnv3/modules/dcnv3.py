# --------------------------------------------------------
# InternImage
# Copyright (c) 2022 OpenGVLab
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import warnings
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, constant_
from ..functions import DCNv3Function, dcnv3_core_pytorch
from networks.utils.bricks import BuildNormalization


class to_channels_first(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.permute(0, 3, 1, 2)


class to_channels_last(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.permute(0, 2, 3, 1)


def build_norm_layer(dim,
                     norm_layer,
                     in_format='channels_last',
                     out_format='channels_last',
                     eps=1e-6):
    layers = []
    if norm_layer == 'BN' or norm_layer == 'batchnorm2d':
        if in_format == 'channels_last':
            layers.append(to_channels_first())
        layers.append(nn.BatchNorm2d(dim))
        if out_format == 'channels_last':
            layers.append(to_channels_last())
    elif norm_layer == 'GN' or norm_layer == 'groupnorm':
        if in_format == 'channels_last':
            layers.append(to_channels_first())
        layers.append(nn.GroupNorm(dim//8, dim))
        if out_format == 'channels_last':
            layers.append(to_channels_last())
    elif norm_layer == 'LN' or norm_layer == 'layernorm':
        if in_format == 'channels_first':
            layers.append(to_channels_last())
        layers.append(nn.LayerNorm(dim, eps=eps))
        if out_format == 'channels_first':
            layers.append(to_channels_first())
    else:
        raise NotImplementedError(
            f'build_norm_layer does not support {norm_layer}')
    return nn.Sequential(*layers)


def build_act_layer(act_layer):
    if act_layer == 'ReLU':
        return nn.ReLU(inplace=True)
    elif act_layer == 'SiLU':
        return nn.SiLU(inplace=True)
    elif act_layer == 'GELU':
        return nn.GELU()

    raise NotImplementedError(f'build_act_layer does not support {act_layer}')


def _is_power_of_2(n):
    if (not isinstance(n, int)) or (n < 0):
        raise ValueError(
            "invalid input for _is_power_of_2: {} (type: {})".format(n, type(n)))

    return (n & (n - 1) == 0) and n != 0


class CenterFeatureScaleModule(nn.Module):
    def forward(self,
                query,
                center_feature_scale_proj_weight,
                center_feature_scale_proj_bias):
        center_feature_scale = F.linear(query,
                                        weight=center_feature_scale_proj_weight,
                                        bias=center_feature_scale_proj_bias).sigmoid()
        return center_feature_scale


class DCNv3_pytorch(nn.Module):
    def __init__(
            self,
            channels=64,
            kernel_size=3,
            dw_kernel_size=None,
            stride=1,
            pad=1,
            dilation=1,
            group=4,
            offset_scale=1.0,
            act_layer='GELU',
            norm_layer='LN',
            center_feature_scale=False,
            remove_center=False,
    ):
        """
        DCNv3 Module
        :param channels
        :param kernel_size
        :param stride
        :param pad
        :param dilation
        :param group
        :param offset_scale
        :param act_layer
        :param norm_layer
        """
        super().__init__()
        if channels % group != 0:
            raise ValueError(
                f'channels must be divisible by group, but got {channels} and {group}')
        _d_per_group = channels // group
        dw_kernel_size = dw_kernel_size if dw_kernel_size is not None else kernel_size
        # you'd better set _d_per_group to a power of 2 which is more efficient in our CUDA implementation
        if not _is_power_of_2(_d_per_group):
            warnings.warn(
                "You'd better set channels in DCNv3 to make the dimension of each attention head a power of 2 "
                "which is more efficient in our CUDA implementation.")

        self.offset_scale = offset_scale
        self.channels = channels
        self.kernel_size = kernel_size
        self.dw_kernel_size = dw_kernel_size
        self.stride = stride
        self.dilation = dilation
        self.pad = pad
        self.group = group
        self.group_channels = channels // group
        self.offset_scale = offset_scale
        self.center_feature_scale = center_feature_scale
        self.remove_center = int(remove_center)

        self.dw_conv = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=dw_kernel_size,
                stride=1,
                padding=(dw_kernel_size - 1) // 2,
                groups=channels),
            build_norm_layer(
                channels,
                norm_layer,
                'channels_first',
                'channels_last'),
            build_act_layer(act_layer))
        self.offset = nn.Linear(
            channels,
            group * (kernel_size * kernel_size - remove_center) * 2)
        self.mask = nn.Linear(
            channels,
            group * (kernel_size * kernel_size - remove_center))
        self.input_proj = nn.Linear(channels, channels)
        self.output_proj = nn.Linear(channels, channels)
        self._reset_parameters()
        
        if center_feature_scale:
            self.center_feature_scale_proj_weight = nn.Parameter(
                torch.zeros((group, channels), dtype=torch.float))
            self.center_feature_scale_proj_bias = nn.Parameter(
                torch.tensor(0.0, dtype=torch.float).view((1,)).repeat(group, ))
            self.center_feature_scale_module = CenterFeatureScaleModule()

    def _reset_parameters(self):
        constant_(self.offset.weight.data, 0.)
        constant_(self.offset.bias.data, 0.)
        constant_(self.mask.weight.data, 0.)
        constant_(self.mask.bias.data, 0.)
        xavier_uniform_(self.input_proj.weight.data)
        constant_(self.input_proj.bias.data, 0.)
        xavier_uniform_(self.output_proj.weight.data)
        constant_(self.output_proj.bias.data, 0.)

    def forward(self, input):
        """
        :param query                       (N, H, W, C)
        :return output                     (N, H, W, C)
        """
        N, H, W, _ = input.shape

        x = self.input_proj(input)
        x_proj = x

        x1 = input.permute(0, 3, 1, 2)
        x1 = self.dw_conv(x1)
        offset = self.offset(x1)
        mask = self.mask(x1).reshape(N, H, W, self.group, -1)
        mask = F.softmax(mask, -1).reshape(N, H, W, -1)

        x = dcnv3_core_pytorch(
            x, offset, mask,
            self.kernel_size, self.kernel_size,
            self.stride, self.stride,
            self.pad, self.pad,
            self.dilation, self.dilation,
            self.group, self.group_channels,
            self.offset_scale, self.remove_center)
        if self.center_feature_scale:
            center_feature_scale = self.center_feature_scale_module(
                x1, self.center_feature_scale_proj_weight, self.center_feature_scale_proj_bias)
            # N, H, W, groups -> N, H, W, groups, 1 -> N, H, W, groups, _d_per_group -> N, H, W, channels
            center_feature_scale = center_feature_scale[..., None].repeat(
                1, 1, 1, 1, self.channels // self.group).flatten(-2)
            x = x * (1 - center_feature_scale) + x_proj * center_feature_scale
        x = self.output_proj(x)

        return x


class DCNv3(nn.Module):
    def __init__(
        self,
        channels=64,
        kernel_size=3,
        dw_kernel_size=None,
        stride=1,
        pad=1,
        dilation=1,
        group=4,
        offset_scale=1.0,
        act_layer='GELU',
        norm_layer='LN',
        center_feature_scale=False,
        remove_center=False,
        resolution_scale=1,
    ):
        """
        DCNv3 Module
        :param channels
        :param kernel_size
        :param stride
        :param pad
        :param dilation
        :param group
        :param offset_scale
        :param act_layer
        :param norm_layer
        """
        super().__init__()
        if norm_layer == "LN":
            norm_layer = 'layernorm'
        if channels % group != 0:
            raise ValueError(
                f'channels must be divisible by group, but got {channels} and {group}')
        _d_per_group = channels // group
        dw_kernel_size = dw_kernel_size if dw_kernel_size is not None else kernel_size
        # you'd better set _d_per_group to a power of 2 which is more efficient in our CUDA implementation
        if not _is_power_of_2(_d_per_group):
            warnings.warn(
                "You'd better set channels in DCNv3 to make the dimension of each attention head a power of 2 "
                "which is more efficient in our CUDA implementation.")

        self.offset_scale = offset_scale
        self.channels = channels
        self.kernel_size = kernel_size
        self.dw_kernel_size = dw_kernel_size
        self.stride = stride
        self.dilation = dilation
        self.pad = pad
        self.group = group
        self.group_channels = channels // group
        self.center_feature_scale = center_feature_scale
        self.remove_center = int(remove_center)
        self.resolution_scale = resolution_scale

        if self.remove_center and self.kernel_size % 2 == 0:
            raise ValueError('remove_center is only compatible with odd kernel size.')

        self.dw_conv = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=dw_kernel_size,
                stride=1,
                padding=(dw_kernel_size - 1) // 2,
                groups=channels),
            build_norm_layer(
                channels,
                norm_layer,
                'channels_first',
                'channels_last'),
            build_act_layer(act_layer))
        self.offset = nn.Linear(
            channels,
            group * (kernel_size * kernel_size - remove_center) * 2)
        self.mask = nn.Linear(
            channels,
            group * (kernel_size * kernel_size - remove_center))
        self.input_proj = nn.Linear(channels, channels)
        self.output_proj = nn.Linear(channels, channels)
        self._reset_parameters()
        
        if center_feature_scale:
            self.center_feature_scale_proj_weight = nn.Parameter(
                torch.zeros((group, channels), dtype=torch.float))
            self.center_feature_scale_proj_bias = nn.Parameter(
                torch.tensor(0.0, dtype=torch.float).view((1,)).repeat(group, ))
            self.center_feature_scale_module = CenterFeatureScaleModule()

    def _reset_parameters(self):
        constant_(self.offset.weight.data, 0.)
        constant_(self.offset.bias.data, 0.)
        constant_(self.mask.weight.data, 0.)
        constant_(self.mask.bias.data, 0.)
        xavier_uniform_(self.input_proj.weight.data)
        constant_(self.input_proj.bias.data, 0.)
        xavier_uniform_(self.output_proj.weight.data)
        constant_(self.output_proj.bias.data, 0.)

    def forward(self, input):
        """
        :param query                       (N, H, W, C)
        :return output                     (N, H, W, C)
        """
        N, H, W, _ = input.shape

        x = self.input_proj(input)
        x_proj = x
        dtype = x.dtype
        x1 = input.permute(0, 3, 1, 2).contiguous()
        x1 = self.dw_conv(x1).contiguous()
        offset = self.offset(x1)
        mask = self.mask(x1).reshape(N, H, W, self.group, -1)
        mask = F.softmax(mask, -1)
        mask = mask.reshape(N, H, W, -1).type(dtype)

        # from clcore import ImageIO;io = ImageIO()
        # import cv2, os
        # import numpy as np
        
        # pos = (306, 136)
        # # offset.shape = (batch, h, w, group*kernel*kernel*2)
        # offset_vis = offset[0, pos[1]//self.resolution_scale, pos[0]//self.resolution_scale, :].cpu().numpy()
        # offset_vis = offset_vis.reshape((self.group, self.kernel_size, self.kernel_size, 2))
        # # offset_vis = offset_vis[0]
        # mask_vis = mask[0, pos[1]//self.resolution_scale, pos[0]//self.resolution_scale, :].cpu().numpy()
        # mask_vis = mask_vis.reshape((self.group, self.kernel_size, self.kernel_size))
        # # mask_vis = mask_vis[0]
        # # for i in range(aa.shape[2]):
        # #     bb = aa[:,:,i].cpu().numpy()
        # #     bb = (bb - bb.min()) / (bb.max() - bb.min()) * 255
        # #     bb = np.stack((bb, bb, bb), 2)
        # #     bb = cv2.circle(bb, pos, 1, (0, 0, 255), 1)
        # if not os.path.exists('/nfs/project/netdisk/100/data/change_data/cp/WHU/out/test2/offset/test_res_{str(self.resolution_scale)}.tif'):
        #     bb = np.ones((offset.shape[1]*self.resolution_scale, offset.shape[2]*self.resolution_scale, 3))
        # else:
        #     bb = cv2.imread('/nfs/project/netdisk/100/data/change_data/cp/WHU/out/test2/offset/test_res_{str(self.resolution_scale)}.tif', -1)
        # for j in range(self.kernel_size): # H
        #     for k in range(self.kernel_size): # W
        #         for m in range(self.group): # group
        #             new_pos = (pos[0] + int(((k - (self.kernel_size - 1) // 2) * self.dilation + offset_vis[m,k,j,0]) * self.resolution_scale), \
        #                 pos[1] + int(((j - (self.kernel_size - 1) // 2) * self.dilation + offset_vis[m,k,j,1]) * self.resolution_scale))
        #             # if self.resolution_scale <= 8:
        #             bb = cv2.circle(bb, new_pos, 1, (int(255.0*(0.75-mask_vis[m,k,j])), 0, int(255.0*(0.25+mask_vis[m,k,j]))), self.resolution_scale//4)
        #             # else:
        #             #     bb = cv2.circle(bb, new_pos, 1, (int(255.0*(0.7-mask_vis[m,j,k])), 0, int(255.0*(0.3+mask_vis[m,j,k]))), 2)
        # bb = cv2.circle(bb, pos, 1, (0, 255, 0), self.resolution_scale//4)            
        # cv2.imwrite(f'/nfs/project/netdisk/100/data/change_data/cp/WHU/out/test2/offset/test_res_{str(self.resolution_scale)}.tif', bb.astype(np.uint8))

        
        x = DCNv3Function.apply(
            x, offset, mask,
            self.kernel_size, self.kernel_size,
            self.stride, self.stride,
            self.pad, self.pad,
            self.dilation, self.dilation,
            self.group, self.group_channels,
            self.offset_scale,
            256,
            self.remove_center)
        
        if self.center_feature_scale:
            center_feature_scale = self.center_feature_scale_module(
                x1, self.center_feature_scale_proj_weight, self.center_feature_scale_proj_bias)
            # N, H, W, groups -> N, H, W, groups, 1 -> N, H, W, groups, _d_per_group -> N, H, W, channels
            center_feature_scale = center_feature_scale[..., None].repeat(
                1, 1, 1, 1, self.channels // self.group).flatten(-2)
            x = x * (1 - center_feature_scale) + x_proj * center_feature_scale
        x = self.output_proj(x)

        return x


class DCNrs(DCNv3):
    def __init__(
        self,
        channels=64,
        kernel_size=3,
        dw_kernel_size=None,
        stride=1,
        pad=1,
        dilation=1,
        group=4,
        offset_scale=1.0,
        act_layer='GELU',
        norm_layer='LN',
        center_feature_scale=False,
        remove_center=False,
        resolution_scale=1,
    ):
        """
        DCNrs Module
        :param channels
        :param kernel_size
        :param stride
        :param pad
        :param dilation
        :param group
        :param offset_scale
        :param act_layer
        :param norm_layer
        """
        super(DCNrs, self).__init__(channels, kernel_size, dw_kernel_size, stride, pad, dilation, \
            group, offset_scale, act_layer, norm_layer, center_feature_scale, remove_center, resolution_scale)
        self.scale = nn.Linear(
            channels,
            group * (kernel_size * kernel_size - remove_center))
        self.softplus = torch.nn.Softplus(1, 4)
        constant_(self.scale.weight.data, 1.) # 在这里将scale的权重初始化为1
        constant_(self.scale.bias.data, 0.)
        # self.tanh = torch.nn.Tanh()

    def forward(self, input):
        """
        :param query                       (N, H, W, C)
        :return output                     (N, H, W, C)
        """
        N, H, W, _ = input.shape

        x = self.input_proj(input)
        x_proj = x
        dtype = x.dtype

        x1 = input.permute(0, 3, 1, 2).contiguous()
        x1 = self.dw_conv(x1).permute(0, 2, 3, 1).contiguous()
        offset = self.offset(x1)
        # mask
        mask = self.mask(x1).reshape(N, H, W, self.group, -1)
        mask = F.softmax(mask, -1)
        mask = mask.reshape(N, H, W, -1).type(dtype)
        # scale
        scale = self.scale(x1)
        """pre best way """
        scale = self.softplus(scale)  # normalize to a non-neg value
        """"""
        # scale = self.group*F.softmax(scale, -1)
        # scale = scale.reshape(N, H, W, -1).type(dtype)
        # scale = 1 + self.tanh(scale)
        # scale = torch.sigmoid(scale)
        # set scale
        offset = offset.reshape((N, H, W, self.group, self.kernel_size, self.kernel_size, 2))
        scale = scale.reshape((N, H, W, self.group, self.kernel_size, self.kernel_size, 1))
        """try"""
        # scale = self.softplus(scale)  # normalize to a non-neg value
        # max_v = scale.max()
        # scale = torch.softmax(scale, 3) * max_v
        """"""
        offset = offset * scale
        offset = offset.reshape((N, H, W, -1))
        # import ipdb;ipdb.set_trace()
        # # visualize scale
        # from clcore import ImageIO;io = ImageIO()
        # import cv2, os
        # import numpy as np
        
        # scale = scale.reshape((N, H, W, -1))
        # scale = scale[0].sum(-1)  # H*W
        # scale = scale.cpu().numpy()
        # scale = 255 * (scale - scale.min()) / (scale.max() - scale.min())
        # scale = cv2.resize(scale, dsize=None, fx=self.resolution_scale, fy=self.resolution_scale, interpolation=cv2.INTER_LINEAR)
        # scale = scale.astype(np.uint8)[:,:,None]
        # scale = cv2.applyColorMap(scale, cv2.COLORMAP_JET)
        # cv2.imwrite(f'/nfs/project/netdisk/100/data/change_data/cp/WHU/out/test2/scale/4_49/test_res_{str(self.resolution_scale)}.png', scale)
        # import ipdb;ipdb.set_trace()
        
        
        # visualize offset
        # from clcore import ImageIO;io = ImageIO()
        # import cv2, os
        # import numpy as np 
        # pos = (306, 136)
        # # offset.shape = (batch, h, w, group*kernel*kernel*2)
        # offset_vis = offset[0, pos[1]//self.resolution_scale, pos[0]//self.resolution_scale, :].cpu().numpy()
        # offset_vis = offset_vis.reshape((self.group, self.kernel_size, self.kernel_size, 2))
        # # offset_vis = offset_vis[0]
        # mask_vis = mask[0, pos[1]//self.resolution_scale, pos[0]//self.resolution_scale, :].cpu().numpy()
        # mask_vis = mask_vis.reshape((self.group, self.kernel_size, self.kernel_size))
        # # mask_vis = mask_vis[0]
        # # for i in range(aa.shape[2]):
        # #     bb = aa[:,:,i].cpu().numpy()
        # #     bb = (bb - bb.min()) / (bb.max() - bb.min()) * 255
        # #     bb = np.stack((bb, bb, bb), 2)
        # #     bb = cv2.circle(bb, pos, 1, (0, 0, 255), 1)
        # if not os.path.exists('/nfs/project/netdisk/100/data/change_data/cp/WHU/out/test2/offset/test_res_{str(self.resolution_scale)}.tif'):
        #     bb = np.ones((offset.shape[1]*self.resolution_scale, offset.shape[2]*self.resolution_scale, 3))
        # else:
        #     bb = cv2.imread('/nfs/project/netdisk/100/data/change_data/cp/WHU/out/test2/offset/test_res_{str(self.resolution_scale)}.tif', -1)
        # for j in range(self.kernel_size): # H
        #     for k in range(self.kernel_size): # W
        #         for m in range(self.group): # group
        #             new_pos = (pos[0] + int(((k - (self.kernel_size - 1) // 2) * self.dilation + offset_vis[m,k,j,0]) * self.resolution_scale), \
        #                 pos[1] + int(((j - (self.kernel_size - 1) // 2) * self.dilation + offset_vis[m,k,j,1]) * self.resolution_scale))
        #             # if self.resolution_scale <= 8:
        #             bb = cv2.circle(bb, new_pos, 1, (int(255.0*(0.75-mask_vis[m,k,j])), 0, int(255.0*(0.25+mask_vis[m,k,j]))), self.resolution_scale//4)
        #             # else:
        #             #     bb = cv2.circle(bb, new_pos, 1, (int(255.0*(0.7-mask_vis[m,j,k])), 0, int(255.0*(0.3+mask_vis[m,j,k]))), 2)
        # bb = cv2.circle(bb, pos, 1, (0, 255, 0), self.resolution_scale//4)            
        # cv2.imwrite(f'/nfs/project/netdisk/100/data/change_data/cp/WHU/out/test2/offset/test_res_{str(self.resolution_scale)}.tif', bb.astype(np.uint8))

        x = DCNv3Function.apply(
            x, offset, mask,
            self.kernel_size, self.kernel_size,
            self.stride, self.stride,
            self.pad, self.pad,
            self.dilation, self.dilation,
            self.group, self.group_channels,
            self.offset_scale,
            256,
            self.remove_center)
        
        if self.center_feature_scale:
            center_feature_scale = self.center_feature_scale_module(
                x1, self.center_feature_scale_proj_weight, self.center_feature_scale_proj_bias)
            # N, H, W, groups -> N, H, W, groups, 1 -> N, H, W, groups, _d_per_group -> N, H, W, channels
            center_feature_scale = center_feature_scale[..., None].repeat(
                1, 1, 1, 1, self.channels // self.group).flatten(-2)
            x = x * (1 - center_feature_scale) + x_proj * center_feature_scale
        x = self.output_proj(x)

        return x


class DCNfuse3(DCNrs):
    def __init__(
        self,
        in_channels=64,
        out_channels=64,
        kernel_size=3,
        dw_kernel_size=None,
        stride=1,
        pad=1,
        dilation=1,
        group=4,
        offset_scale=1.0,
        act_layer='ReLU',
        norm_layer='batchnorm2d',
        center_feature_scale=False,
        remove_center=False,
        resolution_scale=1,
        with_aux=False,
    ):
        """
        DCNrs Module
        :param channels
        :param kernel_size
        :param stride
        :param pad
        :param dilation
        :param group
        :param offset_scale
        :param act_layer
        :param norm_layer
        """
        super(DCNfuse3, self).__init__(in_channels, kernel_size, dw_kernel_size, stride, pad, dilation, \
            group, offset_scale, act_layer, norm_layer, center_feature_scale, remove_center, resolution_scale)
        self.input_proj = None
        self.window_size = 8
        self.dw_conv = nn.Conv2d(5, 1, kernel_size=3, stride=1, padding=1)
        self.proj_diff = nn.Sequential(
            nn.Linear(
                in_channels,
                in_channels),
            nn.LayerNorm(in_channels))
        
        self.norm_act1 = nn.Sequential(
            BuildNormalization(norm_layer, (in_channels, {})),
            build_act_layer(act_layer))
        self.proj_c = nn.Sequential(
            nn.Linear(
                in_channels*2,
                in_channels),
            nn.LayerNorm(in_channels))
        self.output_proj = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1),
            BuildNormalization(norm_layer, (out_channels, {})),
            build_act_layer(act_layer)
        )
        self.group_channels = in_channels*2 // self.group
        self.with_aux = with_aux
        if self.with_aux:
            self.aux_head = nn.Sequential(
                torch.nn.Conv2d(in_channels, 32, kernel_size=1, stride=1, padding=0),
                BuildNormalization(norm_layer, (32, {})),
                build_act_layer(act_layer),
                torch.nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1)
            )

    def forward(self, input1, input2):
        N, C, H, W = input1.shape
        """偏移窗口扩展全图感受野"""
        diff = input1 - input2
        if self.with_aux:
            out_aux = self.aux_head(diff)
        window_pad_step_h = self.window_size // 2; window_pad_step_w = self.window_size // 2
        diff = F.pad(diff, (abs(window_pad_step_h), abs(window_pad_step_h), abs(window_pad_step_w), abs(window_pad_step_w)), \
            mode="replicate") # b, c, h, w -> b, c, h+ws, w+ws
        """通过torch.roll往不同方向偏移，实现窗口的重叠"""
        # to uper-left
        diff_to_ul = torch.roll(diff, shifts=(-window_pad_step_h, -window_pad_step_w), dims=[2,3]).unsqueeze(2) # b, c, 1, h+ws, w+ws
        # to uper-right
        diff_to_ur = torch.roll(diff, shifts=(-window_pad_step_h, window_pad_step_w), dims=[2,3]).unsqueeze(2) # b, c, 1, h+ws, w+ws
        # to down-left
        diff_to_dl = torch.roll(diff, shifts=(window_pad_step_h, -window_pad_step_w), dims=[2,3]).unsqueeze(2) # b, c, 1, h+ws, w+ws
        # to down-right
        diff_to_dr = torch.roll(diff, shifts=(window_pad_step_h, window_pad_step_w), dims=[2,3]).unsqueeze(2) # b, c, 1, h+ws, w+ws
        diff = diff.unsqueeze(2) # b, c, 1, h+ws, w+ws
        diff = torch.cat((diff_to_ul, diff_to_ur, diff_to_dl, diff_to_dr, diff), dim=2)\
            [:,:,:,abs(window_pad_step_h):H+abs(window_pad_step_h),abs(window_pad_step_w):W+abs(window_pad_step_w)] # shift; b, c, h+ws, w+ws -> b, c, h, w
        diff = diff.reshape(N*C, 5, H, W) # b, 5, c, h+ws, w+ws
            
        diff = self.norm_act1(self.dw_conv(diff).reshape(N, C, H, W)).permute(0,2,3,1).contiguous()

        dtype = input1.dtype

        f = self.proj_diff(diff)
        offset = self.offset(f)
        # mask
        mask = self.mask(f).reshape(N, H, W, self.group, -1).contiguous()
        mask = F.softmax(mask, -1)
        mask = mask.reshape(N, H, W, -1).type(dtype).contiguous()
        # scale
        scale = self.scale(f)
        # scale = torch.sigmoid(scale)
        scale = self.softplus(scale)  # normalize to a non-neg value
        # set scale
        offset = offset.reshape((N, H, W, self.group, self.kernel_size, self.kernel_size, 2))
        scale = scale.reshape((N, H, W, self.group, self.kernel_size, self.kernel_size, 1))
        offset = offset * scale
        offset = offset.reshape((N, H, W, -1))
        c = DCNv3Function.apply(
            torch.cat((input1, input2), 1).permute(0,2,3,1).contiguous(), offset, mask,
            self.kernel_size, self.kernel_size,
            self.stride, self.stride,
            self.pad, self.pad,
            self.dilation, self.dilation,
            self.group, self.group_channels,
            self.offset_scale,
            256,
            self.remove_center)
        c = self.proj_c(c)
        c = self.output_proj((c + diff).permute(0,3,1,2).contiguous())
        if self.with_aux:
            return c, out_aux
        return c


class DCNfuse_MN(DCNfuse3):
    """Use MaskNorm in CD"""
    def __init__(
        self,
        in_channels=64,
        out_channels=64,
        kernel_size=3,
        dw_kernel_size=None,
        stride=1,
        pad=1,
        dilation=1,
        group=4,
        offset_scale=1.0,
        act_layer='ReLU',
        norm_layer='batchnorm2d',
        center_feature_scale=False,
        remove_center=False,
        resolution_scale=1,
        with_aux=False,
    ):
        super(DCNfuse_MN, self).__init__(in_channels, out_channels, kernel_size, dw_kernel_size, stride, pad, dilation, \
            group, offset_scale, act_layer, norm_layer, center_feature_scale, remove_center, resolution_scale, with_aux)
        
        self.proj_input = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.masknorm = BuildNormalization('masknorm', (in_channels*2, {}))
        # self.act = build_act_layer(act_layer)
       
    def forward(self, input1, input2):
        N, C, H, W = input1.shape
        """偏移窗口扩展全图感受野"""
        diff = input1 - input2
        if self.with_aux:
            out_aux = self.aux_head(diff)
            mask = torch.round(torch.sigmoid(out_aux))
            input1 = self.proj_input(input1)
            input2 = self.proj_input(input2)
            input = torch.cat((input1, input2), 1)
            input = self.masknorm(input, mask)
            input1, input2 = input[:,0:C,::], input[:,C:,::]
        window_pad_step_h = self.window_size // 2; window_pad_step_w = self.window_size // 2
        diff = F.pad(diff, (abs(window_pad_step_h), abs(window_pad_step_h), abs(window_pad_step_w), abs(window_pad_step_w)), \
            mode="replicate") # b, c, h, w -> b, c, h+ws, w+ws
        """通过torch.roll往不同方向偏移，实现窗口的重叠"""
        # to uper-left
        diff_to_ul = torch.roll(diff, shifts=(-window_pad_step_h, -window_pad_step_w), dims=[2,3]).unsqueeze(2) # b, c, 1, h+ws, w+ws
        # to uper-right
        diff_to_ur = torch.roll(diff, shifts=(-window_pad_step_h, window_pad_step_w), dims=[2,3]).unsqueeze(2) # b, c, 1, h+ws, w+ws
        # to down-left
        diff_to_dl = torch.roll(diff, shifts=(window_pad_step_h, -window_pad_step_w), dims=[2,3]).unsqueeze(2) # b, c, 1, h+ws, w+ws
        # to down-right
        diff_to_dr = torch.roll(diff, shifts=(window_pad_step_h, window_pad_step_w), dims=[2,3]).unsqueeze(2) # b, c, 1, h+ws, w+ws
        diff = diff.unsqueeze(2) # b, c, 1, h+ws, w+ws
        diff = torch.cat((diff_to_ul, diff_to_ur, diff_to_dl, diff_to_dr, diff), dim=2)\
            [:,:,:,abs(window_pad_step_h):H+abs(window_pad_step_h),abs(window_pad_step_w):W+abs(window_pad_step_w)] # shift; b, c, h+ws, w+ws -> b, c, h, w
        diff = diff.reshape(N*C, 5, H, W) # b, 5, c, h+ws, w+ws
            
        diff = self.dw_conv(diff).reshape(N, C, H, W).permute(0,2,3,1).contiguous()
        
        # """通过torch.roll往不同方向偏移，实现窗口的重叠"""
        # """方法2"""
        # # to uper-left
        # diff_to_ul = torch.roll(diff, shifts=(-window_pad_step_h, -window_pad_step_w), dims=[2,3]) # b, c, h+ws, w+ws
        # # to uper-right
        # diff_to_ur = torch.roll(diff, shifts=(-window_pad_step_h, window_pad_step_w), dims=[2,3]) # b, c, h+ws, w+ws
        # # to down-left
        # diff_to_dl = torch.roll(diff, shifts=(window_pad_step_h, -window_pad_step_w), dims=[2,3]) # b, c, h+ws, w+ws
        # # to down-right
        # diff_to_dr = torch.roll(diff, shifts=(window_pad_step_h, window_pad_step_w), dims=[2,3]) # b, c, h+ws, w+ws
        
        # diff = torch.cat((diff_to_ul, diff_to_ur, diff_to_dl, diff_to_dr, diff), dim=1)\
        #     [:,:,abs(window_pad_step_h):H+abs(window_pad_step_h),abs(window_pad_step_w):W+abs(window_pad_step_w)] # shift; b, c, h+ws, w+ws -> b, c, h, w
            
        # diff = self.dw_conv(diff).reshape(N, C, H, W).permute(0,2,3,1).contiguous()
        dtype = input1.dtype

        f = self.proj_diff(diff)
        offset = self.offset(f)
        # mask
        mask = self.mask(f).reshape(N, H, W, self.group, -1).contiguous()
        mask = F.softmax(mask, -1)
        mask = mask.reshape(N, H, W, -1).type(dtype).contiguous()
        # scale
        scale = self.scale(f)
        # scale = torch.sigmoid(scale)
        scale = self.softplus(scale)  # normalize to a non-neg value
        # set scale
        offset = offset.reshape((N, H, W, self.group, self.kernel_size, self.kernel_size, 2))
        scale = scale.reshape((N, H, W, self.group, self.kernel_size, self.kernel_size, 1))
        offset = offset * scale
        offset = offset.reshape((N, H, W, -1))
        c = DCNv3Function.apply(
            torch.cat((input1, input2), 1).permute(0,2,3,1).contiguous(), offset, mask,
            self.kernel_size, self.kernel_size,
            self.stride, self.stride,
            self.pad, self.pad,
            self.dilation, self.dilation,
            self.group, self.group_channels,
            self.offset_scale,
            256,
            self.remove_center)
        c = self.norm_act(self.output_proj(torch.cat((c, diff), 3).permute(0,3,1,2).contiguous()))
        if self.with_aux:
            return c, out_aux
        return c     
class DCNfuse(nn.Module):
    def __init__(
        self,
        channels1=64,
        channels2=64,
        out_channel=64,
        num_masks=64,
        act_layer='ReLU',
        norm_layer='groupnorm',
        upsample=False,
    ):
        super(DCNfuse, self).__init__()
        self.num_masks = num_masks
        self.out_channel = out_channel
        self.upsample = upsample
        if self.upsample:
            self.channel_reduce = nn.Conv2d(channels1, channels2, kernel_size=1, stride=1, padding=0,)
        self.proj_mask = nn.Conv2d(channels2, num_masks, kernel_size=1, stride=1, padding=0)
        self.proj_cls = nn.Conv2d(channels2, out_channel, kernel_size=1, stride=1, padding=0)
        # self.proj_mask2 = nn.Conv2d(channels2, num_masks, kernel_size=1, stride=1, padding=0)
        # self.proj_cls2 = nn.Conv2d(channels2, out_channel, kernel_size=1, stride=1, padding=0)
        
        self.fuse_cls = nn.Conv1d(out_channel*2, out_channel, 3, padding=1)
        self.fuse_mask = nn.Sequential(
            BuildNormalization(norm_layer, (num_masks*2, {})),
            nn.Conv2d(
                num_masks*2,
                num_masks,
                kernel_size=1,
                stride=1,
                padding=0))

        self.dcn = DCNrs(num_masks, norm_layer='GN')
        self.norm_act = nn.Sequential(
            BuildNormalization(norm_layer, (out_channel, {})),
            build_act_layer(act_layer))

    def forward(self, input1, input2):
        """
        :param query                       (N, H, W, C)
        :return output                     (N, H, W, C)
        """
        if self.upsample:
            input1 = self.channel_reduce(input1)
            input1 =  F.interpolate(input1, None, scale_factor=2, mode="bilinear", align_corners=True)
        N, _, H, W = input1.shape
        mask1 = self.proj_mask(input1)
        mask2 = self.proj_mask(input2)
        cls1 = self.proj_cls(input1)
        cls2 = self.proj_cls(input2)
        f_emb1 = torch.bmm(cls1.view(N, self.out_channel, -1), \
            mask1.view(N, self.num_masks, -1).transpose(1, 2))
        f_emb2 = torch.bmm(cls2.view(N, self.out_channel, -1), \
            mask2.view(N, self.num_masks, -1).transpose(1, 2))
        f_emb = torch.softmax(self.fuse_cls(torch.cat((f_emb1, f_emb2), 1)), -1)
        mask = self.fuse_mask(torch.cat((mask1, mask2), 1)).permute(0,2,3,1)
        mask = self.dcn(mask)
        out = self.norm_act(torch.einsum("bcq,bhwq->bchw", f_emb, mask))
        return out


class DCNfuse2(nn.Module):
    def __init__(
        self,
        channels=64,
        mask_dim=64,
        act_layer='GELU',
        norm_layer='layernorm',
    ):
        super(DCNfuse2, self).__init__()
        self.mask_dim = mask_dim
        self.proj_mask = nn.Conv2d(channels, mask_dim, kernel_size=3, stride=1, padding=1)
        self.proj_cls = nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0)
        self.dcn = DCNrs(mask_dim*2)
        
        self.gamma = nn.Parameter(torch.zeros(1))
        # self.beta = nn.Parameter(torch.zeros(1))
        self.proj_emb = nn.Sequential(
            nn.Linear(channels, channels),
            nn.Softmax(1)
        )
        # self.norm_act = nn.Sequential(
        #     BuildNormalization(norm_layer, (channels, {})),
        #     build_act_layer(act_layer))
        self.dcn_proj = nn.Sequential(
            nn.Linear(mask_dim*2, mask_dim),
            build_norm_layer(mask_dim, norm_layer),
            build_act_layer(act_layer))
        self.output_proj = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0),
            BuildNormalization(norm_layer, (channels, {})),
            build_act_layer(act_layer))
        self.input_proj = None
        
    def forward(self, input1, input2):
        """
        :param query                       (N, H, W, C)
        :return output                     (N, H, W, C)
        """
        N, C, H, W = input1.shape
        f_mask1 = self.proj_mask(input1) # （N, num_masks, H, W)
        input1 = self.proj_cls(input1) # （N, num_masks, H, W)
        f_emb1 = self.gamma*torch.bmm(input1.view(N, C, -1), f_mask1.view(N, self.mask_dim, -1).transpose(1, 2)) # （N, C, mask_dim)
        f_mask2 = self.proj_mask(input2) # （N, num_masks, H, W)
        input2 = self.proj_cls(input2) # （N, num_masks, H, W)
        f_emb2 = self.gamma*torch.bmm(input2.view(N, C, -1), f_mask2.view(N, self.mask_dim, -1).transpose(1, 2)) # （N, C, mask_dim)

        f_mask = torch.cat((f_mask1, f_mask2), 1)
        f_emb = (f_emb1 - f_emb2).permute(0,2,1)

        f_mask = self.dcn(f_mask.permute(0,2,3,1))
        f_mask = self.dcn_proj(f_mask)
        f_emb = self.proj_emb(f_emb)
        out = torch.einsum("bqc,bhwq->bchw", f_emb, f_mask)
        return self.output_proj(out)


class DCNcat(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
    ):
        super(DCNcat, self).__init__()
        self.con1x1 = nn.Sequential(
            nn.Conv2d(
                in_channels*2,
                out_channels,
                kernel_size=1,
                stride=1,
                padding=0),
            BuildNormalization('layernorm', (out_channels, {})),
            build_act_layer("GELU"))
        self.dcn = DCNrs(in_channels)
        self.norm = BuildNormalization('layernorm', (out_channels, {}))
        self.act = build_act_layer("GELU")
    
    def forward(self, x1, x2):
        x = torch.cat((x1, x2), 1)
        x = self.con1x1(x)
        residual = x
        x = self.norm(self.dcn(x.permute(0,2,3,1).contiguous()).permute(0,3,2,1))
        return self.act(residual + x)

"""decoder blocks built by DCNrs"""
class DCNdecBlock(nn.Module):
    def __init__(
        self,
        high_channels,
        low_channels,
        num_masks=96,
        act_layer='ReLU',
        norm_layer='batchnorm2d',
        **kwargs
    ):
        super(DCNdecBlock, self).__init__()
        self.proj_high = nn.Sequential(
            nn.Conv2d(
                high_channels,
                low_channels,
                kernel_size=1,
                stride=1,
                padding=0),
            BuildNormalization(norm_layer, (low_channels, {})),
            build_act_layer(act_layer))
        self.low_channels = low_channels
        self.proj_mask = nn.Conv2d(low_channels, num_masks, kernel_size=1, stride=1, padding=0)
        self.num_masks = num_masks
        self.softplus = torch.nn.Softplus(1, 4)
        self.softmax = torch.nn.Softmax(1)
        self.kernel_size = 3
        self.group = 4
        self.group_channels = num_masks // self.group
        self.offset = nn.Linear(
            num_masks,
            self.group * (self.kernel_size * self.kernel_size) * 2)
        self.mask = nn.Linear(
            num_masks,
            self.group * (self.kernel_size * self.kernel_size))
        
        self.scale = nn.Linear(
            num_masks,
            self.group * (self.kernel_size * self.kernel_size))
        self.proj_low = nn.Sequential(
            nn.Conv2d(
                low_channels,
                num_masks,
                kernel_size=1,
                stride=1,
                padding=0),
            )

        self.norm_act1 = nn.Sequential(
            BuildNormalization(norm_layer, (num_masks, {})),
            build_act_layer(act_layer))
        self.norm_act2 = nn.Sequential(
            BuildNormalization(norm_layer, (num_masks, {})),
            build_act_layer(act_layer))
        self.norm_act3 = nn.Sequential(
            BuildNormalization(norm_layer, (low_channels, {})),
            build_act_layer(act_layer))

    def forward(self, f_high, f_low):
        """
        f_low: low level features                    (N, C, 2H, 2W)
        f_high: high level features                  (N, 2C, H, W)
        """
        N, C, H, W = f_low.shape
        f_high = self.proj_high(f_high) # reduce channel （N, C, H, W）
        # use high-level features to get the representation of masks
        f_mask = self.proj_mask(f_high) # （N, num_masks, H, W)
        f_emb = torch.softmax(torch.bmm(f_high.view(N, C, -1), f_mask.view(N, self.num_masks, -1).transpose(1, 2)), 1) # （N, C, num_masks）
        f_mask = F.interpolate(f_mask, None, scale_factor=2, mode="bilinear", align_corners=True)
        f_low = self.proj_low(f_low) # （N, num_masks, H, W)
        # calculate deformable kernel
        f_mask = self.norm_act1(f_mask) # 14
        f_low = self.norm_act2(f_low) # 14
        _f_mask = f_mask.view(N, self.num_masks, -1).permute(0,2,1) # channel last # 14
        _f_low = f_low.view(N, self.num_masks, -1).permute(0,2,1) # channel last # 14
        # _f_mask = self.norm_act1(f_mask).view(N, self.num_masks, -1).permute(0,2,1) # channel last # 11 12 13
        # _f_low = self.norm_act2(f_low).view(N, self.num_masks, -1).permute(0,2,1) # channel last # 11 12 13
        dtype = _f_mask.dtype

        offset = self.offset(_f_low)
        # mask
        mask = self.mask(_f_mask).reshape(N, H, W, self.group, -1)
        mask = F.softmax(mask, -1)
        mask = mask.reshape(N, H, W, -1).type(dtype)
        # scale
        scale = self.scale(_f_mask)
        scale = self.softplus(scale)
        # set scale
        offset = offset.reshape((N, H, W, self.group, self.kernel_size, self.kernel_size, 2))
        scale = scale.reshape((N, H, W, self.group, self.kernel_size, self.kernel_size, 1))
        offset = offset * scale
        offset = offset.reshape((N, H, W, -1))
         # B, H, W, 128
        f_dcn = DCNv3Function.apply(
            # (f_low+f_mask).permute(0,2,3,1).contiguous(), offset, mask, # 11 13 14 laji
            f_low.permute(0,2,3,1).contiguous(), offset, mask, # 12
            self.kernel_size, self.kernel_size,
            1, 1, # stride
            1, 1, # padding
            1, 1, # dilation
            self.group, self.group_channels,
            1, # offset scale
            256,
            False) # B, H, W, 128
        out = self.norm_act3(torch.einsum("bcq,bhwq->bchw", f_emb, f_dcn)) # 11 12 14
        # f_dcn = self.norm_act3(f_dcn.permute(0,3,1,2)) # 13  laji
        # out = torch.einsum("bcq,bqhw->bchw", f_emb, f_dcn) # 13
        return out

"""decoder blocks built by DCNrs"""
class DCNdecBlock2(nn.Module):
    def __init__(
        self,
        high_channels,
        low_channels,
        num_masks=128,
        act_layer='ReLU',
        norm_layer='groupnorm',
        **kwargs
    ):
        super(DCNdecBlock2, self).__init__()
        self.con1x1 = nn.Sequential(
            nn.Conv2d(
                high_channels,
                low_channels,
                kernel_size=1,
                stride=1,
                padding=0),
            BuildNormalization(norm_layer, (low_channels, {})),
            build_act_layer(act_layer))
        self.low_channels = low_channels
        self.num_masks = num_masks
        self.softplus = torch.nn.Softplus(1, 4)
        self.softmax = torch.nn.Softmax(1)
        self.kernel_size = 3
        self.group = 4
        self.group_channels = low_channels*2 // self.group
        self.offset = nn.Linear(
            low_channels,
            self.group * (self.kernel_size * self.kernel_size) * 2)
        self.mask = nn.Linear(
            low_channels,
            self.group * (self.kernel_size * self.kernel_size))
        
        self.scale = nn.Linear(
            low_channels,
            self.group * (self.kernel_size * self.kernel_size))
        self.conv_low = nn.Sequential(
            nn.Conv2d(
                low_channels*2,
                low_channels,
                kernel_size=1,
                stride=1,
                padding=0),
            BuildNormalization(norm_layer, (low_channels, {})),
            build_act_layer(act_layer)
            )
        self.proj_out = nn.Linear(
            low_channels*2,
            low_channels)
        self.norm_act2 = nn.Sequential(
            BuildNormalization(norm_layer, (low_channels, {})),
            build_act_layer(act_layer))

    def forward(self, f_high, f_low1, f_low2):
        """
        f_low: low level features                    (N, C, 2H, 2W)
        f_high: high level features                  (N, 2C, H, W)
        """
        N, C, H, W = f_low1.shape
        f_high = self.con1x1(f_high) # reduce channel （N, C, H, W）
        # use high-level features to get the representation of masks
        f_high = F.interpolate(f_high, None, scale_factor=2, mode="bilinear", align_corners=True)
        f_low = self.conv_low(torch.cat((f_low1, f_low2), 1))
        _f_low = f_low.view(N, self.low_channels, -1).permute(0,2,1) # channel last
        _f_high = f_high.view(N, self.low_channels, -1).permute(0,2,1) # channel last
        dtype = _f_high.dtype

        # calculate deformable kernel
        # offset = self.offset(_f_mask)  # 23
        offset = self.offset(_f_high)
        # mask
        # mask = self.mask(_f_mask).reshape(N, H, W, self.group, -1)  # 23
        mask = self.mask(_f_high).reshape(N, H, W, self.group, -1)
        mask = F.softmax(mask, -1)
        mask = mask.reshape(N, H, W, -1).type(dtype)
        # scale
        scale = self.scale(_f_high)
        scale = self.softplus(scale)
        # set scale
        offset = offset.reshape((N, H, W, self.group, self.kernel_size, self.kernel_size, 2))
        scale = scale.reshape((N, H, W, self.group, self.kernel_size, self.kernel_size, 1))
        offset = offset * scale
        offset = offset.reshape((N, H, W, -1))
         # B, H, W, 128
        f_dcn = DCNv3Function.apply(
            torch.cat((f_low, f_high), 1).permute(0,2,3,1).contiguous(), offset, mask,# 17 18
            self.kernel_size, self.kernel_size,
            1, 1, # stride
            1, 1, # padding
            1, 1, # dilation
            self.group, self.group_channels,
            1, # offset scale
            256,
            False) # B, H, W, 128
        out = self.norm_act2(self.proj_out(f_dcn).permute(0,3,1,2).contiguous())
        return out


class DCNdecBlock3(DCNrs):
    def __init__(
        self,
        in_channels_high=64,
        in_channels_low=64,
        out_channels=64,
        kernel_size=3,
        dw_kernel_size=None,
        stride=1,
        pad=1,
        dilation=1,
        group=4,
        offset_scale=1.0,
        act_layer='ReLU',
        norm_layer='batchnorm2d',
        center_feature_scale=False,
        remove_center=False,
        resolution_scale=1,
    ):
        super(DCNdecBlock3, self).__init__(in_channels_high, kernel_size, dw_kernel_size, stride, pad, dilation, \
            group, offset_scale, act_layer, norm_layer, center_feature_scale, remove_center, resolution_scale)
        self.dw_conv = nn.Sequential(
            nn.Conv2d(
                in_channels_high,
                in_channels_low,
                kernel_size=3,
                stride=1,
                padding=1,
                groups=in_channels_low//4),
            BuildNormalization(norm_layer, (in_channels_low, {})),
            build_act_layer(act_layer))
        self.input_proj = None
        self.proj_high = nn.Sequential(
            nn.Linear(
                in_channels_high,
                in_channels_high),
            nn.LayerNorm(in_channels_high))
        # self.proj_low = nn.Sequential(
        #     nn.Linear(
        #         in_channels_low,
        #         in_channels_low),
        #     nn.LayerNorm(in_channels_low))
        # self.offset = nn.Linear(
        #     in_channels_low,
        #     group * (kernel_size * kernel_size - remove_center) * 2)
        self.output_proj = nn.Conv2d(
                in_channels_low*2,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1)
        self.norm_act = nn.Sequential(
            BuildNormalization(norm_layer, (out_channels, {})),
            build_act_layer(act_layer))
        self.group_channels = in_channels_low*2 // self.group

    def forward(self, f_high, f_low):
        N, C, H, W = f_low.shape

        f_high = F.interpolate(f_high, None, scale_factor=2, mode="bilinear", align_corners=True)
        f = self.proj_high(f_high.permute(0,2,3,1))
        f_high = self.dw_conv(f_high)
        dtype = f_high.dtype
        offset = self.offset(f)
        # mask
        mask = self.mask(f).reshape(N, H, W, self.group, -1).contiguous()
        mask = F.softmax(mask, -1)
        mask = mask.reshape(N, H, W, -1).type(dtype).contiguous()
        # scale
        scale = self.scale(f)
        # scale = torch.sigmoid(scale)
        scale = self.softplus(scale)  # normalize to a non-neg value
        # set scale
        offset = offset.reshape((N, H, W, self.group, self.kernel_size, self.kernel_size, 2))
        scale = scale.reshape((N, H, W, self.group, self.kernel_size, self.kernel_size, 1))
        offset = offset * scale
        offset = offset.reshape((N, H, W, -1))
        out = DCNv3Function.apply(
            torch.cat((f_high, f_low), 1).permute(0,2,3,1).contiguous(), offset, mask,
            self.kernel_size, self.kernel_size,
            self.stride, self.stride,
            self.pad, self.pad,
            self.dilation, self.dilation,
            self.group, self.group_channels,
            self.offset_scale,
            256,
            self.remove_center)
        out = self.norm_act(self.output_proj(out.permute(0,3,1,2).contiguous()))
        return out


__all__ = [
    "DCNv3_pytorch",
    "DCNv3",
    "DCNrs",
    "DCNcat",
    "DCNfuse",
    "DCNfuse2",
    "DCNfuse3",
    "DCNcat",
    "DCNdecBlock",
    "DCNdecBlock2",
    "DCNdecBlock3",
    "DCNfuse_MN",
]