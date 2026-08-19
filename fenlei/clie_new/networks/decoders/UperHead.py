# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from ..utils.ppm import PyramidPoolingModule
from ..utils.bricks import BuildActivation, BuildNormalization
import torch.nn.functional as F


class UPerHead(nn.Module):
    """Unified Perceptual Parsing for Scene Understanding.

    This head is the implementation of `UPerNet
    <https://arxiv.org/abs/1807.10221>`_.

    Args:
        pool_scales (tuple[int]): Pooling scales used in Pooling Pyramid
            Module applied on the last feature. Default: (1, 2, 3, 6).
    """

    def __init__(self, in_channels, mid_channel, num_class, **kwargs):
        super(UPerHead, self).__init__()
        # PSP Module
        self.psp_modules = PyramidPoolingModule(
            in_channels[0],
            mid_channel,
            )
        # FPN Module
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        for i in range(len(in_channels)-1, 0, -1):  # skip the top layer
            self.lateral_convs.append(
                nn.Sequential(
                    nn.Conv2d(in_channels[i], mid_channel, kernel_size=1, stride=1, padding=0, bias=False),
                    BuildNormalization('layernorm', (mid_channel, {})),
                    torch.nn.ReLU(inplace=True),
                )
            )
            self.fpn_convs.append(
                nn.Sequential(
                    nn.Conv2d(mid_channel, mid_channel, kernel_size=1, stride=1, padding=0, bias=False),
                    BuildNormalization('layernorm', (mid_channel, {})),
                    torch.nn.ReLU(inplace=True),
                )
            )

        self.fpn_bottleneck = nn.Sequential(
                    nn.Conv2d(len(in_channels) * mid_channel, mid_channel, kernel_size=1, stride=1, padding=0, bias=False),
                    BuildNormalization('layernorm', (mid_channel, {})),
                    torch.nn.ReLU(inplace=True),
                )
        self.segmentation_head = nn.Sequential(
                nn.Conv2d(mid_channel, 16, kernel_size=1, stride=1, padding=0),
                nn.ReLU(inplace=True),
                torch.nn.Conv2d(16, num_class, kernel_size=3, stride=1, padding=1),
            )

    def forward(self, inputs, target_size):
        """Forward function."""

        # build laterals
        laterals = [
            lateral_conv(inputs[i])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        laterals.append(self.psp_modules(inputs[-1]))

        # build top-down path
        used_backbone_levels = len(laterals)
        laterals_outs = []
        for i in range(used_backbone_levels - 1, 0, -1):
            prev_shape = laterals[i - 1].shape[2:]
            laterals_outs.append(laterals[i - 1] + F.interpolate(laterals[i], size=prev_shape, mode='bilinear', align_corners=True))
        # build outputs
        fpn_outs = [
            self.fpn_convs[i](laterals_outs[i])
            for i in range(used_backbone_levels - 1)
        ]
        # append psp feature
        fpn_outs.append(laterals[-1])

        for i in range(used_backbone_levels - 1, 0, -1):
            fpn_outs[i] = F.interpolate(fpn_outs[i], size=fpn_outs[0].shape[2:], mode='bilinear', align_corners=True)
        fpn_out = torch.cat(fpn_outs, dim=1)
        output = self.fpn_bottleneck(fpn_out) # 2，512，128，128
        output = F.interpolate(output, size=target_size, mode="bilinear", align_corners=True)
        output = self.segmentation_head(output)
        return [output]

if __name__ == '__main__':
    model = UPerHead(in_channels=[128, 256, 512, 1024],
        in_index=[0, 1, 2, 3],
        pool_scales=(1, 2, 3, 6),
        channels=512,
        dropout_ratio=0.1,
        num_classes=7,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        align_corners=False,
        loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0))
    print(model)