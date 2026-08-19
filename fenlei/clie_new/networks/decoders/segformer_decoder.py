# ---------------------------------------------------------------
# Copyright (c) 2021, NVIDIA Corporation. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# ---------------------------------------------------------------
import numpy as np
import torch.nn as nn
import torch
from collections import OrderedDict
import torch.nn.functional as F

from IPython import embed
bn_mom = 0.0003
class MLP(nn.Module):
    """
    Linear Embedding
    """
    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2).contiguous()
        x = self.proj(x)
        return x


class SegFormerDecoder(nn.Module):
    """
    SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers
    """
    def __init__(self, in_channels, num_class, dropout_ratio=0.1,  **kwargs):
        super(SegFormerDecoder, self).__init__()
        """
        Args:
            in_channels(Sequence[int]): Input channels.
            num_class (int): The class number of the task.
        """

        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = in_channels

        embedding_dim = 128

        self.linear_c4 = MLP(input_dim=c4_in_channels, embed_dim=embedding_dim)
        self.linear_c3 = MLP(input_dim=c3_in_channels, embed_dim=embedding_dim)
        self.linear_c2 = MLP(input_dim=c2_in_channels, embed_dim=embedding_dim)
        self.linear_c1 = MLP(input_dim=c1_in_channels, embed_dim=embedding_dim)

        self.linear_fuse = nn.Sequential(
                        nn.Conv2d(embedding_dim*4, embedding_dim, kernel_size=1, stride=1, padding=0),
                        nn.BatchNorm2d(embedding_dim, momentum=bn_mom),
                        nn.ReLU(inplace=True),
                        )
        self.dropout = nn.Dropout2d(dropout_ratio)

        self.linear_pred = nn.Conv2d(embedding_dim, num_class, kernel_size=3, padding=1)

    # def init_weights(self):
    #     """Initialize weights of classification layer."""
    #     normal_init(self.conv_seg, mean=0, std=0.01)


    def forward(self, x):
        #x: len=4, 1/4, 1/8, 1/16, 1/32
        c4, c3, c2, c1 = x

        ############## MLP decoder on C1-C4 ###########
        n, _, h, w = c4.shape
        _c4 = self.linear_c4(c4).permute(0,2,1).reshape(n, -1, c4.shape[2], c4.shape[3])

        _c3 = self.linear_c3(c3).permute(0,2,1).reshape(n, -1, c3.shape[2], c3.shape[3])
        _c3 = F.interpolate(_c3, size=c4.size()[2:],mode='bilinear',align_corners=False)

        _c2 = self.linear_c2(c2).permute(0,2,1).reshape(n, -1, c2.shape[2], c2.shape[3])
        _c2 = F.interpolate(_c2, size=c4.size()[2:],mode='bilinear',align_corners=False)

        _c1 = self.linear_c1(c1).permute(0,2,1).reshape(n, -1, c1.shape[2], c1.shape[3])
        _c1 = F.interpolate(_c1, size=c4.size()[2:],mode='bilinear',align_corners=False)

        _c = self.linear_fuse(torch.cat([_c4, _c3, _c2, _c1], dim=1))

        _c = F.interpolate(_c, scale_factor=4, mode='bilinear', align_corners=True)
        x = self.dropout(_c)
        x = self.linear_pred(x)
        return [x]


__all__ = [
    "SegFormerDecoder",
]
