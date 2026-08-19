"""
Transformer Fuse Block
"""
import torch.nn as nn
import torch
from .bricks import BuildNormalization

class Transformer_Fuse_Block(nn.Module):
    def __init__(self, in_channel1, in_channel2, out_channel, norm_layer="batchnorm2d"):
        super(Transformer_Fuse_Block, self).__init__()
        self.out_channel = out_channel
        self.conv_merge = nn.Sequential(
            nn.Conv2d(in_channel1+in_channel2, out_channel, kernel_size=3, stride=1, dilation=1, padding=1),
            BuildNormalization(norm_layer, (out_channel, {"data_format": "channels_first"})),
            nn.ReLU(),
        )
        self.proj1 = nn.Sequential(
            nn.Conv2d(in_channel1, out_channel*3, kernel_size=1, stride=1, dilation=1, padding=0),
            BuildNormalization(norm_layer, (out_channel*3, {"data_format": "channels_first"})),
            nn.ReLU(),
        )
        self.proj2 = nn.Sequential(
            nn.Conv2d(in_channel2, out_channel*3, kernel_size=1, stride=1, dilation=1, padding=0),
            BuildNormalization(norm_layer, (out_channel*3, {"data_format": "channels_first"})),
            nn.ReLU(),
        )
        self.conv_merge_y2 = nn.Sequential(
            nn.Conv2d(out_channel*4, out_channel, kernel_size=1, stride=1, dilation=1, padding=0),
            BuildNormalization(norm_layer, (out_channel, {"data_format": "channels_first"})),
            nn.ReLU(),
        )
        self.conv_merge_y = nn.Sequential(
            nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, dilation=1, padding=1),
            BuildNormalization(norm_layer, (out_channel, {"data_format": "channels_first"})),
            nn.ReLU(),
        )

    def forward(self, f1, f2):
        bs, C, height, width = f1.size()
        bs, C, _, _ = f2.size()
        assert height == _, 'size mismatch while fuse features.'

        y1 = self.conv_merge(torch.cat((f1, f2), 1))

        qkv1 = self.proj1(f1).reshape(bs, self.out_channel, 3, height, width)
        proj_query1, proj_key1, proj_value1 = qkv1[:,:,0,::], qkv1[:,:,1,::], qkv1[:,:,2,::]
        qkv2 = self.proj2(f2).reshape(bs, self.out_channel, 3, height, width)
        proj_query2, proj_key2, proj_value2 = qkv2[:,:,0,::], qkv2[:,:,1,::], qkv2[:,:,2,::]

        proj_key = torch.cat((proj_key1, proj_key2), 1) #B,2*self.out_channel,H,W
        proj_value = torch.cat((proj_value1, proj_value2), 1) #B,2*self.out_channel,H,W

        proj_query1 = proj_query1.view(bs, self.out_channel, -1).permute(0, 2, 1)  # B * (H*W) * C
        proj_query2 = proj_query2.view(bs, self.out_channel, -1).permute(0, 2, 1)  # B * (H*W) * C

        proj_key = proj_key.view(bs, self.out_channel, -1)  # B * C * (H*W)*2
        proj_value = proj_value.view(bs, self.out_channel, -1)  # B * C * (H*W)*2

        energy1 = torch.bmm(proj_query1, proj_key)  # B * (H*W) * (H*W)*2
        energy1 = energy1 * self.out_channel ** -0.5
        attention1 = torch.softmax(energy1, dim=-1)
        energy2 = torch.bmm(proj_query2, proj_key)  # B * (H*W) * (H*W)*2
        energy2 = energy2 * self.out_channel ** -0.5
        attention2 = torch.softmax(energy2, dim=-1)

        proj_value1 = proj_value1.view(bs, self.out_channel, -1)  # B * C * (H*W)
        proj_value2 = proj_value2.view(bs, self.out_channel, -1)  # B * C * (H*W)
        y2_1 = torch.bmm(proj_value1, attention1).view(bs, self.out_channel*2, height, width)  # B * C * (H*W)*2 -> B * C*2 * H * W
        y2_2 = torch.bmm(proj_value2, attention2).view(bs, self.out_channel*2, height, width)
        y2 = self.conv_merge_y2(torch.cat((y2_1, y2_2), 1))
        return self.conv_merge_y(y1 + y2)