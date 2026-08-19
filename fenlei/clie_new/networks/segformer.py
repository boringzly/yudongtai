import torch.nn as nn
import torch
import torch.nn.functional as F

from .utils.ppm import PyramidPoolingModule
from .backbone import Build_Backbone
from .decoders import SwinDecoder, SegFormerDecoder

class SegFormer(nn.Module):
    def __init__(self, backbone_name, pretrained, num_band, num_class=1, mode="seg", pretrained_path='', **kwargs):
        super(SegFormer, self).__init__()
        # assert "mit" in backbone_name
        if mode == 'change':
            num_band = num_band*2
        #backbone
        self.backbone, channels_blocks, do_upsample = Build_Backbone(
            backbone_name, pretrained, num_band, pretrained_path=pretrained_path
        )
        self.center = PyramidPoolingModule(channels_blocks[0], channels_blocks[0])
        self.decode_head = SwinDecoder(channels_blocks, num_class)
    
    def forward(self, x):
        b, c, h, w = x.shape
        layers = self.backbone(x)
        ppm = self.center(layers[-1])
        y = self.decode_head(ppm, layers, (h, w))
        return y



