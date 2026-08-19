import torch.nn as nn
import torch
import torch.nn.functional as F

from .utils.ppm import PyramidPoolingModule
from .utils.tfb import *
from .backbone import Build_Backbone
from .decoders import SwinDecoder
from .utils.bricks import BuildNormalization
bn_mom = 0.0003

class SwinUNet(nn.Module):
    def __init__(self, backbone_name, pretrained, num_band, num_class=1, mode="seg", pretrained_path='', **kwargs):
        super(SwinUNet, self).__init__()
        # assert "mit" in backbone_name
        if mode == 'change':
            num_band = num_band*2
        #backbone
        self.backbone, self.channels_blocks, do_upsample = Build_Backbone(
            backbone_name, pretrained, num_band, pretrained_path=pretrained_path
        )
        self.center = PyramidPoolingModule(self.channels_blocks[0], self.channels_blocks[0])
        self.decode_head = SwinDecoder(self.channels_blocks, num_class)
    
    def forward(self, x):
        b, c, h, w = x.shape
        layers = self.backbone(x)
        ppm = self.center(layers[-1])
        y = self.decode_head(ppm, layers, (h, w))
        return y



""""SwinUNet with unsupervised domain adaption"""
class SwinUNet_UDA(SwinUNet):
    def __init__(self, backbone_name, pretrained, num_band, num_class=1, mode="seg", pretrained_path='', **kwargs):
        super(SwinUNet_UDA, self).__init__(backbone_name, pretrained, num_band, num_class, mode, pretrained_path)
        # assert "mit" in backbone_name
        # if mode == 'change':
        #     num_band = num_band*2

        self.auto_decoder = nn.Sequential(
                            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True),
                            torch.nn.Conv2d(self.channels_blocks[0], self.channels_blocks[0]//4, kernel_size=1, stride=1, padding=0),
                            nn.BatchNorm2d(self.channels_blocks[0]//4, momentum=bn_mom),
                            nn.ReLU(inplace=True),
                            torch.nn.Conv2d(self.channels_blocks[0]//4, self.channels_blocks[0]//4, kernel_size=3, stride=1, padding=1),
                            nn.BatchNorm2d(self.channels_blocks[0]//4, momentum=bn_mom),
                            nn.ReLU(inplace=True),
                            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                            torch.nn.Conv2d(self.channels_blocks[0]//4, 16, kernel_size=1, stride=1, padding=0),
                            nn.ReLU(inplace=True),
                            torch.nn.Conv2d(16, 3, kernel_size=3, stride=1, padding=1),
                            )
    
    def forward(self, x, trg=None):
        b, c, h, w = x.shape
        layers = self.backbone(x)
        ppm = self.center(layers[-1])
        y = self.decode_head(ppm, layers, (h, w))
        if self.training:
            y_trg = self.auto_decoder(layers[-1])
            return y, y_trg
        else:
            return y