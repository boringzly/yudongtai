import torch.nn as nn
import torch
# import resnet
from .backbone import Build_Backbone
import torch.nn.functional as F
from networks.utils.ocr_module import *

bn_mom = 0.0003


class hrocr(torch.nn.Module):
    def __init__(self, backbone_name, pretrained, num_band=3, num_class=1, mode='seg', **kwargs):
        super(hrocr, self).__init__() ##parent's init func
        self.num_class = num_class
        if mode == 'change':
            num_band = num_band*2
        self.backbone, channels_blocks, do_upsample = Build_Backbone(
            backbone_name, pretrained, num_band
        )
        self.convmerge = nn.Sequential(
                        nn.Conv2d(sum(channels_blocks), 24,kernel_size=3, stride=1, padding=1),
                        nn.BatchNorm2d(24, momentum=bn_mom),
                        nn.ReLU(inplace=True)
                        )
        self.conv3x3_ocr = nn.Sequential(
                        nn.Conv2d(24, 24,kernel_size=3, stride=1, padding=1),
                        nn.BatchNorm2d(24, momentum=bn_mom),
                        nn.ReLU(inplace=True)
                        )
        ocr_mid_channels = 24
        ocr_key_channels = 16
        self.ocr_gather_head = SpatialGather_Module(num_class)

        self.ocr_distri_head = SpatialOCR_Module(in_channels=ocr_mid_channels,
                                                 key_channels=ocr_key_channels,
                                                 out_channels=ocr_mid_channels,
                                                 scale=1,
                                                 dropout=0,
                                                 )
        self.conv_out = nn.Sequential(
                        nn.Conv2d(24,16,kernel_size=3, stride=1, padding=1),
                        nn.BatchNorm2d(16, momentum=bn_mom),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(16, num_class,kernel_size=3, stride=1, padding=1),
                        )
        self.conv_out_aux = nn.Sequential(
                        nn.Conv2d(24, 16,kernel_size=3, stride=1, padding=1),
                        nn.BatchNorm2d(16, momentum=bn_mom),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(16, num_class, kernel_size=3, stride=1, padding=1),
                        )
    
    def forward(self,x):
        x0_h, x0_w = x.size(2), x.size(3)
        layers = self.backbone(x)
        
        x1 = F.interpolate(layers[0], size=(x0_h, x0_w),
                        mode='bilinear', align_corners=ALIGN_CORNERS)
        x2 = F.interpolate(layers[1], size=(x0_h, x0_w),
                        mode='bilinear', align_corners=ALIGN_CORNERS)
        x3 = F.interpolate(layers[2], size=(x0_h, x0_w),
                        mode='bilinear', align_corners=ALIGN_CORNERS)
        x4 = F.interpolate(layers[3], size=(x0_h, x0_w),
                        mode='bilinear', align_corners=ALIGN_CORNERS)

        x = self.convmerge(torch.cat([x1, x2, x3, x4], 1))

        y_aux = self.conv_out_aux(x)
        x = self.conv3x3_ocr(x)
        context = self.ocr_gather_head(x, y_aux)
        x = self.ocr_distri_head(x, context)
        y = self.conv_out(x)
        return y