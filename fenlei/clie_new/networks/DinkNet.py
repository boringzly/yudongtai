"""
Codes of LinkNet based on https://github.com/snakers4/spacenet-three
"""
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import models
import torch.nn.functional as F
from config import cfg
from .backbone import Build_Backbone

from functools import partial

nonlinearity = partial(F.relu, inplace=True)


class Dblock_more_dilate(nn.Module):
    def __init__(self, channel):
        super(Dblock_more_dilate, self).__init__()
        self.dilate1 = nn.Conv2d(channel, channel, kernel_size=3, dilation=1, padding=1)
        self.dilate2 = nn.Conv2d(channel, channel, kernel_size=3, dilation=2, padding=2)
        self.dilate3 = nn.Conv2d(channel, channel, kernel_size=3, dilation=4, padding=4)
        self.dilate4 = nn.Conv2d(channel, channel, kernel_size=3, dilation=8, padding=8)
        self.dilate5 = nn.Conv2d(channel, channel, kernel_size=3, dilation=16, padding=16)
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        dilate1_out = nonlinearity(self.dilate1(x))
        dilate2_out = nonlinearity(self.dilate2(dilate1_out))
        dilate3_out = nonlinearity(self.dilate3(dilate2_out))
        dilate4_out = nonlinearity(self.dilate4(dilate3_out))
        dilate5_out = nonlinearity(self.dilate5(dilate4_out))
        out = x + dilate1_out + dilate2_out + dilate3_out + dilate4_out + dilate5_out
        return out


class Dblock(nn.Module):
    def __init__(self, channel):
        super(Dblock, self).__init__()
        self.dilate1 = nn.Conv2d(channel, channel, kernel_size=3, dilation=1, padding=1)
        self.dilate2 = nn.Conv2d(channel, channel, kernel_size=3, dilation=2, padding=2)
        self.dilate3 = nn.Conv2d(channel, channel, kernel_size=3, dilation=4, padding=4)
        self.dilate4 = nn.Conv2d(channel, channel, kernel_size=3, dilation=8, padding=8)
        # self.dilate5 = nn.Conv2d(channel, channel, kernel_size=3, dilation=16, padding=16)
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        dilate1_out = nonlinearity(self.dilate1(x))
        dilate2_out = nonlinearity(self.dilate2(dilate1_out))
        dilate3_out = nonlinearity(self.dilate3(dilate2_out))
        dilate4_out = nonlinearity(self.dilate4(dilate3_out))
        # dilate5_out = nonlinearity(self.dilate5(dilate4_out))
        out = x + dilate1_out + dilate2_out + dilate3_out + dilate4_out  # + dilate5_out
        return out


class Dblock_GN(nn.Module):
    def __init__(self, channel):
        super(Dblock_GN, self).__init__()
        self.dilate1 = nn.Conv2d(channel, channel, kernel_size=3, dilation=1, padding=1)
        self.dilate2 = nn.Conv2d(channel, channel, kernel_size=3, dilation=2, padding=2)
        self.dilate3 = nn.Conv2d(channel, channel, kernel_size=3, dilation=4, padding=4)
        self.dilate4 = nn.Conv2d(channel, channel, kernel_size=3, dilation=8, padding=8)
        self.gn1 = nn.GroupNorm(4, channel)
        self.gn2 = nn.GroupNorm(4, channel)
        self.gn3 = nn.GroupNorm(4, channel)
        self.gn4 = nn.GroupNorm(4, channel)
        # self.dilate5 = nn.Conv2d(channel, channel, kernel_size=3, dilation=16, padding=16)
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        dilate1_out = nonlinearity(self.gn1(self.dilate1(x)))
        dilate2_out = nonlinearity(self.gn2(self.dilate2(dilate1_out)))
        dilate3_out = nonlinearity(self.gn3(self.dilate3(dilate2_out)))
        dilate4_out = nonlinearity(self.gn4(self.dilate4(dilate3_out)))
        # dilate5_out = nonlinearity(self.dilate5(dilate4_out))
        out = x + dilate1_out + dilate2_out + dilate3_out + dilate4_out  # + dilate5_out
        return out


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, n_filters, do_updample=True):
        super(DecoderBlock, self).__init__()
        self.do_updample = do_updample
        self.conv1 = nn.Conv2d(in_channels, in_channels // 4, 1)
        self.norm1 = nn.BatchNorm2d(in_channels // 4)
        self.relu1 = nonlinearity

        self.deconv2 = nn.ConvTranspose2d(in_channels // 4, in_channels // 4, 3, stride=2, padding=1, output_padding=1)
        self.conv2 = nn.Conv2d(in_channels // 4, in_channels // 4, 3, stride=1, padding=1)
        self.norm2 = nn.BatchNorm2d(in_channels // 4)
        self.relu2 = nonlinearity

        self.conv3 = nn.Conv2d(in_channels // 4, n_filters, 1)
        self.norm3 = nn.BatchNorm2d(n_filters)
        self.relu3 = nonlinearity

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        if self.do_updample:
            x = self.deconv2(x)
        else:
            x = self.conv2(x)
        x = self.norm2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu3(x)
        return x


class DecoderBlock_GN(nn.Module):
    def __init__(self, in_channels, n_filters, do_updample=True):
        super(DecoderBlock_GN, self).__init__()
        self.do_updample = do_updample
        self.conv1 = nn.Conv2d(in_channels, in_channels // 4, 1)
        # self.norm1 = nn.BatchNorm2d(in_channels // 4)
        self.gn1 = nn.GroupNorm(4, in_channels // 4)
        self.relu1 = nonlinearity

        self.deconv2 = nn.ConvTranspose2d(in_channels // 4, in_channels // 4, 3, stride=2, padding=1, output_padding=1)
        self.conv2 = nn.Conv2d(in_channels // 4, in_channels // 4, 3, stride=1, padding=1)
        # self.norm2 = nn.BatchNorm2d(in_channels // 4)
        self.gn2 = nn.GroupNorm(4, in_channels // 4)
        self.relu2 = nonlinearity

        self.conv3 = nn.Conv2d(in_channels // 4, n_filters, 1)
        # self.norm3 = nn.BatchNorm2d(n_filters)
        self.gn3 = nn.GroupNorm(4, n_filters)
        self.relu3 = nonlinearity

    def forward(self, x):
        x = self.conv1(x)
        x = self.gn1(x)
        x = self.relu1(x)
        if self.do_updample:
            x = self.deconv2(x)
        else:
            x = self.conv2(x)
        x = self.gn2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.gn3(x)
        x = self.relu3(x)
        return x


class DinkNet34_less_pool(nn.Module):
    def __init__(self, num_classes=1):
        super(DinkNet34_less_pool, self).__init__()

        filters = [64, 128, 256, 512]
        resnet = models.resnet34(pretrained=True)

        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3

        self.dblock = Dblock_more_dilate(256)

        self.decoder3 = DecoderBlock(filters[2], filters[1])
        self.decoder2 = DecoderBlock(filters[1], filters[0])
        self.decoder1 = DecoderBlock(filters[0], filters[0])

        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, num_classes, 3, padding=1)

    def forward(self, x):
        # Encoder
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x = self.firstmaxpool(x)
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)

        # Center
        e3 = self.dblock(e3)

        # Decoder
        d3 = self.decoder3(e3) + e2
        d2 = self.decoder2(d3) + e1
        d1 = self.decoder1(d2)

        # Final Classification
        out = self.finaldeconv1(d1)
        out = self.finalrelu1(out)
        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        out = self.finalconv3(out)

        return F.sigmoid(out)


class DlinkNet(nn.Module):
    def __init__(self, loss_func, backbone_name, pretrained, os, num_band=3, num_class=10):
        super(DlinkNet, self).__init__()
        self.num_class = num_class
        self.loss_func = loss_func

        self.backbone, channels_blocks, do_upsample = Build_Backbone(backbone_name, pretrained, num_band, os)

        self.dblock = Dblock(channels_blocks[0])

        self.decoder4 = DecoderBlock(channels_blocks[0], channels_blocks[1], do_upsample[0])
        self.decoder3 = DecoderBlock(channels_blocks[1], channels_blocks[2], do_upsample[1])
        self.decoder2 = DecoderBlock(channels_blocks[2], channels_blocks[3], do_upsample[2])
        self.decoder1 = DecoderBlock(channels_blocks[3], channels_blocks[3], do_upsample[3])

        self.finaldeconv1 = nn.ConvTranspose2d(channels_blocks[3], 32, 4, 2, 1)
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, num_class, 3, padding=1)

    def forward(self, x):
        # Encoder
        # method 1
        low_layers = self.backbone(x)
        # Center
        e4 = self.dblock(low_layers[3])
        # import ipdb;ipdb.set_trace()
        # Decoder
        d4 = self.decoder4(e4) + low_layers[2]
        d3 = self.decoder3(d4) + low_layers[1]
        d2 = self.decoder2(d3) + low_layers[0]
        d1 = self.decoder1(d2)

        out = self.finaldeconv1(d1)
        out = self.finalrelu1(out)
        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        out = self.finalconv3(out)

        return out
    # def __init__(self, loss_func, backbone_name, pretrained, os, num_band=3, num_class=10):
    #     super(DlinkNet, self).__init__()

    #     filters = [64, 128, 256, 512]
    #     resnet = models.resnet34(pretrained=False)#resnet34-333f7ec4
    #     old_dict = torch.load('./networks/pretrained/resnet34-333f7ec4.pth') 
    #     model_dict = resnet.state_dict()
    #     old_dict = {k: v for k,v in old_dict.items() if (k in model_dict)}
    #     model_dict.update(old_dict)
    #     resnet.load_state_dict(model_dict) 
    #     self.firstconv = resnet.conv1
    #     self.firstbn = resnet.bn1
    #     self.firstrelu = resnet.relu
    #     self.firstmaxpool = resnet.maxpool
    #     self.encoder1 = resnet.layer1
    #     self.encoder2 = resnet.layer2
    #     self.encoder3 = resnet.layer3
    #     self.encoder4 = resnet.layer4

    #     self.dblock = Dblock(512)

    #     self.decoder4 = DecoderBlock(filters[3], filters[2])
    #     self.decoder3 = DecoderBlock(filters[2], filters[1])
    #     self.decoder2 = DecoderBlock(filters[1], filters[0])
    #     self.decoder1 = DecoderBlock(filters[0], filters[0])

    #     self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
    #     self.finalrelu1 = nonlinearity
    #     self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
    #     self.finalrelu2 = nonlinearity
    #     self.finalconv3 = nn.Conv2d(32, num_class, 3, padding=1)

    # def forward(self, x):
    #     # Encoder
    #     x = self.firstconv(x)
    #     x = self.firstbn(x)
    #     x = self.firstrelu(x)
    #     x = self.firstmaxpool(x)
    #     e1 = self.encoder1(x)
    #     e2 = self.encoder2(e1)
    #     e3 = self.encoder3(e2)
    #     e4 = self.encoder4(e3)

    #     # Center
    #     e4 = self.dblock(e4)

    #     # Decoder
    #     d4 = self.decoder4(e4) + e3
    #     d3 = self.decoder3(d4) + e2
    #     d2 = self.decoder2(d3) + e1
    #     d1 = self.decoder1(d2)

    #     out = self.finaldeconv1(d1)
    #     out = self.finalrelu1(out)
    #     out = self.finalconv2(out)
    #     out = self.finalrelu2(out)
    #     out = self.finalconv3(out)

    #     return F.sigmoid(out)


class DinkNet34_GN(nn.Module):
    def __init__(self, num_channels=3, num_classes=1):
        super(DinkNet34_GN, self).__init__()

        self.resnet, channels_blocks, do_upsample = Build_Backbone(backbone_name=cfg.MODEL_BACKBONE,
                                                                   pretrained=cfg.PRE_TRAINED, band_num=num_channels,
                                                                   os=cfg.MODEL_OUTPUT_STRIDE)
        self.dblock = Dblock(channels_blocks[0])

        self.decoder4 = DecoderBlock_GN(channels_blocks[0], channels_blocks[1], do_upsample[0])
        self.decoder3 = DecoderBlock(channels_blocks[1], channels_blocks[2], do_upsample[1])
        self.decoder2 = DecoderBlock(channels_blocks[2], channels_blocks[3], do_upsample[2])
        self.decoder1 = DecoderBlock(channels_blocks[3], channels_blocks[3], do_upsample[3])

        self.finaldeconv1 = nn.ConvTranspose2d(channels_blocks[3], 32, 4, 2, 1)
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, num_classes, 3, padding=1)

    def forward(self, x):
        # Encoder
        # method 1
        low_layers = self.resnet(x)
        # Center
        e4 = self.dblock(low_layers[3])
        # import ipdb;ipdb.set_trace()
        # Decoder
        d4 = self.decoder4(e4) + low_layers[2]
        d3 = self.decoder3(d4) + low_layers[1]
        d2 = self.decoder2(d3) + low_layers[0]
        d1 = self.decoder1(d2)

        out = self.finaldeconv1(d1)
        out = self.finalrelu1(out)
        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        out = self.finalconv3(out)

        return F.sigmoid(out)


