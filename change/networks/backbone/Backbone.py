# ----------------------------------------
# Written by Chen Pan
# ----------------------------------------

# import networks.ResNet as resnet
from .ResNext import *
from .hrnet import hrnet
from .transformers import *
import torch.nn as nn
import os
from .convnextv2 import *
#from .flashinternimage import build_flashinternimage
from .mamba.models.vmamba import Backbone_VSSM

def Build_Backbone(
    backbone_name="resnet50", pretrained=True, band_num=4, output_stride=8, \
        use_se=False, drop_rate=0, pretrained_path='', **kwargs
):
    if backbone_name == "resnet18":
        net = resnet18(pretrained=pretrained, band_num=band_num, pretrained_path=pretrained_path)
        if output_stride == 32:
            return net, [512, 256, 128, 64], [True, True, True, True]
        elif output_stride == 16:
            return net, [512, 256, 128, 64], [False, True, True, True]
        else:
            return net, [512, 256, 128, 64], [False, False, True, True]
    elif 'vssm_tiny' in backbone_name:
        embed_dim = 768
        net = Backbone_VSSM(in_chans=band_num, out_indices=(0, 1, 2, 3), pretrained=os.path.join(pretrained_path, \
            'mamba/vssm_tiny_0230_ckpt_epoch_262.pth') if pretrained else None, forward_type="v3noz", **kwargs)
        return net, [768, 384, 192, 96], [True, True, True, True]
    elif backbone_name == "resnet34":
        net = resnet34(
            pretrained=pretrained, band_num=band_num, pretrained_path=pretrained_path, output_stride=output_stride, use_se=use_se
        )
        if output_stride == 32:
            return net, [512, 256, 128, 64], [True, True, True, True]
        elif output_stride == 16:
            return net, [512, 256, 128, 64], [False, True, True, True]
        else:
            return net, [512, 256, 128, 64], [False, False, True, True]
    elif backbone_name == "resnet18_dcn":
        net = resnet18_dcn(
            pretrained=pretrained, band_num=band_num, pretrained_path=pretrained_path, output_stride=output_stride, use_se=use_se
        )
        if output_stride == 32:
            return net, [512, 256, 128, 64], [True, True, True, True]
        elif output_stride == 16:
            return net, [512, 256, 128, 64], [False, True, True, True]
        else:
            return net, [512, 256, 128, 64], [False, False, True, True]
    elif backbone_name == "resnet50":
        net = resnet50(
            pretrained=pretrained, band_num=band_num, pretrained_path=pretrained_path, output_stride=output_stride, use_se=use_se
        )
        if output_stride == 32:
            return net, [2048, 1024, 512, 256], [True, True, True, True]
        elif output_stride == 16:
            return net, [2048, 1024, 512, 256], [False, True, True, True]
        else:
            return net, [2048, 1024, 512, 256], [False, False, True, True]
    elif backbone_name == "resnet50_moco":
        net = resnet50(
            pretrained=pretrained, band_num=band_num, pretrained_path=pretrained_path, output_stride=output_stride, use_se=use_se
        )
        if output_stride == 32:
            return net, [2048, 1024, 512, 256], [True, True, True, True]
        elif output_stride == 16:
            return net, [2048, 1024, 512, 256], [False, True, True, True]
        else:
            return net, [2048, 1024, 512, 256], [False, False, True, True]
    elif backbone_name == "resnet101":
        net = resnet101(
            pretrained=pretrained, band_num=band_num, pretrained_path=pretrained_path, output_stride=output_stride, use_se=use_se
        )
        if output_stride == 32:
            return net, [2048, 1024, 512, 256], [True, True, True, True]
        elif output_stride == 16:
            return net, [2048, 1024, 512, 256], [False, True, True, True]
        else:
            return net, [2048, 1024, 512, 256], [False, False, True, True]
    elif backbone_name == "resnet152":
        net = resnet152(
            pretrained=pretrained, band_num=band_num, pretrained_path=pretrained_path, output_stride=output_stride, use_se=use_se
        )
        if output_stride == 32:
            return net, [2048, 1024, 512, 256], [True, True, True, True]
        elif output_stride == 16:
            return net, [2048, 1024, 512, 256], [False, True, True, True]
        else:
            return net, [2048, 1024, 512, 256], [False, False, True, True]
    elif backbone_name == "resnext50_32x4d":
        net = resnext50_32x4d(
            pretrained=pretrained, band_num=band_num, pretrained_path=pretrained_path, output_stride=output_stride, use_se=use_se
        )
        if output_stride == 32:
            return net, [2048, 1024, 512, 256], [True, True, True, True]
        elif output_stride == 16:
            return net, [2048, 1024, 512, 256], [False, True, True, True]
        else:
            return net, [2048, 1024, 512, 256], [False, False, True, True]
    elif backbone_name == "resnext50d_32x4d":
        net = resnext50d_32x4d(
            pretrained=pretrained, band_num=band_num, pretrained_path=pretrained_path, output_stride=output_stride, use_se=use_se
        )
        if output_stride == 32:
            return net, [2048, 1024, 512, 256], [True, True, True, True]
        elif output_stride == 16:
            return net, [2048, 1024, 512, 256], [False, True, True, True]
        else:
            return net, [2048, 1024, 512, 256], [False, False, True, True]
    elif backbone_name == "resnext101_32x4d":
        net = resnext101_32x4d(
            pretrained=pretrained, band_num=band_num, pretrained_path=pretrained_path, output_stride=output_stride, use_se=use_se
        )
        if output_stride == 32:
            return net, [2048, 1024, 512, 256], [True, True, True, True]
        elif output_stride == 16:
            return net, [2048, 1024, 512, 256], [False, True, True, True]
        else:
            return net, [2048, 1024, 512, 256], [False, False, True, True]
    elif backbone_name == "resnext101_32x8d":
        net = resnext101_32x8d(
            pretrained=pretrained, band_num=band_num, pretrained_path=pretrained_path, output_stride=output_stride, use_se=use_se
        )
        if output_stride == 32:
            return net, [2048, 1024, 512, 256], [True, True, True, True]
        elif output_stride == 16:
            return net, [2048, 1024, 512, 256], [False, True, True, True]
        else:
            return net, [2048, 1024, 512, 256], [False, False, True, True]
    elif 'vit' in backbone_name:
        if backbone_name == 'sam_vit_base':
            net = build_sam_vit_base(img_size=512, in_chans=band_num)
            if pretrained:
                pretrained_path = os.path.join(pretrained_path, 'sam_vit_b_01ec64.pth')
                net.load_pretrained(pretrained_path)
            return net, [net.out_chans]*4, [True, True, True, True]
        else:
            raise ValueError(
                "backbone.py: Currently only sam_vit_base is supported."
            )
    elif backbone_name == "resnext101_64x4d":
        net = resnext101_64x4d(
            pretrained=pretrained, band_num=band_num, pretrained_path=pretrained_path, output_stride=output_stride, use_se=use_se
        )
        if output_stride == 32:
            return net, [2048, 1024, 512, 256], [True, True, True, True]
        elif output_stride == 16:
            return net, [2048, 1024, 512, 256], [False, True, True, True]
        else:
            return net, [2048, 1024, 512, 256], [False, False, True, True]
    
    elif "hrnet" in backbone_name:
        net = hrnet(
            model_name=backbone_name, pretrained=pretrained, band_num=band_num, pretrained_path=pretrained_path
        )
        if backbone_name == "hrnet-w18":
            return net, [144, 72, 36, 18], [True, True, True, True]
        elif backbone_name == "hrnet-w30":
            return net, [240, 120, 60, 30], [True, True, True, True]
        elif backbone_name == "hrnet-w32":
            return net, [256, 128, 64, 32], [True, True, True, True]
        elif backbone_name == "hrnet-w40":
            return net, [320, 160, 80, 40], [True, True, True, True]
        elif backbone_name == "hrnet-w44":
            return net, [352, 176, 88, 44], [True, True, True, True]
        elif backbone_name == "hrnet-w48":
            return net, [384, 192, 96, 48], [True, True, True, True]
        elif backbone_name == "hrnet-w64":
            return net, [512, 256, 128, 64], [True, True, True, True]
    # elif backbone_name == 'xception' or backbone_name == 'Xception':
    # 	net = xception.xception(pretrained=False, output_stride=output_stride)
    # 	return net
    
    elif 'Swin' in backbone_name:
        if "Swin_tiny_p4w7" in backbone_name:
            # swin_tiny_patch4_window7_224
            net = BuildSwinTransformer('swin_tiny_patch4_window7_224', in_channels=band_num)
            if pretrained:
                pretrained_path = os.path.join(pretrained_path, 'swin_tiny_patch4_window7_224.pth')
                net.initweights('swin_tiny_patch4_window7_224', pretrained_path)
            return net, [768, 384, 192, 96], [True, True, True, True]
        elif "Swin_small_p4w7" in backbone_name:
            # swin_small_patch4_window7_224
            net = BuildSwinTransformer('swin_small_patch4_window7_224', in_channels=band_num)
            if pretrained:
                pretrained_path = os.path.join(pretrained_path, 'swin/swin_small_patch4_window7_224.pth')
                net.initweights('swin_small_patch4_window7_224', pretrained_path)
            return net, [768, 384, 192, 96], [True, True, True, True]
        elif "Swin_base_p4w7" in backbone_name:
            # swin_base_patch4_window7_224
            net = BuildSwinTransformer('swin_base_patch4_window7_224', in_channels=band_num)
            if pretrained:
                pretrained_path = os.path.join(pretrained_path, 'swin/swin_base_patch4_window7_224.pth')
                net.initweights('swin_base_patch4_window7_224', pretrained_path)
            return net, [1024, 512, 256, 128], [True, True, True, True]
        elif "Swin_base_p4w8_SimMIM" in backbone_name:
            # swin_base_patch4_window7_224
            net = BuildSwinTransformer('swin_base_patch4_window8_512', in_channels=band_num)
            if pretrained:
                pretrained_path = os.path.join(pretrained_path, 'swin/simmim_gf30w_swinb_ep35.pth')
                net.initweights('swin_base_patch4_window7_224', pretrained_path)
            return net, [1024, 512, 256, 128], [True, True, True, True]
        elif "Swin_large_p4w7" in backbone_name:
            # swin_large_patch4_window7_224
            net = BuildSwinTransformer('swin_large_patch4_window12_384_22k', in_channels=band_num)
            if pretrained:
                pretrained_path = os.path.join(pretrained_path, 'swin/swin_large_patch4_window12_384_22k.pth')
                net.initweights('swin_large_patch4_window12_384_22k', pretrained_path)
            return net, [1536, 768, 384, 192], [True, True, True, True]
    elif "convnextv2" in backbone_name:
        net, channels = build_convnext_v2(backbone_name, in_chans=band_num, pretrained=pretrained, pretrained_path=pretrained_path)
        return net, channels, [True, True, True, True]
    elif "flashinternimage" in backbone_name:
        net, out_channels = build_flashinternimage(backbone_name, band_num, pretrained, pretrained_path)
        return net, out_channels, [True, True, True, True]
    else:
        raise ValueError(
            "backbone.py: The backbone named %s is not supported yet." % backbone_name
        )
