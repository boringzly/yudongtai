# ----------------------------------------
# Written by Chen Pan
# ----------------------------------------

# import networks.ResNet as resnet
from .ResNext import *
from .Efficientnet import efficientnet
from .hrnet import hrnet
from .convnext import ConvNeXt
from .transformers import *
from .replknet import create_RepLKNet31B, create_RepLKNet31L
import torch.nn as nn
import os


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
    elif backbone_name == "tv_resnext50_32x4d":
        net = tv_resnext50_32x4d(
            pretrained=pretrained, band_num=band_num, pretrained_path=pretrained_path, output_stride=output_stride, use_se=use_se
        )
        if output_stride == 32:
            return net, [2048, 1024, 512, 256], [True, True, True, True]
        elif output_stride == 16:
            return net, [2048, 1024, 512, 256], [False, True, True, True]
        else:
            return net, [2048, 1024, 512, 256], [False, False, True, True]
    elif backbone_name == "resnest50":
        net = resnest50(pretrained=pretrained, band_num=band_num, pretrained_path=pretrained_path)
        return net, [2048, 1024, 512, 256], [True, True, True, True]
    elif backbone_name == "resnest101":
        net = resnest101(pretrained=pretrained, band_num=band_num, pretrained_path=pretrained_path)
        return net, [2048, 1024, 512, 256], [True, True, True, True]
    elif backbone_name == "resnest200":
        net = resnest200(pretrained=pretrained, band_num=band_num, pretrained_path=pretrained_path)
        return net, [2048, 1024, 512, 256], [True, True, True, True]
    elif backbone_name == "resnest269":
        net = resnest269(pretrained=pretrained, band_num=band_num, pretrained_path=pretrained_path)
        return net, [2048, 1024, 512, 256], [True, True, True, True]
    elif "efficientnet" in backbone_name:
        if backbone_name == "efficientnet-b0":
            net = efficientnet(
                model_name="efficientnet-b0",
                layerlist=[2, 4, 10, 15],
                pretrained=pretrained,
                band_num=band_num,
                drop_connect_rate=drop_rate,
                pretrained_path=pretrained_path
            )
            return net, [320, 112, 40, 24], [True, True, True, True]
        elif backbone_name == "efficientnet-b0_x2":
            net = efficientnet(
                model_name="efficientnet-b0",
                layerlist=[0, 2, 4, 10, 15],
                pretrained=pretrained,
                band_num=band_num,
                drop_connect_rate=drop_rate,
                pretrained_path=pretrained_path
            )
            return net, [320, 112, 40, 24, 16], [True, True, True, True, True]
        elif backbone_name == "efficientnet-b1":
            net = efficientnet(
                model_name="efficientnet-b1",
                layerlist=[4, 7, 15, 22],
                pretrained=pretrained,
                band_num=band_num,
                drop_connect_rate=drop_rate,
                pretrained_path=pretrained_path
            )
            return net, [320, 112, 40, 24], [True, True, True, True]
        elif backbone_name == "efficientnet-b1_x2":
            net = efficientnet(
                model_name="efficientnet-b1",
                layerlist=[1, 4, 7, 15, 22],
                pretrained=pretrained,
                band_num=band_num,
                drop_connect_rate=drop_rate,
                pretrained_path=pretrained_path
            )
            return net, [320, 112, 40, 24, 16], [True, True, True, True, True]
        elif backbone_name == "efficientnet-b2":
            net = efficientnet(
                model_name="efficientnet-b2",
                layerlist=[4, 7, 15, 22],
                pretrained=pretrained,
                band_num=band_num,
                drop_connect_rate=drop_rate,
                pretrained_path=pretrained_path
            )
            return net, [352, 120, 48, 24], [True, True, True, True]
        elif backbone_name == "efficientnet-b2_x2":
            net = efficientnet(
                model_name="efficientnet-b2",
                layerlist=[1, 4, 7, 15, 22],
                pretrained=pretrained,
                band_num=band_num,
                drop_connect_rate=drop_rate,
                pretrained_path=pretrained_path
            )
            return net, [352, 120, 48, 24, 16], [True, True, True, True, True]
        elif backbone_name == "efficientnet-b3":
            net = efficientnet(
                model_name="efficientnet-b3",
                layerlist=[4, 7, 17, 25],
                pretrained=pretrained,
                band_num=band_num,
                drop_connect_rate=drop_rate,
                pretrained_path=pretrained_path
            )
            return net, [384, 136, 48, 32], [True, True, True, True]
        elif backbone_name == "efficientnet-b3_x2":
            net = efficientnet(
                model_name="efficientnet-b3",
                layerlist=[1, 4, 7, 17, 25],
                pretrained=pretrained,
                band_num=band_num,
                drop_connect_rate=drop_rate,
                pretrained_path=pretrained_path
            )
            return net, [384, 136, 48, 32, 24], [True, True, True, True, True]
        elif backbone_name == "efficientnet-b4":
            net = efficientnet(
                model_name="efficientnet-b4",
                layerlist=[5, 9, 20, 30],
                pretrained=pretrained,
                band_num=band_num,
                drop_connect_rate=drop_rate,
                pretrained_path=pretrained_path
            )
            return net, [448, 160, 56, 32], [True, True, True, True]
        elif backbone_name == "efficientnet-b4_x2":
            net = efficientnet(
                model_name="efficientnet-b4",
                layerlist=[1, 5, 9, 20, 30],
                pretrained=pretrained,
                band_num=band_num,
                drop_connect_rate=drop_rate,
                pretrained_path=pretrained_path
            )
            return net, [448, 160, 56, 32, 24], [True, True, True, True, True]
        elif backbone_name == "efficientnet-b5":
            net = efficientnet(
                model_name="efficientnet-b5",
                layerlist=[7, 12, 26, 38],
                pretrained=pretrained,
                band_num=band_num,
                drop_connect_rate=drop_rate,
                pretrained_path=pretrained_path
            )
            return net, [512, 176, 64, 40], [True, True, True, True]
        elif backbone_name == "efficientnet-b5_x2":
            net = efficientnet(
                model_name="efficientnet-b5",
                layerlist=[2, 7, 12, 26, 38],
                pretrained=pretrained,
                band_num=band_num,
                drop_connect_rate=drop_rate,
                pretrained_path=pretrained_path
            )
            return net, [512, 176, 64, 40, 24], [True, True, True, True, True]
        elif backbone_name == "efficientnet-b6":
            net = efficientnet(
                model_name="efficientnet-b6",
                layerlist=[8, 14, 30, 43],
                pretrained=pretrained,
                band_num=band_num,
                drop_connect_rate=drop_rate,
                pretrained_path=pretrained_path
            )
            return net, [576, 200, 72, 40], [True, True, True, True]
        elif backbone_name == "efficientnet-b6_x2":
            net = efficientnet(
                model_name="efficientnet-b6",
                layerlist=[2, 8, 14, 30, 43],
                pretrained=pretrained,
                band_num=band_num,
                drop_connect_rate=drop_rate,
                pretrained_path=pretrained_path
            )
            return net, [576, 200, 72, 40, 32], [True, True, True, True, True]
        elif backbone_name == "efficientnet-b6_x2_l4":
            net = efficientnet(
                model_name="efficientnet-b6",
                layerlist=[2, 8, 14, 30],
                pretrained=pretrained,
                band_num=band_num,
                drop_connect_rate=drop_rate,
                pretrained_path=pretrained_path
            )
            return net, [200, 72, 40, 32], [True, True, True, True]
        elif backbone_name == "efficientnet-b7":
            net = efficientnet(
                model_name="efficientnet-b7",
                layerlist=[10, 17, 37, 53],
                pretrained=pretrained,
                band_num=band_num,
                drop_connect_rate=drop_rate,
                pretrained_path=pretrained_path
            )
            return net, [640, 224, 80, 48], [True, True, True, True]
        elif backbone_name == "efficientnet-b7_x2":
            net = efficientnet(
                model_name="efficientnet-b7",
                layerlist=[3, 10, 17, 37, 53],
                pretrained=pretrained,
                band_num=band_num,
                drop_connect_rate=drop_rate,
                pretrained_path=pretrained_path
            )
            return net, [640, 224, 80, 48, 32], [True, True, True, True, True]
        elif backbone_name == "efficientnet-b8":
            net = efficientnet(
                model_name="efficientnet-b8",
                layerlist=[11, 19, 41, 60],
                pretrained=pretrained,
                band_num=band_num,
                drop_connect_rate=drop_rate,
                pretrained_path=pretrained_path
            )
            return net, [704, 248, 88, 56], [True, True, True, True]
        elif backbone_name == "efficientnet-b8_x2":
            net = efficientnet(
                model_name="efficientnet-b8",
                layerlist=[3, 11, 19, 41, 60],
                pretrained=pretrained,
                band_num=band_num,
                drop_connect_rate=drop_rate,
                pretrained_path=pretrained_path
            )
            return net, [704, 248, 88, 56, 32], [True, True, True, True, True]
    elif "hrnet" in backbone_name:
        if backbone_name == "hrnet-w18":
            net = hrnet(
                model_name=backbone_name, pretrained=pretrained, band_num=band_num, pretrained_path=pretrained_path
            )
            return net, [144, 72, 36, 18], [True, True, True, True]
        elif backbone_name == "hrnet-w30":
            net = hrnet(
                model_name=backbone_name, pretrained=pretrained, band_num=band_num, pretrained_path=pretrained_path
            )
            return net, [240, 120, 60, 30], [True, True, True, True]
        elif backbone_name == "hrnet-w32":
            net = hrnet(
                model_name=backbone_name, pretrained=pretrained, band_num=band_num, pretrained_path=pretrained_path
            )
            return net, [256, 128, 64, 32], [True, True, True, True]
        elif backbone_name == "hrnet-w40":
            net = hrnet(
                model_name=backbone_name, pretrained=pretrained, band_num=band_num, pretrained_path=pretrained_path
            )
            return net, [320, 160, 80, 40], [True, True, True, True]
        elif backbone_name == "hrnet-w44":
            net = hrnet(
                model_name=backbone_name, pretrained=pretrained, band_num=band_num, pretrained_path=pretrained_path
            )
            return net, [352, 176, 88, 44], [True, True, True, True]
        elif backbone_name == "hrnet-w48":
            net = hrnet(
                model_name=backbone_name, pretrained=pretrained, band_num=band_num, pretrained_path=pretrained_path
            )
            return net, [384, 192, 96, 48], [True, True, True, True]
        elif backbone_name == "hrnet-w64":
            net = hrnet(
                model_name=backbone_name, pretrained=pretrained, band_num=band_num, pretrained_path=pretrained_path
            )
            return net, [512, 256, 128, 64], [True, True, True, True]
    # elif backbone_name == 'xception' or backbone_name == 'Xception':
    # 	net = xception.xception(pretrained=False, output_stride=output_stride)
    # 	return net
    elif "mit" in backbone_name:
        if backbone_name == "mit-b0":
            net = mit_b0(band_num, pretrained=pretrained, pretrained_path=pretrained_path)
            return net, [256, 160, 64, 32], [True, True, True, True]
        elif backbone_name == "mit-b1":
            net = mit_b1(band_num, pretrained=pretrained, pretrained_path=pretrained_path)
            return net, [512, 320, 128, 64], [True, True, True, True]
        elif backbone_name == "mit-b2":
            net = mit_b2(band_num, pretrained=pretrained, pretrained_path=pretrained_path)
            return net, [512, 320, 128, 64], [True, True, True, True]
        elif backbone_name == "mit-b3":
            net = mit_b3(band_num, pretrained=pretrained, pretrained_path=pretrained_path)
            return net, [512, 320, 128, 64], [True, True, True, True]
        elif backbone_name == "mit-b4":
            net = mit_b4(band_num, pretrained=pretrained, pretrained_path=pretrained_path)
            return net, [512, 320, 128, 64], [True, True, True, True]
        elif backbone_name == "mit-b5":
            net = mit_b5(band_num, pretrained=pretrained, pretrained_path=pretrained_path)
            return net, [512, 320, 128, 64], [True, True, True, True]
        else:
            raise ValueError(
                "backbone.py: The backbone named %s is not supported yet." % backbone_name
            )
    elif 'Swin' in backbone_name:
        if "Swin_tiny_p4w7" in backbone_name:
            # swin_tiny_patch4_window7_224
            net = BuildSwinTransformer('swin_tiny_patch4_window7_224', in_channels=band_num)
            if pretrained:
                pretrained_path = os.path.join(pretrained_path, 'swin/swin_tiny_patch4_window7_224.pth')
                net.initweights('swin_tiny_patch4_window7_224', pretrained_path)
            return net, [768, 384, 192, 96], [True, True, True, True]
        elif "Swin_small_p4w7" in backbone_name:
            # swin_small_patch4_window7_224
            net = BuildSwinTransformer('swin_small_patch4_window7_224', in_channels=band_num)
            if pretrained:
                pretrained_path = os.path.join(pretrained_path, 'swin/swin_small_patch4_window7_224.pth')
                net.initweights('swin_small_patch4_window7_224', pretrained_path)
            return net, [768, 384, 192, 96], [True, True, True, True]
        else:
            raise ValueError(
                "backbone.py: The backbone named %s is not supported yet." % backbone_name
            )
            # in the following are older swin
            pretrained = os.path.join(pretrained_path, 'swin/swin_tiny_patch4_window7_224.pth')
            net = SwinTransformer(pretrain_img_size=512, patch_size=4, in_channels=3, window_size=7, \
                embed_dims=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24], pretrained=pretrained)
            return net, [768, 384, 192, 96], [True, True, True, True]
    elif "convnext" in backbone_name:
        net = ConvNeXt(in_chans=band_num, depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], \
            drop_path_rate=0.4, layer_scale_init_value=1.0, out_indices=[0, 1, 2, 3],)
        if pretrained:
           net.load_pretrained(backbone_name, pretrained_path) 
        return net, [768, 384, 192, 96], [True, True, True, True]
    elif "RepLKNet" in backbone_name:
        if "RepLKNet31B" in backbone_name:
            net = create_RepLKNet31B(in_channels=band_num)
            if pretrained:
                net.load_pretrained(os.path.join(pretrained_path, "RepLKNet-31B_ImageNet-22K.pth"))
            return net, [1024, 512, 256, 128], [True, True, True, True]
    elif "WaveMLP" in backbone_name:
        if "WaveMLP_S" == backbone_name:
            net = WaveMLP_S(band_num)
            if pretrained:
                net.load_pretrained(os.path.join(pretrained_path, "wavemlp/WaveMLP_S.pth"))
            return net, [512, 320, 128, 64], [True, True, True, True]
        if "WaveMLP_T" == backbone_name:
            net = WaveMLP_T(band_num)
            if pretrained:
                net.load_pretrained(os.path.join(pretrained_path, "wavemlp/WaveMLP_T.pth"))
            return net, [512, 320, 128, 64], [True, True, True, True]
    else:
        raise ValueError(
            "backbone.py: The backbone named %s is not supported yet." % backbone_name
        )

def Build_Backbone_smp(backbone_name="res50", pretrained=True, band_num=4, pretrained_path='', output_stride=7, use_se=False, drop_rate=0, **kwargs):
    weights_offline=None
    weights_offline_path=None
    if pretrained:
        weights_offline='path',
        weights_offline_path=pretrained_path
    if output_stride == 8:
           dilation_list = [2,2]
           upsample_list = [False, False, True, True, True]
    elif output_stride == 16:
           dilation_list = [2]
           upsample_list = [False, True, True, True, True]
    else:
           dilation_list = []
           upsample_list = [True, True, True, True, True]

    backbone, out_channels = get_encoder(
            backbone_name,
            in_channels=band_num,
            depth=5,
            weights=None,
            weights_offline=weights_offline,
            weights_offline_path=weights_offline_path,
            dilation_list=dilation_list
        )
    channels_blocks = [out_channels[4-i] for i in range(5)]
    return backbone, channels_blocks, upsample_list


def Build_Backbone_Pair(
    backbone_name="resnet50",
    pretrained=True,
    num_band=4,
    output_stride=8,
    use_se=False,
    drop_rate=0,
    para_share=True,
    smp_encoders=True
):
    if smp_encoders:
        if para_share:  # parameter share or not
            backbone1, channels_blocks, do_upsample = Build_Backbone_smp(
                backbone_name, pretrained, num_band, output_stride, drop_rate=0.1
            )
            backbone2 = backbone1
        else:
            backbone1, channels_blocks, do_upsample = Build_Backbone_smp(
                backbone_name, pretrained, num_band, output_stride, drop_rate=0.1
            )
            backbone2, channels_blocks2, do_upsample2 = Build_Backbone_smp(
                backbone_name, pretrained, num_band, output_stride, drop_rate=0.1
            )
    else:
        if para_share:  # parameter share or not
            backbone1, channels_blocks, do_upsample = Build_Backbone(
                backbone_name, pretrained, num_band, output_stride, drop_rate=0.1
            )
            backbone2 = backbone1
        else:
            backbone1, channels_blocks, do_upsample = Build_Backbone(
                backbone_name, pretrained, num_band, output_stride, drop_rate=0.1
            )
            backbone2, channels_blocks2, do_upsample2 = Build_Backbone(
                backbone_name, pretrained, num_band, output_stride, drop_rate=0.1
            )
    backbone_pair = nn.ModuleList([backbone1, backbone2])
    return backbone_pair, channels_blocks, do_upsample
