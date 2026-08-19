# ----------------------------------------
# Written by Chen Pan
# ----------------------------------------


from networks.UNet import *
from networks.FCS import *
from networks.swinunet import SwinUNet
from networks.nestedUnet import NestedUNet
from networks.TransUNetCD import TransUNetCD
from networks.swinsunet import SwinSUNet
from networks.DeepLabV3P import DeepLabV3Plus
from networks.maskcd import MaskCD
from networks.FBCD import FBCD
from networks.maskformer import *
model_dict = {
        "unet": unet, # UNet，支持语义分割与变化检测两个任务，参考论文：U-Net:Convolutional Networks for Biomedical Image Segmentation
        "res_unet": res_unet, # 改进版UNet，将编码器替换成任意成熟的基础网络结构
        "FCS": FCS, # 全卷积孪生网络，针对变化检测任务，参照论文：FCCDN: Feature constraint network for VHR image change detection
        "DeepLabV3Plus": DeepLabV3Plus, # DeepLabV3+，语义分割网络，参照论文：Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation
        "SwinUNet": SwinUNet, # SwinUNet，基于swin transformer构建的UNet，支持语义分割与变化检测任务，参照论文：Swin-Unet: Unet-like Pure Transformer for Medical Image Segmentation
        "NestedUNet": NestedUNet, # UNet++，密集链接的UNet，支持语义分割与变化检测任务，参照论文：UNet++: Redesigning Skip Connections to Exploit Multiscale Features in Image Segmentation
        "TransUNetCD": TransUNetCD, # TransUNetCD，基于transformer的变化检测网络，参照论文：TransUNetCD: A Hybrid Transformer Network for Change Detection in Optical Remote-Sensing Images
        "SwinSUNet": SwinSUNet, # 基于swin transformer结构的变化检测网络，参照论文：SwinSUNet: Pure Transformer Network for Remote Sensing Image Change Detection
        "MaskFormer": MaskFormer,
        "Mask2Former": Mask2Former,
        "MaskCD": MaskCD,
        "FBCD":FBCD,
}

matrix_leanring_model_list = [
        "NestedUNet",
]

def GenerateNet(cfg):
    if cfg.MODEL_NAME not in matrix_leanring_model_list and cfg.WITH_MATRIX_LEARNING:
        raise ValueError(f"Please set cfg.WITH_MATRIX_LEARNING=False. Only models in {matrix_leanring_model_list} support Matrix learning!")
    return model_dict[cfg.MODEL_NAME](
            backbone_name = cfg.MODEL_BACKBONE,
            pretrained = cfg.PRETRAINED,
            output_stride = cfg.MODEL_OUTPUT_STRIDE,
            num_band = cfg.BAND_NUM,
            num_class = cfg.MODEL_NUM_CLASSES,
            use_se = cfg.USE_SE,
            deep_supervise = cfg.DEEP_SUPERVISE,
            pretrained_path = cfg.PRETRAINED_PATH,
            mode = cfg.TASK_TYPE,
            with_matrix_learning = cfg.WITH_MATRIX_LEARNING,
            with_aux = cfg.WITH_AUX,
        )
