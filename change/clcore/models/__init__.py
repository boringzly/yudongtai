from .unet import Unet
from .unetplusplus import UnetPlusPlus
from .manet import MAnet
from .linknet import Linknet
from .fpn import FPN
from .pspnet import PSPNet
from .deeplabv3 import DeepLabV3, DeepLabV3Plus
from .pan import PAN

from . import encoders

from typing import Optional
import torch


def create_model(
    arch: str,
    encoder_name: str = "resnet34",
    encoder_weights: Optional[str] = "imagenet",
    encoder_weights_offline: Optional[str] = None,
    encoder_weights_offline_path: Optional[str] = None,
    in_channels: int = 3,
    classes: int = 1,
    **kwargs,
) -> torch.nn.Module:
    """Models wrapper. Allows to create any model just with parametes

    """

    archs = [Unet, UnetPlusPlus, MAnet, Linknet, FPN, PSPNet, DeepLabV3, DeepLabV3Plus, PAN]
    archs_dict = {a.__name__.lower(): a for a in archs}
    try:
        model_class = archs_dict[arch.lower()]
    except KeyError:
        raise KeyError("Wrong architecture type `{}`. Avalibale options are: {}".format(
            arch, list(archs_dict.keys()),
        ))
    return model_class(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        encoder_weights_offline=encoder_weights_offline,
        encoder_weights_offline_path=encoder_weights_offline_path,
        in_channels=in_channels,
        classes=classes,
        **kwargs,
    )
