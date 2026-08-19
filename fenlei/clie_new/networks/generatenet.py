# ----------------------------------------
# Written by Chen Pan
# ----------------------------------------


import torch
# import torch.nn as nn
from networks.UNet import unet, res_unet, res_unet_aux, res_unet_CSA_Aux, res_unet_dense, res_unet_dense_aux, \
		res_unet_dense_scse_aux, res_unet_ddense_scse_aux, res_unet_old, SERes_UNet
from networks.UNet_new import *
from networks.PAN import PAN
from networks.DinkNet import DlinkNet
from networks.hrocr import hrocr
from networks.segformer import SegFormer
from networks.maskformer import MaskFormer, Mask2Former 
from networks.swinunet import SwinUNet, SwinUNet_UDA
from networks.Agriculture import sk_cnn_src, skcnn_TabNet
from networks.nestedUnet import Deep_NestedUNet_U2

def GenerateNet(cfg):
	if cfg.MODEL_NAME == 'unet':
		return unet(cfg.LOSS_FUNC, cfg.BAND_NUM, cfg.MODEL_NUM_CLASSES)
	elif cfg.MODEL_NAME == 'res_unet':
		return res_unet(cfg.MODEL_BACKBONE, cfg.PRE_TRAINED, cfg.MODEL_OUTPUT_STRIDE,cfg.BAND_NUM, cfg.MODEL_NUM_CLASSES)
	elif cfg.MODEL_NAME == 'SERes_UNet':
		return SERes_UNet(cfg.MODEL_BACKBONE, cfg.PRE_TRAINED, cfg.MODEL_OUTPUT_STRIDE,cfg.BAND_NUM, cfg.MODEL_NUM_CLASSES)
	elif cfg.MODEL_NAME == 'res_unet_old':
		return res_unet_old(cfg.MODEL_BACKBONE, cfg.PRE_TRAINED, cfg.MODEL_OUTPUT_STRIDE,cfg.BAND_NUM, cfg.MODEL_NUM_CLASSES)
	elif cfg.MODEL_NAME == 'res_unet_aux':
		return res_unet_aux(cfg.MODEL_BACKBONE, cfg.PRE_TRAINED, cfg.MODEL_OUTPUT_STRIDE,cfg.BAND_NUM, cfg.MODEL_NUM_CLASSES)
	elif cfg.MODEL_NAME == 'res_unet_CSA_aux':
		return res_unet_CSA_Aux(cfg.LOSS_FUNC, cfg.MODEL_BACKBONE, cfg.PRE_TRAINED, cfg.MODEL_OUTPUT_STRIDE,cfg.BAND_NUM, cfg.MODEL_NUM_CLASSES)
	elif cfg.MODEL_NAME == 'res_unet_dense':
		return res_unet_dense(cfg.LOSS_FUNC, cfg.MODEL_BACKBONE, cfg.PRE_TRAINED, cfg.MODEL_OUTPUT_STRIDE,cfg.BAND_NUM, cfg.MODEL_NUM_CLASSES)
	elif cfg.MODEL_NAME == 'res_unet_dense_aux':
		return res_unet_dense_aux(cfg.LOSS_FUNC, cfg.MODEL_BACKBONE, cfg.PRE_TRAINED, cfg.MODEL_OUTPUT_STRIDE,cfg.BAND_NUM, cfg.MODEL_NUM_CLASSES)
	elif cfg.MODEL_NAME == 'res_unet_dense_scse_aux':
		return res_unet_dense_scse_aux(cfg.LOSS_FUNC, cfg.MODEL_BACKBONE, cfg.PRE_TRAINED, cfg.MODEL_OUTPUT_STRIDE,cfg.BAND_NUM, cfg.MODEL_NUM_CLASSES)
	elif cfg.MODEL_NAME == 'res_unet_ddense_scse_aux':
		return res_unet_ddense_scse_aux(cfg.LOSS_FUNC, cfg.MODEL_BACKBONE, cfg.PRE_TRAINED, cfg.MODEL_OUTPUT_STRIDE,cfg.BAND_NUM, cfg.MODEL_NUM_CLASSES)
	elif cfg.MODEL_NAME == 'Deep_NestedUNet_U2':
		return Deep_NestedUNet_U2(cfg.MODEL_BACKBONE, cfg.PRE_TRAINED, cfg.BAND_NUM, cfg.MODEL_NUM_CLASSES)
	elif cfg.MODEL_NAME == 'PAN':
		return PAN(cfg.LOSS_FUNC, cfg.MODEL_BACKBONE, cfg.PRE_TRAINED, cfg.MODEL_OUTPUT_STRIDE,cfg.BAND_NUM, cfg.MODEL_NUM_CLASSES)
	elif cfg.MODEL_NAME == 'DlinkNet':
		return DlinkNet(cfg.LOSS_FUNC, cfg.MODEL_BACKBONE, cfg.PRE_TRAINED, cfg.MODEL_OUTPUT_STRIDE, cfg.BAND_NUM, cfg.MODEL_NUM_CLASSES)
	elif cfg.MODEL_NAME == 'hrocr':
		return hrocr(cfg.MODEL_BACKBONE, cfg.PRE_TRAINED, cfg.BAND_NUM, cfg.MODEL_NUM_CLASSES)
	elif cfg.MODEL_NAME == 'SegFormer':
		return SegFormer(cfg.MODEL_BACKBONE, cfg.PRE_TRAINED, cfg.BAND_NUM, cfg.MODEL_NUM_CLASSES)
	elif cfg.MODEL_NAME == 'SwinUNet':
		return SwinUNet(cfg.MODEL_BACKBONE, cfg.PRE_TRAINED, cfg.BAND_NUM, cfg.MODEL_NUM_CLASSES)
	elif cfg.MODEL_NAME == 'SwinUNet_UDA':
		return SwinUNet_UDA(cfg.MODEL_BACKBONE, cfg.PRE_TRAINED, cfg.BAND_NUM, cfg.MODEL_NUM_CLASSES)
	elif cfg.MODEL_NAME == "Mask2Former":
		return Mask2Former(cfg.MODEL_BACKBONE, cfg.PRE_TRAINED, cfg.BAND_NUM, cfg.MODEL_NUM_CLASSES)
	elif cfg.MODEL_NAME == "MaskFormer":
		return MaskFormer(cfg.MODEL_BACKBONE, cfg.PRE_TRAINED, cfg.BAND_NUM, cfg.MODEL_NUM_CLASSES)
	elif cfg.MODEL_NAME == "res_unet_new":
		return res_unet_new(cfg.MODEL_BACKBONE, cfg.PRE_TRAINED, num_band=cfg.BAND_NUM, num_class=cfg.MODEL_NUM_CLASSES)
	elif cfg.MODEL_NAME == "unet_new":
		return unet_new(num_band=cfg.BAND_NUM, num_class=cfg.MODEL_NUM_CLASSES)
	elif cfg.MODEL_NAME == 'Agriculture':
		return sk_cnn_src(cfg.BAND_NUM, cfg.MODEL_NUM_CLASSES)
	elif cfg.MODEL_NAME == 'Agriculture_model_2':
		return skcnn_TabNet(cfg.BAND_NUM, cfg.MODEL_NUM_CLASSES)
	else:
		raise ValueError('GenerateNet.py: network %s is not support yet'%cfg.MODEL_NAME)
