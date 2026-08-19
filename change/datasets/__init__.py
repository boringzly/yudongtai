import torch
import numpy as np
import random
from torch.utils.data.distributed import DistributedSampler
from .BaseDataset import BaseChangeDataset, BaseChangeDataset_ds, BaseChangeDataset_seg, \
    BaseSegDataset, BaseSegDataset_tmp, ChangeDataset_MaskFormer, SegDataset_MaskFormer, \
        UDADataset, BaseChangeDataset_tmp, InstChangeDataset, InstChangeDataset_ds

from .sar_opt_dataset import BaseSAROptDataset, SAROptSegDataset, SAROptChangeDataset

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def build_dataloader(nrank, seed, cfg):
    # set dataloader
    " data_loader for train abd val "
    train_data_params = {
        'data_root': cfg.TRAIN_IMG_PATH,
        'img_suffix': cfg.IMG_SUFFIX,
        'gt_suffix': cfg.GT_SUFFIX,
        'name_list': cfg.TRAIN_NAME_LIST
    }
    val_data_params = {
        'data_root': cfg.VAL_IMG_PATH,
        'img_suffix': cfg.IMG_SUFFIX,
        'gt_suffix': cfg.GT_SUFFIX,
        'name_list': cfg.VAL_NAME_LIST
    }
    data_stat_files = {
        'mean_file': cfg.MEAN_FILE,
        'std_file': cfg.STD_FILE
    }
    aug_params = None
    if cfg.USE_DATA_ENHANCEMENT and cfg.TASK_TYPE != "InstCD":
        aug_params = {
            'aug_p': cfg.aug_p,
            'crop': cfg.aug_crop,
            'crop_size': cfg.aug_crop_size,
            'flip': cfg.aug_flip, 
            'transpose': cfg.aug_transpose,
            'scale': cfg.aug_scale,
            'scale_limit': cfg.aug_scale_limit,
            'shift_scale_rotate': cfg.aug_shift_scale_rotate,
            'ssr_limits': cfg.aug_ssr_limits,
            'optical_distortion': cfg.aug_optical_distortion,
            'grid_distortion': cfg.aug_grid_distortion,
            'elasticTransform': cfg.aug_elasticTransform,
            'mask_out': cfg.aug_mask_out,
            'hsv': cfg.aug_hsv,
            'hsv_limits': cfg.aug_hsv_limit,
            'contrast': cfg.aug_contrast,
            'CLAHE': cfg.aug_CLAHE,
            'blur': cfg.aug_blur,
            'motionBlur': cfg.aug_motionBlur,
            'medianBlur': cfg.aug_medianBlur,
            'brightness': cfg.aug_brightness,
            'bright_limit': cfg.aug_bright_limit,
            'gaussnoise': cfg.aug_gaussnoise,
            'fancy_pca': cfg.aug_fancy_pca, 
            "degradation": cfg.with_degradation,
            'exchangeT': cfg.with_exchangeT,
            'revrb': cfg.revrb,
        }
    elif cfg.USE_DATA_ENHANCEMENT:
            aug_params = {
            'aug_p': cfg.aug_p,
            'crop_size': cfg.aug_crop_size if cfg.aug_crop else 0,
            'flip': cfg.aug_flip, 
            'transpose': cfg.aug_transpose,
            'ssr_limits': cfg.aug_ssr_limits,
            'ssr': cfg.aug_shift_scale_rotate,
            'distortion': cfg.aug_optical_distortion,
            'hsv': cfg.aug_hsv,
            'hsv_limits': cfg.aug_hsv_limit,
            'noise': cfg.aug_gaussnoise,
            'fancypca': cfg.aug_fancy_pca, 
            'CLAHE': cfg.aug_CLAHE,
            'degradation': cfg.with_degradation,
            'exchangeT': cfg.with_exchangeT,
            'revrb': cfg.revrb,
        } 

    # 针对部分固定输入的基础网络，重采样val来与train保持一致
    resize_val= cfg.MODEL_BACKBONE in ["sam_vit_base"] # 添加固定尺寸输入的基础网络名字
    extra_data_params = {
        "resize_val": resize_val,
        "val_size": cfg.aug_crop_size,
        "band_num": cfg.BAND_NUM,
    }
    g = torch.Generator()
    g.manual_seed(seed)
    if cfg.IS_TEMP_SETTING:
        data_train = BaseSegDataset_tmp(train_data_params, data_stat_files, extra_data_params,
                aug_params=aug_params, mode='train', upsample=cfg.UPSAMPLE
                )
        if cfg.EPOCH_VAL_FREQ <= cfg.TRAIN_EPOCHS:
            data_val = BaseSegDataset_tmp(val_data_params, data_stat_files, extra_data_params, \
                aug_params=aug_params, mode='val', upsample=cfg.UPSAMPLE)
    elif cfg.TASK_TYPE == "seg" and cfg.DATA_NAME == 'dfc25':
        data_train = SAROptSegDataset(train_data_params, data_stat_files, extra_data_params, with_fft=cfg.with_fft,
                aug_params=aug_params, mode='train', upsample=cfg.UPSAMPLE
                )
        if cfg.EPOCH_VAL_FREQ <= cfg.TRAIN_EPOCHS:
            data_val = SAROptSegDataset(val_data_params, data_stat_files, extra_data_params, \
                aug_params=aug_params, with_fft=cfg.with_fft, mode='val', upsample=cfg.UPSAMPLE)
    elif cfg.TASK_TYPE == "change" and cfg.DATA_NAME == 'dfc25':
        data_train = SAROptChangeDataset(train_data_params, data_stat_files, extra_data_params, with_fft=cfg.with_fft,
                aug_params=aug_params, mode='train', upsample=cfg.UPSAMPLE
                )
        if cfg.EPOCH_VAL_FREQ <= cfg.TRAIN_EPOCHS:
            data_val = SAROptChangeDataset(val_data_params, data_stat_files, extra_data_params, \
                aug_params=aug_params, with_fft=cfg.with_fft, mode='val', upsample=cfg.UPSAMPLE)
    elif "UDA" in cfg.MODEL_NAME:
        train_data_params.update({"target_root": cfg.UDA_IMG_PATH})
        data_train = UDADataset(train_data_params, data_stat_files, extra_data_params,
                aug_params=aug_params, mode='train', upsample=cfg.UPSAMPLE
                )
        if cfg.EPOCH_VAL_FREQ <= cfg.TRAIN_EPOCHS:
            data_val = BaseSegDataset_tmp(val_data_params, data_stat_files, extra_data_params, \
                aug_params=aug_params, mode='val', upsample=cfg.UPSAMPLE)
    elif cfg.TASK_TYPE == "seg":
        data_train = BaseSegDataset(train_data_params, data_stat_files, extra_data_params,
                aug_params=aug_params, mode='train', upsample=cfg.UPSAMPLE
                )
        if cfg.EPOCH_VAL_FREQ <= cfg.TRAIN_EPOCHS:
            data_val = BaseSegDataset(val_data_params, data_stat_files, extra_data_params, \
                aug_params=aug_params, mode='val', upsample=cfg.UPSAMPLE)
    elif cfg.TASK_TYPE == "InstCD":
        if cfg.DEEP_SUPERVISE or '_DS' in cfg.MODEL_NAME or cfg.MODEL_NAME \
            in ["ChangeFormer"] or "MaskDecoder" in cfg.MODEL_NAME or "CADec" in cfg.MODEL_NAME:
            data_train = InstChangeDataset_ds(train_data_params, data_stat_files, extra_data_params, with_fft=cfg.with_fft,
                aug_params=aug_params, mode='train', upsample=cfg.UPSAMPLE
                )
            if cfg.EPOCH_VAL_FREQ <= cfg.TRAIN_EPOCHS:
                data_val = InstChangeDataset_ds(val_data_params, data_stat_files, extra_data_params, \
                    aug_params=aug_params, with_fft=cfg.with_fft, mode='val', upsample=cfg.UPSAMPLE)
        else:
            data_train = InstChangeDataset(train_data_params, data_stat_files, extra_data_params,
                    aug_params=aug_params, with_fft=cfg.with_fft, mode='train', upsample=cfg.UPSAMPLE
                    )
            if cfg.EPOCH_VAL_FREQ <= cfg.TRAIN_EPOCHS:
                data_val = InstChangeDataset(val_data_params, data_stat_files, extra_data_params, \
                    aug_params=aug_params, with_fft=cfg.with_fft, mode='val', upsample=cfg.UPSAMPLE)
    elif cfg.TASK_TYPE == "change":
        if cfg.DEEP_SUPERVISE or '_DS' in cfg.MODEL_NAME or cfg.MODEL_NAME \
            in ["ChangeFormer"] or "MaskDecoder" in cfg.MODEL_NAME or "CADec" in cfg.MODEL_NAME:
            data_train = BaseChangeDataset_ds(train_data_params, data_stat_files, extra_data_params,
                aug_params=aug_params, with_fft=cfg.with_fft, mode='train', upsample=cfg.UPSAMPLE
                )
            if cfg.EPOCH_VAL_FREQ <= cfg.TRAIN_EPOCHS:
                data_val = BaseChangeDataset_ds(val_data_params, data_stat_files, extra_data_params, \
                    aug_params=aug_params, with_fft=cfg.with_fft, mode='val', upsample=cfg.UPSAMPLE)
        elif cfg.MODEL_NAME in ['FCCDN', "Deep_FCCDN"] and cfg.DATA_NAME in ["SECOND"]:
            data_train = BaseChangeDataset_seg(train_data_params, data_stat_files, extra_data_params, \
                aug_params=aug_params, with_fft=cfg.with_fft, mode='train', upsample=cfg.UPSAMPLE
                )
            if cfg.EPOCH_VAL_FREQ <= cfg.TRAIN_EPOCHS:
                data_val = BaseChangeDataset_seg(val_data_params, data_stat_files, extra_data_params, \
                    aug_params=aug_params, with_fft=cfg.with_fft, mode='val', upsample=cfg.UPSAMPLE
                    )
        # elif cfg.MODEL_NAME in ['MaskFormer']:
        #     data_train = ChangeDataset_MaskFormer(train_data_params, data_stat_files, \
        #         aug_params=aug_params, with_fft=cfg.with_fft, mode='train'
        #         )
        #     if cfg.EPOCH_VAL_FREQ <= cfg.TRAIN_EPOCHS:
        #         data_val = ChangeDataset_MaskFormer( val_data_params, data_stat_files, \
        #             aug_params=aug_params, with_fft=cfg.with_fft, mode='val'
        #             )
        else:
            data_train = BaseChangeDataset(train_data_params, data_stat_files, extra_data_params, \
                aug_params=aug_params, with_fft=cfg.with_fft, mode='train'
                )
            if cfg.EPOCH_VAL_FREQ <= cfg.TRAIN_EPOCHS:
                data_val = BaseChangeDataset(val_data_params, data_stat_files, extra_data_params, \
                    aug_params=aug_params, with_fft=cfg.with_fft, mode='val'
                    )
            
    elif cfg.TASK_TYPE == "SAROPT":
        data_train = BaseSAROptDataset(train_data_params, data_stat_files, extra_data_params, with_fft=cfg.with_fft,
                aug_params=aug_params, mode='train', upsample=cfg.UPSAMPLE
                )
        if cfg.EPOCH_VAL_FREQ <= cfg.TRAIN_EPOCHS:
            data_val = BaseSAROptDataset(val_data_params, data_stat_files, extra_data_params, \
                aug_params=aug_params, with_fft=cfg.with_fft, mode='val', upsample=cfg.UPSAMPLE)
    train_sampler = DistributedSampler(data_train) if nrank > 1 else None
    use_shuffle = False if nrank > 1 else True
    if torch.__version__ >= '1.8.0':
        pin_memory = True
    else:
        pin_memory = False
    data_loader_train = torch.utils.data.DataLoader(data_train, cfg.TRAIN_BATCHES, sampler=train_sampler, \
        pin_memory=pin_memory, shuffle=use_shuffle, num_workers=cfg.DATA_WORKERS, drop_last=True,
        worker_init_fn=seed_worker, generator=g, persistent_workers=True, prefetch_factor=10    #persistent_workers=False解决不了shared_memory的问题
        )
    if cfg.VAL_BATCHES < 0:
        cfg.VAL_BATCHES = cfg.TRAIN_BATCHES
    if cfg.EPOCH_VAL_FREQ <= cfg.TRAIN_EPOCHS:
        val_sampler = DistributedSampler(data_val) if nrank > 1 else None
        data_loader_val = torch.utils.data.DataLoader(data_val, cfg.VAL_BATCHES, sampler=val_sampler, pin_memory=pin_memory, \
            num_workers=cfg.DATA_WORKERS, drop_last=False, worker_init_fn=seed_worker, generator=g, persistent_workers=True, prefetch_factor=10
            )
    return data_loader_train, data_train, train_sampler, data_loader_val if cfg.EPOCH_VAL_FREQ <= cfg.TRAIN_EPOCHS else None, \
        data_val if cfg.EPOCH_VAL_FREQ <= cfg.TRAIN_EPOCHS else None

def build_test_dataloader(nrank, seed, cfg):
    """TODO"""
    pass