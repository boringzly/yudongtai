# ----------------------------------------
# Written by Pan Chen
# ----------------------------------------
import torch
import argparse
import os
import sys
import cv2
import time
import warnings


class Configuration():
	def __init__(self):
		# path
		self.TRAIN_IMG_PATH = './data/image_train/'
		self.TRAIN_IMG_SUFFIX = '.tif'
		self.TRAIN_GT_PATH = './data/label_train/'
		self.TRAIN_GT_SUFFIX = '.tif'
		self.VAL_IMG_PATH = './data/image_val/'
		self.VAL_IMG_SUFFIX = '.tif'
		self.VAL_GT_PATH = './data/label_val/'
		self.VAL_GT_SUFFIX = '.tif'
		self.image_input_dir = None
		self.TEST_IMG_SUFFIX = '.tif'
		self.image_input_file = None
		self.image_output_dir = './output'
		self.TEST_OUT_SUFFIX = '.tif'
		self.mean_value = [0.5, 0.5, 0.5]
		self.std_value = [0.2, 0.2, 0.2]
		self.LOSS_WEIGHT_FILE = None
		self.COLOR_TABLE_FILE = None
		self.job_id = 'default'

		# name
		self.EXPERIMENT_NAME = 'test'
		self.DATA_NAME = 'test_data'

		# augment
		self.USE_DATA_ENHANCEMENT = False
		self.aug_p = 0.1
		self.aug_crop = False
		self.aug_crop_size = 640
		self.aug_flip = True
		self.aug_transpose = True
		self.aug_shift_scale_rotate = False
		self.aug_ssr_limits = [0.1, 0.1, 45]
		self.aug_optical_distortion = False
		self.aug_grid_distortion = False
		self.aug_elasticTransform = False
		self.aug_hsv = False
		self.aug_hsv_limit = [10, 10, 20]
		self.aug_contrast = False
		self.aug_CLAHE = False
		self.aug_blur = False
		self.aug_motionBlur = False
		self.aug_medianBlur = False
		self.aug_brightness = False
		self.aug_bright_limit = 10
		self.aug_gaussnoise = False
		self.aug_fancy_pca = False
		self.aug_val = False

		# model
		self.DATA_WORKERS = 1 
		self.MODEL_NAME = 'efficientunet'  # 'PAN' 'Unet' 'Res_U' 'U_Dink' 'Dlink' 'Res_U_V2'
		self.MODEL_BACKBONE = 'res34'  # res101_atrous
		#self.MODEL_NAME = 'res_unet'	#'PAN' 'Unet' 'Res_U' 'U_Dink' 'Dlink' 'Res_U_V2'
		#self.MODEL_BACKBONE = 'efficientnet-b1'#res101_atrous
		self.MODEL_NUM_CLASSES = 1
		self.BAND_NUM = 3
		self.DROPOUT_LIST = [0., 0., 0.]
		self.CHANNELS_BLOCK = [2048, 1024, 512, 256, 64]  # [512, 256, 128, 64, 64]
		self.MODEL_OUTPUT_STRIDE = 32
		self.PRE_TRAINED = False
		self.USE_SE = False
		self.UPSAMPLE = False

		# train setting
		self.FREEZE_BACKBONE_EPOCH = -1
		# self.FREEZE_BN_EPOCH = 10000
		# self.FREEZE_BN_MODE = 'backbone_freeze'#'all_freeze' 'no_freeze'
		self.MODEL_SAVE_DIR = './checkpoints/'
		self.LOG_DIR = './log/'
		self.MODEL_SAVING_FREQ = 1000
		self.VAL_NUM_DURING_TRAIN = 20
		self.VAL_FREQ_DURING_TRAIN = 20
		self.TRAIN_CKPT = None
		self.LOAD_CKPT_MODE = 'resume'  # resume pretrain
		self.TRAIN_MOMENTUM = 0.9
		self.TRAIN_WEIGHT_DECAY = 0
		self.TRAIN_BN_MOM = 0.0003
		self.TRAIN_POWER = 0.9
		self.TRAIN_GPUS = '[0]'
		self.INPUT_DEVICE = 1
		self.OUTPUT_DEVICE = 1
		self.OPTIM_DEVICE = 1
		self.TRAIN_BATCHES = 6
		self.TRAIN_SHUFFLE = True
		self.TRAIN_MINEPOCH = 0
		self.TRAIN_EPOCHS = 100
		self.TRAIN_TBLOG = True
		self.VAL_BATCHES = -1
		self.VAL_WITH_TRAIN_SET = False
		self.VAL_WITH_BACKGROUND = False
		self.LOSS_WEIGHT_WITH_BACKGROUND = False
		self.EPOCH_VAL_FREQ = 1
		self.OPTIM_KEYWORD = 'train_loss'  # train_loss val_loss train_acc val_acc
		self.EVAL_NODE = 'no'  # 'no' 'all' 'backbone'
		self.PIN_MEMORY = False

		# lr
		self.TRAIN_LR = 1e-4
		self.LR_WARMUP_STEPS = 0
		self.SCHEDULER_PATIENCE = 10
		self.LR_STEP_SIZE = 1
		self.LR_SCHEDULER = 'step_lr'  # 'step_lr' 'multistep_lr' 'cos_lr' 'plateau_lr'
		self.USE_LR_JUMP = False
		self.LR_JUMP_STEP = 5000
		self.FINAL_LR = 0.1
		self.LR_STEPS = [1e5, 5e5, 1e6, 5e6]
		self.OPTIMIZER = 'SGD'  # Adam SGD
		self.LOAD_OPTI_PARAMS = False
		self.LR_SCHEDULER_GAMMA = 0.2
		self.COS_LR_T = 10  # epoches from max to min
		self.COS_T_MULT = 1
		self.SNAPSHOT_ENSEMBLE_NUM = 1
		self.WITH_SWA = False
		self.SWA_START = 10
		self.SWA_FREQ = 10

		# loss
		self.LOSS_FUNC = 'CE'
		self.LOSSW_A = 5
		self.LOSSW_B = 5
		self.FOCALLOSS_GAMMA = 1.5
		self.OHEM = False
		self.OHEM_THRES = 0.6
		self.OHEM_KEEP = 200000
		self.AUX_LOSS_WEIGHTS = []

		# adabound
		self.BETA1 = 0.9
		self.BETA2 = 0.999
		self.GAMMA = 0.001
		self.TBLOG_FREQ = 40

		# test setting
		self.TEST_DATA_RESIZE = False
		#self.model_file = './tools/res_unet_test_southern_china_latest_clie.pth'
		self.model_file = './tools/fullclass_chinaall_3b.pth'
		self.TEST_GPUS = [0]
		self.TEST_BATCHES = 4
		self.VISUALIZE_FM = False
		self.WITH_TTA = False
		self.ROAD_TEST = False
		self.WITH_CLAHE = False
		self.TTACH = False
		self.TTA_SCALE = [0.5, 1, 1.5, 2]

		# auto test
		self.USE_DIST = False
		self.USE_CUDNN_BENCHMARK = True
		self.USE_GPU = True
		self.TEST_GPUS = [0]
		self.USE_FULL_OUT = False
		self.USE_SINGLE_OUT = False
		self.USE_COLOR_OUT = True
		self.USE_SHAPEFILE_OUT = False
		self.USE_BLOCK_INFO_OUT = False
		self.USE_OUTPUT_ROOT_RESET = False
		self.USE_TRANSPARENT_BACKGROUND = True
		self.TEST_GPU_SYNC = True
		self.LIST_FILE = None
		self.THRESHOLD = 0.5
		self.FOREGROUND_IDX = 0
		self.RESUME = False
		self.RETRY = 4
		self.FORCE_TO_SIMGLE_CLASS = None
		self.TIFF_COMPRESS = 'LZW'
		self.NON_BLOCKING = True
		self.USE_BINARY = True
		self.EXCEPTION_VALUE = 0
		self.IMG_SIZE = 2048
		self.PIXEL_OVERLAP = 0 
		self.BAND_LIST_FILE = ''
		self.GDAL_DRIVER = 'GTiff'
		self.DEBUG_FILE = './debug.txt'
		self.ERROR_LOG = './error.log'
		self.GEOJSON_OUT = False
		self.tta_flip = True
		self.tta_rotate = True
		self.with_tensorrt = False
		self.build_overviews = False
		self.DYNAMIC_MEAN = False

		self.params = []
		self.current_time = ''

		self.IS_AGRICULTURE_INFER = False
		self.AGRICULTURE_WINDOW_SIZE = 7

	def chack_path(self):
		self.current_time = (time.strftime("%y%m%d%H%M"), time.localtime())
		if not os.path.exists('./train_log'):
			os.makedirs('./train_log')
		params_log = './train_log/' + self.EXPERIMENT_NAME + '_' + self.DATA_NAME + '_' + self.MODEL_NAME + '_' + self.MODEL_BACKBONE + '_' + \
					 self.current_time[0] + '.log'
		print('user config:')
		with open(params_log, 'w') as f:
			self.params.sort()
			for line in self.params:
				print(line)
				f.write(line + '\n')

		if not os.path.isdir(self.LOG_DIR):
			os.makedirs(self.LOG_DIR)
		if not os.path.isdir(self.MODEL_SAVE_DIR):
			os.makedirs(self.MODEL_SAVE_DIR)

	def set_parse(self, kwargs):
		for k, v in kwargs.items():
			self.params.append(k + ':' + str(v))
			if not hasattr(self, k):
				warnings.warn("Warnings: cfg has not attribute %s" % k)
			setattr(self, k, v)


cfg = Configuration()
