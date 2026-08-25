def script_method(fn, _rcb=None):
    return fn


def script(obj, optimize=True, _frames_up=0, _rcb=None):
    return obj


import torch.jit
script_method1 = torch.jit.script_method
script1 = torch.jit.script
torch.jit.script_method = script_method
torch.jit.script = script

import sys
import os
import numpy as np
import cv2
import shutil
from .raster2line import raster2LineShp
from CLInferEngine.logger import logger

import torch as t
from .configs import opt
import networks
import ttach as tta_func
from config import cfg
from .info import info
from .dataset import InferDataMgr, reopen_infer_data_worker
from .imageio import ImageReader, ImageWriter, ImageUtils
from .utils import Utils
from .tta import TTAEncoder, TTADecoder
from .road_process import process_road
from CLInferEngine.clie_interfaces import model_interface, model_v1_interface
from torch.utils.data import DataLoader
from torch.autograd import Variable
from tqdm import tqdm
import time
import fire
import signal
from CLInferEngine.api import get_block_data
from CLInferEngine.api.zipshp import make_zip
from .agriculture_infer import img2points_group
import glob

# from secure.SecureCheck import secure_check
# secure_check()

from hist_stretch import percent_stretch_image
from osgeo import gdal, osr, ogr
try:
    from MessageClient.ProgressMessageSender import ProgressMessageSender
except:
    print('failed to load ProgressMessageSender package.')
    ProgressMessageSender = None
prg_sender = None

from pathlib import Path


def _available_cpu_count():
    """返回容器实际可用 CPU 数，兼容 affinity 与 cgroup v1/v2。"""
    counts = [os.cpu_count() or 1]
    if hasattr(os, 'sched_getaffinity'):
        try:
            counts.append(len(os.sched_getaffinity(0)))
        except (OSError, ValueError):
            pass

    try:
        with open('/sys/fs/cgroup/cpu.max', 'r', encoding='ascii') as cpu_max_file:
            quota_text, period_text = cpu_max_file.read().strip().split()[:2]
        if quota_text != 'max':
            quota = int(quota_text)
            period = int(period_text)
            if quota > 0 and period > 0:
                counts.append(max(1, quota // period))
    except (FileNotFoundError, OSError, ValueError):
        try:
            with open('/sys/fs/cgroup/cpu/cpu.cfs_quota_us', 'r', encoding='ascii') as quota_file:
                quota = int(quota_file.read().strip())
            with open('/sys/fs/cgroup/cpu/cpu.cfs_period_us', 'r', encoding='ascii') as period_file:
                period = int(period_file.read().strip())
            if quota > 0 and period > 0:
                counts.append(max(1, quota // period))
        except (FileNotFoundError, OSError, ValueError):
            pass
    return max(1, min(counts))


def _recommended_data_workers(gpu_count):
    """每张 GPU 默认准备 4 个切片 worker，并允许通过环境变量覆盖。"""
    available_cpus = _available_cpu_count()
    configured = os.environ.get('CLIE_DATA_WORKERS')
    if configured is not None:
        try:
            return max(0, min(available_cpus, int(configured)))
        except ValueError:
            pass
    workers_per_gpu = 4
    target = workers_per_gpu * max(1, gpu_count)
    return max(1, min(16, available_cpus, target))


def _classification_batch_per_gpu():
    """L20 分类默认每卡 8 张切片，可通过 CLIE_BATCH_PER_GPU 在 1～16 间调整。"""
    configured = os.environ.get('CLIE_BATCH_PER_GPU', '8')
    try:
        return max(1, min(16, int(configured)))
    except (TypeError, ValueError):
        return 8

def resample_to_2m_gdal(image_path):
    image_name = os.path.basename(image_path)
    if "GF02" not in image_name and "GF07" not in image_name and "GF2" not in image_name and "GF7" not in image_name:
        print("影像名称不包含 GF02 或 GF07，无需重采样")
        return image_path

    tmp_dir = Path.cwd() / "tmp"
    tmp_dir.mkdir(exist_ok=True)
    output_path_tmp = str(tmp_dir / f"tmp_{image_name}")
    output_path_final = str(tmp_dir / image_name)

    # 1. 打开原始影像
    src_ds = gdal.Open(image_path)
    if src_ds is None:
        raise RuntimeError(f"无法打开影像：{image_path}")

    proj_wkt = src_ds.GetProjection()
    src_gt = src_ds.GetGeoTransform()
    src_srs = osr.SpatialReference()
    src_srs.ImportFromWkt(proj_wkt)

    # 2. 判断是否是地理坐标系（单位为度）
    if src_srs.IsGeographic():
        # 自动计算 UTM 投影
        center_lon = src_gt[0] + 0.5 * src_gt[1] * src_ds.RasterXSize
        center_lat = src_gt[3] + 0.5 * src_gt[5] * src_ds.RasterYSize
        utm_zone = int((center_lon + 180) / 6) + 1
        is_north = center_lat >= 0

        utm_srs = osr.SpatialReference()
        utm_srs.SetUTM(utm_zone, is_north)
        utm_srs.SetWellKnownGeogCS("WGS84")
        dst_srs_wkt = utm_srs.ExportToWkt()
        print(f"原始坐标系为地理坐标系，临时投影为 UTM zone {utm_zone} ({'北半球' if is_north else '南半球'})")
    else:
        dst_srs_wkt = proj_wkt  # 保持原始投影（单位为米）

    # 3. 临时重采样为 2m（单位为米）
    warp_tmp = gdal.Warp(
        destNameOrDestDS=output_path_tmp,
        srcDSOrSrcDSTab=image_path,
        dstSRS=dst_srs_wkt,
        xRes=2,
        yRes=2,
        resampleAlg="bilinear",
        multithread=True,
        format="GTiff",
        creationOptions=["COMPRESS=LZW", "TILED=YES", "BIGTIFF=YES"],
        targetAlignedPixels=True
    )
    if warp_tmp is None:
        raise RuntimeError("重采样失败：GDAL Warp 临时阶段返回 None")
    del warp_tmp
    
    # 4. 将重采样结果再投影回原始投影
    warp_final = gdal.Warp(
        destNameOrDestDS=output_path_final,
        srcDSOrSrcDSTab=output_path_tmp,
        dstSRS=proj_wkt,
        resampleAlg="bilinear",
        multithread=True,
        format="GTiff",
        creationOptions=["COMPRESS=LZW", "TILED=YES", "BIGTIFF=YES"]
    )
    if warp_final is None:
        raise RuntimeError("投影回原始坐标失败")

    print(f"重采样完成：{output_path_final}")
    return output_path_final

def clip_and_mask_by_vector(image_path):
    wkt_str = os.environ.get("VECTOR_WKT")
    if not wkt_str:
        print("未设置 VECTOR_WKT，跳过裁剪")
        return image_path

    # 创建 tmp 文件夹
    tmp_dir = Path.cwd() / "tmp"
    tmp_dir.mkdir(exist_ok=True)
    output_path = tmp_dir / f"{Path(image_path).name}"

    # 打开影像
    src_ds = gdal.Open(image_path)
    if src_ds is None:
        raise RuntimeError("无法打开影像")
    src_proj_wkt = src_ds.GetProjection()
    src_gt = src_ds.GetGeoTransform()

    # 加载 WKT 几何（EPSG:3857）
    geom = ogr.CreateGeometryFromWkt(wkt_str)
    if geom is None:
        raise ValueError("VECTOR_WKT 无法解析为有效的 WKT 字符串")

    # 设置 WKT 坐标参考（默认是 EPSG:3857 墨卡托）
    wkt_sr = osr.SpatialReference()
    wkt_sr.ImportFromEPSG(3857)

    # 获取影像的坐标参考
    image_sr = osr.SpatialReference()
    image_sr.ImportFromWkt(src_proj_wkt)

    # 若坐标系不一致，则进行投影转换
    if not wkt_sr.IsSame(image_sr):
        transform = osr.CoordinateTransformation(wkt_sr, image_sr)
        geom.Transform(transform)

    # 获取投影后的矢量外接矩形
    minx, maxx, miny, maxy = geom.GetEnvelope()
    pixel_width = abs(src_gt[1])
    pixel_height = abs(src_gt[5])
    x_res = int((maxx - minx) / pixel_width)
    y_res = int((maxy - miny) / pixel_height)

    # 裁剪：用 gdal.Warp 裁剪出外接矩形区域
    tmp_cut_path = tmp_dir / f"rect_{Path(image_path).name}"
    warp_options = gdal.WarpOptions(
        format='GTiff',
        outputBounds=(minx, miny, maxx, maxy),
        width=x_res,
        height=y_res,
        dstSRS=image_sr.ExportToWkt(),
        resampleAlg='bilinear',
        creationOptions=['COMPRESS=LZW', 'TILED=YES', 'BIGTIFF=YES']
    )
    cut_result = gdal.Warp(str(tmp_cut_path), src_ds, options=warp_options)
    if cut_result is None:
        raise RuntimeError(f"裁剪分类影像失败: {tmp_cut_path}")
    cut_result = None

    # 打开裁剪后图像，复制为输出影像
    cut_ds = gdal.Open(str(tmp_cut_path))
    if cut_ds is None:
        raise RuntimeError(f"无法打开裁剪后的分类影像: {tmp_cut_path}")
    driver = gdal.GetDriverByName("GTiff")
    if driver is None:
        raise RuntimeError("GDAL GTiff 驱动不可用")
    masked_ds = driver.CreateCopy(
        str(output_path), cut_ds, 0,
        options=['COMPRESS=LZW', 'TILED=YES', 'BIGTIFF=YES']
    )
    if masked_ds is None:
        raise RuntimeError(f"创建 BigTIFF 裁剪结果失败: {output_path}")

    # 创建内存数据源和图层
    mem_ds = ogr.GetDriverByName('Memory').CreateDataSource('mem')
    mem_layer = mem_ds.CreateLayer('poly', srs=image_sr, geom_type=ogr.wkbPolygon)
    feat = ogr.Feature(mem_layer.GetLayerDefn())
    feat.SetGeometry(geom)
    mem_layer.CreateFeature(feat)

    # Rasterize 为掩膜（区域内为1）
    gdal.RasterizeLayer(
        masked_ds,
        [i + 1 for i in range(masked_ds.RasterCount)],
        mem_layer,
        burn_values=[1] * masked_ds.RasterCount,
        options=["ALL_TOUCHED=TRUE"]
    )

    # 掩掉区域外像素
    mask = masked_ds.GetRasterBand(1).ReadAsArray()
    for i in range(masked_ds.RasterCount):
        band = masked_ds.GetRasterBand(i + 1)
        data = band.ReadAsArray()
        data[mask == 0] = 0
        band.WriteArray(data)

    masked_ds.FlushCache()
    masked_ds = None

    print(str(output_path))
    return str(output_path)

class ProgressMessageSenderWrap():

    def __init__(self, bootstrap_servers='', topic='', taskId=None):
        try:
            self.prg_sender = ProgressMessageSender(bootstrap_servers, topic, taskId)
            if self.prg_sender.is_none():
                self.prg_sender = None
        except:
            self.prg_sender = None
        # 进度区间映射：默认 0→100（独立模式），子进程模式时由父进程通过环境变量设置
        self._progress_min = 0
        self._progress_max = 100
        self._last_subprocess_progress_key = None
        self._last_subprocess_infer_filename = None
        self._last_subprocess_infer_progress = None
        self._last_subprocess_send_time = 0.0
        try:
            self._subprocess_infer_progress_step = max(
                1, int(os.environ.get('CLIE_INFER_PROGRESS_STEP', '2'))
            )
        except (TypeError, ValueError):
            self._subprocess_infer_progress_step = 2
        try:
            self._subprocess_progress_max_interval = max(
                1.0, float(os.environ.get('CLIE_PROGRESS_MAX_INTERVAL', '3'))
            )
        except (TypeError, ValueError):
            self._subprocess_progress_max_interval = 3.0

    def set_progress_range(self, progress_min, progress_max):
        """设置进度映射区间。当 max < 100 时自动抑制 completed 状态（子进程模式）。"""
        self._progress_min = float(progress_min)
        self._progress_max = float(progress_max)

    def _map_progress(self, pct):
        """将 CLIE 内部 0→100 进度线性映射到父进程的进度区间"""
        if self._progress_min == 0 and self._progress_max == 100:
            return int(pct)
        return round(
            self._progress_min + (self._progress_max - self._progress_min) * (pct / 100.0),
            2,
        )

    def _is_subprocess_mode(self):
        return self._progress_max < 100

    def get_task_id(self):
        if self.prg_sender is not None:
            return self.prg_sender.get_task_id()

    def send(self, message_dict):
        if self._is_subprocess_mode():
            # 不生成瓦片缩略图；子进程仅按整数进度发送轻量消息。
            if 'progress' not in message_dict:
                return False
            message_dict.pop('inferPreviewFilename', None)
            # 子进程结束不代表整个分类步骤结束，避免提前把前端状态置为 completed。
            message_dict['runningStatus'] = 'running'
            message_dict['progress'] = self._map_progress(message_dict['progress'])
            infer_progress = message_dict.get('inferProgress')
            infer_filename = message_dict.get('inferFilename')
            now = time.monotonic()
            if infer_progress is None:
                # 初始化、创建写出指针等阶段消息只过滤完全相同的重复项。
                progress_key = (
                    infer_filename,
                    message_dict.get('progress'),
                    message_dict.get('runningInfo'),
                )
                if progress_key == self._last_subprocess_progress_key:
                    return False
                self._last_subprocess_progress_key = progress_key
            else:
                infer_progress = int(float(infer_progress))
                filename_changed = infer_filename != self._last_subprocess_infer_filename
                progress_advanced = (
                    self._last_subprocess_infer_progress is None
                    or infer_progress - self._last_subprocess_infer_progress
                    >= self._subprocess_infer_progress_step
                )
                interval_elapsed = (
                    now - self._last_subprocess_send_time
                    >= self._subprocess_progress_max_interval
                )
                if not (filename_changed or progress_advanced or interval_elapsed or infer_progress >= 100):
                    return False
                self._last_subprocess_infer_filename = infer_filename
                self._last_subprocess_infer_progress = infer_progress
            self._last_subprocess_send_time = now
        elif 'progress' in message_dict:
            message_dict['progress'] = self._map_progress(message_dict['progress'])

        if self.prg_sender is not None:
            return self.prg_sender.send(message_dict)
        print(f'[SIMULATED SEND] {message_dict}', flush=True)
        return False

    def close(self):
        if self.prg_sender is not None and hasattr(self.prg_sender, 'close'):
            self.prg_sender.close()

    def set_title(self, title=None, titleId=None):
        if self.prg_sender is not None:
            self.prg_sender.set_title(title, titleId)

    def set_source(self, source=None, rank=None):
        if self.prg_sender is not None:
            self.prg_sender.set_source(source, rank)

    def calc_progress_value(self, index, total, min_value=0, max_value=100):
        if self.prg_sender is not None:
            return self.prg_sender.calc_progress_value(index, total, min_value, max_value)
        if total > 0:
            return min_value + (max_value - min_value) * (index / total)
        return min_value

def get_band_count(tif_path):
    dataset = gdal.Open(tif_path)
    band_count = dataset.RasterCount
    dataset = None
    return band_count


"""
--model_file
--image_input_dir
--image_input_file
--image_output_dir
--job_id
"""
required_keys = ["color_table", "discription", "mean_value", "std_value", "model_num_classes", "threshold", "img_size", \
    "band_num", "model_output_stride", "foreground_idx", "model_name", "model_backbone", "model_state_dict"]
required_keys.sort()

def infer(kwargs):
    info()
    cfg.set_parse(kwargs)
    "start initialize server"

    bootstrap_servers = Utils.get_environ_var('KAFKA_SERVER_IP_PORT')
    topic = Utils.get_environ_var('KAFKA_TOPIC')
    task_id = Utils.get_environ_var('KAFKA_TASK_ID')
    global prg_sender
    prg_sender = ProgressMessageSenderWrap(bootstrap_servers, topic, task_id)

    # 从环境变量获取步骤标识（若未设置则使用默认值），与 change/fenlei 的消息格式对齐
    _clie_title = Utils.get_environ_var('CLIE_TITLE') or '语义分割推理'
    _clie_title_id = Utils.get_environ_var('CLIE_TITLE_ID') or 'clie_semantic_seg'
    _clie_source = Utils.get_environ_var('CLIE_SOURCE') or 'clie_module'
    _clie_rank_str = Utils.get_environ_var('CLIE_RANK')
    _clie_rank = int(_clie_rank_str) if _clie_rank_str is not None else -1
    prg_sender.set_title(title=_clie_title, titleId=_clie_title_id)
    prg_sender.set_source(source=_clie_source, rank=_clie_rank)

    # 从环境变量读取进度映射区间（子进程模式下由父进程 classification_core 设置）
    _prg_min_str = Utils.get_environ_var('CLIE_PROGRESS_MIN')
    _prg_max_str = Utils.get_environ_var('CLIE_PROGRESS_MAX')
    if _prg_min_str is not None and _prg_max_str is not None:
        prg_sender.set_progress_range(_prg_min_str, _prg_max_str)

    
    if not os.path.exists('./log/'):
        os.mkdir('./log/')
    log_path = os.path.join('./log/', cfg.job_id + '.log')
    logger.setFilePathAndLogLevel(log_path)
    logger.setStageAndProcess("Start", '100%').info("开始")
    requested_gpu = bool(cfg.USE_GPU)
    num_gpus = t.cuda.device_count() if requested_gpu and t.cuda.is_available() else 0
    cfg.USE_GPU = num_gpus > 0
    use_multi_gpu = num_gpus > 1
    batch_per_gpu = _classification_batch_per_gpu()
    if num_gpus > 0:
        cfg.TEST_BATCHES = (
            batch_per_gpu
            if use_multi_gpu and cfg.USE_DIST
            else batch_per_gpu * num_gpus
        )
    else:
        cfg.TEST_BATCHES = 1
    data_workers = _recommended_data_workers(num_gpus)
    cfg.PIN_MEMORY = cfg.USE_GPU

    local_rank = 0
    if use_multi_gpu and cfg.USE_DIST:
        _USE_DIST = True
        t.distributed.init_process_group(backend='nccl')
        local_rank = t.distributed.get_rank()
        t.cuda.set_device(local_rank)
        device = t.device('cuda', local_rank)
    else:
        _USE_DIST = False
        device = t.device('cuda', 0) if cfg.USE_GPU else t.device('cpu')
        if cfg.USE_GPU:
            t.cuda.set_device(0)

    if cfg.USE_GPU:
        t.backends.cudnn.benchmark = cfg.USE_CUDNN_BENCHMARK

    if cfg.USE_GPU:
        gpu_names = [t.cuda.get_device_name(index) for index in range(num_gpus)]
        logger.info(
                '分类推理设备: %s 张 GPU, batch_per_gpu=%s, batch_size=%s, data_workers=%s, devices=%s' % (
                num_gpus, batch_per_gpu, cfg.TEST_BATCHES, data_workers, gpu_names
            )
        )
    else:
        logger.warning(
            '未检测到可用 GPU，分类将回退到 CPU；batch_size=%s, data_workers=%s' % (
                cfg.TEST_BATCHES, data_workers
            )
        )

    rank_suffix = ''
    if _USE_DIST:
        rank_suffix = '_rank{}'.format(local_rank)
    # load weights and configs
    pth_path = cfg.model_file
    if os.path.isfile(pth_path):
        cfg.model_file = pth_path
    else:
        pth_list = glob.glob(os.path.join(pth_path, '*.pth'))
        cfg.model_file = pth_list[0]
    if local_rank == 0:
        logger.setStageAndProcess('Prepare', '0%').info("加载参数配置")
        logger.info('Loading Model from %s' % cfg.model_file)
    pretrained_dict = t.load(cfg.model_file, map_location='cpu')
    try:
        color_table = pretrained_dict["color_table"]
        cfg.mean_value = pretrained_dict["mean_value"]
        cfg.std_value = pretrained_dict["std_value"]
        cfg.MODEL_NUM_CLASSES = int(pretrained_dict["model_num_classes"])
        cfg.THRESHOLD = float(pretrained_dict["threshold"])
        cfg.IMG_SIZE = int(pretrained_dict["img_size"])
        cfg.BAND_NUM = int(pretrained_dict["band_num"])
        cfg.MODEL_OUTPUT_STRIDE = int(pretrained_dict["model_output_stride"])
        cfg.FOREGROUND_IDX = int(pretrained_dict["foreground_idx"])
        cfg.MODEL_NAME = pretrained_dict["model_name"]
        cfg.MODEL_BACKBONE = pretrained_dict["model_backbone"]
        cfg.PIXEL_OVERLAP = 0
        if 'upsample' in pretrained_dict.keys():
            cfg.UPSAMPLE = pretrained_dict["upsample"]
    except:
        logger.info('pth键不匹配，需要键包括：')
        logger.info(required_keys)
        logger.info('当前pth键为：')
        keys = sorted(pretrained_dict.keys())
        logger.info(keys)
        logger.info("即将加载默认参数")
    cfg.IMG_SIZE=1024
    band_list = [1+i for i in range(cfg.BAND_NUM)]
    try:
        cfg.IS_AGRICULTURE_INFER = pretrained_dict["Agriculture_bool"]
        cfg.AGRICULTURE_WINDOW_SIZE = pretrained_dict["window_size"]
    except:
        cfg.IS_AGRICULTURE_INFER = False
    cfg.set_parse(kwargs)
    if cfg.IS_AGRICULTURE_INFER:
        # 农业模型内部把单张切片再次拆点，目前只支持 batch=1。
        cfg.TEST_BATCHES = 1
        logger.info('农业分类模型强制使用 batch_size=1')
    elif num_gpus > 0:
        # cfg.set_parse 可能覆盖配置，普通分类在这里重新应用按卡 batch 设置。
        cfg.TEST_BATCHES = (
            batch_per_gpu
            if use_multi_gpu and cfg.USE_DIST
            else batch_per_gpu * num_gpus
        )
    else:
        cfg.TEST_BATCHES = 1
    # color_table = Utils.generate_color_table(cfg.COLOR_TABLE_FI
    task_input = cfg.image_input_file
    test_output_root = cfg.image_output_dir
    ### TO DO
    # set suffix with os.environ
    ###
    cfg.TEST_IMG_SUFFIX = [".tif", ".IMG", ".img", '.tiff']
    test_input_root = None
    test_input_filelist = None
    #  import pdb; pdb.set_trace()
    if os.path.isdir(task_input):
        test_input_root = task_input
    elif os.path.isfile(task_input):
        test_input_filelist = [task_input]
    else:
        logger.error('image_input_dir: 输入路径不存在，程序退出。')
    """
    TODO: add log file with the filename=log/cfg.job_id.log
    """
    "initialize server"

    if cfg.USE_OUTPUT_ROOT_RESET and (not _USE_DIST):
        Utils.check_path(test_output_root, reset=False)
    else:
        try:
            Utils.check_path(test_output_root, reset=False)
        except:
            logger.warning(
                'Failed to create {} ! It maybe has been created by another thread.'.format(test_output_root))
    if cfg.USE_FULL_OUT:
        full_out_root = test_output_root + '/out_full{}/'.format(rank_suffix)
        try:
            Utils.check_path(full_out_root, reset=False)
        except:
            logger.warning(
                'Failed to create {} ! It maybe has been created by another thread.'.format(full_out_root))
    if cfg.USE_SINGLE_OUT:
        single_out_root = test_output_root + '/out_single{}/'.format(rank_suffix)
        try:
            Utils.check_path(single_out_root, reset=False)
        except:
            logger.warning(
                'Failed to create {} ! It maybe has been created by another thread.'.format(single_out_root))
    if cfg.USE_COLOR_OUT:
        color_out_root = test_output_root
        try:
            Utils.check_path(color_out_root, reset=False)
        except:
            logger.warning(
                'Failed to create {} ! It maybe has been created by another thread.'.format(color_out_root))
    if cfg.USE_SHAPEFILE_OUT:
        if not (cfg.USE_FULL_OUT or cfg.USE_SINGLE_OUT or cfg.USE_COLOR_OUT):
            logger.error('ArcGIS shape file MUST be generated by full_out or single_out or color_out!')
            sys.exit(1)
        shapefile_out_root = test_output_root + '/out_shapefile{}/'.format(rank_suffix)
        try:
            Utils.check_path(shapefile_out_root, reset=False)
        except:
            logger.warning(
                'Failed to create {} ! It maybe has been created by another thread.'.format(shapefile_out_root))
    if cfg.USE_BLOCK_INFO_OUT:
        block_info_out_path = os.path.join(test_output_root, 'blocks')
        try:
            Utils.check_path(block_info_out_path, reset=False)
        except:
            logger.warning(
                'Failed to create {} ! It maybe has been created by another thread.'.format(block_info_out_path))
    if _USE_DIST:
        t.distributed.barrier()


    output_suffix = Utils.generate_suffix(cfg.GDAL_DRIVER)
    width = cfg.IMG_SIZE
    height = cfg.IMG_SIZE
    if local_rank == 0:
        logger.setStageAndProcess("Prepare", '30%').info("定义模型")
    model = networks.GenerateNet(cfg)
    module_model_state_dict = dict()
    # import ipdb;ipdb.set_trace()
    for item, value in pretrained_dict['model_state_dict'].items():
        item = item.replace('module.', '')
        if (item in model.state_dict()) and (value.shape == model.state_dict()[item].shape):
            module_model_state_dict[item] = value
    try:
        model.load_state_dict(module_model_state_dict, strict=True)
        model.eval()
    except:
        logger.error('加载模型权重失败')
        return
    if local_rank == 0:
        logger.setStageAndProcess('Prepare', '40%').info("加载模型权重")
    if cfg.USE_GPU:
        if use_multi_gpu:
            if _USE_DIST:
                model.to(device)
                model = t.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)
            else:
                model.to(device)
                model = t.nn.DataParallel(
                    model,
                    device_ids=list(range(num_gpus)),
                    output_device=0,
                )
        else:
            model.to(device)
    if cfg.with_tensorrt:
        from torch2trt import torch2trt
        with torch.no_grad():
            input_data = torch.rand((1, cfg.BAND_NUM, cfg.IMG_SIZE,  cfg.IMG_SIZE), dtype=torch.float).cuda()
            model = torch2trt(model, [input_data], max_batch_size=cfg.TEST_BATCHES, fp16_mode=True)
    if local_rank == 0:
        logger.setStageAndProcess('Prepare', '50%').info("构建模型成功")


    if cfg.WITH_TTA:
        tta_params = {
            'tta_flip': cfg.tta_flip,
            'tta_rotate': cfg.tta_rotate
        }
        tta_encoder = TTAEncoder(tta_params)
        tta_decoder = TTADecoder(tta_params)

    if local_rank == 0:
        logger.setStageAndProcess('Prepare', '60%').info("遍历文件夹")
    if test_input_filelist is not None:
        basename_list, filename_list = Utils.generate_filelist_for_aiserver(test_input_filelist)
        if local_rank == 0:
            logger.info('找到 ' + str(len(basename_list)) + ' 个文件在列表 ' + str(test_input_filelist))
    elif cfg.LIST_FILE is not None:
        basename_list, filename_list = Utils.generate_filelist(cfg.LIST_FILE, cfg.image_input_dir)
        if local_rank == 0:
            logger.info('找到 ' + str(len(basename_list)) + ' 个文件在列表 ' + cfg.LIST_FILE)
    else:
        basename_list, filename_list = Utils.generate_baselist(test_input_root, cfg.TEST_IMG_SUFFIX)
        # filename_list = Utils.generate_list(test_input_root, basename_list, cfg.TEST_IMG_SUFFIX)
        if local_rank == 0:
            logger.info('找到 ' + str(len(basename_list)) + ' 个文件在目录 ' + test_input_root)
    if local_rank == 0:
        logger.setStageAndProcess('Prepare', '80%').info("遍历文件夹成功")
    if _USE_DIST:
        basename_list_list, filename_list_list = Utils.split_filename_list_for_dist(basename_list, filename_list,
                                                                                    split=num_gpus)
    else:
        basename_list_list, filename_list_list = Utils.split_filename_list_for_dist(basename_list, filename_list,
                                                                                    split=1)
    basename_list_list, filename_list_list = Utils.set_resume_list(basename_list_list, filename_list_list,
                                                                   resume=cfg.RESUME)
    if local_rank == 0:
        logger.setStageAndProcess('Prepare', '100%').info('共加载' + str(len(basename_list_list[local_rank])) + '张影像')
    if cfg.BAND_LIST_FILE != '':
        band_list = Utils.load_band_list_file(cfg.BAND_LIST_FILE)
    # mean_value = Utils.load_mean_file(cfg.MEAN_FILE)
    # std_value = Utils.load_std_file(cfg.STD_FILE)

    img_blank = np.zeros((height, width), dtype=np.uint8)

    _infer_task_prg_min = 3
    _infer_task_prg_max = 99
    print('[INFO] Test task started.')

    if local_rank == 0:
        logger.info('开始预测')
    pbar0 = tqdm(total=len(basename_list_list[local_rank]), bar_format='{l_bar}{bar:30}{r_bar}')
    for idx in range(len(basename_list_list[local_rank])):
        if os.path.exists(cfg.DEBUG_FILE):
            import ipdb
            ipdb.set_trace()

        _sub_task_prg_min = prg_sender.calc_progress_value(idx, len(basename_list_list[local_rank]), min_value=_infer_task_prg_min, max_value=_infer_task_prg_max)
        _sub_task_prg_max = prg_sender.calc_progress_value(idx + 1, len(basename_list_list[local_rank]), min_value=_infer_task_prg_min, max_value=_infer_task_prg_max)
        
        basename = basename_list_list[local_rank][idx]
        filename = filename_list_list[local_rank][idx]
        
        filename = resample_to_2m_gdal(filename)
        # filename = clip_and_mask_by_vector(filename)
        
        tqdm.write('[INFO] %s Index: %d / %d' % (time.strftime('[%y%m%d_%H:%M:%S]'), idx + 1, len(basename_list_list[local_rank])))
        tqdm.write('[INFO] %s  Reading: %s' % (time.strftime('[%y%m%d_%H:%M:%S]'), filename))

        logger.info('%s Index: %d / %d' % (
        time.strftime('[%y%m%d_%H:%M:%S]'), idx + 1, len(basename_list_list[local_rank])))
        logger.info('%s  Reading: %s' % (time.strftime('[%y%m%d_%H:%M:%S]'), filename))

        message_dict = {'progress': _sub_task_prg_min, 'runningStatus': 'running', 'runningInfo': '构建CLIE数据加载器'}
        prg_sender.send(message_dict)


        test_data = InferDataMgr(filename, band_list, width, height, cfg.PIXEL_OVERLAP, cfg.mean_value, cfg.std_value,  cfg.WITH_CLAHE, \
            upsample=cfg.UPSAMPLE, dynamic_mean=cfg.DYNAMIC_MEAN)
        # import pdb;pdb.set_trace()
        if test_data.image_reader.nbands < cfg.BAND_NUM:
            logger.info(
                '影像: {} 无法正常预测，影像波段数: {} 低于模型要求波段数: {} '.format(filename, test_data.image_reader.nbands, cfg.BAND_NUM))
            continue
        _retry = 0
        while test_data.image_reader is None:
            try:
                test_data = InferDataMgr(filename, band_list, width, height, cfg.PIXEL_OVERLAP, cfg.mean_value, cfg.std_value,  cfg.WITH_CLAHE, \
                    upsample=cfg.UPSAMPLE, dynamic_mean=cfg.DYNAMIC_MEAN)
            except:
                test_data.image_reader = None
                _retry += 1
                if _retry > cfg.RETRY:
                    break
                logger.warning(time.strftime('[%y%m%d_%H:%M:%S]') + ' Retry ' + str(
                    _retry) + ' times for reading ' + filename)
        if test_data.image_reader is None:
            logger.warning(time.strftime('[%y%m%d_%H:%M:%S]') + '  Skip predicting ' + filename)
            if cfg.ERROR_LOG is not None:
                with open(cfg.ERROR_LOG, 'a') as f:
                    f.write('[WARNING] ' + time.strftime('[%y%m%d_%H:%M:%S]') + '  Skip predicting ' + filename)
            continue

        dataloader_kwargs = {
            'batch_size': cfg.TEST_BATCHES,
            'shuffle': False,
            'num_workers': data_workers,
            'pin_memory': cfg.PIN_MEMORY,
        }
        if data_workers > 0:
            dataloader_kwargs.update({
                'prefetch_factor': 1,
                'persistent_workers': True,
                'worker_init_fn': reopen_infer_data_worker,
            })
        test_dataloader = DataLoader(test_data, **dataloader_kwargs)

        message_dict = {'progress': _sub_task_prg_min, 'runningStatus': 'running', 'runningInfo': '构建文件写出指针'}
        prg_sender.send(message_dict)

        "sever taskClear"
        pictureRange = Utils.get_rect_geoinfo(test_data.image_reader.coord, test_data.image_reader.width,
                                              test_data.image_reader.height)

        datas = get_block_data(pictureRange, test_data.image_reader.coord[1], test_data.image_reader.coord[5],
                                  cfg.IMG_SIZE - cfg.PIXEL_OVERLAP, \
                                  estTime=0.04 * len(test_dataloader))
        logger.setStageAndProcess("Block", '100%').info(datas)
        if local_rank == 0 and cfg.USE_BLOCK_INFO_OUT:
            with open(os.path.join(test_output_root, 'blocks.txt'), 'w') as f:
                f.write(datas)

        logger.setStageAndProcess('影像:' + os.path.basename(filename) + '预测进度', '0%').info("预测开始")
        if cfg.USE_FULL_OUT:
            full_out_filename = full_out_root + '/' + basename + output_suffix
            try:
                image_writer_full = ImageWriter(full_out_filename, test_data.image_reader.width,
                                                test_data.image_reader.height, \
                                                test_data.image_reader.nbands + 1, compress=cfg.TIFF_COMPRESS)
                image_writer_full.set_proj(test_data.image_reader.proj)
                image_writer_full.set_coord(test_data.image_reader.coord)
                if cfg.USE_TRANSPARENT_BACKGROUND:
                    image_writer_full.set_no_data_value(0)
                if cfg.build_overviews:
                    image_writer_full.build_overviews()
            except:
                tqdm.write(
                    '[WARNING] ' + time.strftime('[%y%m%d_%H:%M:%S]') + '  Failed to create ' + full_out_filename)
                if cfg.ERROR_LOG is not None:
                    with open(cfg.ERROR_LOG, 'a') as f:
                        f.write('[WARNING] ' + time.strftime(
                            '[%y%m%d_%H:%M:%S]') + '  Failed to create ' + full_out_filename)
                image_writer_full = None

        if cfg.USE_SINGLE_OUT:
            single_out_filename = single_out_root + '/' + basename + output_suffix
            try:
                image_writer_single = ImageWriter(single_out_filename, test_data.image_reader.width,
                                                  test_data.image_reader.height, 1, compress=cfg.TIFF_COMPRESS)
                image_writer_single.set_proj(test_data.image_reader.proj)
                image_writer_single.set_coord(test_data.image_reader.coord)
                if cfg.USE_TRANSPARENT_BACKGROUND:
                    image_writer_single.set_no_data_value(0)
                if cfg.build_overviews:
                    image_writer_single.build_overviews()
            except:
                tqdm.write(
                    '[WARNING] ' + time.strftime('[%y%m%d_%H:%M:%S]') + '  Failed to create ' + single_out_filename)
                if cfg.ERROR_LOG is not None:
                    with open(cfg.ERROR_LOG, 'a') as f:
                        f.write('[WARNING] ' + time.strftime(
                            '[%y%m%d_%H:%M:%S]') + '  Failed to create ' + single_out_filename)
                image_writer_single = None

        if cfg.USE_COLOR_OUT:
            color_out_filename = color_out_root + '/' + basename + output_suffix
            try:
                image_writer_color = ImageWriter(color_out_filename, test_data.image_reader.width,
                                                 test_data.image_reader.height, 1, compress=cfg.TIFF_COMPRESS)
                image_writer_color.set_proj(test_data.image_reader.proj)
                image_writer_color.set_coord(test_data.image_reader.coord)
                image_writer_color.set_color_table(color_table)
                if cfg.USE_TRANSPARENT_BACKGROUND:
                    image_writer_color.set_no_data_value(0)
                if cfg.build_overviews:
                    image_writer_color.build_overviews()
            except:
                tqdm.write(
                    '[WARNING] ' + time.strftime('[%y%m%d_%H:%M:%S]') + '  Failed to create ' + color_out_filename)
                if cfg.ERROR_LOG is not None:
                    with open(cfg.ERROR_LOG, 'a') as f:
                        f.write('[WARNING] ' + time.strftime(
                            '[%y%m%d_%H:%M:%S]') + '  Failed to create ' + color_out_filename)
                image_writer_color = None

        message_dict = {'progress': _sub_task_prg_min, 'runningStatus': 'running', 'runningInfo': f'推理任务-{idx + 1}/{len(basename_list_list[local_rank])}',
                        'inferProgress': 0, 'inferFilename': filename}
        prg_sender.send(message_dict)
        coord = list(test_data.image_reader.coord)
        proj = osr.SpatialReference(wkt=test_data.image_reader.proj)
        epsg = proj.GetAttrValue('AUTHORITY', 1)
        spatial_ref = test_data.image_reader.dataset.GetSpatialRef()
        proj_name = spatial_ref.GetName()

        # pbar1 = tqdm(total=len(test_dataloader), bar_format='{l_bar}{bar:30}{r_bar}')
        for idx_x, (img, img_mask, idx_i, idx_j, status) in enumerate(test_dataloader):
            logger.setStageAndProcess('影像:' + os.path.basename(filename) + '预测进度', str(100 * (idx_x + 1) / len(test_dataloader)) + '%').info(
                "预测id:" + str(idx_i) + "_" + str(idx_j))

            # 初始化预测结果数组
            pred = np.zeros((len(status), height, width), dtype=np.uint8)

            # 找出状态为'good'的索引
            good_indices = [i for i, s in enumerate(status) if s == 'good']

            if good_indices:  # 只有当存在good状态的切片时才进行预测
                if cfg.WITH_TTA:
                    # TTA预测只对good状态的切片进行
                    good_img = img[good_indices]
                    good_img = good_img.data.cpu().numpy().transpose(0, 2, 3, 1)
                    img_tta_set = tta_encoder.tta(good_img)
                    for i in range(len(img_tta_set)):
                        img_tta_set[i] = t.from_numpy(img_tta_set[i].transpose(0, 3, 1, 2))

                    with t.no_grad():
                        scores_tta = []
                        for i in range(len(img_tta_set)):
                            data = Variable(img_tta_set[i])
                            if cfg.USE_GPU:
                                if _USE_DIST:
                                    data = data.cuda(local_rank, non_blocking=cfg.NON_BLOCKING)
                                else:
                                    data = data.cuda()
                            scores = model(data)
                            scores_tta.append(scores.data.cpu().numpy().transpose(0, 2, 3, 1))
                        scores = tta_decoder.tta(scores_tta)
                        scores_sum = 0
                        for i in range(len(scores)):
                            scores[i] = t.from_numpy(scores[i].transpose(0, 3, 1, 2))
                            scores_sum += t.sigmoid(scores[i])

                        if cfg.MODEL_NUM_CLASSES > 1:
                            good_pred = scores_sum.data.max(1)[1].cpu().numpy()
                        else:
                            good_pred = scores_sum.data.cpu().numpy()
                            good_pred = good_pred[:, 0, :, :]
                            if cfg.USE_BINARY:
                                good_pred[good_pred < cfg.THRESHOLD] = 0
                                good_pred[good_pred >= cfg.THRESHOLD] = 1
                            else:
                                good_pred *= 255
                            good_pred = good_pred.astype(np.uint8)

                    # 将good切片的预测结果放回对应位置
                    for i, orig_idx in enumerate(good_indices):
                        pred[orig_idx] = good_pred[i]

                elif cfg.IS_AGRICULTURE_INFER:
                    # 农业推理只对good状态的切片进行
                    t1 = time.time()
                    window_size = cfg.AGRICULTURE_WINDOW_SIZE
                    good_img = img[good_indices]
                    good_img = good_img.data.cpu().numpy()
                    if cfg.TEST_BATCHES != 1:
                        tqdm.write('[ERROR] THE BATCH OF AGRICULTURE_INFER MUST BE 1')
                        break

                    agricluture_clip = 4
                    data_all = img2points_group(good_img, window_size)
                    img_pred_clip = np.zeros([1, height, width])

                    for i in range(data_all.shape[0] // agricluture_clip):
                        data = data_all[i * agricluture_clip:(i + 1) * agricluture_clip, :, :, :, :].reshape(
                            data_all.shape[1] * agricluture_clip, data_all.shape[2], window_size, window_size)
                        with torch.no_grad():
                            data = Variable(data)
                        if cfg.USE_GPU:
                            data = data.cuda()
                        if cfg.MODEL_NAME == 'Agriculture':
                            score = model(data)
                        else:
                            score = model(data)[0]
                        if cfg.MODEL_NUM_CLASSES > 1:
                            pred_batch = score.data.max(1)[1].cpu().numpy()
                        else:
                            pred_batch = torch.sigmoid(score).data.cpu().numpy()
                            pred_batch = pred_batch[:, 0, :, :]
                            pred_batch[pred_batch >= cfg.THRESHOLD] = 1
                            pred_batch[pred_batch < cfg.THRESHOLD] = 0
                        img_pred_clip[:, i * agricluture_clip:(i + 1) * agricluture_clip, :] = pred_batch.reshape(
                            agricluture_clip, data_all.shape[1], 1).transpose(2, 0, 1)

                    if (data_all.shape[0] // agricluture_clip * agricluture_clip) != data_all.shape[0]:
                        data = data_all[(data_all.shape[0] // agricluture_clip * agricluture_clip):, :, :, :, :].view(
                            data_all.shape[1] * (
                                        data_all.shape[0] - (data_all.shape[0] // agricluture_clip * agricluture_clip)),
                            data_all.shape[2], window_size, window_size)
                        with torch.no_grad():
                            data = Variable(data)
                        if cfg.USE_GPU:
                            data = data.cuda()
                        if cfg.MODEL_NAME == 'Agriculture':
                            score = model(data)
                        else:
                            score = model(data)[0]
                        if cfg.MODEL_NUM_CLASSES > 1:
                            pred_batch = score.data.max(1)[1].cpu().numpy()
                        else:
                            pred_batch = torch.sigmoid(score).data.cpu().numpy()
                            pred_batch = pred_batch[:, 0, :, :]
                            pred_batch[pred_batch >= cfg.THRESHOLD] = 1
                            pred_batch[pred_batch < cfg.THRESHOLD] = 0
                        img_pred_clip[:, (data_all.shape[0] // agricluture_clip * agricluture_clip):,
                        :] = pred_batch.reshape(
                            (data_all.shape[0] - (data_all.shape[0] // agricluture_clip * agricluture_clip)),
                            data_all.shape[1], 1).transpose(2, 0, 1)

                    # 将good切片的预测结果放回对应位置
                    for i, orig_idx in enumerate(good_indices):
                        pred[orig_idx] = img_pred_clip[i]

                else:
                    # 普通推理只对good状态的切片进行
                    with t.no_grad():
                        good_data = Variable(img[good_indices])
                        if cfg.USE_GPU:
                            if _USE_DIST:
                                good_data = good_data.cuda(local_rank, non_blocking=cfg.NON_BLOCKING)
                            else:
                                good_data = good_data.cuda(non_blocking=cfg.NON_BLOCKING)

                        if cfg.TTACH and cfg.FOREGROUND_IDX == 7:
                            tta_transformes = tta_func.Compose([
                                tta_func.Scale(scales=cfg.TTA_SCALE, interpolation='bilinear')
                            ])
                            tta_model = tta_func.SegmentationTTAWrapper(model, tta_transformes, merge_mode='mean')
                            good_score = tta_model(good_data)
                        else:
                            good_score = model(good_data)

                        if cfg.MODEL_NUM_CLASSES > 1:
                            good_pred = torch.argmax(good_score, 1).cpu().numpy()
                        else:
                            good_score = t.sigmoid(good_score)
                            good_pred = good_score.data.cpu().numpy()
                            good_pred = good_pred[:, 0, :, :]
                            if cfg.USE_BINARY:
                                good_pred[good_pred >= cfg.THRESHOLD] = 1
                                good_pred[good_pred < cfg.THRESHOLD] = 0
                            else:
                                good_pred *= 255
                            good_pred = good_pred.astype(np.uint8)

                    # 将good切片的预测结果放回对应位置
                    for i, orig_idx in enumerate(good_indices):
                        pred[orig_idx] = good_pred[i]
            #
            for i in range(pred.shape[0]):
                if status[i] == 'good':
                    img_pred = pred[i]
                else:
                    img_pred = img_blank
                img_pred[:, :][img_mask[i] == 0] = 0
                if cfg.FOREGROUND_IDX > 0:
                    img_pred[img_pred == 1] = cfg.FOREGROUND_IDX
                height_down = cfg.PIXEL_OVERLAP // 2
                height_up = height - cfg.PIXEL_OVERLAP // 2
                width_down = cfg.PIXEL_OVERLAP // 2
                width_up = width - cfg.PIXEL_OVERLAP // 2
                img_pred = img_pred[height_down:height_up, width_down:width_up, None]

                if cfg.FORCE_TO_SIMGLE_CLASS is not None:
                    if isinstance(cfg.FORCE_TO_SIMGLE_CLASS, list):
                        for ii in range(cfg.MODEL_NUM_CLASSES + 1):
                            if ii not in cfg.FORCE_TO_SIMGLE_CLASS:
                                img_pred[img_pred == ii] = 0
                            else:
                                img_pred[img_pred == ii] = cfg.FORCE_TO_SIMGLE_CLASS[0]
                    else:
                        img_pred[img_pred != cfg.FORCE_TO_SIMGLE_CLASS] = 0
                        # img_pred[img_pred == cfg.FORCE_TO_SIMGLE_CLASS] = 1

                writer_height_down = int(idx_i[i]) * (height - cfg.PIXEL_OVERLAP)
                writer_height_up = (int(idx_i[i]) + 1) * (height - cfg.PIXEL_OVERLAP)
                writer_width_down = int(idx_j[i]) * (width - cfg.PIXEL_OVERLAP)
                writer_width_up = (int(idx_j[i]) + 1) * (width - cfg.PIXEL_OVERLAP)
                if writer_height_up > test_data.image_reader.height:
                    img_pred = img_pred[
                               :(height - cfg.PIXEL_OVERLAP - writer_height_up + test_data.image_reader.height), :, :]
                    writer_height_up = test_data.image_reader.height
                if writer_width_up > test_data.image_reader.width:
                    img_pred = img_pred[:,
                               :(width - cfg.PIXEL_OVERLAP - writer_width_up + test_data.image_reader.width), :]
                    writer_width_up = test_data.image_reader.width

                # import pudb;pu.db
                if cfg.EXCEPTION_VALUE is not None:
                    _img = test_data.image_reader.read_image( \
                            read_range=(writer_width_down, writer_height_down, writer_width_up - writer_width_down, writer_height_up - writer_height_down))
                    _exp_val = [cfg.EXCEPTION_VALUE] * len(test_data.image_reader.band_list)
                    _loc = np.all((_img == _exp_val), axis=2)
                    img_pred[_loc] = 0

                # """dict_vec_temp out"""
                # if cfg.USE_VEC_TEMP_OUT:
                #     vec_temp_filename = os.path.join(vec_temp_out_root,
                #                                      basename + "_" + idx_i[i] + "_" + idx_j[i] + ".shp")
                #     crop_out = img_pred[:, :, 0]

                #     geo_info = Utils.get_clip_geoinfo(test_data.image_reader.coord, height - cfg.PIXEL_OVERLAP,
                #                                       width - cfg.PIXEL_OVERLAP, int(idx_i[i]), int(idx_j[i]))
                #     dict_vec_temp = '{\n"type": "FeatureCollection",\n"name": "pred",\n"features": [\n'
                #     dict_vec_temp += ImageUtils.polygonize_mem(crop_out, test_data.image_reader.proj, geo_info,
                #                                                vec_temp_filename, 1)
                #     dict_vec_temp += ']\n}\n'
                    
                    # 切片数量 切片id 名称 进度

                if cfg.USE_FULL_OUT and (image_writer_full is not None):
                    _img = test_data.image_reader.read_image(
                        read_range=(writer_width_down, writer_height_down, writer_width_up - writer_width_down,
                                    writer_height_up - writer_height_down))
                    _img = np.append(_img, img_pred, axis=2)
                    image_writer_full.write_image(_img, write_offset=(writer_width_down, writer_height_down))
                if cfg.USE_SINGLE_OUT and (image_writer_single is not None):
                    image_writer_single.write_image(img_pred, write_offset=(writer_width_down, writer_height_down))
                if cfg.USE_COLOR_OUT and (image_writer_color is not None):
                    image_writer_color.write_image(img_pred, write_offset=(writer_width_down, writer_height_down))

                processing_tag = str(int((idx_x+1) / (len(test_dataloader)) * 100)) + '%'
                total_tag = str((idx / len(basename_list_list[local_rank]) * 100))[:6] + '%'
                if len(basename_list_list[local_rank]) == 1:
                    total_tag = processing_tag
                logger.setStageAndProcess("Process", processing_tag).info(str(idx_x+1) + '/' + str(len(test_dataloader)) + ' ' + idx_i[i] + "_" + idx_j[i]  + ' ' + os.path.basename(filename)  + ' ' + total_tag)
                if cfg.USE_BLOCK_INFO_OUT:
                    with open(os.path.join(test_output_root, 'blocks', basename_list_list[local_rank][idx] + '_' + idx_i[i] + "_" + idx_j[i] + '.txt'), 'w') as f:
                        f.write(str(idx_x+1) + '/' + str(len(test_dataloader)))

            x_tl = coord[0] + writer_width_down * coord[1]
            y_tl = coord[3] + writer_height_down * coord[5]
            x_tr = coord[0] + (writer_width_down + img_pred.shape[1]) * coord[1]
            y_tr = coord[3] + writer_height_down * coord[5]
            x_dl = coord[0] + writer_width_down * coord[1]
            y_dl = coord[3] + (writer_height_down + img_pred.shape[0]) * coord[5]
            x_dr = coord[0] + (writer_width_down + img_pred.shape[1]) * coord[1]
            y_dr = coord[3] + (writer_height_down + img_pred.shape[0]) * coord[5]
            x1, y1 = x_tl, y_tl
            x2, y2 = x_tr, y_tr
            x3, y3 = x_dr, y_dr
            x4, y4 = x_dl, y_dl
            '''
            epsg
            4326 wgs84
            3857 web mercator
            '''
            _infer_geo_info = None
            if proj_name in ['WGS 84', 'CGCS2000', 'WGS 84 / Pseudo-Mercator']:
                if proj_name == 'WGS 84 / Pseudo-Mercator':
                    x1, y1 = Utils.webmercator_to_lonlat(x1, y1)
                    x2, y2 = Utils.webmercator_to_lonlat(x2, y2)
                    x3, y3 = Utils.webmercator_to_lonlat(x3, y3)
                    x4, y4 = Utils.webmercator_to_lonlat(x4, y4)
                _infer_geo_info = f'POLYGON(({x1} {y1}, {x2} {y2}, {x3} {y3}, {x4} {y4}, {x1} {y1}))'
            infer_geo_info = f'[{_infer_geo_info}]'

            _prg_idx = prg_sender.calc_progress_value(idx_x + 1, len(test_dataloader), min_value=_sub_task_prg_min, max_value=_sub_task_prg_max)
            _sub_prg_idx = prg_sender.calc_progress_value(idx_x + 1, len(test_dataloader), min_value=0, max_value=99)
            message_dict = {'progress': _prg_idx, 'runningStatus': 'running',
                            'runningInfo': f'分类推理 {os.path.basename(filename)}: {idx_x + 1}/{len(test_dataloader)}批次 ({_sub_prg_idx}%)',
                            'inferProgress': _sub_prg_idx, 'inferFilename': filename,
                            'inferBatchCurrent': idx_x + 1, 'inferBatchTotal': len(test_dataloader),
                            'inferGeoInfo': infer_geo_info}
            prg_sender.send(message_dict)

        logger.setStageAndProcess("Process", '100%').info('完成预测: ' + os.path.basename(filename))
       
        #     pbar1.update(1)

        # pbar1.close()
        if cfg.USE_FULL_OUT:
            full_out_filename = full_out_root + '/' + basename + output_suffix
            tqdm.write('[INFO] %s  Writing: %s' % (time.strftime('[%y%m%d_%H:%M:%S]'), full_out_filename))
            try:
                image_writer_full.close()
            except Exception as exc:
                tqdm.write('[WARNING] ' + time.strftime('[%y%m%d_%H:%M:%S]') + '  Failed to write ' + full_out_filename)
                if cfg.ERROR_LOG is not None:
                    with open(cfg.ERROR_LOG, 'a') as f:
                        f.write('[WARNING] ' + time.strftime(
                            '[%y%m%d_%H:%M:%S]') + '  Failed to write ' + full_out_filename)
                raise RuntimeError(f'分类结果写盘失败: {full_out_filename}: {exc}') from exc

        if cfg.USE_SINGLE_OUT:
            single_out_filename = single_out_root + '/' + basename + output_suffix
            tqdm.write('[INFO] %s  Writing: %s' % (time.strftime('[%y%m%d_%H:%M:%S]'), single_out_filename))
            try:
                image_writer_single.close()
            except Exception as exc:
                tqdm.write(
                    '[WARNING] ' + time.strftime('[%y%m%d_%H:%M:%S]') + '  Failed to write ' + single_out_filename)
                if cfg.ERROR_LOG is not None:
                    with open(cfg.ERROR_LOG, 'a') as f:
                        f.write('[WARNING] ' + time.strftime(
                            '[%y%m%d_%H:%M:%S]') + '  Failed to write ' + single_out_filename)
                raise RuntimeError(f'分类结果写盘失败: {single_out_filename}: {exc}') from exc

        if cfg.USE_COLOR_OUT:
            color_out_filename = color_out_root + '/' + basename + output_suffix
            tqdm.write('[INFO] %s  写入文件: %s' % (time.strftime('[%y%m%d_%H:%M:%S]'), color_out_filename))
            try:
                image_writer_color.close()
                # here is the road process part
                if cfg.ROAD_TEST:
                    logger.info('道路中心线提取中')
                    # import pudb;pu.db
                    pred_path = color_out_root + basename + output_suffix
                    line_path = color_out_root.replace('_color', '_centerlineshp')
                    zip_path = color_out_root.replace('_color', '_compress.zip')
                    center_path = color_out_root + '{}_center{}'.format(basename, output_suffix)
                    # filter and extract center line
                    process_road(pred_path, pred_path, center_path, 0.2, True)
                    shp_path = os.path.join(line_path, basename + '.shp')
                    Utils.check_path(line_path, reset=True)
                    raster2LineShp(center_path, shp_path)
            except Exception as exc:
                logger.warning(
                    time.strftime('[%y%m%d_%H:%M:%S]') + '  Failed to write ' + color_out_filename)
                if cfg.ERROR_LOG is not None:
                    with open(cfg.ERROR_LOG, 'a') as f:
                        f.write('[WARNING] ' + time.strftime(
                            '[%y%m%d_%H:%M:%S]') + '  Failed to write ' + color_out_filename)
                raise RuntimeError(f'分类结果写盘或后处理失败: {color_out_filename}: {exc}') from exc
            #filter_color_from_tif(color_out_filename)
            logger.setStageAndProcess("tifName", "100").info(color_out_filename)

        test_data.image_reader.close()
        if cfg.USE_SHAPEFILE_OUT:
            shapefile_out_filename = shapefile_out_root + '/' + basename + '.shp'
            logger.info('%s  写入文件: %s' % (time.strftime('[%y%m%d_%H:%M:%S]'), shapefile_out_filename))
            try:
                if cfg.USE_FULL_OUT and (image_writer_full is not None):
                    ImageUtils.polygonize(full_out_filename, shapefile_out_filename, test_data.image_reader.nbands + 1)
                elif cfg.USE_SINGLE_OUT and (image_writer_single is not None):
                    ImageUtils.polygonize(single_out_filename, shapefile_out_filename, 1)
                elif cfg.USE_COLOR_OUT and (image_writer_color is not None):
                    ImageUtils.polygonize(color_out_filename, shapefile_out_filename, 1)
                else:
                    logger.error('Can not find valid raster file.')
            except:
                logger.warning(
                    time.strftime('[%y%m%d_%H:%M:%S]') + '  Failed to write ' + shapefile_out_filename)
                if cfg.ERROR_LOG is not None:
                    with open(cfg.ERROR_LOG, 'a') as f:
                        f.write('[WARNING] ' + time.strftime(
                            '[%y%m%d_%H:%M:%S]') + '  Failed to write ' + shapefile_out_filename)
            # filter_shapefile_by_class(shapefile_out_filename)
            logger.setStageAndProcess("shapefileName", "100").info(shapefile_out_filename + " " + color_out_filename)

        pbar0.update(1)
        message_dict = {'progress': _sub_task_prg_max, 'runningStatus': 'running', 'runningInfo': f'推理任务-{idx + 1}/{len(basename_list_list[local_rank])}',
                        'inferProgress': 100, 'inferFilename': filename}
        prg_sender.send(message_dict)

    pbar0.close()
    
    if cfg.USE_SHAPEFILE_OUT:
        # shpsavepath = task_input.split('/')[-2] + '.zip'
        make_zip(shapefile_out_root, os.path.join(test_output_root, f'shp{rank_suffix}.zip'))

    message_dict = {'progress': 100, 'runningStatus': 'completed', 'runningInfo': '推理任务完成'}
    prg_sender.send(message_dict)
    prg_sender.close()

    logger.info('Test task finished.')
    logger.setStageAndProcess("Finish", "100").info("结束")


def filter_color_from_tif(tif_path):
    class_id = os.getenv("FILTER_CLASS_ID")
    if class_id is None:
        print("[INFO] 未设置环境变量 FILTER_CLASS_ID，跳过处理")
        return None

    class_id_str = class_id.zfill(2)
    class_id = int(class_id_str.lstrip("0"))  # 把 '05' 变成 5

    ds = gdal.Open(tif_path, gdal.GA_Update)
    if ds is None:
        raise RuntimeError(f"无法打开图像: {tif_path}")

    band = ds.GetRasterBand(1)
    arr = band.ReadAsArray()

    # 创建掩码：保留class_id的，其他都变为0
    mask = (arr == class_id)
    filtered = np.where(mask, class_id, 0).astype(arr.dtype)

    band.WriteArray(filtered)
    band.FlushCache()

    # 清理 GDAL 资源
    ds = None
    print(f"[INFO] 已在图像中保留类别 {class_id_str}，其他像素已清零。")

def filter_shapefile_by_class(shp_path, class_field="objects"):
    class_id = get_target_rgb_from_env()
    if class_id is None:
        return

    driver = ogr.GetDriverByName("ESRI Shapefile")
    if not os.path.exists(shp_path):
        raise FileNotFoundError(f"找不到shp文件: {shp_path}")

    # 原始数据源
    src_ds = driver.Open(shp_path, 0)
    src_layer = src_ds.GetLayer()
    srs = src_layer.GetSpatialRef()

    # 创建临时输出文件
    tmp_shp_path = shp_path.replace(".shp", "_filtered.shp")
    if os.path.exists(tmp_shp_path):
        driver.DeleteDataSource(tmp_shp_path)
    dst_ds = driver.CreateDataSource(tmp_shp_path)
    dst_layer = dst_ds.CreateLayer("filtered", srs, geom_type=ogr.wkbPolygon)

    # 拷贝字段结构
    layer_defn = src_layer.GetLayerDefn()
    for i in range(layer_defn.GetFieldCount()):
        dst_layer.CreateField(layer_defn.GetFieldDefn(i))
    dst_defn = dst_layer.GetLayerDefn()

    # 按字段值拷贝要素
    for feat in src_layer:
        field_val = str(feat.GetField(class_field)).zfill(2)
        if field_val == class_id:
            out_feat = ogr.Feature(dst_defn)
            out_feat.SetGeometry(feat.GetGeometryRef().Clone())
            for i in range(dst_defn.GetFieldCount()):
                out_feat.SetField(i, feat.GetField(i))
            dst_layer.CreateFeature(out_feat)

    # 关闭数据源
    src_ds = None
    dst_ds = None

    # 替换原文件
    for ext in [".shp", ".shx", ".dbf", ".prj", ".cpg"]:
        original_file = shp_path.replace(".shp", ext)
        if os.path.exists(original_file):
            os.remove(original_file)
        filtered_file = tmp_shp_path.replace(".shp", ext)
        if os.path.exists(filtered_file):
            os.rename(filtered_file, original_file)

    print(f"[INFO] shp处理完成，仅保留类别 {class_id} 对应要素")
