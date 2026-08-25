import os
import glob
import cv2
import shutil
import gc
from pathlib import Path
from time import sleep
import numpy as np
from osgeo import gdal, ogr, osr
from datasets.DataUtils import DataUtils
from torch.utils.data import get_worker_info
import random
import torch

from torch.autograd import Variable as V
from torch.utils import data
from torchvision import transforms as T
from shapely.geometry import Polygon
from shapely.ops import transform as shapely_transform
import pyproj

from tqdm import tqdm
from networks.GenerateNet import GenerateNet
from utils.hist_stretch import percent_stretch_image
from config import cfg
from functools import partial

import warnings
warnings.filterwarnings('ignore')
from utils.paek_smooth import smooth_polygons_gdf_used
from schedule import post_status, post_progress


# from secure.SecureCheck import secure_check
# secure_check()


def _available_cpu_count():
    """返回当前容器实际可用的 CPU 数，兼容 cpuset、cgroup v1/v2。"""
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


def _recommended_dataloader_workers():
    """按并行影像任务数均分约 80% 的容器 CPU，给主进程和 GDAL 留出余量。"""
    available_cpus = _available_cpu_count()
    try:
        parallel_jobs = max(1, int(os.environ.get('CHANGE_DETECTION_PARALLEL_JOBS', '1')))
    except (TypeError, ValueError):
        parallel_jobs = 1
    workers_per_job = int(available_cpus * 0.8) // parallel_jobs
    return max(1, min(48, workers_per_job))

def CalHistogram(img):
    
    img_dtype = img.dtype
    img_hist = img.reshape(-1)
    img_min, img_max = img_hist.min(), img_hist.max()
    n_bins = 2 ** 16
    if (img_dtype == np.uint8 ):
        n_bins = 256
    if  (img_dtype == np.uint16 ):
        n_bins = 2**16
    elif (img_dtype == np.uint32):
        n_bins = 2**32 
    if (img_dtype == np.uint8 ) or (img_dtype == np.uint16 ) or (img_dtype == np.uint32):
        hist = np.bincount(img_hist, minlength = n_bins)
        hist[0] = 0  
        hist[-1] = 0    
        s_values = np.arange(n_bins)
    else:
        hist,s_values= np.histogram(img_hist,bins = n_bins,range = (img_min,img_max)) 
        hist[0] = 0
        hist[-1] = 0   
    img_hist = None  
    return hist, s_values

def GetPercentStretchValue(img,left_clip = 0.001,right_clip = 0.001):
    
    right_clip = 1.0 - right_clip
    hist, s_values = CalHistogram(img)
    s_quantiles = np.cumsum(hist).astype(np.float64)
    s_quantiles /= (s_quantiles[-1] + 1.0E-5)    
    left_clip_index = np.argmin(np.abs(s_quantiles-left_clip))
    right_clip_index = np.argmin(np.abs(s_quantiles-right_clip))  
    img_min_clip,img_max_clip = s_values[[left_clip_index,right_clip_index]]
    return img_min_clip,img_max_clip 

def percent_stretch_image1(input_image_data, left_clip = 0.001,right_clip = 0.001):
    
    if input_image_data is None:
        return None  
    n_dim = input_image_data.ndim
    img_bands = 1 if n_dim == 2 else input_image_data.shape[n_dim-1]
    xsize = input_image_data.shape[1]
    ysize = input_image_data.shape[0]
    indtype = input_image_data.dtype
    if indtype == np.uint8:
        to_8bit = True
    if img_bands > 1:
        out_8bit_data = np.zeros((ysize,xsize,img_bands),dtype = np.uint8)
    else:
        out_8bit_data = np.zeros((ysize,xsize),dtype = np.uint8)  
    for i_band in range(img_bands):
        if img_bands == 1:
            input_image_data_raw = input_image_data#[:,:,i_band]
        else:
            input_image_data_raw = input_image_data[:,:,i_band]
        img_clip_min,img_clip_max = GetPercentStretchValue(input_image_data_raw,left_clip=left_clip,right_clip = right_clip)    
        input_image_data_raw = np.clip(input_image_data_raw,img_clip_min,img_clip_max)
        input_image_data_raw = (input_image_data_raw -  img_clip_min)/(img_clip_max - img_clip_min) * 255
        input_image_data_raw = input_image_data_raw.astype(np.uint8)               
        if img_bands > 1:
            out_8bit_data[:,:,i_band] = input_image_data_raw
        else:
            out_8bit_data = input_image_data_raw
    return out_8bit_data


def truncated_linear_stretch(image, truncated_value=2):
    """无符号整型16位转无符号整型8位程序
    @author: Xiangyu Tian

    Args:
        image (numpy.ndarray): 影像矩阵信息
        truncated_value (int): np.percentile函数的百分比信息

    Returns:
        image_stretch (numpy.ndarray): 转8位之后的矩阵
    """
    # 如果是多波段
    if(len(image.shape) == 3):
        image_stretch = np.zeros([image.shape[0],image.shape[1],image.shape[2]])
        image_stretch = np.uint16(image_stretch)
        for i in range(1):
            gray = gray_process(image[:,:,i], truncated_value)
            image_stretch[:,:,i] = gray
        image_stretch = np.array(image_stretch)
    # 如果是单波段
    else:
        image_stretch = gray_process(image, truncated_value)
    return image_stretch

def gray_process(gray, truncated_value, max_out = 255, min_out = 0):
    """无符号整型16位转无符号整型8位程序 单波段应用
    @author: Xiangyu Tian

    Args:
        image (numpy.ndarray): 影像矩阵信息
        truncated_value (int): np.percentile函数的百分比信息

    Returns:
        gray (numpy.ndarray): 转8位之后的矩阵
    """
    temp = gray.copy()
    if np.all(gray==0):
        return gray
    temp = temp.ravel()[np.flatnonzero(temp)]
    if temp.shape[0] == 0:
        import pdb;pdb.set_trace()
    truncated_down = np.percentile(temp, truncated_value)
    truncated_up = np.percentile(temp, 100 - truncated_value)
    gray = (gray - truncated_down) / (truncated_up - truncated_down) * (max_out - min_out) + min_out
    gray[gray < min_out] = min_out
    gray[gray > max_out] = max_out
    if(max_out <= 255):
        gray = np.uint8(gray)
    elif(max_out <= 65535):
        gray = np.uint16(gray)
    return gray

# gdal读取图像
def readimage(dataset, read_range):
    nband = dataset.RasterCount
    band_list = [i + 1 for i in range(nband)]
    if nband == 3 or nband == 4:
        band_list[0] = 3
        band_list[2] = 1
    img = None
    count, addition = 0  ,100 // len(band_list)
    # import ipdb;ipdb.set_trace()
    for band in band_list:
        _data = dataset.GetRasterBand(band)
        _img = _data.ReadAsArray(*read_range)[:, :, None]
        if 'int16' in _img.dtype.name:
            if _img.max() == 0:
                _img = _img.astype(np.uint8)
            else:
                _img = truncated_linear_stretch(image=_img)
                _img = _img.astype(np.uint8)
        if img is not None:
            img = np.append(img, _img, axis=2)
        else:
            img = _img
        count += addition
    if img.shape[2] == 1:
        img = img[:, :, 0]
    return img

def mask_nodata(pre_img, post_img):
    if pre_img.min() == 255 or post_img.min() == 255:
        return np.zeros((pre_img.shape[0], pre_img.shape[1]), dtype=pre_img.dtype)
    # import ipdb; ipdb.set_trace()
    assert pre_img.shape == post_img.shape
    # 任意一张是nodata就应该被掩盖掉
    temp = pre_img * post_img  # [3,1024,1024]
    mask = np.where(np.mean(temp, axis=2)==0, 0, 1)
    return mask

def style_transfer(source_image, target_image):
    h, w, c = source_image.shape
    out = []
    for i in range(c):
        source_image_f = np.fft.fft2(source_image[:,:,i])
        source_image_fshift = np.fft.fftshift(source_image_f)
        target_image_f = np.fft.fft2(target_image[:,:,i])
        target_image_fshift = np.fft.fftshift(target_image_f)
        
        change_length = 1
        source_image_fshift[int(h/2)-change_length:int(h/2)+change_length, 
                            int(h/2)-change_length:int(h/2)+change_length] = \
            target_image_fshift[int(h/2)-change_length:int(h/2)+change_length,
                                int(h/2)-change_length:int(h/2)+change_length]
            
        source_image_ifshift = np.fft.ifftshift(source_image_fshift)
        source_image_if = np.fft.ifft2(source_image_ifshift)
        source_image_if = np.abs(source_image_if)
        
        source_image_if[source_image_if>255] = np.max(source_image[:,:,i])
        out.append(source_image_if)
    out = np.array(out)
    out = out.swapaxes(1,0).swapaxes(1,2)
    
    # # 结果中含有>255的值,拉伸或者强制=255效果都不好,但是cv2.imwrite再read效果好
    # # 有时间探究一下原因
    # # 生成数字+字母
    # token = string.ascii_letters + string.digits
    # # 随机选择指定长度随机码
    # token = random.sample(token,15)
    # token_str = ''.join(token)
    # temp_path = "temp_{}.png".format(token_str)
    # if not os.path.exists(temp_path): cv2.imwrite(temp_path, out)
    # out = cv2.imread(temp_path)
    # if os.path.exists(temp_path): os.remove(temp_path)
    out = out.astype(np.uint8)
    return out


def get_union_extent(gt1, gt2, size1, size2):
    # 要求统一投影
    xmin1 = gt1[0]
    ymax1 = gt1[3]
    xmax1 = xmin1 + gt1[1] * size1[0]
    ymin1 = ymax1 + gt1[5] * size1[1]

    xmin2 = gt2[0]
    ymax2 = gt2[3]
    xmax2 = xmin2 + gt2[1] * size2[0]
    ymin2 = ymax2 + gt2[5] * size2[1]

    xmin = max(xmin1, xmin2)
    ymin = min(ymin1, ymin2)
    xmax = min(xmax1, xmax2)
    ymax = max(ymax1, ymax2)
    return xmin, ymin, xmax, ymax

def fill_hole(shp_file_path, output_shp_file_name):
    """
    @function: 检查shp文件中Polygon的内环部分进行填充；（在原shp文件的同级目录下生成一个新的名为output_shp_file_name的shp文件）
    @description:
    * 直接判断Polygon里面包含的Shape数量，是2的话就是包含内外环，获取外环重新构建一个多边形；
    @params:
    * output_shp_file_name: 输出的shp文件名，不要加.shp后缀
    @return:
    * None
    """
    gdal.SetConfigOption("SHAPE_ENCODING", "")
    driver = ogr.GetDriverByName("ESRI Shapefile")
    shp_file_path = Path(shp_file_path)
    datasource = driver.Open(str(shp_file_path), 1)
    output_shp_file_name = Path(output_shp_file_name)
    if output_shp_file_name.exists():
        driver.DeleteDataSource(str(output_shp_file_name))
    layer = datasource.GetLayer()
    src_srs = layer.GetSpatialRef()  # 获取原始的坐标系或投影
    tgt_srs = osr.SpatialReference()  # 获取目标的坐标系或投影， web mercator
    # tgt_srs.ImportFromEPSG(3857)
    tgt_srs.ImportFromWkt(src_srs.ExportToWkt())
    transform = osr.CoordinateTransformation(src_srs, tgt_srs)

    tgt_datasource = driver.CreateDataSource(str(output_shp_file_name))
    tgt_geomtype = ogr.wkbPolygon
    tgt_layer = tgt_datasource.CreateLayer(str(output_shp_file_name), srs=tgt_srs, geom_type=tgt_geomtype, options=["ENCODING=GBK"])
    layerDefinition = layer.GetLayerDefn()  # 获取图层的字段信息
    for i in range(layerDefinition.GetFieldCount()):
        tgt_layer.CreateField(layerDefinition.GetFieldDefn(i))

    # feature = layer.GetFeature(396)
    # geom = feature.GetGeometryRef()
    # from ipdb import set_trace; set_trace()
    for feature in layer:
        geom = feature.GetGeometryRef()
        geom_tgt = geom.Clone()
        # geom_tgt.Transform(transform)
        if geom_tgt.GetGeometryCount() < 2:
            feature.SetGeometry(geom_tgt)
            tgt_layer.CreateFeature(feature)
        else:
            geom_out_ring = geom_tgt.GetGeometryRef(0)
            geom_tgt_polygon = ogr.Geometry(ogr.wkbPolygon)
            geom_tgt_polygon.AddGeometry(geom_out_ring)
            feature.SetGeometry(geom_tgt_polygon)
            tgt_layer.CreateFeature(feature)

    datasource.Destroy()
    tgt_datasource.Destroy()

def simplify_preserve_toplogy(shp_file_path, output_shp_file_name, tolerance):
        """
        @function: 简化多边形
        @description:
        * None；
        @params:
        * output_shp_file_name: 输出的shp文件名，不限制
        * tolerance：阈值
        @return:
        * None
        """
        # print(f'当前容差是： {tolerance}')
        gdal.SetConfigOption("SHAPE_ENCODING", "")
        driver = ogr.GetDriverByName("ESRI Shapefile")
        shp_file_path = Path(shp_file_path)
        datasource = driver.Open(str(shp_file_path), 1)
        output_shp_file_name = Path(output_shp_file_name)
        if output_shp_file_name.suffix == '':
            output_shp_file_name = output_shp_file_name.with_suffix('.shp')
        if len(output_shp_file_name.parts) == 1:
            output_shp_file_name = shp_file_path.parent / output_shp_file_name
        if output_shp_file_name.exists():
            driver.DeleteDataSource(str(output_shp_file_name))
        layer = datasource.GetLayer()
        src_srs = layer.GetSpatialRef()  # 获取原始的坐标系或投影
        tgt_srs = osr.SpatialReference()  # 获取目标的坐标系或投影， web mercator
        tgt_srs.ImportFromWkt(src_srs.ExportToWkt())

        tgt_datasource = driver.CreateDataSource(str(output_shp_file_name))
        tgt_geomtype = ogr.wkbPolygon
        tgt_layer = tgt_datasource.CreateLayer(str(output_shp_file_name), srs=tgt_srs, geom_type=tgt_geomtype,
                                               options=["ENCODING=GBK"])
        layerDefinition = layer.GetLayerDefn()  # 获取图层的字段信息
        for i in range(layerDefinition.GetFieldCount()):
            tgt_layer.CreateField(layerDefinition.GetFieldDefn(i))

        for feature in layer:
            geom = feature.GetGeometryRef()
            geom_tgt = geom.Clone()
            geom_tgt = geom_tgt.SimplifyPreserveTopology(tolerance)
            feature.SetGeometry(geom_tgt)
            tgt_layer.CreateFeature(feature)

        datasource.Destroy()
        tgt_datasource.Destroy()
        # print(f'{output_shp_file_name} 创建完成...')

def create_empty_shapefile(output_path, logger=None):
    """
    创建一个空的shapefile（没有几何要素）
    """
    try:
        # 确保输出目录存在
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建空的shapefile
        driver = ogr.GetDriverByName('ESRI Shapefile')
        
        # 删除已存在的文件
        if Path(output_path).exists():
            driver.DeleteDataSource(str(output_path))
        
        # 获取输入图像的空间参考
        # 这里使用默认的WGS84，如果需要可以根据实际情况调整
        srs = osr.SpatialReference()
        srs.ImportFromEPSG(4326)
        
        # 创建数据源
        data_source = driver.CreateDataSource(str(output_path))
        layer = data_source.CreateLayer('empty', srs, ogr.wkbPolygon)
        
        # 添加字段（与原推理输出保持一致）
        field_defn = ogr.FieldDefn("precode", ogr.OFTString)
        layer.CreateField(field_defn)
        
        field_defn = ogr.FieldDefn('preyear', ogr.OFTInteger)
        layer.CreateField(field_defn)
        
        field_defn = ogr.FieldDefn("currcode", ogr.OFTString)
        layer.CreateField(field_defn)
        
        field_defn = ogr.FieldDefn("curryear", ogr.OFTInteger)
        layer.CreateField(field_defn)
        
        # 关闭数据源
        data_source = None
        
        if logger is not None:
            logger.info(f"创建了空的shapefile: {output_path}")
            
    except Exception as e:
        if logger is not None:
            logger.error(f"创建空shapefile失败: {str(e)}")
        raise

def create_completion_marker(output_path, logger=None):
    """创建完成标记文件"""
    try:
        import time
        marker_file = str(Path(output_path).with_suffix('.completed.txt'))
        
        with open(marker_file, 'w') as f:
            f.write(f"Completed at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Output file: {output_path}\n")
            f.write(f"Status: No spatial intersection, empty result generated\n")
        
        if logger is not None:
            logger.info(f"创建完成标记文件: {marker_file}")
        return marker_file
    except Exception as e:
        if logger is not None:
            logger.error(f"创建标记文件失败: {str(e)}")
        return None

class OnlineDataset(data.Dataset):
    """Dataloader

    节省内存提高预测效率
    """
    
    def __init__(self, path_info, band_num, with_fft):
        super(OnlineDataset, self).__init__()
        self.path_info = path_info
        self.preimg_path = self.path_info['preImgPath']
        self.postimg_path = self.path_info['postImgPath']
        self.index_list = self.path_info['index_list']
        self.overlap = self.path_info['overlap']
        self.clipsize = self.path_info['clipsize']
        self.with_fft = with_fft
        self.predataset = {} # gdal.Open(self.preimg_path)
        self.postdataset = {} # gdal.Open(self.postimg_path)
        self.band_num = band_num
        self.datautils = DataUtils()
        mean_file = self.path_info['mean_file']
        std_file = self.path_info['std_file']
        """这里需要加入读取均值方差的功能"""
        # self.mean_value = [0.26069229475394134, 0.298781168513767, 0.2869695704458281, 0.434965101964887]
        # self.std_value = [0.055944613439789126, 0.062278333852758584, 0.07335243334796868, 0.1069423216555217]
        self.mean_value =  self.datautils.load_mean_file(mean_file)
        self.std_value =  self.datautils.load_std_file(std_file)
        normalize = T.Normalize(mean=self.mean_value, std=self.std_value)
        self.transforms = T.Compose([
            T.ToTensor(),
            normalize
        ])

    def __getitem__(self, index):
        # 根据计算好索引列表读取对应区域的影像切片（前时相）
        clip_x, clip_y, pad_x_right, pad_y_down = self.index_list[index]
        
        clipsize_x, clipsize_y = self.clipsize[0], self.clipsize[1]
        # 根据index坐标索引读取小块图像
        if clip_x != 0 and self.overlap[0] > 0:
            clip_x = clip_x - self.overlap[0] // 2
        elif clip_x == 0:
            clipsize_x = self.clipsize[0] - self.overlap[0] // 2
        if clip_y != 0 and self.overlap[1] > 0:
            clip_y = clip_y - self.overlap[1] // 2
        elif clip_y == 0:
            clipsize_y = self.clipsize[1] - self.overlap[1] // 2
        if pad_x_right > 0:
            clipsize_x = self.clipsize[0] - int(pad_x_right)
        else:
            pad_x_right = 0
        if pad_y_down > 0:
            clipsize_y = clipsize_y - int(pad_y_down)
        else:
            pad_y_down = 0

        # 惰性加载大图数据，每个进程只打开一次
        worker_info = get_worker_info()
        key = 'main' if worker_info is None else f'worker_{worker_info.id}'

        if key not in self.predataset:
            pre_ds = gdal.Open(self.preimg_path, gdal.GA_ReadOnly)
            if pre_ds is None:
                raise RuntimeError(f"GDAL failed to open {self.preimg_path}")
            post_ds = gdal.Open(self.postimg_path, gdal.GA_ReadOnly)
            if post_ds is None:
                raise RuntimeError(f"GDAL failed to open {self.postimg_path}")
            self.predataset[key] = pre_ds
            self.postdataset[key] = post_ds
            
        # import ipdb;ipdb.set_trace()
        # 处理大图的尺寸不足以裁剪出一块切片的时候
        big_width, big_height = self.predataset[key].RasterXSize, self.predataset[key].RasterYSize
        if clipsize_x > big_width:
            clipsize_x = big_width
            pad_x_right = pad_x_right + clipsize_x - big_width
            self.index_list[index][2] = pad_x_right
        if clipsize_y > big_height:
            clipsize_y = big_height
            pad_y_down = pad_y_down + clipsize_y - big_height
            self.index_list[index][3] = pad_y_down

        pre_img = readimage(self.predataset[key], (clip_x, clip_y, clipsize_x, clipsize_y))[:,:,0:self.band_num]
        # import ipdb;ipdb.set_trace()
        post_img = readimage(self.postdataset[key], (clip_x, clip_y, clipsize_x, clipsize_y))[:,:,0:self.band_num]

        # 交换得到正确的RGB图像
        # pre_img = pre_img[..., [i - 1 for i in self.path_info['rgb']]]
        # post_img = post_img[..., [i - 1 for i in self.path_info['rgb']]]
        
        # 处理边缘
        if clip_x == 0 and clip_y != 0:
            pre_img = np.pad(pre_img, ((0,int(pad_y_down)), (self.overlap[1]//2, int(pad_x_right)), (0,0)))
            post_img = np.pad(post_img, ((0,int(pad_y_down)), (self.overlap[1]//2, int(pad_x_right)), (0,0)))
        elif clip_x == 0 and clip_y == 0:
            pre_img = np.pad(pre_img, ((self.overlap[0]//2,int(pad_y_down)), (self.overlap[1]//2, int(pad_x_right)), (0,0)))
            post_img = np.pad(post_img, ((self.overlap[0]//2,int(pad_y_down)), (self.overlap[1]//2, int(pad_x_right)), (0,0)))
        elif clip_x != 0 and clip_y == 0:
            pre_img = np.pad(pre_img, ((self.overlap[0]//2, int(pad_y_down)), (0,int(pad_x_right)), (0,0)))
            post_img = np.pad(post_img, ((self.overlap[0]//2, int(pad_y_down)), (0,int(pad_x_right)), (0,0)))
        else:
            pre_img = np.pad(pre_img, ((0, int(pad_y_down)), (0,int(pad_x_right)), (0,0)))
            post_img = np.pad(post_img, ((0, int(pad_y_down)), (0,int(pad_x_right)), (0,0)))

        # 若由于分辨率差异或者进位错误导致两张切片图像的尺寸不一致，将后时相影像尺寸转为前时相影像的尺寸
        # if pre_img.shape != post_img.shape:
        #     post_img = cv2.resize(post_img, (pre_img.shape[1], pre_img.shape[0]), interpolation=cv2.INTER_NEAREST)
        # cv2.imwrite('./test/' + pre_name+ '.png', np.concatenate((pre_img, post_img), axis=1).astype(np.uint8))
        nodata_mask = mask_nodata(pre_img, post_img)
        # use FFT
        if self.with_fft:
            pre_img = style_transfer(pre_img, post_img)
        # percentage strech
        if False:
            pre_img = percent_stretch_image(pre_img)
            post_img = percent_stretch_image(post_img)
        # 影像规则化
        pre_img = self.transforms(np.ascontiguousarray(pre_img, dtype = np.uint8))
        post_img = self.transforms(np.ascontiguousarray(post_img, dtype = np.uint8))

        return pre_img, post_img, nodata_mask, self.index_list[index]

    def __len__(self):
        return len(self.index_list)

def get_extent(ds):
    gt = ds.GetGeoTransform()
    xsize, ysize = ds.RasterXSize, ds.RasterYSize
    xmin = gt[0]
    ymax = gt[3]
    xmax = xmin + gt[1] * xsize
    ymin = ymax + gt[5] * ysize
    return xmin, ymin, xmax, ymax

def get_polygon_from_extent(extent):
    xmin, ymin, xmax, ymax = extent
    return Polygon([
        (xmin, ymin),
        (xmax, ymin),
        (xmax, ymax),
        (xmin, ymax),
        (xmin, ymin)
    ])

def get_srs(ds):
    srs = osr.SpatialReference()
    srs.ImportFromWkt(ds.GetProjection())
    return srs

def transform_polygon(polygon, src_srs, dst_srs):
    try:
        project = pyproj.Transformer.from_crs(
            pyproj.CRS.from_wkt(src_srs.ExportToWkt()),
            pyproj.CRS.from_wkt(dst_srs.ExportToWkt()),
            always_xy=True
        ).transform
        return shapely_transform(project, polygon)
    except Exception as e:
        print("投影转换失败：", e)
        return None

def get_pixel_offset(dataset, x_geo, y_geo):
    gt = dataset.GetGeoTransform()
    col = int((x_geo - gt[0]) / gt[1])
    row = int((y_geo - gt[3]) / gt[5])
    return row, col

def compute_intersection_info(ds1, ds2):

    bounds1 = get_extent(ds1)
    bounds2 = get_extent(ds2)
    
    # 计算空间交集 (left, bottom, right, top)
    overlap_left = max(bounds1[0], bounds2[0])
    overlap_right = min(bounds1[2], bounds2[2])
    overlap_bottom = max(bounds1[1], bounds2[1])
    overlap_top = min(bounds1[3], bounds2[3])

    if overlap_left >= overlap_right or overlap_bottom >= overlap_top:
        print("❌ 两幅图像无空间交集")
        return None

    # 左上角在两张图中的像素位置
    row1, col1 = get_pixel_offset(ds1, overlap_left, overlap_top)
    row2, col2 = get_pixel_offset(ds2, overlap_left, overlap_top)

    # 获取像素大小（单位：米或度）
    pixel_width = ds1.GetGeoTransform()[1]
    pixel_height = abs(ds1.GetGeoTransform()[5])

    # 计算交集区域的像素尺寸
    width_px = int((overlap_right - overlap_left) / pixel_width)
    height_px = int((overlap_top - overlap_bottom) / pixel_height)

    return {
        "offset_img1": (row1, col1),
        "offset_img2": (row2, col2),
        "size": (height_px, width_px),
        "bounds": (overlap_left, overlap_bottom, overlap_right, overlap_top)
    }

def generate_indexlist(width, height, ori_w=0, ori_h=0, subsize=(512, 512), overlap=(0, 0), debug=False):
    """
    Args:
       width: width of the input big tif 
       height: height of the input big tif 
       ori_w: starting width of the input big tif 
       ori_h: starting height of the input big tif 
       subsize: clip size
       overlap: overlap between two clips
    Return:
        params_saveimg: saves a tuple, the content is:
            (the start position of left, the start position of up, 
            the pixels to be padded in right, the pixels to be padded in bottom)
        nx & ny: the number of patches in one row & col
    """
    overlap_half_x = overlap[0] // 2
    overlap_half_y = overlap[1] // 2
    slide_x = subsize[0] - overlap[0]
    slide_y = subsize[1] - overlap[1]
    params_saveimg = []
    left= 0
    nx = 0
    while left < width:
        if debug:
            import ipdb;ipdb.set_trace()
        if left + slide_x + overlap_half_x > width:
            pad_x_right = left + slide_x + overlap_half_x - width
        elif left + slide_x + overlap_half_x == width:
            pad_x_right = -1
        else:
            pad_x_right = 0
        up = 0
        ny = 0
        while up < height:
            if up + slide_y + overlap_half_y > height:
                pad_y_down = up + slide_y + overlap_half_y - height
            elif up + slide_y + overlap_half_y == height:
                pad_y_down = -1
            else:
               pad_y_down = 0
            clip_left = left + ori_w
            clip_up = up + ori_h
            params_saveimg.append([clip_left, clip_up, pad_x_right, pad_y_down])
            up = up + slide_y
            ny += 1
        left = left + slide_x
        nx += 1
    return params_saveimg, nx, ny

def test_with_TTA(net, imgs, num_classes, device):
    im_bs = imgs.shape[0] // 8
    
    images = torch.Tensor(imgs).to(device)

    with torch.no_grad():
        scores = net(images)
        
    if num_classes > 1:
        output_change = torch.softmax(scores[0], axis=1)[:,1].cpu().numpy()
    else:
        output_change = torch.sigmoid(scores[0])[:,0].cpu().numpy()
    
    mask_change = output_change[0:im_bs*2] + output_change[im_bs*2:im_bs*4][:,::-1] + output_change[im_bs*4:im_bs*6][:,:,::-1] + output_change[im_bs*6:][:,::-1,::-1]
    mask_change = mask_change[0:im_bs] + np.rot90(mask_change[im_bs:], axes = (1, 2))[:,::-1,::-1]
    mask_change = np.rint(mask_change/8).astype(np.uint8)
    return mask_change

def test_normal(net, imgs, num_classes, device):
    images = torch.Tensor(imgs).to(device)
    with torch.no_grad():
        scores = net(images)
    if num_classes > 1:
        output_change = torch.softmax(scores[0], 1)[:,1].cpu().numpy()
    else:
        output_change = torch.sigmoid(scores[0])[:,0].cpu().numpy()
    output_change = np.rint(output_change).astype(np.uint8)
    return output_change

def set_color_table(dataset, color_table=None):
    if color_table is not None:
        ct = gdal.ColorTable()
        for i, color in enumerate(color_table):
            ct.SetColorEntry(i, color)

        _band = dataset.GetRasterBand(1)
        _band.SetRasterColorInterpretation(gdal.GCI_PaletteIndex)
        _band.SetColorTable(ct)

def set_no_data_value(dataset, no_data_value=None):
    if no_data_value is not None:
        _band = dataset.GetRasterBand(1)
        _band.SetNoDataValue(no_data_value)
        
def build_overviews(dataset, overviewlist=[2,4,8,16,32,64,128]):
    gdal.ErrorReset()
    overview_result = dataset.BuildOverviews('NEAREST', overviewlist=overviewlist)
    flush_result = dataset.FlushCache()
    if (overview_result not in (None, 0)
            or flush_result not in (None, 0)
            or gdal.GetLastErrorType() >= gdal.CE_Failure):
        error_message = gdal.GetLastErrorMsg() or 'unknown GDAL error'
        raise RuntimeError(f'完成变化检测 BigTIFF 写盘失败: {error_message}')


def _resolve_checkpoint_path(module_dir, requested_path, fallback_filename, model_label):
    """优先使用工作流传入路径，相对路径同时按模块目录解析。"""
    candidates = []
    if requested_path:
        requested_checkpoint = Path(requested_path)
        candidates.append(requested_checkpoint)
        if not requested_checkpoint.is_absolute():
            candidates.append(module_dir / requested_checkpoint)
    candidates.append(module_dir / fallback_filename)
    checkpoint_path = next((path.resolve() for path in candidates if path.is_file()), None)
    if checkpoint_path is None:
        searched_paths = ', '.join(str(path) for path in candidates)
        raise FileNotFoundError(f'{model_label}不存在，已检查: {searched_paths}')
    return checkpoint_path


def test_lib_big_memeff(pre_img_path='', post_img_path='', output_path='', logger=None, callback_url=None, job_id=None,
                        temp_dir_suffix="tmp", progress_callback=None, model_path=None,
                        use_two_models=False, second_model_path=None):
    """
    主要的推理函数

    参数:
    - temp_dir_suffix: 临时目录后缀，用于区分不同进程的临时目录，默认为"tmp"
    """
    # 首先检查两幅图像是否有空间交集
    if logger is not None:
        logger.info("开始处理，首先检查前后时相图像的空间交集...")
    
    try:
        # 打开前后时相图像
        pre_data = gdal.Open(pre_img_path)
        post_data = gdal.Open(post_img_path)
        
        if pre_data is None:
            gdal_error = gdal.GetLastErrorMsg() or 'unknown GDAL error'
            raise RuntimeError(f"无法打开前时相图像: {pre_img_path}: {gdal_error}")

        if post_data is None:
            gdal_error = gdal.GetLastErrorMsg() or 'unknown GDAL error'
            raise RuntimeError(f"无法打开后时相图像: {post_img_path}: {gdal_error}")
            
        # 计算空间交集
        #logger.info("===========计算空间交集=======")
        overlap_info = compute_intersection_info(pre_data, post_data)        
        if overlap_info is None:
            raise RuntimeError("两幅图像无空间交集，无法执行变化检测")
        
        # 如果有交集，继续正常处理
        pre_data = None
        post_data = None
        
    except Exception as e:
        pre_data = None
        post_data = None
        error_message = f"检查图像交集时出错: {e}"
        if logger is not None:
            logger.error(error_message)
        raise RuntimeError(error_message) from e
    
    # 模型配置
    if logger is not None:
        logger.info("两幅图像有空间交集，开始构建模型------")
    
    module_dir = Path(__file__).resolve().parent
    use_two_models = (
        use_two_models
        if isinstance(use_two_models, bool)
        else str(use_two_models).strip().lower() in {'1', 'true', 'yes', 'y', 'on'}
    )
    checkpoint_path = _resolve_checkpoint_path(
        module_dir,
        model_path,
        'FBCD_test_select0207_best_acc.pth',
        '变化检测主模型',
    )
    second_checkpoint_path = None
    if use_two_models:
        second_checkpoint_path = _resolve_checkpoint_path(
            module_dir,
            second_model_path,
            'FBCD_test_Levir_CD_best_acc.pth',
            '变化检测第二模型',
        )
        if second_checkpoint_path == checkpoint_path:
            raise ValueError(
                f'双模型模式需要两个不同的权重文件，当前都解析为: {checkpoint_path}'
            )
    if logger is not None:
        logger.info('变化检测主模型: %s', checkpoint_path)
        if use_two_models:
            logger.info('变化检测第二模型: %s；融合方式: OR', second_checkpoint_path)
        else:
            logger.info('变化检测使用单模型模式')

    # 一个进程只使用一张可见 GPU。文件夹批处理由外层按影像启动独立进程，
    # 避免 VMamba 动态绑定的 forward 方法被 DataParallel 复制后跨设备引用。
    visible_gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    gpu_count = 1 if visible_gpu_count > 0 else 0
    kwargs = {
        'PRE_IMG_PATH': pre_img_path,
        'POST_IMG_PATH': post_img_path,
        'TEST_OUT_PATH': output_path,
        'IMG_SUFFIX': '.tif',
        'GT_SUFFIX': '.tif',
        'TEST_CKPT': str(checkpoint_path),
        'USE_TWO_MODELS': use_two_models,
        'TEST_CKPT_2': str(second_checkpoint_path) if second_checkpoint_path else None,
        'MEAN_FILE': str(module_dir / 'mean_value.txt'),
        'STD_FILE': str(module_dir / 'std_value.txt'),
        'MODEL_NAME': 'FBCD',
        'MODEL_BACKBONE': 'vssm_tiny',
        'MODEL_NUM_CLASSES': 1,
        # 每个影像进程固定使用一张 GPU；不同影像由外层进程并行调度。
        'TEST_BATCHES': 1,
        # 根据 Pod CPU 配额和并行影像数均分约 80%：60 CPU、双卡并行约每卡 24。
        # 其余 CPU 留给主进程、GPU 喂数、GDAL 写出和消息线程。
        'TEST_NUM_WORKERS': _recommended_dataloader_workers(),
        'TEST_PREFETCH_FACTOR': 1,
        'WITH_TTA': False,
        'with_fft': True,
        'THRESHOLD': 0.5,
        'BAND_NUM': 3,
        # 2560 像素切片在多进程 FFT 时会产生严重的内存带宽争用，首批数据数分钟无法返回。
        # 1280 仍满足网络下采样倍数要求，单切片数据量降为 1/4，可更快开始 GPU 推理。
        'TEST_IMG_SIZE': 1280,
        'TEST_PIXEL_OVERLAP': 0,
        'PRETRAINED': False,
        'TEMP_DIR_SUFFIX': temp_dir_suffix,  # 添加临时目录后缀配置
    }
    #import pdb;pdb.set_trace()
    cfg.set_parse(kwargs)
    if logger is not None:
        logger.info("开始推理过程------")

    # 每个调用只写当前影像的结果；文件夹模式在外层按 GPU 并行不同影像。
    test_lib(0, gpu_count, cfg, logger, progress_callback=progress_callback)

    # 修改临时目录清理逻辑，使用指定的后缀
    temp_dir_path = Path(os.path.join(cfg.TEST_OUT_PATH, temp_dir_suffix))
    if temp_dir_path.exists():
        try:
            shutil.rmtree(str(temp_dir_path), ignore_errors=True)
            if logger is not None:
                logger.info(f"已清理临时目录: {temp_dir_path}")
        except Exception as e:
            if logger is not None:
                logger.warning(f"清理临时目录失败: {temp_dir_path}, 错误: {str(e)}")

    if logger is not None:
        logger.info("推理完成")
6
def generate_namelist_from_file(name_list_file, file_root,  suffix):
    name_list = []
    with open(name_list_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line.endswith(suffix):
                continue
            name_list.append(os.path.join(file_root, line))
    return name_list


def _load_change_model(checkpoint_path, cfg, device):
    net = GenerateNet(cfg)
    pretrained_dict = torch.load(checkpoint_path, map_location='cpu')
    module_model_state_dict = {}
    for item, value in pretrained_dict['model_state_dict'].items():
        if item.startswith('module.'):
            item = item[7:]
        module_model_state_dict[item] = value
    net.load_state_dict(module_model_state_dict, strict=True)
    net.to(device)
    net.eval()
    return net


def test_lib(local_rank, gpu_count, cfg, logger, progress_callback=None):
    if logger is not None:
        logger.info('test_lib_begin')
    if gpu_count > 0:
        torch.cuda.set_device(0)
        device = torch.device('cuda', 0)
    else:
        device = torch.device('cpu')

    if logger is not None:
        if gpu_count > 0:
            gpu_names = [torch.cuda.get_device_name(index) for index in range(gpu_count)]
            logger.info(
                '变化检测推理设备: %s 张 GPU, batch_size=%s, devices=%s',
                gpu_count,
                cfg.TEST_BATCHES,
                gpu_names,
            )
        else:
            logger.warning('未检测到可用 GPU，变化检测将回退到 CPU')

    print('rank {} is ready!'.format(local_rank))

    # 获取临时目录后缀，默认为 'tmp'
    temp_dir_suffix = getattr(cfg, 'TEMP_DIR_SUFFIX', 'tmp')

    # get filename list
    pre_namelist = [cfg.PRE_IMG_PATH]
    basenmae = [os.path.split(name)[-1] for name in pre_namelist]
    post_namelist = [cfg.POST_IMG_PATH]
    save_shpname = cfg.TEST_OUT_PATH
    from pathlib import Path
    cfg.TEST_OUT_PATH = str(Path(cfg.TEST_OUT_PATH).parent)
    out_namelist = [os.path.join(cfg.TEST_OUT_PATH, name[0:-len(cfg.IMG_SUFFIX)] + cfg.GT_SUFFIX) for name in basenmae]
    # 使用动态临时目录后缀
    out_shp_namelist = [os.path.join(cfg.TEST_OUT_PATH, temp_dir_suffix, name[0:-len(cfg.IMG_SUFFIX)] + '.shp') for name
                        in basenmae]
    out_shp_namelist1 = [save_shpname]

    # unused: mean_value file and std_value file
    mean_file = cfg.MEAN_FILE
    std_file = cfg.STD_FILE

    # make output path
    if local_rank == 0:
        if not os.path.exists(cfg.TEST_OUT_PATH):
            os.makedirs(cfg.TEST_OUT_PATH)
        # 使用动态临时目录后缀创建临时目录
        temp_dir_path = os.path.join(cfg.TEST_OUT_PATH, temp_dir_suffix)
        if not os.path.exists(temp_dir_path):
            os.makedirs(temp_dir_path)
            if logger is not None:
                logger.info(f"创建临时目录: {temp_dir_path}")

    # 默认只加载主模型；双模型开关打开时才额外占用显存加载第二模型。
    net = _load_change_model(cfg.TEST_CKPT, cfg, device)
    net2 = None
    if getattr(cfg, 'USE_TWO_MODELS', False):
        net2 = _load_change_model(cfg.TEST_CKPT_2, cfg, device)
        if logger is not None:
            logger.info('两个变化检测模型均已加载，推理结果将执行 OR 融合')

    # inference loop
    for n in range(len(pre_namelist)):
        print(f"rank {local_rank}: Inference {n + 1} of {len(pre_namelist)}: {basenmae[n]}")
        # get the indexlist that stores the position of small patches in big image
        if not os.path.exists(post_namelist[n]):
            print(f'File not exists: {post_namelist[n]}')
            continue
        pre_data = gdal.Open(pre_namelist[n])
        big_img_width = pre_data.RasterXSize
        big_img_height = pre_data.RasterYSize
        post_data = gdal.Open(post_namelist[n])
        big_img_width_post = post_data.RasterXSize
        big_img_height_post = post_data.RasterYSize
        gt = list(pre_data.GetGeoTransform())
        proj = pre_data.GetProjection()
        gt_post = list(post_data.GetGeoTransform())
        proj_post = post_data.GetProjection()
        # 处理pre和post尺寸以及分辨率不一致情况
        #import ipdb;ipdb.set_trace()
        res_gapx = 0 if abs(gt[1] - gt_post[1]) < 1e-8 else 1
        res_gapy = 0 if abs(gt[5] - gt_post[5]) < 1e-8 else 1
        gt_rot_pre = str(gt[2]) + str(gt[4])
        gt_rot_post = str(gt_post[2]) + str(gt_post[4])
        gt_str_range_pre = str(gt[0]) + str(gt[3])
        gt_str_range_post = str(gt_post[0]) + str(gt_post[3])
        #print(gt_str_range_pre != gt_str_range_post)
        #print(proj == proj_post)
        #print(gt_rot_pre == gt_rot_post)
        if  gt_str_range_pre == gt_str_range_post and (
                res_gapx + res_gapy == 0 and gt_rot_pre == gt_rot_post):
            if logger is not None:
                logger.info('前后时相影像网格一致，开始生成切片并并行预处理')
        elif gt_str_range_pre != gt_str_range_post and (
                res_gapx + res_gapy == 0 and gt_rot_pre == gt_rot_post):
            overlap_info = compute_intersection_info(pre_data, post_data)
            if overlap_info == None:
                continue
            offset1 = overlap_info["offset_img1"]
            offset2 = overlap_info["offset_img2"]
            clip_height = overlap_info["size"][0]
            clip_width = overlap_info["size"][1]
            # 使用动态临时目录后缀
            pre_data = gdal.Translate(
                os.path.join(cfg.TEST_OUT_PATH, temp_dir_suffix, str(local_rank) + '_pre.tif'),
                pre_data,
                srcWin=[offset1[1], offset1[0], clip_width, clip_height]  # [col, row, width, height]
            )
            post_data = gdal.Translate(
                os.path.join(cfg.TEST_OUT_PATH, temp_dir_suffix, str(local_rank) + '_post.tif'),
                post_data,
                srcWin=[offset2[1], offset2[0], clip_width, clip_height]  # [col, row, width, height]
            )
            pre_namelist[n] = os.path.join(cfg.TEST_OUT_PATH, temp_dir_suffix, str(local_rank) + '_pre.tif')
            post_namelist[n] = os.path.join(cfg.TEST_OUT_PATH, temp_dir_suffix, str(local_rank) + '_post.tif')
            big_img_width = min(pre_data.RasterXSize, post_data.RasterXSize)
            big_img_height = min(pre_data.RasterYSize, post_data.RasterYSize)
            gt_post[0] = overlap_info['bounds'][0]
            gt_post[3] = overlap_info['bounds'][3]
        else:
            if logger is not None:
                logger.info("重采样前后时相影像")
            print("******************")
            print("重采样")
            extent1 = get_extent(pre_data)
            extent2 = get_extent(post_data)

            srs1 = get_srs(pre_data)
            srs2 = get_srs(post_data)

            poly1 = get_polygon_from_extent(extent1)
            poly2 = get_polygon_from_extent(extent2)

            # 将 poly2 转换为 poly1 所在的投影
            poly1_in_srs2 = transform_polygon(poly1, srs1, srs2)
            if poly1_in_srs2 is None:
                raise RuntimeError("无法将第一张图的范围投影到第二张图的坐标系")

            intersection = poly2.intersection(poly1_in_srs2)
            if intersection.is_empty:
                if logger is not None:
                    logger.info("两图无相交区域")
                continue

            xmin, ymin, xmax, ymax = intersection.bounds
            gdal.SetCacheMax(512 * 1024 * 1024)
            if logger is not None:
                logger.info('begin warp')

            # 使用动态临时目录后缀
            pre_data = gdal.Warp(
                os.path.join(cfg.TEST_OUT_PATH, temp_dir_suffix, str(local_rank) + '_pre.tif'),
                pre_data,
                options=gdal.WarpOptions(
                    format='GTiff',
                    creationOptions=["TILED=YES", "BLOCKXSIZE=512", "BLOCKYSIZE=512", "COMPRESS=NONE", "BIGTIFF=YES",
                                     "PROFILE=GEOTIFF"],
                    srcSRS=proj_post,
                    dstSRS=proj_post,
                    xRes=gt_post[1],
                    yRes=gt_post[5],
                    outputBounds=intersection.bounds,
                    resampleAlg="NearestNeighbour",  # "NearestNeighbour",#gdal.GRIORA_Bilinear,
                    warpOptions=['NUM_THREADS=16'],
                    dstNodata=0))

            post_data = gdal.Warp(
                os.path.join(cfg.TEST_OUT_PATH, temp_dir_suffix, str(local_rank) + '_post.tif'),
                post_data,
                options=gdal.WarpOptions(
                    format='GTiff',
                    creationOptions=["TILED=YES", "BLOCKXSIZE=512", "BLOCKYSIZE=512", "COMPRESS=NONE", "BIGTIFF=YES",
                                     "PROFILE=GEOTIFF"],
                    dstSRS=proj_post,
                    xRes=gt_post[1],
                    yRes=gt_post[5],
                    outputBounds=intersection.bounds,
                    resampleAlg="NearestNeighbour",  # "NearestNeighbour",#gdal.GRIORA_Bilinear,
                    warpOptions=['NUM_THREADS=16'],
                    dstNodata=0))
            if logger is not None:
                logger.info('end warp')
            gt_post = post_data.GetGeoTransform()
            big_img_width = pre_data.RasterXSize
            big_img_height = pre_data.RasterYSize
            pre_namelist[n] = os.path.join(cfg.TEST_OUT_PATH, temp_dir_suffix, str(local_rank) + '_pre.tif')
            post_namelist[n] = os.path.join(cfg.TEST_OUT_PATH, temp_dir_suffix, str(local_rank) + '_post.tif')
        del pre_data
        del post_data
        index_list, nx, ny = generate_indexlist(big_img_width, big_img_height, ori_w=0, ori_h=0, \
                                                subsize=(cfg.TEST_IMG_SIZE, cfg.TEST_IMG_SIZE),
                                                overlap=(cfg.TEST_PIXEL_OVERLAP, cfg.TEST_PIXEL_OVERLAP))
        # pack path parameters
        path_info = {
            "preImgPath": pre_namelist[n],
            "postImgPath": post_namelist[n],
            "index_list": index_list,
            "clipsize": (cfg.TEST_IMG_SIZE, cfg.TEST_IMG_SIZE),
            "overlap": (cfg.TEST_PIXEL_OVERLAP, cfg.TEST_PIXEL_OVERLAP),
            "mean_file": mean_file,
            "std_file": std_file
        }
        #import ipdb;ipdb.set_trace()
        # test_dataloader
        test_data = OnlineDataset(path_info, cfg.BAND_NUM, cfg.with_fft)
        if logger is not None:
            logger.info(
                'DataLoader配置: batch_size=%s, num_workers=%s, prefetch_factor=%s',
                cfg.TEST_BATCHES,
                cfg.TEST_NUM_WORKERS,
                cfg.TEST_PREFETCH_FACTOR,
            )
        data_loader_test = torch.utils.data.DataLoader(
            test_data,
            batch_size=cfg.TEST_BATCHES,
            shuffle=False,
            num_workers=cfg.TEST_NUM_WORKERS,
            pin_memory=gpu_count > 0,
            prefetch_factor=cfg.TEST_PREFETCH_FACTOR,
            persistent_workers=cfg.TEST_NUM_WORKERS > 0,
        )
        # create result file

        driver = gdal.GetDriverByName('GTiff')
        if driver is None:
            raise RuntimeError('GDAL GTiff 驱动不可用')
        gdal.ErrorReset()
        out_data = driver.Create(out_namelist[n], big_img_width, big_img_height, 1, gdal.GDT_Byte,
                                 options=['COMPRESS=LZW', 'TILED=YES', 'BIGTIFF=YES'])
        if out_data is None or gdal.GetLastErrorType() >= gdal.CE_Failure:
            error_message = gdal.GetLastErrorMsg() or 'unknown GDAL error'
            raise RuntimeError(f'创建变化检测 BigTIFF 失败: {error_message}')
        out_data.SetProjection(proj_post)
        out_data.SetGeoTransform(gt_post)
        set_color_table(out_data, [(0, 0, 0), (255, 255, 255)])
        set_no_data_value(out_data, 0)
        # start prediction
        total_patches = len(test_data)
        processed_patches = 0
        if local_rank == 0:
            pbar = tqdm(desc=basenmae[n], total=total_patches)

        # cur_batch = 0
        # pre_img, post_img, nodata_mask, indexs = None, None, None, None

        for idx, (pre_img, post_img, nodata_mask, indexs) in enumerate(data_loader_test):
            batch_patches = int(nodata_mask.size(0))
            # 跳过无效数据
            non_zero_mask = nodata_mask.view(nodata_mask.size(0), -1).abs().sum(dim=1) != 0
            if non_zero_mask.sum() == 0:
                processed_patches += batch_patches
                if local_rank == 0:
                    pbar.update(batch_patches)
                    if progress_callback is not None and total_patches > 0:
                        progress_callback(processed_patches, total_patches)
                continue
            pre_img = pre_img[non_zero_mask]
            post_img = post_img[non_zero_mask]
            valid_nodata_mask = nodata_mask[non_zero_mask]
            valid_indexs = [component[non_zero_mask] for component in indexs]

            img = np.concatenate((pre_img, post_img), 1)
            if cfg.WITH_TTA:
                img90 = np.array(np.rot90(img, axes=(2, 3)))  # rotate 90
                img1 = np.ascontiguousarray(np.concatenate([img, img90]), dtype=np.float32)  # concat raw and rot90
                img2 = np.ascontiguousarray(np.array(img1)[:, :, ::-1], dtype=np.float32)
                img3 = np.ascontiguousarray(np.array(img1)[:, :, :, ::-1], dtype=np.float32)
                img4 = np.ascontiguousarray(np.array(img2)[:, :, :, ::-1], dtype=np.float32)
                images = np.ascontiguousarray(np.concatenate([img1, img2, img3, img4]), dtype=np.float32)
                output = test_with_TTA(net, images, cfg.MODEL_NUM_CLASSES, device)
                if net2 is not None:
                    output2 = test_with_TTA(net2, images, cfg.MODEL_NUM_CLASSES, device)
            else:
                output = test_normal(net, img, cfg.MODEL_NUM_CLASSES, device)  # (b,h,w)
                if net2 is not None:
                    output2 = test_normal(net2, img, cfg.MODEL_NUM_CLASSES, device)
            if net2 is not None:
                output = np.bitwise_or(output, output2)
            for i in range(output.shape[0]):
                # maskout nodata area
                pred = output[i] * valid_nodata_mask[i].numpy().astype(output[i].dtype)
                clip_x, clip_y, pad_x, pad_y = (
                    valid_indexs[0][i],
                    valid_indexs[1][i],
                    valid_indexs[2][i],
                    valid_indexs[3][i],
                )
                overlap_half = cfg.TEST_PIXEL_OVERLAP // 2
                if pad_x == -1:
                    pred = pred[:, overlap_half:]
                elif pad_x == 0 and overlap_half > 0:
                    pred = pred[:, overlap_half:-overlap_half]
                elif pad_x == 0 and overlap_half == 0:
                    pass
                else:
                    pred = pred[:, overlap_half:-int(pad_x)]

                if pad_y == -1:
                    pred = pred[overlap_half:, :]
                elif pad_y == 0 and overlap_half > 0:
                    pred = pred[overlap_half:-overlap_half, :]
                elif pad_y == 0 and overlap_half == 0:
                    pass
                else:
                    pred = pred[overlap_half:-int(pad_y), :]
                _band = out_data.GetRasterBand(1)
                gdal.ErrorReset()
                write_result = _band.WriteArray(pred, int(clip_x.data), int(clip_y.data))
                flush_result = _band.FlushCache()
                if (write_result not in (None, 0)
                        or flush_result not in (None, 0)
                        or gdal.GetLastErrorType() >= gdal.CE_Failure):
                    error_message = gdal.GetLastErrorMsg() or 'unknown GDAL error'
                    raise RuntimeError(f'写入变化检测 BigTIFF 失败: {error_message}')
            processed_patches += batch_patches
            if local_rank == 0:
                pbar.update(batch_patches)
                if progress_callback is not None and total_patches > 0:
                    # 按真实切片数汇报；多卡 batch>1 时进度与 ETA 仍保持准确。
                    progress_callback(processed_patches, total_patches)
        build_overviews(out_data)
        del test_data.predataset
        del test_data.postdataset
        # raster to shapefile
        if logger is not None:
            logger.info('raster to shapefile')
        driver = ogr.GetDriverByName('ESRI Shapefile')

        shape_dataset = driver.CreateDataSource(out_shp_namelist[n])
        if shape_dataset is None:
            print('[FATAL] OGR create file failed. [%s]' % out_shp_namelist[n])
            del out_data
            continue
        proj_ref = out_data.GetProjectionRef()
        proj_shp = osr.SpatialReference()
        proj_shp.ImportFromWkt(proj_ref)
        # layer = shape_dataset.CreateLayer('mask', proj_shp, ogr.wkbPolygon)
        layer = shape_dataset.CreateLayer(str(Path(out_shp_namelist[0]).stem), proj_shp, ogr.wkbPolygon)
        # field_name = ogr.FieldDefn('shape', ogr.OFTInteger)
        # layer.CreateField(field_name)

        pre_code_field = ogr.FieldDefn("precode", ogr.OFTString)
        pre_year_field = ogr.FieldDefn('preyear', ogr.OFTInteger)
        post_code_field = ogr.FieldDefn("currcode", ogr.OFTString)
        post_year_field = ogr.FieldDefn("curryear", ogr.OFTInteger)
        layer.CreateField(pre_code_field)
        layer.CreateField(pre_year_field)
        layer.CreateField(post_code_field)
        layer.CreateField(post_year_field)

        band = out_data.GetRasterBand(1)
        if logger is not None:
            logger.info('polygonize begin')
        gdal.Polygonize(band, band, layer, 0, [], callback=None)
        if logger is not None:
            logger.info('polygonize')

        for feature in layer:
            feature.SetField("precode", "0")
            try:
                pre_year_value = str(Path(cfg.PRE_IMG_PATH).name).split('_')[1]
                pre_year_value = int(pre_year_value)
            except:
                pre_year_value = 0

            feature.SetField("currcode", "0")
            try:
                curr_year_value = str(Path(cfg.POST_IMG_PATH).name).split('_')[1]
                curr_year_value = int(curr_year_value)
            except:
                curr_year_value = 0
            feature.SetField("curryear", curr_year_value)
            feature.SetField("preyear", int(curr_year_value) - 1)
            layer.SetFeature(feature)

        # 检查是否有变化图斑：无变化时保留空的 SHP 文件，供后续分类使用
        feature_count = layer.GetFeatureCount()
        del shape_dataset
        del out_data
        if feature_count == 0:
            if logger is not None:
                logger.info('未检测到变化图斑，保留空矢量文件供后续分类')
            # 将空的中间 SHP 复制到最终输出路径，供下游分类使用
            import shutil as _shutil
            for ext in ['.shp', '.shx', '.dbf', '.prj', '.cpg']:
                src = str(Path(out_shp_namelist[0]).with_suffix(ext))
                dst = str(Path(out_shp_namelist1[0]).with_suffix(ext))
                if os.path.exists(src):
                    _shutil.copy2(src, dst)
            continue

        if logger is not None:
            logger.info('矢量优化')
        fill_hole_shp = str(Path(out_shp_namelist[0]).parent / (Path(out_shp_namelist[0]).stem + '_fillhole.shp'))
        fill_hole(out_shp_namelist[0], fill_hole_shp)
        output_shp = out_shp_namelist1[0]
        smooth_polygons_gdf_used(fill_hole_shp, output_shp)
        if logger is not None:
            logger.info('矢量优化完成')

        # os.remove(out_namelist[n])

    # 在最后清理时使用动态临时目录
    if local_rank == 0:
        temp_dir_path = Path(os.path.join(cfg.TEST_OUT_PATH, temp_dir_suffix))
        if temp_dir_path.exists():
            try:
                shutil.rmtree(str(temp_dir_path), ignore_errors=True)
                if logger is not None:
                    logger.info(f"已清理临时目录: {temp_dir_path}")
            except Exception as e:
                if logger is not None:
                    logger.warning(f"清理临时目录失败: {temp_dir_path}, 错误: {str(e)}")

    # VMamba 内部存在绑定方法形成的引用环；同一 GPU 进程继续领取下一张影像前，
    # 主动回收上一张的模型和 CUDA 缓存，避免批量任务显存逐张累积。
    del net
    if net2 is not None:
        del net2
    gc.collect()
    if gpu_count > 0:
        torch.cuda.empty_cache()
