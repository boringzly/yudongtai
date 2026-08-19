#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Func    : 指定分类体系的变化检测类别识别
# @Time    : 2024/7/16 15:46
# @File    : GPT4_CD.py
# @Software: PyCharm
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["MAX_PIXELS"] = "1003520"
os.environ["VIDED_MAX_PIXELS"] = "50176"
os.environ["FPS_MAX_FRAMES"] = "12"
os.environ["PYTHONWARNINGS"] = "ignore"
import warnings

warnings.filterwarnings("ignore")
from pathlib import Path
import base64
from io import BytesIO
import argparse
import json
import logging
from copy import deepcopy
import cv2
import shutil
import numpy as np
import shapefile
from tqdm import tqdm
from PIL import Image
import multiprocessing as mp
from functools import partial

# from osgeo import gdal
from pyproj import Transformer
from shapely.geometry import box, mapping, Polygon
import geopandas as gpd

tmp_save_folder = "./tmp_save"
Path(tmp_save_folder).mkdir(parents=True, exist_ok=True)


def encode_image(image):
    """
    把array格式的图片编码成base64格式
    :param image:
    :return:
    """
    image = Image.fromarray(image)
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return img_str


def create_empty_shp(source_shp_path, target_shp_path):
    """
    创建一个空的shp文件，包含指定字段，几何类型为Polygon
    """
    # 读取原始shp文件获取CRS信息
    source_gdf = gpd.read_file(source_shp_path)

    # 创建一个空的GeoDataFrame，明确指定几何类型为Polygon
    # 使用空的Polygon几何体确保几何类型正确
    empty_geometry = Polygon()
    empty_gdf = gpd.GeoDataFrame({
        'preyear': [],
        'curryear': [],
        'pre_code': [],
        'xian': [],
        'curr_code': []
    }, geometry=[], crs=source_gdf.crs)

    # 设置字段数据类型
    empty_gdf['preyear'] = empty_gdf['preyear'].astype(str)
    empty_gdf['curryear'] = empty_gdf['curryear'].astype(str)
    empty_gdf['pre_code'] = empty_gdf['pre_code'].astype(int)
    empty_gdf['xian'] = empty_gdf['xian'].astype(str)
    empty_gdf['curr_code'] = empty_gdf['curr_code'].astype(int)

    # 保存为新的shp文件，明确指定几何类型
    empty_gdf.to_file(target_shp_path, encoding="utf-8", geometry_type='Polygon')

    return empty_gdf


def map_class_code(original_code):
    """
    将原始分类代码（30类）映射为目标分类代码（7类）
    新的映射规则：
    0: 背景（0）
    1: 耕地（1，2）
    2: 林地（3-13）
    3: 草地（14-17）
    4: 水域（25、30）
    5: 建设用地（18、19、20、21、22、23、24、26、28、29）
    6: 未利用地（27）
    """
    # 背景
    if original_code == 0:
        return 0

    # 耕地（1，2）
    if original_code in [1, 2]:
        return 1

    # 林地（3-13）
    if original_code in range(3, 14):  # 3-13
        return 2

    # 草地（14-17）
    if original_code in range(14, 18):  # 14-17
        return 3

    # 水域（25、30）
    if original_code in [25, 30]:
        return 4

    # 建设用地（18、19、20、21、22、23、24、26、28、29）
    if original_code in [18, 19, 20, 21, 22, 23, 24, 26, 28, 29]:
        return 5

    # 未利用地（27）
    if original_code == 27:
        return 6

    # 默认返回背景
    return 0


def process_chunk(args):
    """
    处理单个数据块的函数，包含类别映射
    """
    gdf_chunk, tif_file = args
    from rasterstats import zonal_stats

    # 使用categorical统计获取每个类别的像元数量
    stats = zonal_stats(
        gdf_chunk,
        tif_file,
        categorical=True,
        nodata=0
    )

    major_classes = []
    for stat in stats:
        if stat:
            # 找到最多的原始类别
            original_major_class = max(stat, key=stat.get)
            # 映射到目标分类
            mapped_class = map_class_code(original_major_class)
            major_classes.append(mapped_class)
        else:
            # 如果没有统计到数据，默认为5（建设用地）
            major_classes.append(5)

    return list(gdf_chunk.index), major_classes


def parallel_zonal_stats(gdf, tif_file, num_processes=4):
    """
    并行处理zonal_stats
    """
    # 将数据框分成多个块
    total_rows = len(gdf)
    chunk_size = total_rows // num_processes
    chunks = []

    for i in range(num_processes):
        start_idx = i * chunk_size
        if i == num_processes - 1:
            # 最后一个进程处理剩余的所有行
            end_idx = total_rows
        else:
            end_idx = start_idx + chunk_size
        chunks.append(gdf.iloc[start_idx:end_idx])

    # 准备参数
    args_list = [(chunk, tif_file) for chunk in chunks]

    # 使用进程池并行处理
    with mp.Pool(processes=num_processes) as pool:
        results = pool.map(process_chunk, args_list)

    # 合并结果
    all_indices = []
    all_classes = []
    for indices, classes in results:
        all_indices.extend(indices)
        all_classes.extend(classes)

    # 按原始索引排序
    sorted_results = sorted(zip(all_indices, all_classes), key=lambda x: x[0])
    final_classes = [cls for idx, cls in sorted_results]

    return final_classes


def main_classification(pre_image, post_image, mask_shp, save_shp, cls_txt='asset/cls_txt.txt', image_size=512,
                        logger=None, callback_url=None, job_id=None):
    """
    调用GPT4v完成指定分类体系的变化检测类别识别
    :param image_size: 输入gpt4v的图片大小
    :param cls_txt: 分类体系txt文件
    :param pre_image: 前期影像
    :param post_image: 后期影像
    :param mask_shp: 变化区域shp文件
    :return:
    """
    import geopandas as gpd
    logger.info(f"开始处理中：")
    # import ipdb;ipdb.set_trace()
    Path(tmp_save_folder).mkdir(parents=True, exist_ok=True)
    temp_dir = './temp'
    Path(temp_dir).mkdir(parents=True, exist_ok=True)
    # 读取分类体系
    # with open(cls_txt, 'r') as f:
    #     cls = f.readlines()
    # cls = [c.strip() for c in cls]
    shp_file_path = mask_shp
    # pre_img_path = pre_image
    # post_img_path = post_image
    out_shp_file = save_shp

    gdf = gpd.read_file(shp_file_path)
    logger.info(f"开始检测是否无变化")
    logger.info(len(gdf))
    # 检测mask是否为空（没有变化区域）
    if len(gdf) == 0:
        logger.info("mask为空，没有变化区域，创建空的shp文件")
        create_empty_shp(shp_file_path, out_shp_file)
        logger.info("处理完成：创建了空的shp文件")
        return

    gdf["pre_cls"] = ""
    gdf["post_cls"] = ""
    patch_size = 1024
    gdf["uid"] = range(len(gdf))
    origin_crs = gdf.crs

    '''
    input_gdb_path_neimeng = "./asset/gdb/2022广东和内蒙古/内蒙古/全省现状.gdb"
    # input_gdb_path_neimeng = "./asset/gdb/2022广东和内蒙古/广东/全省现状_最新拷贝20240202.gdb"
    input_gdb_path_guangdong = "./asset/gdb/2022广东和内蒙古/广东/全省现状.gdb"

    print(f"读取土地利用分类体系中......")
    landuse_gdf = gpd.read_file(input_gdb_path_neimeng)
    if gdf.crs != landuse_gdf.crs:
        gdf = gdf.to_crs(landuse_gdf.crs)
    print(f"求相交操作中......")
    overlay = gpd.overlay(gdf, landuse_gdf, how="intersection")
    if len(overlay) == 0:
        landuse_gdf = gpd.read_file(input_gdb_path_guangdong)
        if gdf.crs != landuse_gdf.crs:
            gdf = gdf.to_crs(landuse_gdf.crs)
        overlay = gpd.overlay(gdf, landuse_gdf, how="intersection")
    overlay["area"] = overlay.geometry.area
    classified = (
        overlay.loc[overlay.groupby("uid")["area"].idxmax()][["uid", "ld2022"]]
        .reset_index(drop=True)
    )
    gdf = gdf.merge(classified, on="uid", how="left")
    gdf["pre_code"] = gdf["ld2022"]
    gdf.drop(columns=["uid", "ld2022"], inplace=True)
    gdf = gdf.to_crs(origin_crs)
    mapping_dict = {
        111: 11,
        112: 11,
        113: 11,
        114: 11,
        115: 11,
        121: 12,
        122: 12,
        123: 12,
        124: 12,
        125: 12
    }
    gdf["pre_code"] = gdf["pre_code"].map(mapping_dict).fillna(gdf["pre_code"])
    # gdf["pre_code"] = 1
    # gdf["curr_code"] = 5
    '''
    import pandas as pd
    gdf["pre_code"] = pd.to_numeric(gdf["pre_code"], errors='coerce').fillna(0).astype(int)

    logger.info("后时相影像预测中")
    import subprocess
    work_dir = r"./clie_new"
    os.chdir(work_dir)
    base_name, ext = os.path.splitext(post_image)
    file_name = os.path.basename(base_name)
    mask_image = f"/cresdashare/docker/tmp/mask/{file_name}_mask{ext}"
    subprocess.run([
        "/root/miniconda/bin/python", "shp2mask_v1.py",
        "--input_shp", shp_file_path,
        "--reference_tif", post_image,
        "--output_mask", mask_image])
    """
    logger.info("MASK")
    # 检测mask是否全为0
    try:
        import rasterio
        with rasterio.open(mask_image) as src:
            mask_data = src.read(1)
            if np.all(mask_data == 0):
                logger.info("mask全为0，没有变化区域，创建空的shp文件")
                create_empty_shp(shp_file_path, out_shp_file)
                os.chdir("../")
                logger.info("处理完成：创建了空的shp文件")
                return
    except Exception as e:
        logger.warning(f"检测mask失败，继续执行后续处理: {e}")
    logger.info(f"JIANCE0")
    """
    subprocess.run([
        "/root/miniconda/bin/python", "run.py", "run",
        "--image_input_file", post_image,
        "--image_output_dir", "/cresdashare/docker/tmp/class_output"
    ])
    tif_file = f"/cresdashare/docker/tmp/class_output/{Path(post_image).stem + '.tif'}"
    from rasterstats import zonal_stats
    import rasterio
    with rasterio.open(tif_file) as src:
        raster_src = src.crs
        raster_bounds = src.bounds

    if gdf.crs != raster_src:
        gdf = gdf.to_crs(raster_src)

    logger.info(f"predicting post class with 4 processes....")

    # 使用并行处理替代原来的串行处理
    # 注意：这里会使用修改后的process_chunk函数，其中包含类别映射
    major_classes = parallel_zonal_stats(gdf, tif_file, num_processes=4)

    gdf["curr_code"] = major_classes
    gdf["curr_code"] = gdf["curr_code"].astype(int)

    gdf = gdf.to_crs(origin_crs)
    gdf.to_file(out_shp_file, encoding="utf-8")
    os.chdir("../")
    logger.info(f"predicting finished...")

    # shutil.rmtree(tmp_save_folder)
    # logger.info(f"shanchu finished")


class_dict = {
    "耕地": 1,
    "林地": 2,
    "草地": 3,
    "水域": 4,
    "建设用地": 5,
    "未利用土地": 6,
}

if __name__ == '__main__':
    # parser = argparse.ArgumentParser(description='指定分类体系的变化检测类别识别')
    # parser.add_argument('--cls_txt', type=str, help='分类体系txt文件')
    # parser.add_argument('--pre_image', type=str, help='前期影像')
    # parser.add_argument('--post_image', type=str, help='后期影像')
    # parser.add_argument('--mask_shp', type=str, help='变化区域shp文件')
    # parser.add_argument('--image_size', type=int, default=512, help='输入大模型的图片大小')
    # args = parser.parse_args()
    # main(args.cls_txt, args.pre_image, args.post_image, args.mask_shp, args.image_size)
    main_classification('asset/ecomn_2022_150000_change_inner.tif', 'asset/ecomn_2023_12_17.tif', 'asset/output.shp',
                        'asset/output_10_result.shp')