import os
import sys
import multiprocessing as mp
import logging
import shutil
import subprocess
import geopandas as gpd
from shapely.geometry import Polygon
from pathlib import Path



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
    将原始分类代码映射为目标分类代码
    映射规则与Label_map.py中的一致
    """
    label_mapping = {
        0: 0,
        1: 1, 2: 1,
        3: 2, 4: 2,
        5: 3,
        11: 4,
        6: 5, 7: 5, 8: 5, 9: 5,
        10: 6, 12: 6, 13: 6
    }
    return label_mapping.get(original_code, 0)


def process_chunk(args):
    """
    处理单个数据块的函数，包含类别映射
    """
    gdf_chunk, tif_file = args
    from rasterstats import zonal_stats
    import rasterio

    with rasterio.open(tif_file) as src:
        raster_data = src.read(1)
        affine = src.transform
    
        stats = zonal_stats(
            gdf_chunk,
            raster_data,
            affine=affine,
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


def run_one_image_change_detection(input_path1, input_path2, shp_file_path, out_shp_file, logger):
    logger.info('接收影像路径和shp')
    if not os.path.exists(input_path1) or not os.path.exists(input_path2):
        logger.info('输入文件不存在')

    # # 1. 复制文件到带 time 标识的子目录，避免同名覆盖
    # tmp_dir = "./tmp/"
    # os.makedirs(tmp_dir, exist_ok=True)
    # time1_dir = os.path.join(tmp_dir, "time1")
    # time2_dir = os.path.join(tmp_dir, "time2")
    # os.makedirs(time1_dir, exist_ok=True)
    # os.makedirs(time2_dir, exist_ok=True)

    # original_filename1 = os.path.basename(input_path1)
    # original_filename2 = os.path.basename(input_path2)
    # new_input_path1 = os.path.join(time1_dir, original_filename1)       
    # new_input_path2 = os.path.join(time2_dir, original_filename2)

    # logger.info(f'复制前时相: {input_path1} -> {new_input_path1}')
    # logger.info(f'复制后时相: {input_path2} -> {new_input_path2}')
    # shutil.copy2(input_path1, new_input_path1)
    # shutil.copy2(input_path2, new_input_path2)
    # input_path1 = new_input_path1
    # input_path2 = new_input_path2


    # 2. 读取变化掩码
    os.makedirs(os.path.dirname(out_shp_file), exist_ok=True)
    gdf = gpd.read_file(shp_file_path)
    logger.info(f'变化地块数量：{len(gdf)}')
    if len(gdf) == 0:
        logger.info("mask为空，没有变化区域，创建空的shp文件")
        create_empty_shp(shp_file_path, out_shp_file)
        logger.info("处理完成：创建了空的shp文件")
        return

    gdf["uid"] = range(len(gdf))
    origin_crs = gdf.crs
    gdf["pre_code"] = 0
    gdf["curr_code"] = 0

    # 3. 切换到 clie_new 工作目录
    work_dir = r"./clie_new"
    os.chdir(work_dir)

    # 4. 生成后时相的掩码（用于后续分类，可选）
    mask_dir = "./tmp/mask/"
    os.makedirs(mask_dir, exist_ok=True)
    base_name, ext = os.path.splitext(input_path2)
    file_name = os.path.basename(base_name)
    mask_image = os.path.join(mask_dir, f"{file_name}_mask{ext}")
    subprocess.run([
        sys.executable, "shp2mask_v1.py",
        "--input_shp", shp_file_path,
        "--reference_tif", input_path2,
        "--output_mask", mask_image
    ])


    # 5. 前时相分类
    logger.info("前时相影像分类预测中...")
    out_dir_time1 = "./tmp/class_output/time1"
    os.makedirs(out_dir_time1, exist_ok=True)
    subprocess.run([
        sys.executable, "run.py", "run",
        "--image_input_file", input_path1,
        "--image_output_dir", out_dir_time1
    ])
    tif_file1 = f"{out_dir_time1}/{Path(input_path1).stem}.tif"

    # 6. 后时相分类
    logger.info("后时相影像分类预测中...")
    out_dir_time2 = "./tmp/class_output/time2"
    os.makedirs(out_dir_time2, exist_ok=True)
    subprocess.run([
        sys.executable, "run.py", "run",
        "--image_input_file", input_path2,
        "--image_output_dir", out_dir_time2
    ])
    tif_file2 = f"{out_dir_time2}/{Path(input_path2).stem}.tif"

    if not os.path.exists(tif_file1) or not os.path.exists(tif_file2):
        logger.error("前时相或后时相分类结果缺失")
        return

    # 7. 统一坐标系（以后时相分类结果为基准）
    import rasterio
    with rasterio.open(tif_file2) as src:
        raster_crs = src.crs
    if gdf.crs != raster_crs:
        gdf = gdf.to_crs(raster_crs)

    # 8. 并行计算前后时相类别
    logger.info("计算前时相各地块类别...")
    pre_classes = parallel_zonal_stats(gdf, tif_file1, num_processes=4)
    gdf["pre_code"] = pre_classes

    logger.info("计算后时相各地块类别...")
    curr_classes = parallel_zonal_stats(gdf, tif_file2, num_processes=4)
    gdf["curr_code"] = curr_classes
    gdf["curr_code"] = gdf["curr_code"].astype(int)

    # 9. 保存结果
    gdf = gdf.to_crs(origin_crs)
    gdf.to_file(out_shp_file, encoding="utf-8")
    os.chdir("../")
    logger.info(f"处理完成，结果保存至：{out_shp_file}")



if __name__ == '__main__':
#    input_path1 = '/mnt/prod/k3s/working/cd_seg/testdata/pre/R_417_203_1668_815.tif'
 #   input_path2 = '/mnt/prod/k3s/working/cd_seg/testdata/post/R_417_203_1668_815.tif'
  #  mask_path = '/mnt/prod/k3s/working/cd_seg/output/R_417_203_1668_815.shp'
   # output_path = '/mnt/prod/k3s/working/cd_seg/output_2/Export_Output_2_result2.shp'

    input_path1 = '/share/test/pre/107445_107671_24095_24317.tif'
    input_path2 = '/share/test/post/107445_107671_24095_24317.tif'
    mask_path = '/output/R_test.shp'
    output_path = '/output/Export_Output_2_result2.shp'

    
    # 初始化logger
    log_file = "./server_log"
    logger = logging.getLogger('mylog')
    logger.setLevel(logging.INFO)

    file_handlers = logging.FileHandler(log_file)
    file_handlers.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    # th = handlers.TimedRotatingFileHandler(filename=log_file, when='D', backupCount=30,
    #                                        encoding='utf-8')  # 往文件里写入#指定间隔时间自动生成文件的处理器

    fmt1 = '%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s'
    fmt = logging.Formatter(fmt1)
    # th.setFormatter(fmt1)

    ###
    file_handlers.setFormatter(fmt)
    console_handler.setFormatter(fmt)

    # logger.addHandler(th)
    logger.addHandler(file_handlers)
    logger.addHandler(console_handler)

    logger.info(f'前时相影像路径：{input_path1}')
    logger.info(f'后时相影像路径：{input_path2}')
    logger.info(f'变化结果路径：{mask_path}')
    logger.info(f'结果保存路径：{output_path}')
    run_one_image_change_detection(input_path1, input_path2, mask_path, output_path, logger)

    
