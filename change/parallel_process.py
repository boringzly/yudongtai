import multiprocessing
import os
import geopandas as gpd
import pandas as pd
from get_intersecting_raster_names import get_intersecting_raster_names
from process_shp_by_county import process_shp_by_county
from test_lib_batch_memeff_single_image import test_lib_big_memeff

def merge_shapefiles(shapefile_paths, output_path):
    """
    使用GeoPandas合并Shapefile文件
    """
    print(f"开始合并 {len(shapefile_paths)} 个Shapefile...")

    gdfs = []
    # 读取所有文件
    for i, path in enumerate(shapefile_paths, 1):
        print(f"读取文件 {i}: {os.path.basename(path)}")

        # 尝试不同编码
        for encoding in ['utf-8', 'gbk', 'latin1']:
            try:
                gdf = gpd.read_file(path, encoding=encoding)
                print(f"  ✓ 读取成功 ({len(gdf)} 要素，编码: {encoding})")
                gdfs.append(gdf)
                break
            except Exception as e:
                if encoding == 'latin1':  # 最后一个编码
                    print(f"  ✗ 读取失败: {e}")

    if not gdfs:
        print("❌ 没有成功读取任何文件")
        return False

    # 合并数据
    print("合并数据...")
    merged_gdf = pd.concat(gdfs, ignore_index=True)
    print(f"合并后总要素数: {len(merged_gdf)}")

    # 保存文件
    print(f"保存到: {output_path}")
    merged_gdf.to_file(output_path, encoding='utf-8')

    # 创建编码文件确保中文正常显示
    cpg_path = output_path.replace('.shp', '.cpg')
    with open(cpg_path, 'w') as f:
        f.write('UTF-8')

    print("✅ 合并完成!")
    return True

def process_image_batch(process_func, batch_tasks):
    """
    处理一批影像任务的通用函数
    参数:
    process_func: function - 处理单个影像对的函数
    batch_tasks: list - 任务列表，每个元素是 (post_img, pre_img, county_id, output_path) 的元组
    返回:
    list - 处理生成的Shapefile路径列表
    """
    result_shps = []

    # 直接处理传入的批次任务
    for i, (post_img, pre_img, county_id, output_path) in enumerate(batch_tasks):
        try:
            # 调用处理函数，传入四个参数
            output_shp = process_func(post_img, pre_img, county_id, output_path)
            if output_shp and os.path.exists(output_shp):
                result_shps.append(output_shp)
                print(f"任务 {i} 完成: 后时相={os.path.basename(post_img)}, 前时相={os.path.basename(pre_img)}, 县区={county_id}")
        except Exception as e:
            print(f"任务 {i} 出错 (后时相={post_img}, 前时相={pre_img}, 县区={county_id}): {e}")

    return result_shps


def parallel_process_images(image_dict, output_dir, process_func, num_processes=3):
    """
    多进程并行处理影像并合并结果
    参数:
    image_list: list - 影像路径列表
    process_func: function - 处理单个影像的函数
    num_processes: int - 进程数，默认为3
    返回:
    str - 合并后的Shapefile路径
    """
    if not image_dict:
        return None

    # 首先提取所有的前时相影像（二级键）
    pre_images = []
    for post_img, pre_dict in image_dict.items():
        for pre_img in pre_dict.keys():
            if pre_img not in pre_images:
                pre_images.append(pre_img)

    # 将前时相影像列表分为num_processes份
    if len(pre_images) > num_processes:
        batch_size = len(pre_images) // num_processes
    else:
        batch_size = 1

    batches = []

    for i in range(num_processes):
        start_idx = i * batch_size
        if i == num_processes - 1:
            # 最后一个进程处理剩余的所有前时相影像
            batch_pre_images = pre_images[start_idx:]
        else:
            batch_pre_images = pre_images[start_idx:start_idx + batch_size]

        # 为每个批次构建完整的任务数据
        batch_tasks = []
        for pre_img in batch_pre_images:
            # 找到包含这个前时相影像的所有后时相影像和县区ID
            for post_img, pre_dict in image_dict.items():
                if pre_img in pre_dict:
                    county_id = pre_dict[pre_img]
                    output_filename = f"output_{os.path.basename(post_img).split('.')[0]}_{os.path.basename(pre_img).split('.')[0]}_{county_id}.shp"
                    output_shp_path = os.path.join(output_dir, output_filename)
                    batch_tasks.append((post_img, pre_img, county_id, output_shp_path))

        batches.append((process_func, batch_tasks))

    print(f"总共分成 {len(batches)} 个批次")
    for i, batch in enumerate(batches):
        print(f"批次 {i}: {len(batch)} 个任务")

    # 使用多进程处理
    with multiprocessing.Pool(processes=num_processes) as pool:
        results = pool.starmap(process_image_batch, batches)

    # 收集所有生成的shp文件
    all_shp_files = []
    for batch_result in results:
        all_shp_files.extend(batch_result)
    # import ipdb;ipdb.set_trace()
    # 合并所有shp文件
    if all_shp_files:
        merged_output = "merged_result.shp"
        merge_shapefiles(all_shp_files, merged_output)

        # 可选：删除临时文件
        for shp_file in all_shp_files:
            try:
                # 删除shp及相关文件
                base_name = os.path.splitext(shp_file)[0]
                for ext in ['.shp', '.shx', '.dbf', '.prj']:
                    file_to_delete = base_name + ext
                    if os.path.exists(file_to_delete):
                        os.remove(file_to_delete)
            except:
                pass

        return merged_output

    return None

def prediction_function(post_img, pre_img, county_id, shp_path):
    """
    预测函数 - 创建包含随机矩形框的WGS84 Shapefile
    参数:
    image_path: str - 输入影像路径
    output_shp_path: str - 输出shp路径
    返回:
    str - 生成的shp文件路径
    """
    gdb_path = "./assets/2023年广东省分县现状.gdb"
    test_lib_big_memeff(pre_img, post_img, shp_path, None)
    process_shp_by_county(shp_path, gdb_path)
    return shp_path


if __name__ == "__main__":
    # 影像列表
    shp_path = os.getenv("PATH_SHP") # 分幅矢量
    raster_path = os.getenv("PATH_POST_IMG") # 后时相影像
    output_dir = os.getenv("PATH_RESULT") # 最终输出图斑目
    image_dict = get_intersecting_raster_names(shp_path, raster_path)

    result_shp = parallel_process_images(
        image_dict=image_dict,
        output_dir=output_dir,
        process_func=prediction_function,
        num_processes=8
    )

    print(f"合并后的结果文件: {result_shp}")
