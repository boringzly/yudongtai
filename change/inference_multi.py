import os

os.environ["PROJ_LIB"] = "./utils/proj"
import sys
import json
import logging
from pathlib import Path
import torch.multiprocessing as mp
import requests

# 设置多进程
mp.set_start_method('spawn', force=True)

from test_lib_batch_memeff_single_image_nomp import test_lib_big_memeff
from merge import merge_shp
from process_shp_by_county import process_prediction_with_division
from schedule import post_status, post_progress


def get_gdb_path_from_api(admin_province_code, release_year, logger):
    """通过API动态获取GDB文件路径"""
    try:
        # API参数
        params = {
            'ak': 'mf85056077e36b72b8ef9170acd8d95b9e',
            'op': 'select_with_ref',
            'page_count': 200000,
            'page_num': 1,
            'file_type': 'gdb',
            'release_status': 'online',
            'usage_status': 'ready',
            'admin_province_code': admin_province_code,
            'release_year': release_year,
            'type': 'current'
        }

        # API地址
        api_url = "http://172.20.46.51:7010/sj_assets/v6/api/ecomn/raster_result"

        logger.info(f"请求GDB文件路径 API: {api_url}")
        logger.info(f"请求参数: admin_province_code={admin_province_code}, release_year={release_year}")

        # 发送请求
        response = requests.get(api_url, params=params, timeout=30)
        response.raise_for_status()

        data = response.json()

        # 检查返回结果
        if 'result' in data and 'item_list' in data['result'] and data['result']['item_list']:
            item_list = data['result']['item_list']
            if len(item_list) == 1:
                file_path = item_list[0]['file_path']
                file_path = f"/cresdashare{file_path}"
                logger.info(f"成功获取GDB文件路径: {file_path}")
                return file_path
            else:
                from datetime import datetime
                sorted_items = sorted(item_list,
                                      key=lambda x: datetime.strptime(x.get('release_time', '1970-01-01 00:00:00'),
                                                                      '%Y-%m-%d %H:%M:%S'), reverse=True)
                latest_file_path = sorted_items[0]['file_path']
                latest_release_time = sorted_items[0]['release_time']
                latest_file_path = f"/cresdashare{latest_file_path}"
                logger.info(f"从 {len(item_list)} 个结果中选择最新的GDB文件: {latest_file_path}")
                logger.info(f"最新文件的发布时间: {latest_release_time}")
                return latest_file_path
        else:
            logger.warning(f"API返回结果为空，使用默认GDB路径")
            return "./assets/2023年广东省分县现状.gdb"

    except Exception as e:
        logger.error(f"获取GDB文件路径失败: {str(e)}，使用默认路径")
        return "./assets/2023年广东省分县现状.gdb"


def setup_logger(task_id):
    """设置日志记录器"""
    log_file = "./server_log"
    logger = logging.getLogger('mylog')
    logger.setLevel(logging.INFO)

    # 检查是否已有处理器，避免重复添加
    if logger.handlers:
        # 返回一个带任务标识的logger适配器
        class TaskLoggerAdapter(logging.LoggerAdapter):
            def process(self, msg, kwargs):
                return f'[{task_id}] {msg}', kwargs

        return TaskLoggerAdapter(logger, {})

    file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
    console_handler = logging.StreamHandler()

    fmt = logging.Formatter(
        f'[{task_id}] %(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s'
    )
    file_handler.setFormatter(fmt)
    console_handler.setFormatter(fmt)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def set_cpu_affinity(cpu_list, logger):
    """设置当前进程的CPU亲和性"""
    try:
        import psutil
        process = psutil.Process()
        process.cpu_affinity(cpu_list)
        logger.info(f"设置CPU亲和性成功: CPUs {cpu_list}")
        return True
    except Exception as e:
        logger.warning(f"设置CPU亲和性失败: {str(e)}")
        return False


def add_fields_with_paths(shp_path, save_shp_path, pre_file, post_file):
    """添加字段的辅助函数"""
    import geopandas as gpd
    try:
        gdf = gpd.read_file(shp_path)
        if "pre_file" not in gdf.columns:
            gdf["pre_file"] = pre_file
        else:
            gdf["pre_file"] = gdf["pre_file"].fillna(pre_file)

        if "post_file" not in gdf.columns:
            gdf["post_file"] = post_file
        else:
            gdf["post_file"] = gdf["post_file"].fillna(post_file)

        gdf.to_file(save_shp_path, driver="ESRI Shapefile", encoding="utf-7")
        return True
    except Exception as e:
        # 如果读取失败，可能是空文件，尝试创建新的gdf
        try:
            # 获取原始文件的空间参考
            from osgeo import ogr
            driver = ogr.GetDriverByName('ESRI Shapefile')
            data_source = driver.Open(shp_path, 0)
            if data_source:
                layer = data_source.GetLayer()
                srs = layer.GetSpatialRef()
                data_source = None
            
            # 创建一个空的GeoDataFrame
            import pandas as pd
            from shapely.geometry import Polygon
            gdf = gpd.GeoDataFrame(columns=['pre_file', 'post_file', 'geometry'], geometry='geometry')
            gdf['pre_file'] = [pre_file]
            gdf['post_file'] = [post_file]
            # 创建一个无效的几何体
            gdf['geometry'] = [Polygon()]
            
            # 设置坐标系
            if srs:
                gdf.crs = srs.ExportToWkt()
            
            gdf.to_file(save_shp_path, driver="ESRI Shapefile", encoding="utf-7")
            return True
        except Exception as e2:
            return False


def create_completion_marker(output_path, logger):
    """创建完成标记文件"""
    try:
        import time
        marker_file = Path(output_path).with_suffix('.completed.txt')
        with open(marker_file, 'w') as f:
            f.write(f"Completed at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Output file: {output_path}\n")
        logger.info(f"创建完成标记文件: {marker_file}")
        return marker_file
    except Exception as e:
        logger.error(f"创建标记文件失败: {str(e)}")
        return None


def wait_for_all_markers(all_output_list, timeout=None, logger=None):
    """等待所有标记文件生成"""
    import time

    start_time = time.time()

    # 将输出文件路径转换为标记文件路径
    all_marker_files = [str(Path(f).with_suffix('.completed.txt')) for f in all_output_list]

    while True:
        existing_markers = [f for f in all_marker_files if Path(f).exists()]
        missing_markers = [f for f in all_marker_files if not Path(f).exists()]

        if logger:
            logger.info(f'等待标记文件生成... 已有 {len(existing_markers)}/{len(all_marker_files)} 个标记文件')
            if missing_markers:
                logger.info(f'缺失标记文件: {[Path(f).name for f in missing_markers[:3]]}...')  # 只显示前3个

        if len(existing_markers) == len(all_marker_files):
            return existing_markers

        time.sleep(30)


def process_inference_batch(task_data):
    """处理推理任务批次"""
    task_id = task_data.get('task_id', 'unknown')
    logger = setup_logger(task_id)

    try:
        # 设置CPU亲和性（如果提供了配置）
        cpu_affinity = task_data.get('cpu_affinity')
        if cpu_affinity:
            set_cpu_affinity(cpu_affinity, logger)

        # 从任务数据中获取用户输入的省份编码和年份
        admin_province_code = task_data.get('admin_province_code', '440000')
        release_year = task_data.get('release_year', 2024)

        logger.info(f"使用用户输入的GDB参数 - 省份编码: {admin_province_code}, 年份: {release_year}")

        # 动态获取GDB文件路径
        gdb_path = get_gdb_path_from_api(admin_province_code, release_year, logger)

        logger.info(f"使用GDB文件: {gdb_path}")

        file_list = task_data['file_list']
        input_path2_list = task_data['input_path2']
        output_list = task_data['output_list']
        job_id = task_data['job_id']
        callback_url = task_data['callback_url']
        is_final_merge = task_data.get('is_final_merge', False)
        temp_dir_suffix = task_data.get('temp_dir_suffix', f'tmp_{task_id}')
        temp_base_dir = task_data.get('temp_base_dir', '')  # 获取临时目录信息

        # 记录磁盘使用信息
        if temp_base_dir:
            logger.info(f'任务使用临时目录: {temp_base_dir}')

        logger.info(f'开始处理任务，文件数量: {len(file_list)}，临时目录后缀: {temp_dir_suffix}')

        # 处理当前批次的文件
        success_count = 0
        for i, (pre_file, input_path2, output_path) in enumerate(zip(file_list, input_path2_list, output_list)):
            logger.info(f'处理文件 {i + 1}/{len(file_list)}: {Path(pre_file).name} -> {output_path}')

            try:
                # 将输出路径转换为 Path 对象
                output_path = Path(output_path)
                # 确保输出目录存在
                output_path.parent.mkdir(parents=True, exist_ok=True)

                # 检查输出文件是否已经存在（可能之前已经处理过）
                marker_file = output_path.with_suffix('.completed.txt')
                if marker_file.exists():
                    logger.info(f'跳过已处理文件: {Path(pre_file).name} (标记文件已存在)')
                    success_count += 1
                    continue

                # 执行推理
                test_lib_big_memeff(
                    pre_file,
                    input_path2,
                    str(output_path),
                    logger,
                    callback_url,
                    job_id,
                    temp_dir_suffix=temp_dir_suffix
                )

                # 检查输出文件是否存在
                if output_path.exists():
                    # 检查文件是否为空（没有几何要素）
                    try:
                        import geopandas as gpd
                        gdf = gpd.read_file(str(output_path))
                        
                        if len(gdf) > 0 and not gdf.geometry.is_empty.all():
                            # 非空文件：添加字段并进行县级处理
                            add_fields_with_paths(str(output_path), str(output_path), pre_file, input_path2)
                            
                            # 使用用户指定的GDB文件进行县级处理
                            county_list = task_data.get('county_list', [])
                            logger.info(f"开始县级处理: {output_path}")
                            process_prediction_with_division(output_path, gdb_path, None, county_list)
                            logger.info(f"县级处理完成: {output_path}")
                        else:
                            # 空文件（无空间交集），只添加字段
                            logger.info(f"输出文件为空（无空间交集），仅添加字段: {output_path}")
                            add_fields_with_paths(str(output_path), str(output_path), pre_file, input_path2)
                    except Exception as e:
                        logger.warning(f"读取输出文件失败，跳过后续处理: {str(e)}")
                        # 仍然尝试添加字段
                        try:
                            add_fields_with_paths(str(output_path), str(output_path), pre_file, input_path2)
                        except:
                            pass

                    # 创建完成标记文件（如果test_lib_big_memeff没有创建的话）
                    if not marker_file.exists():
                        marker_file = create_completion_marker(str(output_path), logger)
                    
                    if marker_file and marker_file.exists():
                        success_count += 1
                        logger.info(f'完成文件 ({success_count}/{len(file_list)}): {Path(pre_file).name}')
                    else:
                        logger.warning(f'标记文件创建失败: {output_path}')
                else:
                    logger.warning(f'输出文件未生成: {output_path}')
                    
                    # 即使输出文件未生成，也可能创建了标记文件（在无空间交集的情况下）
                    if marker_file.exists():
                        logger.info(f'标记文件存在但输出文件未生成，视为处理完成: {output_path}')
                        success_count += 1

            except Exception as e:
                logger.error(f'处理文件 {pre_file} 时出错: {str(e)}')
                import traceback
                logger.error(traceback.format_exc())
                continue

        logger.info(f'任务 {task_id} 处理完成，成功: {success_count}/{len(file_list)}')

        # 如果这是负责最终合并的任务
        if is_final_merge:
            logger.info('开始执行最终合并...')

            # 等待所有标记文件（表示后处理完成）
            all_output_list = task_data.get('all_output_list', [])
            final_output_path = task_data.get('final_output_path')
            county_list = task_data.get('county_list', [])

            if not all_output_list:
                logger.error('未提供完整的输出文件列表')
                if callback_url:
                    error_message = {"errorCode": 4006, "message": "输出文件列表为空"}
                    post_progress(callback_url, post_status['error'], -1, error_message, job_id)
                return

            logger.info('等待所有后处理完成（检测标记文件）...')
            if callback_url:
                post_progress(callback_url, "等待所有后处理完成", 80, None, job_id)

            # 等待标记文件而不是原始文件
            existing_markers = wait_for_all_markers(all_output_list, logger=logger)

            if len(existing_markers) == len(all_output_list):
                logger.info(f'所有 {len(existing_markers)} 个文件的后处理都已完成，开始合并')
                if callback_url:
                    post_progress(callback_url, "开始合并结果", 90, None, job_id)

                # 确保最终输出目录存在
                Path(final_output_path).parent.mkdir(parents=True, exist_ok=True)

                # 合并所有shapefile，包括空的
                merge_shp(all_output_list, final_output_path)
                from tichu import filter_shp_by_area
                filter_shp_by_area(final_output_path, final_output_path)
                logger.info('文件合并完成')
                if callback_url:
                    post_progress(callback_url, post_status['finish'], 100, None, job_id)
                logger.info('任务全部完成')
            else:
                logger.error(f'后处理未全部完成，仅完成 {len(existing_markers)}/{len(all_output_list)} 个文件')
                if callback_url:
                    error_message = {"errorCode": 4006,
                                     "message": f"后处理未完成: {len(existing_markers)}/{len(all_output_list)} 个文件"}
                    post_progress(callback_url, post_status['error'], -1, error_message, job_id)

    except Exception as e:
        logger.error(f'任务 {task_id} 处理失败: {str(e)}')
        import traceback
        logger.error(traceback.format_exc())
        if callback_url:
            error_message = {"errorCode": 4005, "message": f"推理服务错误: {str(e)}"}
            post_progress(callback_url, post_status['error'], -1, error_message, job_id)


def main():
    if len(sys.argv) != 2:
        print("用法: python inference_multi.py <task_file>")
        sys.exit(1)

    task_file = sys.argv[1]

    try:
        with open(task_file, 'r') as f:
            task_data = json.load(f)

        process_inference_batch(task_data)

        # 清理临时文件
        if os.path.exists(task_file):
            os.remove(task_file)

    except Exception as e:
        print(f"推理服务启动失败: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()