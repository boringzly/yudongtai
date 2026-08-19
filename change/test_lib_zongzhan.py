import os
import threading
import hashlib
import math

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["PROJ_LIB"] = "./utils/proj"
import sys
import json
import tempfile
import subprocess
from utils.parse_file import parse_all_file
from utils.generate_dstmdl import generate_dst_mdl
from merge import merge_shp

# encoding: utf-8
import logging
import time
import os

os.environ["PROJ_LIB"] = "./utils/proj"
from pathlib import Path
import traceback
from logging import handlers
import flask
import requests
import requests_mock

from flask import Flask, jsonify, make_response, request
from flask_cors import CORS
from gevent.pywsgi import WSGIServer
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor

from schedule import post_status, post_progress

plt.switch_backend('agg')

thread_pool_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="test_")
app = Flask(__name__)

# 省份名称到行政编码的映射字典
PROVINCE_MAPPING = {
    'guangdong': '440000',  # 广东省
    'guangxi': '450000',  # 广西壮族自治区
    'beijing': '110000',  # 北京市
    'tianjin': '120000',  # 天津市
    'hebei': '130000',  # 河北省
    'shanxi': '140000',  # 山西省
    'neimenggu': '150000',  # 内蒙古自治区
    'liaoning': '210000',  # 辽宁省
    'jilin': '220000',  # 吉林省
    'heilongjiang': '230000',  # 黑龙江省
    'shanghai': '310000',  # 上海市
    'jiangsu': '320000',  # 江苏省
    'zhejiang': '330000',  # 浙江省
    'anhui': '340000',  # 安徽省
    'fujian': '350000',  # 福建省
    'jiangxi': '360000',  # 江西省
    'shandong': '370000',  # 山东省
    'henan': '410000',  # 河南省
    'hubei': '420000',  # 湖北省
    'hunan': '430000',  # 湖南省
    'hainan': '460000',  # 海南省
    'chongqing': '500000',  # 重庆市
    'sichuan': '510000',  # 四川省
    'guizhou': '520000',  # 贵州省
    'yunnan': '530000',  # 云南省
    'xizang': '540000',  # 西藏自治区
    'shaanxi': '610000',  # 陕西省
    'gansu': '620000',  # 甘肃省
    'qinghai': '630000',  # 青海省
    'ningxia': '640000',  # 宁夏回族自治区
    'xinjiang': '650000',  # 新疆维吾尔自治区
    'taiwan': '710000',  # 台湾省
    'xianggang': '810000',  # 香港特别行政区
    'aomen': '820000'  # 澳门特别行政区
}

# 省份编码到名称的反向映射
PROVINCE_CODE_TO_NAME = {v: k for k, v in PROVINCE_MAPPING.items()}


def parse_province_and_year_from_path(file_path):
    """
    从文件路径中解析省份和年份

    Args:
        file_path: 文件路径，如 /cresdashare/data2/pre_image_dev/2024/guangxi/F49D004007.tif

    Returns:
        tuple: (province_code, release_year)
    """
    try:
        path_parts = Path(file_path).parts
        release_year = None
        province_name = None

        # 查找年份（4位数字）
        for part in path_parts:
            if part.isdigit() and len(part) == 4 and 2000 <= int(part) <= 2100:
                release_year = int(part)
                break

        # 查找省份名称
        for part in path_parts:
            part_lower = part.lower()
            if part_lower in PROVINCE_MAPPING:
                province_name = part_lower
                break

        # 转换为行政编码
        province_code = PROVINCE_MAPPING.get(province_name, '440000')  # 默认广东省
        release_year = release_year if release_year else 2024  # 默认2024年

        return province_code, release_year

    except Exception as e:
        logger.error(f"解析路径失败 {file_path}: {str(e)}")
        return '440000', 2024  # 返回默认值


def deduplicate_file_list_simple(file_list, logger):
    """
    简单的文件去重，保留第一个出现的文件

    Args:
        file_list: 要处理的文件列表
        logger: 日志记录器

    Returns:
        list: 去重后的文件列表
    """
    seen_files = {}

    logger.info(f"开始简单去重处理，共有 {len(file_list)} 个文件")

    for file_path in file_list:
        # 提取文件名（不含路径）
        file_name = Path(file_path).name

        # 如果文件名已经存在，保留第一个出现的文件
        if file_name in seen_files:
            existing_path = seen_files[file_name]
            logger.info(f'跳过重复文件: {file_name}')
            logger.info(f'  已存在: {existing_path}')
            logger.info(f'  跳过: {file_path}')
        else:
            seen_files[file_name] = file_path
            logger.info(f'保留文件: {file_name} -> {file_path}')

    # 提取去重后的文件路径列表
    deduplicated_list = list(seen_files.values())

    logger.info(f'去重结果: {len(deduplicated_list)}/{len(file_list)} 个文件')

    return deduplicated_list


def distribute_tasks(file_list, num_processes):
    """
    将文件列表分配给多个进程，采用循环分配策略以平衡负载

    Args:
        file_list: 要处理的文件列表
        num_processes: 进程数量

    Returns:
        list: 每个进程的文件列表
    """
    batches = [[] for _ in range(num_processes)]

    # 循环分配，确保负载均衡
    for i, file_path in enumerate(file_list):
        batch_idx = i % num_processes
        batches[batch_idx].append(file_path)

    return batches


def get_available_cpus(exclude_cpu0=True):
    """
    获取容器内实际可用的CPU核心列表，排除CPU0

    Args:
        exclude_cpu0: 是否排除CPU0，默认True

    Returns:
        list: 可用的CPU核心列表（不包含CPU0）
    """
    try:
        # 方法1: 使用os.sched_getaffinity (Python 3.3+)
        if hasattr(os, 'sched_getaffinity'):
            available_cpus = sorted(list(os.sched_getaffinity(0)))
            logger.info(f"通过sched_getaffinity获取可用CPU: {available_cpus}")
        else:
            # 方法2: 通过taskset命令获取
            try:
                result = subprocess.run(
                    ['taskset', '-p', str(os.getpid())],
                    capture_output=True, text=True, check=True
                )
                output = result.stdout.strip()
                if 'affinity mask:' in output:
                    mask_str = output.split('affinity mask:')[1].strip()
                    mask = int(mask_str, 16)
                    available_cpus = []
                    cpu_index = 0
                    while mask:
                        if mask & 1:
                            available_cpus.append(cpu_index)
                        mask >>= 1
                        cpu_index += 1
                    logger.info(f"通过taskset获取可用CPU: {available_cpus}")
                else:
                    raise ValueError("无法解析taskset输出")
            except (subprocess.CalledProcessError, FileNotFoundError, ValueError):
                # 方法3: 读取cpuset
                cpuset_files = [
                    "/sys/fs/cgroup/cpuset/cpuset.effective_cpus",
                    "/sys/fs/cgroup/cpuset/cpuset.cpus"
                ]

                for cpuset_file in cpuset_files:
                    if os.path.exists(cpuset_file):
                        try:
                            with open(cpuset_file, 'r') as f:
                                cpuset_content = f.read().strip()
                            logger.info(f"通过{cpuset_file}获取CPU信息: {cpuset_content}")

                            # 解析cpuset格式，如 "0-3,8-11"
                            available_cpus = []
                            for part in cpuset_content.split(','):
                                if '-' in part:
                                    start, end = map(int, part.split('-'))
                                    available_cpus.extend(range(start, end + 1))
                                else:
                                    available_cpus.append(int(part))
                            break
                        except Exception as e:
                            logger.warning(f"读取{cpuset_file}失败: {str(e)}")
                else:
                    # 方法4: 最后备选，使用所有CPU
                    import psutil
                    available_cpus = list(range(psutil.cpu_count()))
                    logger.warning(f"使用所有CPU作为备选: {available_cpus}")

        # 排除CPU0
        if exclude_cpu0 and 0 in available_cpus:
            available_cpus.remove(0)
            logger.info(f"已排除CPU0，剩余可用CPU: {available_cpus}")

        # 确保列表不为空
        if not available_cpus:
            logger.warning("可用CPU列表为空，将包含CPU0")
            available_cpus = [0]

        return sorted(available_cpus)

    except Exception as e:
        logger.error(f"获取可用CPU失败: {str(e)}")
        import psutil
        available_cpus = list(range(psutil.cpu_count()))
        if exclude_cpu0 and 0 in available_cpus:
            available_cpus.remove(0)
        return available_cpus


def get_cpu_affinity(process_index, total_processes, available_cpus, job_id, cpus_per_process=20):
    """
    为每个进程分配特定的CPU核心
    基于任务ID的哈希值分配不同的大区

    Args:
        process_index: 进程索引 (0-based)
        total_processes: 总进程数
        available_cpus: 可用的CPU核心列表
        job_id: 任务ID
        cpus_per_process: 每个进程分配的CPU核心数

    Returns:
        list: 分配的CPU核心编号列表
    """
    # 使用任务ID的哈希值决定CPU大区
    task_hash = int(hashlib.md5(job_id.encode()).hexdigest(), 16) % 2  # 0或1

    if task_hash == 0:
        # 任务1使用CPU 1-60
        task_base_cpus = list(range(1, 61))
        logger.info(f"任务 {job_id} 分配到CPU大区: 1-60 (任务1)")
    else:
        # 任务2使用CPU 61-120
        task_base_cpus = list(range(61, 121))
        logger.info(f"任务 {job_id} 分配到CPU大区: 61-120 (任务2)")

    # 为每个进程分配连续的CPU块
    start_idx = process_index * cpus_per_process
    end_idx = start_idx + cpus_per_process

    if end_idx <= len(task_base_cpus):
        allocated_cpus = task_base_cpus[start_idx:end_idx]
    else:
        # 如果超出范围，从头开始循环
        allocated_cpus = task_base_cpus[start_idx % len(task_base_cpus):]
        if len(allocated_cpus) < cpus_per_process:
            allocated_cpus += task_base_cpus[:cpus_per_process - len(allocated_cpus)]

    # 过滤掉不在可用CPU列表中的核心
    allocated_cpus = [cpu for cpu in allocated_cpus if cpu in available_cpus]

    # 确保分配的CPU列表不为空
    if not allocated_cpus:
        logger.warning(f"进程 {process_index} 没有分配到CPU，使用默认CPU 0")
        allocated_cpus = [0]

    logger.info(f"任务 {job_id} 进程 {process_index} 分配CPU: {allocated_cpus}")
    return allocated_cpus


def get_temp_base_dir(process_index, job_id, logger):
    """根据进程索引选择临时目录"""
    if process_index == 0:  # 第一个进程使用/docker
        temp_base = Path("/cresdashare/docker/tmp") / str(job_id)
        logger.info(f"进程 {process_index} 使用 /docker 临时目录: {temp_base}")
    elif process_index == 1:  # 第二个进程使用/jieyi
        temp_base = Path("/cresdashare/jieyi/tmp") / str(job_id)
        logger.info(f"进程 {process_index} 使用 /tmp 临时目录: {temp_base}")
    else:  # 第三个进程使用/
        temp_base = Path("/cresdashare/jieyi/tmp1") / str(job_id)
        logger.info(f"进程 {process_index} 使用 /tmp 临时目录: {temp_base}")

    # 创建目录
    temp_base.mkdir(parents=True, exist_ok=True)
    return temp_base


def run_one_image_change_detection_multi_process(input_path1, input_path2, output_path, callback_url, job_id,
                                                 sheng_code):
    """
    多进程版本的变化检测处理函数
    根据文件数量动态调整进程数：文件数>=3时使用3进程，否则使用2进程

    Args:
        sheng_code: 省份编码，如 '440000'
    """
    try:
        logger.info('接收到url和job_id')
        logger.info(f'用户输入的省份编码: {sheng_code}')

        if not os.path.exists(input_path1) or not os.path.exists(input_path2):
            post_progress(callback_url, post_status['输入文件不存在'], 4001, None, job_id)
            logger.info('输入文件不存在')
            return

        if not output_path:
            post_progress(callback_url, post_status['缺少输出路径'], 400301, None, job_id)
            logger.info('缺少输出路径')
            return

        from get_intersecting_raster_names import get_intersecting_raster_names_and_counties
        
        logger.info('相交区判定')
        
        # 获取input_path2的父目录
        input_dir = Path(input_path2).parent
        # 构建standard文件夹路径
        standard_dir = input_dir / "standard"
        tiff_files = []
        # 检查standard文件夹是否存在
        if not standard_dir.exists():
            logger.info(f"standard文件夹不存在,使用输入影像")
            tiff_files.append(Path(input_path2))
        else:
            # 获取standard文件夹下的所有tiff影像
            # tiff_files = list(standard_dir.glob("*.tif")) + list(standard_dir.glob("*.tiff"))
            for line in open(os.path.join(standard_dir, 'path.txt')):
                path = line.strip()
                if path.endswith((".tif", ".tiff")):
                    tiff_files.append(Path(path))

            if not tiff_files:
                logger.error(f"standard文件夹中没有找到TIFF影像: {standard_dir}")
                post_progress(callback_url, post_status['没有找到TIFF影像'], 4004, None, job_id)
                return

        logger.info(f"找到 {len(tiff_files)} 个TIFF影像")

        # 遍历所有TIFF影像，进行相交判断并汇总结果
        all_file_list = []
        all_county_list = []
        year = os.path.basename(input_path1)
        if year == "2025":
            logger.info("使用2025年分幅")
            for tiff_file in tiff_files:
                try:
                    file_list, county_list = get_intersecting_raster_names_and_counties(
                        "./assets/2025ditu_with_pac1/2025ditu_with_pac.shp", str(tiff_file))
        
                    if file_list:
                        all_file_list.extend(file_list)
                        all_county_list.extend(county_list)
                        logger.info(f"影像 {tiff_file.name} 找到 {len(file_list)} 个相交文件")
                except Exception as e:
                    logger.error(f"处理影像 {tiff_file} 时出错: {str(e)}")
        else:
            logger.info("使用2024年分幅")
            for tiff_file in tiff_files:
                try:
                    file_list, county_list = get_intersecting_raster_names_and_counties(
                        "./assets/2020ditu_with_pac/2020ditu_with_pac.shp", str(tiff_file))
                    
                    if file_list:
                        all_file_list.extend(file_list)
                        all_county_list.extend(county_list)
                        logger.info(f"影像 {tiff_file.name} 找到 {len(file_list)} 个相交文件")
                except Exception as e:
                    logger.error(f"处理影像 {tiff_file} 时出错: {str(e)}")

        # 去重
        file_list = list(set(all_file_list))
        county_list = list(set(all_county_list))

        logger.info(f'总共找到 {len(file_list)} 个相交文件')
        logger.info(f'总共涉及 {len(county_list)} 个县区')
        logger.info(f'相交文件列表: {file_list}')

        if not file_list:
            logger.info('没有找到相交的栅格文件')
            post_progress(callback_url, post_status['没有找到相交区域'], 4002, None, job_id)
            return

        # 修改：使用简单的去重逻辑，保留第一个出现的文件
        original_count = len(file_list)
        file_list = deduplicate_file_list_simple(file_list, logger)
        if len(file_list) < original_count:
            logger.info(f'文件去重完成: 从 {original_count} 个文件去重到 {len(file_list)} 个文件')

        # 验证用户输入的省份编码
        if sheng_code not in PROVINCE_CODE_TO_NAME:
            logger.warning(f"用户输入的省份编码 {sheng_code} 不在映射表中，使用默认值 440000")
            sheng_code = '440000'

        # 获取省份名称用于日志
        province_name = PROVINCE_CODE_TO_NAME.get(sheng_code, '未知省份')
        logger.info(f'使用用户输入的省份编码: {sheng_code} ({province_name})')

        # 获取第一个文件的年份
        if file_list:
            sample_file = file_list[0]
            _, release_year = parse_province_and_year_from_path(sample_file)
            logger.info(f'使用GDB参数: 省份编码={sheng_code}, 年份={release_year}')
        else:
            release_year = 2024
            logger.info(f'使用默认GDB参数: 省份编码={sheng_code}, 年份={release_year}')

        # 根据文件数量动态决定进程数
        file_count = len(file_list)
        if file_count >= 3:
            num_processes = 3
        else:
            num_processes = 2

        logger.info(f'文件数量: {file_count}, 使用 {num_processes} 个进程')

        input_path2_list = [input_path2] * len(file_list)
        output_list = []

        # 分配任务到各个进程
        file_batches = distribute_tasks(file_list, num_processes)
        input2_batches = distribute_tasks(input_path2_list, num_processes)

        # 为每个进程批次创建对应的输出路径
        output_batches = [[] for _ in range(num_processes)]
        for i in range(num_processes):
            temp_base = get_temp_base_dir(i, job_id, logger)
            for pre_file in file_batches[i]:
                output_path_for_file = temp_base / (Path(pre_file).stem + "_1.shp")
                output_batches[i].append(output_path_for_file)
                # 确保目录存在
                output_path_for_file.parent.mkdir(parents=True, exist_ok=True)

        logger.info('=== 任务分配详情 ===')
        for i in range(num_processes):
            logger.info(f'进程 {i} 分配了 {len(file_batches[i])} 个文件:')
            for j, file_path in enumerate(file_batches[i]):
                logger.info(f'  文件 {j + 1}: {Path(file_path).name}')
            logger.info(f'  输出目录: {get_temp_base_dir(i, job_id, logger)}')
            logger.info('')

        # 创建一个平坦的输出列表用于最终合并
        all_output_list = []
        for batch in output_batches:
            all_output_list.extend(batch)

        # 获取可用的CPU核心列表（排除CPU0）
        available_cpus = get_available_cpus(exclude_cpu0=True)
        logger.info(f"容器内可用CPU核心(排除CPU0): {available_cpus} (共{len(available_cpus)}个)")

        # 创建任务数据和临时文件
        task_files = []
        processes = []

        # 配置CPU亲和性参数
        cpus_per_process = 20  # 每个进程分配的CPU数

        # 最后一个进程负责最终合并
        for i in range(num_processes):
            if not file_batches[i]:  # 跳过空批次
                continue

            # 为当前进程分配CPU核心
            cpu_affinity = get_cpu_affinity(i, num_processes, available_cpus, job_id, cpus_per_process)

            task_data = {
                'file_list': file_batches[i],
                'input_path2': input2_batches[i],
                'output_list': [str(p) for p in output_batches[i]],
                'job_id': job_id,
                'callback_url': callback_url,
                'task_id': f'batch{i + 1}',
                'temp_dir_suffix': f'tmp_{job_id}_batch{i + 1}',
                'is_final_merge': (i == num_processes - 1),  # 最后一个进程负责合并
                'all_output_list': [str(p) for p in all_output_list],  # 使用新的平坦列表
                'final_output_path': output_path,
                'county_list': county_list,
                'cpu_affinity': cpu_affinity,  # 添加CPU亲和性配置
                'temp_base_dir': str(get_temp_base_dir(i, job_id, logger)),  # 添加临时目录信息
                'admin_province_code': sheng_code,  # 使用用户输入的省份编码
                'release_year': release_year  # 使用统一的年份
            }

            # 创建临时任务文件
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                json.dump(task_data, f, ensure_ascii=False, indent=2)
                task_files.append(f.name)

        logger.info(f'创建了 {len(task_files)} 个任务文件')

        # 配置环境变量并启动进程
        for i, task_file in enumerate(task_files):
            env = os.environ.copy()
            env['CUDA_VISIBLE_DEVICES'] = '0'  # 如果有多个GPU，可以改为 str(i % gpu_count)
            env['TORCH_MULTIPROCESSING_START_METHOD'] = 'spawn'
            env['LD_LIBRARY_PATH'] = '/root/anaconda3/lib:' + env.get('LD_LIBRARY_PATH', '')
            env['CUDA_LAUNCH_BLOCKING'] = '1'

            # 使用taskset设置CPU亲和性
            cpu_affinity = get_cpu_affinity(i, num_processes, available_cpus, job_id, cpus_per_process)
            cpu_affinity_str = ','.join(map(str, cpu_affinity))

            logger.info(f'进程 {i + 1} 分配CPU核心: {cpu_affinity_str}')

            # 使用taskset命令启动进程，绑定到特定CPU核心
            cmd = [
                'taskset', '-c', cpu_affinity_str,
                sys.executable, 'inference_multi.py', task_file
            ]

            process = subprocess.Popen(cmd, env=env)

            processes.append(process)
            logger.info(f'启动进程 {i + 1}/{len(task_files)}: PID={process.pid}, CPUs={cpu_affinity_str}')

            # 错开启动时间，避免资源竞争
            if i < len(task_files) - 1:
                time.sleep(10)

        logger.info(f'{num_processes} 个进程全部启动成功')
        post_progress(callback_url, post_status['running'], 10, None, job_id)

        # 等待所有进程完成
        try:
            logger.info('等待所有进程执行完成...')

            # 监控进程状态
            start_time = time.time()
            check_interval = 30
            last_check = start_time

            while any(p.poll() is None for p in processes):
                current_time = time.time()
                if current_time - last_check >= check_interval:
                    elapsed_time = int(current_time - start_time)
                    running_count = sum(1 for p in processes if p.poll() is None)
                    logger.info(f'进程运行中... 已运行 {elapsed_time} 秒, 仍有 {running_count} 个进程在运行')

                    # 计算进度（简单估算）
                    completed_count = len(processes) - running_count
                    progress = min(20 + int(completed_count / len(processes) * 60), 70)

                    try:
                        post_progress(callback_url, f"处理中... {completed_count}/{len(processes)} 进程完成",
                                      progress, None, job_id)
                    except Exception as e:
                        logger.error(f'进度更新失败: {str(e)}')

                    last_check = current_time

                time.sleep(10)

            # 检查所有进程的返回码
            failed_processes = []
            for i, process in enumerate(processes):
                returncode = process.returncode
                if returncode != 0:
                    failed_processes.append((i + 1, returncode))
                    logger.error(f'进程 {i + 1} 失败，返回码: {returncode}')
                else:
                    logger.info(f'进程 {i + 1} 成功完成')

            if failed_processes:
                error_msg = f"有 {len(failed_processes)} 个进程执行失败: {failed_processes}"
                logger.error(error_msg)
                # post_progress(callback_url, post_status['error'], -1,
                #              {"errorCode": 4008, "message": error_msg}, job_id)
                return

            logger.info('所有进程执行完成')

        except Exception as e:
            logger.error(f'进程执行异常: {str(e)}')
            # 终止所有进程
            for process in processes:
                try:
                    process.kill()
                except:
                    pass
            post_progress(callback_url, post_status['error'], -1,
                          {"errorCode": 4008, "message": f"进程执行异常: {str(e)}"}, job_id)
            return

        finally:
            # 清理临时文件
            for task_file in task_files:
                try:
                    if os.path.exists(task_file):
                        os.remove(task_file)
                except Exception as cleanup_error:
                    logger.warning(f'清理临时文件失败: {cleanup_error}')

        logger.info('多进程变化检测处理完成')

    except Exception as e:
        logger.error(f'多进程处理出错: {str(e)}')
        logger.error(traceback.format_exc())
        error_message = {"errorCode": 4005, "message": f"多进程任务运行中报错: {str(e)}"}
        post_progress(callback_url, post_status['error'], -1, error_message, job_id)


# 异步运行
@app.route('/cal_Change_Detection/', methods=['POST'])
def run_change_detection():
    data = request.json
    logger.info('***start***')
    input_path1 = data.get("input_path1")
    input_path2 = data.get("input_path2")
    output_path = data.get("output_path")
    callback_url = data.get("callback_url")
    job_id = data.get("job_id")
    sheng_code = data.get("sheng")  # 获取用户输入的省份编码

    print(f'前时相影像路径：{input_path1}')
    print(f'后时相影像路径：{input_path2}')
    print(f'输出结果影像路径：{output_path}')
    print(f'用户输入的省份编码：{sheng_code}')
    #input_path2 = "/cresdashare/docker/tmp/ecomn_2024_13_363.tif"
    #print(f'修改后时相影像路径：{input_path2}')
    # 验证sheng参数
    if not sheng_code:
        logger.warning("未提供省份编码参数'sheng'，使用默认值440000")
        sheng_code = '440000'

    # 检查省份编码是否有效
    if sheng_code not in PROVINCE_CODE_TO_NAME:
        logger.warning(f"省份编码{sheng_code}无效，使用默认值440000")
        sheng_code = '440000'

    try:
        # 使用多进程版本的处理函数，传入sheng_code参数
        thread_pool_executor.submit(run_one_image_change_detection_multi_process,
                                    input_path1, input_path2, output_path, callback_url, job_id, sheng_code)
    except Exception as e:
        print("error", e)
        error_message = {"errorCode": 4005, "message": "任务运行中报错"}
        post_progress(callback_url, post_status['error'], -1, error_message, job_id)
    print("success 200")
    return "success", 200


if __name__ == '__main__':
    # 初始化logger
    log_file = "./server_log"
    logger = logging.getLogger('mylog')
    logger.setLevel(logging.INFO)

    file_handlers = logging.FileHandler(log_file)
    file_handlers.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    fmt1 = '%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s'
    fmt = logging.Formatter(fmt1)

    file_handlers.setFormatter(fmt)
    console_handler.setFormatter(fmt)

    logger.addHandler(file_handlers)
    logger.addHandler(console_handler)

    logger.info('Server starting...')
    CORS(app, supports_credentials=True)

    try:
        WSGIServer(('0.0.0.0', 5003), app).serve_forever()
    except KeyboardInterrupt:
        exit(0)
    except Exception as e:
        print(e)
        exit(1)
