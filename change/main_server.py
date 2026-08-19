import os

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
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor

from schedule import post_status, post_progress

plt.switch_backend('agg')

thread_pool_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="test_")


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


def get_cpu_affinity(process_index, total_processes, available_cpus, cpus_per_process=10):
    """
    为每个进程分配特定的CPU核心
    基于实际可用的CPU核心列表进行分配

    Args:
        process_index: 进程索引 (0-based)
        total_processes: 总进程数
        available_cpus: 可用的CPU核心列表
        cpus_per_process: 每个进程分配的CPU核心数

    Returns:
        list: 分配的CPU核心编号列表
    """
    total_available = len(available_cpus)

    # 计算每个进程应该分配的核心数
    if total_processes * cpus_per_process > total_available:
        cpus_per_process = max(1, total_available // total_processes)
        logger.warning(f"CPU核心不足，调整为每个进程 {cpus_per_process} 个核心")

    # 计算起始索引（在可用CPU列表中的索引）
    start_idx = (process_index * cpus_per_process) % total_available
    end_idx = start_idx + cpus_per_process

    if end_idx > total_available:
        # 如果超出范围，从头开始循环分配
        wrapped_cpus = available_cpus[start_idx:] + available_cpus[:end_idx - total_available]
    else:
        wrapped_cpus = available_cpus[start_idx:end_idx]

    logger.info(f"进程 {process_index} 分配CPU: {wrapped_cpus}")
    return wrapped_cpus


def run_one_image_change_detection_multi_process(input_path1, input_path2, output_path, callback_url, job_id):
    """
    多进程版本的变化检测处理函数
    根据文件数量动态调整进程数：文件数>=8时使用8进程，否则使用2进程
    """
    try:
        logger.info('开始处理变化检测任务')
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
        file_list, county_list = get_intersecting_raster_names_and_counties(
            "./assets/2020ditu_with_pac/2020ditu_with_pac.shp", input_path2)
        logger.info(f'找到 {len(file_list)} 个相交文件')
        logger.info(file_list)

        if not file_list:
            logger.info('没有找到相交的栅格文件')
            post_progress(callback_url, post_status['没有找到相交区域'], 4002, None, job_id)
            return

        # 根据文件数量动态决定进程数
        file_count = len(file_list)
        if file_count >= 4:
            num_processes = 4
        else:
            num_processes = 2

        logger.info(f'文件数量: {file_count}, 使用 {num_processes} 个进程')

        input_path2_list = [input_path2] * len(file_list)
        output_list = []

        # 创建输出路径
        for pre_file in file_list:
            output_list.append(
                Path("/docker/tmp/kty/" + str(job_id) + '/') /
                (Path(pre_file).stem + "_1.shp")
            )

        # 创建输出目录
        for output_file in output_list:
            output_file.parent.mkdir(parents=True, exist_ok=True)

        # 分配任务到各个进程
        file_batches = distribute_tasks(file_list, num_processes)
        input2_batches = distribute_tasks(input_path2_list, num_processes)
        output_batches = distribute_tasks(output_list, num_processes)

        # 获取可用的CPU核心列表（排除CPU0）
        available_cpus = get_available_cpus(exclude_cpu0=True)
        logger.info(f"容器内可用CPU核心(排除CPU0): {available_cpus} (共{len(available_cpus)}个)")

        # 创建任务数据和临时文件
        task_files = []
        processes = []

        # 配置CPU亲和性参数
        cpus_per_process = 15  # 每个进程分配的CPU数

        # 最后一个进程负责最终合并
        for i in range(num_processes):
            if not file_batches[i]:  # 跳过空批次
                continue

            # 为当前进程分配CPU核心
            cpu_affinity = get_cpu_affinity(i, num_processes, available_cpus, cpus_per_process)

            task_data = {
                'file_list': file_batches[i],
                'input_path2': input2_batches[i],
                'output_list': [str(p) for p in output_batches[i]],
                'job_id': job_id,
                'callback_url': callback_url,
                'task_id': f'batch{i + 1}',
                'temp_dir_suffix': f'tmp_{job_id}_batch{i + 1}',
                'is_final_merge': (i == num_processes - 1),  # 最后一个进程负责合并
                'all_output_list': [str(p) for p in output_list],
                'final_output_path': output_path,
                'county_list': county_list,
                'cpu_affinity': cpu_affinity  # 添加CPU亲和性配置
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
            cpu_affinity = get_cpu_affinity(i, num_processes, available_cpus, cpus_per_process)
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
                time.sleep(2)

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
                post_progress(callback_url, post_status['error'], -1,
                              {"errorCode": 4008, "message": error_msg}, job_id)
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
        #error_message = {"errorCode": 4005, "message": f"多进程任务运行中报错: {str(e)}"}
        #post_progress(callback_url, post_status['error'], -1, error_message, job_id)


def run_change_detection():
    """直接运行变化检测任务"""
    try:
        logger.info('***开始变化检测任务***')

        # 直接赋值参数
        input_path1 = "/cresdashare/data2/pre_image_dev/2024/guangdong"
        input_path2 = "/cresdashare/data2/inner_data/27/237/ecomn_2024_27_237.tif"
        output_path = "./output/test1017/CDfinally.shp"
        callback_url = None  # 没有回调URL
        job_id = "direct_run_job_002"  # 固定作业ID

        logger.info(f'前时相影像路径：{input_path1}')
        logger.info(f'后时相影像路径：{input_path2}')
        logger.info(f'输出结果影像路径：{output_path}')

        # 使用多进程版本的处理函数
        run_one_image_change_detection_multi_process(
            input_path1, input_path2, output_path, callback_url, job_id
        )

        logger.info("变化检测任务完成")
        return "success"

    except Exception as e:
        logger.error(f"变化检测任务错误: {str(e)}")
        return f"error: {str(e)}"


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

    # 直接运行变化检测任务
    result = run_change_detection()
    logger.info(f"任务执行结果: {result}")
