import os
import sys
import json
import yaml
import shutil
import tempfile
import time

try:
    from MessageClient.ProgressMessageSender import ProgressMessageSender
except:
    print('failed to load ProgressMessageSender package.')
    ProgressMessageSender = None
prg_sender = None


class NonFatalTaskWarning(RuntimeError):
    """记录告警但不让当前工作流步骤以失败状态退出。"""


def _is_nonfatal_warning(exc):
    if isinstance(exc, NonFatalTaskWarning):
        return True
    message = str(exc).lower()
    warning_markers = (
        'maximum tiff file size exceeded',
        'tiffappendtostrip',
        'bigtiff',
        '未生成结果文件',
    )
    return any(marker in message for marker in warning_markers)


class ProgressMessageSenderWrap():
    
    def __init__(self, bootstrap_servers='', topic='', taskId=None):
        try:
            self.prg_sender = ProgressMessageSender(bootstrap_servers, topic, taskId)
            if self.prg_sender.is_none():
                self.prg_sender = None
        except:
            self.prg_sender = None

    def get_task_id(self):
        if self.prg_sender is not None:
            return self.prg_sender.get_task_id()

    def set_title(self, title=None, titleId=None):
        if self.prg_sender is not None:
            self.prg_sender.set_title(title, titleId)

    def set_source(self, source=None, rank=None):
        if self.prg_sender is not None:
            self.prg_sender.set_source(source, rank)

    # def send(self, message_dict):
    #     if self.prg_sender is not None:
    #         self.prg_sender.send(message_dict)

    # def calc_progress_value(self, index, total, min_value=0, max_value=100):
    #     if self.prg_sender is not None:
    #         return self.prg_sender.calc_progress_value(index, total, min_value, max_value)
    def send(self, message_dict):
        if self.prg_sender is not None:
            return self.prg_sender.send(message_dict)
        else:
            print(f'[SIMULATED SEND] {message_dict}', flush=True)  # 模拟发送
            return False

    def close(self):
        if self.prg_sender is not None and hasattr(self.prg_sender, 'close'):
            self.prg_sender.close()

    def calc_progress_value(self, index, total, min_value=0, max_value=100):
        if self.prg_sender is not None:
            return self.prg_sender.calc_progress_value(index, total, min_value, max_value)
        if total > 0:
            return min_value + (max_value - min_value) * (index / total)
        return min_value

def init_progress_message_sender(kafka_server_ip_port, kafka_topic, kafka_task_id):
    global prg_sender
    bootstrap_servers = kafka_server_ip_port or os.environ.get('KAFKA_SERVER_IP_PORT', '')
    topic = kafka_topic or os.environ.get('KAFKA_TOPIC', '')
    task_id = kafka_task_id or os.environ.get('KAFKA_TASK_ID')
    prg_sender = ProgressMessageSenderWrap(bootstrap_servers, topic, task_id)

def init_progress_message_title(step_id, step_name):
    title = None
    titleId = None
    work_path = os.getcwd()
    metadata_path = f'metadata.yml'
    try:
        with open(metadata_path, 'r', encoding='utf-8') as file:
            data = yaml.safe_load(file)
        title = data['name']
        titleId = data['id']
    except FileNotFoundError:
        print(f"ERROR: Can not find metadata.yml in {work_path}")
    except Exception as e:
        print(f"ERROR: {e}")
    if step_id is not None:
        titleId = step_id
    if step_name is not None:
        title = step_name
    global prg_sender
    prg_sender.set_title(title, titleId)

def init_progress_message_source(rank=None):
    _source = 'module'
    _rank = -1
    if rank is not None:
        _rank = rank
    global prg_sender
    prg_sender.set_source(_source, _rank)


# ========== Swap 变量输出 ==========

def swap_write(key, value):
    print(f"##SWAP:{key}={json.dumps(value, ensure_ascii=False)}")


_SHAPEFILE_SIDECARS = (
    '.shp', '.shx', '.dbf', '.prj', '.cpg', '.qix', '.sbn', '.sbx', '.fix', '.shp.xml'
)


def _remove_shapefile_dataset(shp_path):
    """删除一个确定路径的 Shapefile 全套文件，避免复用上次运行的旧结果。"""
    stem = os.path.splitext(str(shp_path))[0]
    for extension in _SHAPEFILE_SIDECARS:
        candidate = stem + extension
        if os.path.isfile(candidate):
            os.remove(candidate)


def _remove_file_if_exists(path):
    if os.path.isfile(path):
        os.remove(path)


_LOCAL_SCRATCH_RESERVE_BYTES = 10 * 1024 ** 3


def _copy_file_atomically(source_path, destination_path):
    """先复制到目标目录的临时文件，再原子替换，避免下游读到半个结果。"""
    destination_path = str(destination_path)
    destination_dir = os.path.dirname(destination_path) or '.'
    os.makedirs(destination_dir, exist_ok=True)
    file_descriptor, temporary_path = tempfile.mkstemp(
        prefix=f'.{os.path.basename(destination_path)}.',
        suffix='.copying',
        dir=destination_dir,
    )
    os.close(file_descriptor)
    try:
        shutil.copy2(str(source_path), temporary_path)
        os.replace(temporary_path, destination_path)
    finally:
        _remove_file_if_exists(temporary_path)


def _copy_shapefile_dataset(source_shp, destination_shp):
    """复制 Shapefile 及全部存在的附属文件。"""
    source_stem = os.path.splitext(str(source_shp))[0]
    destination_stem = os.path.splitext(str(destination_shp))[0]
    if not os.path.isfile(source_stem + '.shp'):
        raise FileNotFoundError(f'本地变化检测 SHP 不存在: {source_stem}.shp')

    _remove_shapefile_dataset(destination_shp)
    try:
        for extension in _SHAPEFILE_SIDECARS:
            source_file = source_stem + extension
            if os.path.isfile(source_file):
                _copy_file_atomically(source_file, destination_stem + extension)
    except Exception:
        _remove_shapefile_dataset(destination_shp)
        raise


def _create_local_pair_scratch(pre_path, post_path):
    """为单对影像创建本地临时目录，并预留输入副本、临时结果和根盘安全空间。"""
    scratch_root = os.environ.get('CHANGE_DETECTION_SCRATCH_DIR', '/tmp')
    os.makedirs(scratch_root, exist_ok=True)
    input_bytes = os.path.getsize(pre_path) + os.path.getsize(post_path)
    # 除两份输入副本外，再按输入总大小预留重采样/结果空间，并保留至少 10 GiB。
    required_bytes = input_bytes * 2 + _LOCAL_SCRATCH_RESERVE_BYTES
    free_bytes = shutil.disk_usage(scratch_root).free
    if free_bytes < required_bytes:
        raise OSError(
            f'本地临时盘空间不足: 可用 {free_bytes / 1024 ** 3:.1f} GiB，'
            f'预计至少需要 {required_bytes / 1024 ** 3:.1f} GiB'
        )
    return tempfile.mkdtemp(prefix='change_detection_', dir=scratch_root)


def _format_file_size(size_bytes):
    size = float(max(0, size_bytes))
    for unit in ('B', 'KB', 'MB', 'GB', 'TB'):
        if size < 1024 or unit == 'TB':
            return f'{size:.1f}{unit}'
        size /= 1024


def _format_duration(seconds):
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f'{hours}小时{minutes}分'
    if minutes:
        return f'{minutes}分{secs}秒'
    return f'{secs}秒'


# ========== DatasetBuilder ==========

from DatasetBuilder import DatasetBuilder

# ========== 算法主体 ==========

def change_detection(pre_image, post_image, model_path, dst_path, output_dataset):
    global prg_sender

    # 1. 报告启动
    prg_sender.send({'progress': 0, 'runningStatus': 'running', 'runningInfo': '变化检测算法启动'})

    # 2. 检查输入
    if not os.path.exists(pre_image):
        raise FileNotFoundError(f'前时相影像不存在: {pre_image}')
    if not os.path.exists(post_image):
        raise FileNotFoundError(f'后时相影像不存在: {post_image}')

    # 3. 创建输出目录
    os.makedirs(dst_path, exist_ok=True)

    # 4. 确定输出文件名
    from pathlib import Path
    pre_stem = Path(pre_image).stem
    post_stem = Path(post_image).stem
    output_shp = os.path.join(dst_path, f'{pre_stem}_{post_stem}_change.shp')
    _remove_shapefile_dataset(output_shp)

    # 5. 调用核心变化检测推理
    prg_sender.send({'progress': 10, 'runningStatus': 'running', 'runningInfo': '加载模型并执行变化检测推理'})

    from test_lib_batch_memeff_single_image_nomp import test_lib_big_memeff
    import logging
    logger = logging.getLogger('change_detection')
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s: %(message)s'))
        logger.addHandler(handler)

    def _cd_progress(current, total):
        pct = prg_sender.calc_progress_value(current, total, 15, 90)
        prg_sender.send({
            'progress': pct,
            'runningStatus': 'running',
            'runningInfo': f'变化检测推理中 ({current}/{total} patches)'
        })

    test_lib_big_memeff(
        pre_img_path=pre_image,
        post_img_path=post_image,
        output_path=output_shp,
        logger=logger,
        callback_url=None,
        job_id=None,
        temp_dir_suffix="tmp",
        progress_callback=_cd_progress
    )

    # 6. 检查结果是否生成。缺少产物记为告警，由入口返回 completed，避免中断工作流。
    if not os.path.exists(output_shp):
        swap_write('output_shp', output_shp)
        raise NonFatalTaskWarning(f'变化检测未生成结果文件: {output_shp}')

    # 7. 输出 Swap 变量（输出文件路径供下游使用）
    swap_write('output_shp', output_shp)

    # 7. 报告进度
    prg_sender.send({'progress': 99, 'runningStatus': 'running', 'runningInfo': '创建输出数据集'})

    # 8. 输出 Dataset
    if output_dataset is not None:
        result_files = [Path(output_shp).name]
        ds = DatasetBuilder(output_dataset)
        ds.add("result", dst_path, "vector", result_files)
        ds.set_render(["result"])
        ds.save()

    # 9. 报告完成
    prg_sender.send({'progress': 100, 'runningStatus': 'completed', 'runningInfo': '变化检测算法完成'})

# ========== 批量变化检测（文件夹模式）==========

def change_detection_folder(pre_folder, post_folder, model_path, dst_path, output_dataset):
    """批量变化检测：遍历前时相文件夹中所有影像文件，在后时相文件夹中匹配同名文件进行变化检测"""
    global prg_sender

    from pathlib import Path
    import logging

    pre_folder = Path(pre_folder)
    post_folder = Path(post_folder)

    # 1. 报告启动
    prg_sender.send({'progress': 0, 'runningStatus': 'running', 'runningInfo': '批量变化检测算法启动'})

    # 2. 检查输入
    if not pre_folder.exists():
        raise FileNotFoundError(f'前时相文件夹不存在: {pre_folder}')
    if not post_folder.exists():
        raise FileNotFoundError(f'后时相文件夹不存在: {post_folder}')

    # 3. 获取前时相文件夹中所有影像文件
    pre_files = sorted(list(pre_folder.glob("*.tif")) + list(pre_folder.glob("*.tiff")))

    if not pre_files:
        raise RuntimeError(f'前时相文件夹中无影像文件: {pre_folder}')

    # 4. 匹配前后时相文件对
    valid_pairs = []
    skipped_unmatched = []
    for pre_path in pre_files:
        stem = pre_path.stem
        post_candidates = list(post_folder.glob(f"{stem}.tif")) + list(post_folder.glob(f"{stem}.tiff"))
        if post_candidates:
            valid_pairs.append((pre_path, post_candidates[0]))
        else:
            skipped_unmatched.append(stem)

    total = len(valid_pairs)
    if total == 0:
        raise RuntimeError('未找到任何匹配的前后时相影像对')

    swap_write('total_pairs', total)
    if skipped_unmatched:
        swap_write('skipped_unmatched', skipped_unmatched)

    # 5. 创建输出目录
    os.makedirs(dst_path, exist_ok=True)
    shp_dir = os.path.join(dst_path, "shp")
    tif_dir = os.path.join(dst_path, "tif")
    os.makedirs(shp_dir, exist_ok=True)
    os.makedirs(tif_dir, exist_ok=True)

    # 6. 设置日志
    logger = logging.getLogger('change_detection_batch')
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s: %(message)s'))
        logger.addHandler(handler)

    from test_lib_batch_memeff_single_image_nomp import test_lib_big_memeff

    output_shp_list = []
    failed_list = []

    for idx, (pre_path, post_path) in enumerate(valid_pairs):
        stem = pre_path.stem
        output_shp = os.path.join(shp_dir, f'{stem}.shp')
        pair_started_at = time.monotonic()

        progress_pct = prg_sender.calc_progress_value(idx, total, 5, 95)
        prg_sender.send({
            'progress': progress_pct,
            'runningStatus': 'running',
            'runningInfo': f'变化检测中 ({idx+1}/{total}): {stem}'
        })

        logger.info(f'Processing ({idx+1}/{total}): pre={pre_path}, post={post_path} -> {output_shp}')

        def _make_batch_cb(_stem, _idx, _tif_path, _pair_started_at, _inference_started_at):
            def _cb(current, total_patches):
                inner_pct = current / max(total_patches, 1)
                pct = prg_sender.calc_progress_value(_idx + inner_pct, total, 5, 95)
                elapsed_pair = max(time.monotonic() - _pair_started_at, 0.001)
                inference_elapsed = max(time.monotonic() - _inference_started_at, 0.001)
                pair_eta = inference_elapsed / max(current, 1) * max(total_patches - current, 0)
                estimated_pair_total = elapsed_pair + pair_eta
                job_eta = pair_eta + estimated_pair_total * max(total - _idx - 1, 0)

                tif_size = os.path.getsize(_tif_path) if os.path.isfile(_tif_path) else 0
                estimated_tif_size = int(tif_size / max(current, 1) * total_patches)
                running_info = (
                    f'变化检测 ({_idx+1}/{total}) {_stem}: '
                    f'{current}/{total_patches}切片，TIFF已写{_format_file_size(tif_size)}'
                    f'（预计{_format_file_size(estimated_tif_size)}），'
                    f'本图剩余约{_format_duration(pair_eta)}，任务剩余约{_format_duration(job_eta)}'
                )
                logger.info(running_info)
                prg_sender.send({
                    'progress': pct,
                    'runningStatus': 'running',
                    'runningInfo': running_info
                })
            return _cb

        pair_scratch_dir = None
        try:
            shared_tif_src = os.path.join(shp_dir, f'{stem}.tif')
            tif_dst = os.path.join(tif_dir, f'{stem}.tif')
            _remove_shapefile_dataset(output_shp)
            _remove_file_if_exists(shared_tif_src)
            _remove_file_if_exists(tif_dst)

            run_pre_path = str(pre_path)
            run_post_path = str(post_path)
            run_output_shp = output_shp
            try:
                pair_scratch_dir = _create_local_pair_scratch(pre_path, post_path)
                local_pre_dir = os.path.join(pair_scratch_dir, 'input', 'pre')
                local_post_dir = os.path.join(pair_scratch_dir, 'input', 'post')
                local_output_dir = os.path.join(pair_scratch_dir, 'output')
                os.makedirs(local_pre_dir, exist_ok=True)
                os.makedirs(local_post_dir, exist_ok=True)
                os.makedirs(local_output_dir, exist_ok=True)
                run_pre_path = os.path.join(local_pre_dir, pre_path.name)
                run_post_path = os.path.join(local_post_dir, post_path.name)
                run_output_shp = os.path.join(local_output_dir, f'{stem}.shp')

                prg_sender.send({
                    'progress': progress_pct,
                    'runningStatus': 'running',
                    'runningInfo': f'正在将第 {idx+1}/{total} 对影像复制到本地临时盘'
                })
                logger.info('复制前时相影像到本地: %s -> %s', pre_path, run_pre_path)
                shutil.copy2(str(pre_path), run_pre_path)
                logger.info('复制后时相影像到本地: %s -> %s', post_path, run_post_path)
                shutil.copy2(str(post_path), run_post_path)
                logger.info('本地暂存完成，工作目录: %s', pair_scratch_dir)
            except Exception as staging_error:
                if pair_scratch_dir is not None:
                    shutil.rmtree(pair_scratch_dir, ignore_errors=True)
                pair_scratch_dir = None
                run_pre_path = str(pre_path)
                run_post_path = str(post_path)
                run_output_shp = output_shp
                logger.warning('本地暂存不可用，回退到共享盘直接处理: %s', staging_error)
                prg_sender.send({
                    'progress': progress_pct,
                    'runningStatus': 'running',
                    'runningInfo': f'本地暂存不可用，第 {idx+1}/{total} 对影像改用共享盘处理'
                })

            active_tif_path = os.path.join(os.path.dirname(run_output_shp), f'{stem}.tif')
            inference_started_at = time.monotonic()
            test_lib_big_memeff(
                pre_img_path=run_pre_path,
                post_img_path=run_post_path,
                output_path=run_output_shp,
                logger=logger,
                callback_url=None,
                job_id=None,
                temp_dir_suffix="tmp",
                progress_callback=_make_batch_cb(
                    stem,
                    idx,
                    active_tif_path,
                    pair_started_at,
                    inference_started_at,
                )
            )
            if not os.path.exists(run_output_shp):
                raise NonFatalTaskWarning(f'变化检测未生成结果文件: {run_output_shp}')

            if pair_scratch_dir is not None:
                prg_sender.send({
                    'progress': prg_sender.calc_progress_value(idx + 1, total, 5, 95),
                    'runningStatus': 'running',
                    'runningInfo': f'第 {idx+1}/{total} 对推理完成，正在复制结果回共享盘'
                })
                if os.path.exists(active_tif_path):
                    _copy_file_atomically(active_tif_path, tif_dst)
                _copy_shapefile_dataset(run_output_shp, output_shp)
            elif os.path.exists(shared_tif_src):
                shutil.move(shared_tif_src, tif_dst)

            if not os.path.exists(output_shp):
                raise NonFatalTaskWarning(f'变化检测结果复制后缺失: {output_shp}')
            output_shp_list.append(output_shp)
        except Exception as e:
            logger.warning(f'处理 {stem} 出现告警，已跳过: {e}')
            failed_list.append({'file': stem, 'error': str(e)})
        finally:
            if pair_scratch_dir is not None:
                shutil.rmtree(pair_scratch_dir, ignore_errors=True)
                logger.info('已清理本地临时目录: %s', pair_scratch_dir)

    # 7. 先判断是否存在有效结果，避免全失败任务先显示为 95%。
    swap_write('processed_count', len(output_shp_list))
    if failed_list:
        swap_write('warning_list', failed_list)
    if not output_shp_list:
        raise RuntimeError(f'批量变化检测全部失败，0/{total} 对影像生成结果')

    # 8. 部分失败可以继续；保留每个有效的变化检测结果，不执行合并。
    prg_sender.send({
        'progress': 95,
        'runningStatus': 'running',
        'runningInfo': f'变化检测推理完成，正在整理 {len(output_shp_list)} 个结果'
    })
    swap_write('output_shp', shp_dir)    # 下游分类通过 glob 按 stem 匹配单个 SHP
    swap_write('output_shp_list', output_shp_list)
    swap_write('output_tif', tif_dir)

    # 9. 输出 Dataset
    prg_sender.send({'progress': 97, 'runningStatus': 'running', 'runningInfo': '正在生成变化检测输出数据集'})

    if output_dataset is not None:
        result_files = sorted(p.name for p in Path(shp_dir).glob("*.shp") if p.is_file())
        ds = DatasetBuilder(output_dataset)
        ds.add("result", shp_dir, "vector", result_files)
        ds.set_render(["result"])
        ds.save()

    prg_sender.send({'progress': 99, 'runningStatus': 'running', 'runningInfo': '变化检测结果已生成，等待步骤完成'})

    # 10. 单个影像失败按告警处理，不中断整个批量工作流。
    result_msg = f'批量变化检测完成，成功处理 {len(output_shp_list)}/{total} 对影像'
    if failed_list:
        result_msg += f'，{len(failed_list)} 对出现告警并已跳过'
    prg_sender.send({'progress': 100, 'runningStatus': 'completed', 'runningInfo': result_msg})


# ========== 入口函数 ==========

def entry(pre_image, post_image, model_path, dst_path, output_dataset, step_id, step_name,
          kafka_server_ip_port, kafka_topic, kafka_task_id):
    init_progress_message_sender(kafka_server_ip_port, kafka_topic, kafka_task_id)
    init_progress_message_title(step_id, step_name)
    init_progress_message_source()
    prg_sender.send({
        'progress': 0,
        'runningStatus': 'running',
        'runningInfo': '变化检测任务已接收，正在初始化'
    })

    try:
        # 自动检测：如果输入为文件夹（目录），则进入批量处理模式
        if os.path.isdir(pre_image) and os.path.isdir(post_image):
            change_detection_folder(pre_image, post_image, model_path, dst_path, output_dataset)
        else:
            change_detection(pre_image, post_image, model_path, dst_path, output_dataset)
    except Exception as exc:
        if _is_nonfatal_warning(exc):
            warning_info = f'变化检测完成（存在告警）：{exc}'
            swap_write('warning', str(exc))
            prg_sender.send({
                'progress': 100,
                'runningStatus': 'completed',
                'runningInfo': warning_info
            })
            return
        prg_sender.send({
            'progress': 100,
            'runningStatus': 'failed',
            'runningInfo': f'变化检测失败: {exc}'
        })
        raise
    finally:
        prg_sender.close()
