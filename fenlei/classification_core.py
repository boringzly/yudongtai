import os
import sys
import json
import yaml
import tempfile
import shutil
import subprocess
import logging
import traceback
from pathlib import Path

# ========== Kafka 客户端 ==========
try:
    from MessageClient.ProgressMessageSender import ProgressMessageSender
except:
    print('failed to load ProgressMessageSender package.')
    ProgressMessageSender = None
prg_sender = None


def _ensure_log_dir(dst_path):
    resolved_dst = Path(dst_path).resolve()
    task_root = resolved_dst
    for candidate in (resolved_dst, *resolved_dst.parents):
        if candidate.name.lower() == 'working':
            task_root = candidate.parent
            break
    log_dir = task_root / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    return str(log_dir)


def _configure_persistent_logger(name, dst_path, filename='classification.log'):
    """同时写 stdout 和任务输出目录，Pod 退出后仍可追溯。"""
    log_path = os.path.join(_ensure_log_dir(dst_path), filename)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s: %(message)s')

    if not any(type(handler) is logging.StreamHandler for handler in logger.handlers):
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    target_path = os.path.normcase(os.path.abspath(log_path))
    for handler in list(logger.handlers):
        if isinstance(handler, logging.FileHandler):
            handler_path = os.path.normcase(os.path.abspath(handler.baseFilename))
            if handler_path == target_path:
                break
            logger.removeHandler(handler)
            handler.close()
    else:
        file_handler = logging.FileHandler(log_path, mode='a', encoding='utf-8')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    return logger, log_path


def _write_diagnostic_report(dst_path, filename, payload):
    try:
        report_path = os.path.join(_ensure_log_dir(dst_path), filename)
        temp_path = report_path + '.tmp'
        with open(temp_path, 'w', encoding='utf-8') as report_file:
            json.dump(payload, report_file, ensure_ascii=False, indent=2)
        os.replace(temp_path, report_path)
        return report_path
    except Exception as report_error:
        print(f'[WARNING] 保存诊断报告失败 {filename}: {report_error}', file=sys.stderr, flush=True)
        return None


def _run_subprocess_logged(command, cwd, env, logger):
    """实时转发 CLIE 输出，并把 stdout/stderr 完整写入持久化日志。"""
    logger.info('启动分类子进程: %s', ' '.join(str(part) for part in command))
    process = subprocess.Popen(
        command,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding='utf-8',
        errors='replace',
        bufsize=1,
    )
    try:
        for output_line in process.stdout:
            logger.info('[CLIE] %s', output_line.rstrip('\r\n'))
    finally:
        if process.stdout is not None:
            process.stdout.close()
    return_code = process.wait()
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, command)
    return return_code


def _resolve_classification_model(model_path, work_dir):
    """优先使用工作流传入模型；兼容旧版默认路径并记录实际使用文件。"""
    candidates = []
    if model_path:
        requested_model = Path(model_path)
        candidates.append(requested_model)
        if not requested_model.is_absolute():
            candidates.append(Path(work_dir) / requested_model)
            candidates.append(Path(work_dir) / 'tools' / requested_model.name)
    candidates.append(Path(work_dir) / 'tools' / 'fullclass_chinaall_3b.pth')
    for candidate in candidates:
        if candidate.is_file() or candidate.is_dir():
            return str(candidate.resolve())
    searched_paths = ', '.join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f'分类模型不存在，已检查: {searched_paths}')


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
        '分类结果缺失',
        '变化检测结果文件不存在',
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
            print(f'[SIMULATED SEND] {message_dict}')  # 模拟发送
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
    if bootstrap_servers:
        os.environ['KAFKA_SERVER_IP_PORT'] = str(bootstrap_servers)
    if topic:
        os.environ['KAFKA_TOPIC'] = str(topic)
    if task_id:
        os.environ['KAFKA_TASK_ID'] = str(task_id)

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


def _send_completed(running_info):
    global prg_sender
    message = {'progress': 100, 'runningStatus': 'completed', 'runningInfo': running_info}
    print(f'[PROGRESS] sending completed message: {running_info}', flush=True)
    sent = prg_sender.send(message)
    print(f'[PROGRESS] completed message sent={sent}', flush=True)
    prg_sender.close()


def _send_failed(running_info):
    global prg_sender
    message = {'progress': 100, 'runningStatus': 'failed', 'runningInfo': running_info}
    print(f'[PROGRESS] sending failed message: {running_info}', flush=True)
    sent = prg_sender.send(message)
    print(f'[PROGRESS] failed message sent={sent}', flush=True)


# ========== Swap 变量输出 ==========

def swap_write(key, value):
    print(f"##SWAP:{key}={json.dumps(value, ensure_ascii=False)}", flush=True)


_SHAPEFILE_SIDECARS = (
    '.shp', '.shx', '.dbf', '.prj', '.cpg', '.qix', '.sbn', '.sbx', '.fix', '.shp.xml'
)


def _remove_shapefile_dataset(shp_path):
    stem = os.path.splitext(str(shp_path))[0]
    for extension in _SHAPEFILE_SIDECARS:
        candidate = stem + extension
        if os.path.isfile(candidate):
            os.remove(candidate)


def _remove_file_if_exists(path):
    if os.path.isfile(path):
        os.remove(path)


_ACTIVE_CLASSIFICATION_TEMP_DIRS = set()


def _create_classification_temp_dir():
    temp_dir = tempfile.mkdtemp(prefix='clie_tmp_')
    _ACTIVE_CLASSIFICATION_TEMP_DIRS.add(temp_dir)
    return temp_dir


def _cleanup_classification_temp_dir(temp_dir):
    if not temp_dir:
        return
    shutil.rmtree(temp_dir, ignore_errors=True)
    _ACTIVE_CLASSIFICATION_TEMP_DIRS.discard(temp_dir)


def _cleanup_all_classification_temp_dirs():
    for temp_dir in list(_ACTIVE_CLASSIFICATION_TEMP_DIRS):
        _cleanup_classification_temp_dir(temp_dir)


def _cleanup_clie_working_rasters(temp_dir):
    """及时清理 CLIE 重采样产生的临时 BigTIFF，避免批量运行时持续累积。"""
    shutil.rmtree(os.path.join(temp_dir, 'tmp'), ignore_errors=True)


def _write_empty_classification_shp(output_shp, crs):
    """创建带稳定字段的 UTF-8 空结果，确保无变化时仍有可用输出。"""
    from osgeo import ogr, osr

    os.makedirs(os.path.dirname(output_shp), exist_ok=True)
    _remove_shapefile_dataset(output_shp)
    driver = ogr.GetDriverByName('ESRI Shapefile')
    if driver is None:
        raise RuntimeError('OGR ESRI Shapefile 驱动不可用')
    data_source = driver.CreateDataSource(output_shp)
    if data_source is None:
        raise RuntimeError(f'无法创建空分类结果: {output_shp}')

    spatial_ref = None
    if crs is not None:
        spatial_ref = osr.SpatialReference()
        crs_wkt = crs.to_wkt() if hasattr(crs, 'to_wkt') else str(crs)
        if spatial_ref.SetFromUserInput(crs_wkt) != 0:
            data_source = None
            raise RuntimeError(f'无法解析空分类结果坐标系: {crs}')

    layer = data_source.CreateLayer(
        os.path.splitext(os.path.basename(output_shp))[0],
        srs=spatial_ref,
        geom_type=ogr.wkbPolygon,
        options=['ENCODING=UTF-8'],
    )
    if layer is None:
        data_source = None
        raise RuntimeError(f'无法创建空分类结果图层: {output_shp}')

    field_specs = (
        ('uid', ogr.OFTInteger64, None),
        ('pre_code', ogr.OFTInteger, None),
        ('pre_name', ogr.OFTString, 32),
        ('curr_code', ogr.OFTInteger, None),
        ('curr_name', ogr.OFTString, 32),
    )
    for name, field_type, width in field_specs:
        field = ogr.FieldDefn(name, field_type)
        if width is not None:
            field.SetWidth(width)
        if layer.CreateField(field) != 0:
            data_source = None
            raise RuntimeError(f'无法创建空分类结果字段 {name}: {output_shp}')
    data_source = None


# ========== DatasetBuilder ==========

from DatasetBuilder import DatasetBuilder
from classification_schema import format_classification_result

# ========== 算法主体 ==========

def _map_class_code(original_code):
    label_mapping = {
        0: 0, 1: 1, 2: 1, 3: 2, 4: 2,
        5: 3, 11: 4,
        6: 5, 7: 5, 8: 5, 9: 5,
        10: 6, 12: 6, 13: 6
    }
    return label_mapping.get(original_code, 0)


def _process_chunk(args):
    import rasterio
    from rasterstats import zonal_stats

    gdf_chunk, tif_file = args
    with rasterio.open(tif_file) as src:
        raster_data = src.read(1)
        affine = src.transform
        stats = zonal_stats(gdf_chunk, raster_data, affine=affine, categorical=True, nodata=0)
    major_classes = []
    for stat in stats:
        if stat:
            original_major_class = max(stat, key=stat.get)
            mapped_class = _map_class_code(original_major_class)
            major_classes.append(mapped_class)
        else:
            major_classes.append(0)
    return list(gdf_chunk.index), major_classes


def classification(pre_image, post_image, mask_shp, model_path, dst_path, output_dataset):
    global prg_sender

    # 1. 报告启动
    prg_sender.send({'progress': 0, 'runningStatus': 'running', 'runningInfo': '类别识别算法启动'})

    # 2. 引入依赖
    from pathlib import Path
    import rasterio
    import geopandas as gpd
    import numpy as np
    logger, log_path = _configure_persistent_logger('classification', dst_path)
    diagnostic_dir = _ensure_log_dir(dst_path)
    logger.info('持久化日志: %s', log_path)

    # 3. 读取变化检测结果
    prg_sender.send({'progress': 5, 'runningStatus': 'running', 'runningInfo': '读取变化检测结果SHP'})

    if not os.path.exists(pre_image):
        raise FileNotFoundError(f'前时相影像不存在: {pre_image}')
    if not os.path.exists(post_image):
        raise FileNotFoundError(f'后时相影像不存在: {post_image}')
    if not os.path.exists(mask_shp):
        raise NonFatalTaskWarning(f'变化检测结果文件不存在: {mask_shp}')

    gdf = gpd.read_file(mask_shp)
    logger.info(f'变化地块数量：{len(gdf)}')

    pre_stem = Path(pre_image).stem
    post_stem = Path(post_image).stem
    out_shp_file = os.path.join(dst_path, f'{pre_stem}_{post_stem}_classified.shp')

    # 4. 如果无变化区域，创建字段完整的空输出后返回。
    if len(gdf) == 0:
        logger.info("mask为空，没有变化区域可分类；空 SHP 是有效的无变化结果")
        _write_empty_classification_shp(out_shp_file, gdf.crs)
        swap_write('output_shp', out_shp_file)
        swap_write('classified_count', 0)
        if output_dataset is not None:
            ds = DatasetBuilder(output_dataset)
            ds.add("result", dst_path, "vector", [Path(out_shp_file).name])
            ds.set_render(["result"])
            ds.save()
        _write_diagnostic_report(dst_path, 'classification_summary.json', {
            'status': 'completed',
            'mode': 'single',
            'mask_shp': mask_shp,
            'output_shp': out_shp_file,
            'feature_count': 0,
            'empty_result': True,
            'reason': '变化检测结果不包含变化图斑',
        })
        _send_completed('无变化区域可分类')
        return

    gdf["uid"] = range(len(gdf))
    origin_crs = gdf.crs
    gdf["pre_code"] = 0
    gdf["curr_code"] = 0

    # 5. 准备 clie_new 推理环境（容器内 /app/module 为只读，所有输出重定向到可写目录）
    work_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "clie_new")
    run_py = os.path.join(work_dir, "run.py")
    model_file = _resolve_classification_model(model_path, work_dir)
    logger.info('分类模型: %s', model_file)

    tmprun_dir = _create_classification_temp_dir()
    logger.info(f"分类临时目录: {tmprun_dir}")

    clie_env = os.environ.copy()
    clie_env['PYTHONPATH'] = work_dir + os.pathsep + clie_env.get('PYTHONPATH', '')
    clie_env['CLIE_LOG_DIR'] = tmprun_dir
    # 将步骤标识注入子进程环境，使 CLIE 消息归属到当前 classification 步骤下
    clie_env['CLIE_TITLE'] = '变化检测类别识别'
    clie_env['CLIE_TITLE_ID'] = 'change_classification'
    clie_env['CLIE_SOURCE'] = 'module'
    clie_env['CLIE_RANK'] = '-1'
    # Kafka 连接参数已在父进程环境中，通过 os.environ.copy() 继承，无需重复设置

    # 6. 后时相影像分类
    prg_sender.send({'progress': 15, 'runningStatus': 'running', 'runningInfo': '后时相影像分类预测中'})
    class_output_dir = os.path.join(tmprun_dir, "class_output")
    os.makedirs(class_output_dir, exist_ok=True)

    # CLIE 子进程进度映射到父进程的 15%→35% 区间
    clie_env['CLIE_PROGRESS_MIN'] = '15'
    clie_env['CLIE_PROGRESS_MAX'] = '35'
    try:
        _run_subprocess_logged([
            sys.executable, run_py, "run",
            "--image_input_file", post_image,
            "--image_output_dir", class_output_dir,
            "--model_file", model_file,
            "--error_log", os.path.join(diagnostic_dir, "clie_post_error.log"),
            "--debug_file", os.path.join(diagnostic_dir, "clie_post_debug.txt"),
        ], cwd=tmprun_dir, env=clie_env, logger=logger)
    except subprocess.CalledProcessError as exc:
        _cleanup_classification_temp_dir(tmprun_dir)
        raise NonFatalTaskWarning(
            f'后时相分类子任务退出码 {exc.returncode}，已按告警跳过'
        ) from exc
    tif_file2 = os.path.join(class_output_dir, f"{post_stem}.tif")

    if not os.path.exists(tif_file2):
        _cleanup_classification_temp_dir(tmprun_dir)
        raise NonFatalTaskWarning(f'后时相分类结果缺失: {tif_file2}')
    _cleanup_clie_working_rasters(tmprun_dir)

    # 7. 前时相影像分类
    prg_sender.send({'progress': 35, 'runningStatus': 'running', 'runningInfo': '前时相影像分类预测中'})
    class_output_dir1 = os.path.join(tmprun_dir, "class_output", "time1")
    os.makedirs(class_output_dir1, exist_ok=True)

    # CLIE 子进程进度映射到父进程的 35%→55% 区间
    clie_env['CLIE_PROGRESS_MIN'] = '35'
    clie_env['CLIE_PROGRESS_MAX'] = '55'
    try:
        _run_subprocess_logged([
            sys.executable, run_py, "run",
            "--image_input_file", pre_image,
            "--image_output_dir", class_output_dir1,
            "--model_file", model_file,
            "--error_log", os.path.join(diagnostic_dir, "clie_pre_error.log"),
            "--debug_file", os.path.join(diagnostic_dir, "clie_pre_debug.txt"),
        ], cwd=tmprun_dir, env=clie_env, logger=logger)
    except subprocess.CalledProcessError as exc:
        _cleanup_classification_temp_dir(tmprun_dir)
        raise NonFatalTaskWarning(
            f'前时相分类子任务退出码 {exc.returncode}，已按告警跳过'
        ) from exc
    tif_file1 = os.path.join(class_output_dir1, f"{pre_stem}.tif")
    if not os.path.exists(tif_file1):
        _cleanup_classification_temp_dir(tmprun_dir)
        raise NonFatalTaskWarning(f'前时相分类结果缺失: {tif_file1}')
    _cleanup_clie_working_rasters(tmprun_dir)

    # 8. 并行计算前后时相类别
    prg_sender.send({'progress': 55, 'runningStatus': 'running', 'runningInfo': '计算前后时相各地块类别'})

    # 统一坐标系
    with rasterio.open(tif_file2) as src:
        raster_crs = src.crs
    if gdf.crs != raster_crs:
        gdf = gdf.to_crs(raster_crs)

    # 使用并行zonal_stats
    import multiprocessing as mp

    def parallel_zonal_stats(gdf, tif_file, num_processes=4):
        total_rows = len(gdf)
        num_processes = min(num_processes, total_rows)
        chunk_size = max(total_rows // num_processes, 1)
        chunks = []
        for i in range(num_processes):
            start_idx = i * chunk_size
            if i == num_processes - 1:
                end_idx = total_rows
            else:
                end_idx = start_idx + chunk_size
            chunks.append(gdf.iloc[start_idx:end_idx])
        args_list = [(chunk, tif_file) for chunk in chunks]
        with mp.Pool(processes=num_processes) as pool:
            results = pool.map(_process_chunk, args_list)
        all_indices, all_classes = [], []
        for indices, classes in results:
            all_indices.extend(indices)
            all_classes.extend(classes)
        sorted_results = sorted(zip(all_indices, all_classes), key=lambda x: x[0])
        return [cls for idx, cls in sorted_results]

    logger.info("计算前时相各地块类别...")
    pre_classes = parallel_zonal_stats(gdf, tif_file1, num_processes=4)
    gdf["pre_code"] = pre_classes

    prg_sender.send({'progress': 75, 'runningStatus': 'running', 'runningInfo': '计算后时相各地块类别'})

    logger.info("计算后时相各地块类别...")
    curr_classes = parallel_zonal_stats(gdf, tif_file2, num_processes=4)
    gdf["curr_code"] = curr_classes
    gdf["curr_code"] = gdf["curr_code"].astype(int)

    # 9. 保存结果
    prg_sender.send({'progress': 90, 'runningStatus': 'running', 'runningInfo': '保存分类结果'})

    gdf = gdf.to_crs(origin_crs)
    gdf = format_classification_result(gdf)
    os.makedirs(dst_path, exist_ok=True)
    gdf.to_file(out_shp_file, encoding="utf-8")

    swap_write('output_shp', out_shp_file)
    swap_write('classified_count', len(gdf))

    # 10. 输出 Dataset
    prg_sender.send({'progress': 99, 'runningStatus': 'running', 'runningInfo': '创建输出数据集'})

    if output_dataset is not None:
        result_files = [Path(out_shp_file).name]
        ds = DatasetBuilder(output_dataset)
        ds.add("result", dst_path, "vector", result_files)
        ds.set_render(["result"])
        ds.save()

    _write_diagnostic_report(dst_path, 'classification_summary.json', {
        'status': 'completed',
        'mode': 'single',
        'mask_shp': mask_shp,
        'output_shp': out_shp_file,
        'feature_count': len(gdf),
        'empty_result': len(gdf) == 0,
    })

    # 11. 报告完成
    _send_completed('类别识别算法完成')
    _cleanup_classification_temp_dir(tmprun_dir)

# ========== 批量类别识别（文件夹模式）==========

def classification_folder(pre_folder, post_folder, mask_folder, model_path, dst_path, path_working, output_dataset):
    """批量类别识别：遍历前时相文件夹中所有影像文件，在后时相文件夹和掩码文件夹中匹配同名文件进行类别识别"""
    global prg_sender

    from pathlib import Path
    import rasterio
    import geopandas as gpd
    import numpy as np
    import multiprocessing as mp

    pre_path = Path(pre_folder)
    post_path = Path(post_folder)
    mask_path = Path(mask_folder)
    out_path = Path(dst_path)

    # 1. 报告启动
    prg_sender.send({'progress': 0, 'runningStatus': 'running', 'runningInfo': '批量类别识别算法启动'})

    # 2. 检查输入文件夹
    if not pre_path.exists():
        raise FileNotFoundError(f'前时相文件夹不存在: {pre_folder}')
    if not post_path.exists():
        raise FileNotFoundError(f'后时相文件夹不存在: {post_folder}')
    if not mask_path.exists():
        raise FileNotFoundError(f'掩码文件夹不存在: {mask_folder}')

    # 3. 获取前时相影像文件列表
    pre_files = sorted(list(pre_path.glob("*.tif")) + list(pre_path.glob("*.tiff")))

    if not pre_files:
        raise RuntimeError(f'前时相文件夹中无影像文件: {pre_folder}')

    # 4. 匹配文件三元组（pre_tif, post_tif, mask_shp）
    valid_triples = []
    skipped_unmatched = []
    for pf in pre_files:
        stem = pf.stem
        post_candidates = list(post_path.glob(f"{stem}.tif")) + list(post_path.glob(f"{stem}.tiff"))
        mask_candidates = list(mask_path.glob(f"{stem}.shp"))
        if post_candidates and mask_candidates:
            valid_triples.append((pf, post_candidates[0], mask_candidates[0]))
        else:
            missing = []
            if not post_candidates:
                missing.append('post')
            if not mask_candidates:
                missing.append('mask')
            skipped_unmatched.append({'stem': stem, 'missing': missing})

    total = len(valid_triples)
    if total == 0:
        raise RuntimeError('未找到任何匹配的影像-掩码三元组')

    swap_write('total_triples', total)
    if skipped_unmatched:
        swap_write('skipped_unmatched', skipped_unmatched)

    # 5. 创建输出目录
    os.makedirs(dst_path, exist_ok=True)
    if path_working is None:
        path_working = dst_path
    shp_dir = os.path.join(path_working, "shp_cls")
    png_dir = os.path.join(path_working, "png_cls")
    os.makedirs(shp_dir, exist_ok=True)
    os.makedirs(png_dir, exist_ok=True)

    # 6. 设置持久化日志
    logger, log_path = _configure_persistent_logger('classification_batch', dst_path)
    diagnostic_dir = _ensure_log_dir(dst_path)
    logger.info('持久化日志: %s', log_path)

    # 7. 准备 clie_new 推理环境
    work_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "clie_new")
    run_py = os.path.join(work_dir, "run.py")
    model_file = _resolve_classification_model(model_path, work_dir)
    logger.info('分类模型: %s', model_file)

    tmprun_dir = _create_classification_temp_dir()
    logger.info(f"分类临时目录: {tmprun_dir}")

    clie_env = os.environ.copy()
    clie_env['PYTHONPATH'] = work_dir + os.pathsep + clie_env.get('PYTHONPATH', '')
    clie_env['CLIE_LOG_DIR'] = tmprun_dir
    # 将步骤标识注入子进程环境，使 CLIE 消息归属到当前 classification 步骤下
    clie_env['CLIE_TITLE'] = '变化检测类别识别'
    clie_env['CLIE_TITLE_ID'] = 'change_classification'
    clie_env['CLIE_SOURCE'] = 'module'
    clie_env['CLIE_RANK'] = '-1'
    # Kafka 连接参数已在父进程环境中，通过 os.environ.copy() 继承

    output_shp_list = []
    failed_list = []
    empty_result_list = []
    feature_counts = {}

    for idx, (pre_file, post_file, mask_file) in enumerate(valid_triples):
        stem = pre_file.stem
        post_stem = post_file.stem
        output_shp = os.path.join(shp_dir, f'{stem}.shp')

        progress_pct = prg_sender.calc_progress_value(idx, total, 5, 95)
        prg_sender.send({
            'progress': progress_pct,
            'runningStatus': 'running',
            'runningInfo': f'类别识别中 ({idx+1}/{total}): {stem}'
        })

        logger.info(f'Processing ({idx+1}/{total}): pre={pre_file}, post={post_file}, mask={mask_file} -> {output_shp}')

        try:
            _remove_shapefile_dataset(output_shp)
            # --- 单组类别识别核心逻辑 ---

            # a. 读取变化检测结果
            gdf = gpd.read_file(str(mask_file))
            logger.info(f'  变化地块数量：{len(gdf)}')
            feature_counts[stem] = len(gdf)

            # b. 如果无变化区域，保留空 SHP 文件供后续使用
            if len(gdf) == 0:
                logger.info(f'  mask为空，没有变化区域，保留有效空分类结果: {stem}')
                _write_empty_classification_shp(output_shp, gdf.crs)
                output_shp_list.append(output_shp)
                empty_result_list.append(stem)
                continue

            gdf["uid"] = range(len(gdf))
            origin_crs = gdf.crs
            gdf["pre_code"] = 0
            gdf["curr_code"] = 0

            # c. 后时相影像分类
            logger.info(f'  后时相影像分类预测中...')
            class_output_dir = os.path.join(png_dir, f"{stem}_post")
            os.makedirs(class_output_dir, exist_ok=True)
            shutil.rmtree(os.path.join(class_output_dir, 'out_preview'), ignore_errors=True)
            tif_file2 = os.path.join(class_output_dir, f"{post_stem}.tif")
            _remove_file_if_exists(tif_file2)

            # CLIE 子进程进度映射到当前 pair 的前半段
            _pair_start = prg_sender.calc_progress_value(idx, total, 5, 95)
            _pair_end = prg_sender.calc_progress_value(idx + 1, total, 5, 95)
            _pair_mid = _pair_start + (_pair_end - _pair_start) // 2
            clie_env['CLIE_PROGRESS_MIN'] = str(int(_pair_start))
            clie_env['CLIE_PROGRESS_MAX'] = str(int(_pair_mid))
            _run_subprocess_logged([
                sys.executable, run_py, "run",
                "--image_input_file", str(post_file),
                "--image_output_dir", class_output_dir,
                "--model_file", model_file,
                "--error_log", os.path.join(diagnostic_dir, f"clie_{stem}_post_error.log"),
                "--debug_file", os.path.join(diagnostic_dir, f"clie_{stem}_post_debug.txt"),
            ], cwd=tmprun_dir, env=clie_env, logger=logger)
            if not os.path.exists(tif_file2):
                raise FileNotFoundError(f'后时相分类结果缺失: {tif_file2}')
            _cleanup_clie_working_rasters(tmprun_dir)

            # d. 前时相影像分类
            logger.info(f'  前时相影像分类预测中...')
            class_output_dir1 = os.path.join(png_dir, f"{stem}_pre")
            os.makedirs(class_output_dir1, exist_ok=True)
            shutil.rmtree(os.path.join(class_output_dir1, 'out_preview'), ignore_errors=True)
            tif_file1 = os.path.join(class_output_dir1, f"{stem}.tif")
            _remove_file_if_exists(tif_file1)

            # CLIE 子进程进度映射到当前 pair 的后半段
            clie_env['CLIE_PROGRESS_MIN'] = str(int(_pair_mid))
            clie_env['CLIE_PROGRESS_MAX'] = str(int(_pair_end))
            _run_subprocess_logged([
                sys.executable, run_py, "run",
                "--image_input_file", str(pre_file),
                "--image_output_dir", class_output_dir1,
                "--model_file", model_file,
                "--error_log", os.path.join(diagnostic_dir, f"clie_{stem}_pre_error.log"),
                "--debug_file", os.path.join(diagnostic_dir, f"clie_{stem}_pre_debug.txt"),
            ], cwd=tmprun_dir, env=clie_env, logger=logger)
            if not os.path.exists(tif_file1):
                raise FileNotFoundError(f'前时相分类结果缺失: {tif_file1}')
            _cleanup_clie_working_rasters(tmprun_dir)

            # e. 统一坐标系
            with rasterio.open(tif_file2) as src:
                raster_crs = src.crs
            if gdf.crs != raster_crs:
                gdf = gdf.to_crs(raster_crs)

            # f. 并行计算前后时相类别
            def _parallel_zonal_stats(_gdf, _tif_file, num_processes=4):
                total_rows = len(_gdf)
                num_processes = min(num_processes, total_rows)
                chunk_size = max(total_rows // num_processes, 1)
                chunks = []
                for i in range(num_processes):
                    start_idx = i * chunk_size
                    if i == num_processes - 1:
                        end_idx = total_rows
                    else:
                        end_idx = start_idx + chunk_size
                    chunks.append(_gdf.iloc[start_idx:end_idx])
                args_list = [(chunk, _tif_file) for chunk in chunks]
                with mp.Pool(processes=num_processes) as pool:
                    results = pool.map(_process_chunk, args_list)
                all_indices, all_classes = [], []
                for indices, classes in results:
                    all_indices.extend(indices)
                    all_classes.extend(classes)
                sorted_results = sorted(zip(all_indices, all_classes), key=lambda x: x[0])
                return [cls for idx, cls in sorted_results]

            logger.info(f'  计算前时相各地块类别...')
            pre_classes = _parallel_zonal_stats(gdf, tif_file1, num_processes=4)
            gdf["pre_code"] = pre_classes

            logger.info(f'  计算后时相各地块类别...')
            curr_classes = _parallel_zonal_stats(gdf, tif_file2, num_processes=4)
            gdf["curr_code"] = curr_classes
            gdf["curr_code"] = gdf["curr_code"].astype(int)

            # g. 保存结果
            gdf = gdf.to_crs(origin_crs)
            gdf = format_classification_result(gdf)
            os.makedirs(os.path.dirname(output_shp), exist_ok=True)
            gdf.to_file(output_shp, encoding="utf-8")

            logger.info(f'  处理完成，结果保存至：{output_shp}')
            output_shp_list.append(output_shp)

        except Exception as e:
            logger.exception('处理 %s 失败，已跳过: %s', stem, e)
            failed_list.append({'file': stem, 'error': str(e)})
        finally:
            _cleanup_clie_working_rasters(tmprun_dir)

    if failed_list:
        swap_write('warning_list', failed_list)
    if empty_result_list:
        swap_write('empty_result_list', empty_result_list)
    if not output_shp_list:
        _write_diagnostic_report(dst_path, 'classification_summary.json', {
            'status': 'failed',
            'mode': 'batch',
            'total': total,
            'processed_count': 0,
            'failed_count': len(failed_list),
            'failed_results': failed_list,
            'mask_feature_counts': feature_counts,
        })
        _cleanup_classification_temp_dir(tmprun_dir)
        raise RuntimeError(f'批量类别识别全部失败，0/{total} 组数据生成结果')

    # 8. 合并单个分类 SHP 为最终结果
    merged_shp = ""
    if output_shp_list:
        prg_sender.send({'progress': 93, 'runningStatus': 'running', 'runningInfo': '合并分类结果'})
        merged_shp = os.path.join(dst_path, "classification_result.shp")
        import sys as _sys
        _change_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'change')
        if _change_dir not in _sys.path:
            _sys.path.insert(0, _change_dir)
        from merge import merge_shp
        merged_shp = merge_shp(output_shp_list, merged_shp)
        if not merged_shp or not os.path.exists(merged_shp):
            raise RuntimeError('合并分类结果后未生成有效 SHP')

    # 9. 输出 Swap 变量
    result_shp = merged_shp if merged_shp else shp_dir
    swap_write('output_shp', result_shp)
    swap_write('output_png', png_dir)
    swap_write('output_shp_list', output_shp_list)
    swap_write('merged_shp', merged_shp)
    swap_write('processed_count', len(output_shp_list))
    # 10. 输出 Dataset
    prg_sender.send({'progress': 99, 'runningStatus': 'running', 'runningInfo': '创建输出数据集'})

    if output_dataset is not None:
        if merged_shp:
            merged_dir = os.path.dirname(merged_shp)
            result_dir = merged_dir
            result_files = [Path(merged_shp).name]
        else:
            result_dir = dst_path
            result_files = []

        # 这里只写步骤临时 dataset0。正式的步骤 dataset 和 final.dataset
        # 由工作流引擎在步骤退出后统一发布、转换相对路径并聚合。
        ds = DatasetBuilder(output_dataset)
        ds.add("result", result_dir, "vector", result_files)
        ds.set_render(["result"])
        ds.save()

    _write_diagnostic_report(dst_path, 'classification_summary.json', {
        'status': 'completed_with_warnings' if failed_list else 'completed',
        'mode': 'batch',
        'total': total,
        'processed_count': len(output_shp_list),
        'failed_count': len(failed_list),
        'empty_count': len(empty_result_list),
        'empty_results': empty_result_list,
        'failed_results': failed_list,
        'mask_feature_counts': feature_counts,
        'output_shp': result_shp,
    })

    # 11. 单组数据失败按告警处理，不中断整个批量工作流。
    result_msg = f'批量类别识别完成，成功处理 {len(output_shp_list)}/{total} 组数据'
    if empty_result_list:
        result_msg += f'，其中 {len(empty_result_list)} 组无变化区域、结果为空'
    if failed_list:
        result_msg += f'，{len(failed_list)} 组出现告警并已跳过'
    _send_completed(result_msg)
    _cleanup_classification_temp_dir(tmprun_dir)


# ========== 入口函数 ==========

def entry(pre_image, post_image, mask_shp, model_path, dst_path, path_working, output_dataset, step_id, step_name,
          kafka_server_ip_port, kafka_topic, kafka_task_id):
    task_logger, log_path = _configure_persistent_logger('classification_task', dst_path)
    task_logger.info('类别识别任务开始；持久化日志: %s', log_path)
    task_logger.info(
        '输入: pre=%s, post=%s, mask=%s, dst=%s',
        pre_image,
        post_image,
        mask_shp,
        dst_path,
    )
    print(f'[LOG] 类别识别日志已保存到: {log_path}', flush=True)
    init_progress_message_sender(kafka_server_ip_port, kafka_topic, kafka_task_id)
    init_progress_message_title(step_id, step_name)
    init_progress_message_source()
    prg_sender.send({
        'progress': 0,
        'runningStatus': 'running',
        'runningInfo': '类别识别任务已接收，正在初始化'
    })

    try:
        # 自动检测：如果输入为文件夹（目录），则进入批量处理模式
        if os.path.isdir(pre_image) and os.path.isdir(post_image) and os.path.isdir(mask_shp):
            classification_folder(pre_image, post_image, mask_shp, model_path, dst_path, path_working, output_dataset)
        else:
            classification(pre_image, post_image, mask_shp, model_path, dst_path, output_dataset)
    except Exception as exc:
        if _is_nonfatal_warning(exc):
            task_logger.exception('类别识别以告警状态结束: %s', exc)
            _write_diagnostic_report(dst_path, 'classification_warning.json', {
                'status': 'completed_with_warning',
                'error_type': type(exc).__name__,
                'error': str(exc),
                'traceback': traceback.format_exc(),
            })
            warning_info = f'类别识别完成（存在告警）：{exc}'
            swap_write('warning', str(exc))
            _send_completed(warning_info)
            return
        task_logger.exception('类别识别失败: %s', exc)
        _write_diagnostic_report(dst_path, 'classification_failure.json', {
            'status': 'failed',
            'error_type': type(exc).__name__,
            'error': str(exc),
            'traceback': traceback.format_exc(),
        })
        _send_failed(f'类别识别失败: {exc}')
        raise
    finally:
        _cleanup_all_classification_temp_dirs()
        prg_sender.close()
