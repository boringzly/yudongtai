import os
import sys
import json
import yaml
import tempfile
import shutil

# ========== Kafka 客户端 ==========
try:
    from MessageClient.ProgressMessageSender import ProgressMessageSender
except:
    print('failed to load ProgressMessageSender package.')
    ProgressMessageSender = None
prg_sender = None

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
    prg_sender = ProgressMessageSenderWrap(kafka_server_ip_port, kafka_topic, kafka_task_id)
    if kafka_server_ip_port:
        os.environ['KAFKA_SERVER_IP_PORT'] = str(kafka_server_ip_port)
    if kafka_topic:
        os.environ['KAFKA_TOPIC'] = str(kafka_topic)
    if kafka_task_id:
        os.environ['KAFKA_TASK_ID'] = str(kafka_task_id)

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


# ========== Swap 变量输出 ==========

def swap_write(key, value):
    print(f"##SWAP:{key}={json.dumps(value, ensure_ascii=False)}", flush=True)

# ========== DatasetBuilder ==========

from DatasetBuilder import DatasetBuilder

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
            major_classes.append(5)
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
    import subprocess
    import logging

    logger = logging.getLogger('classification')
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s: %(message)s'))
        logger.addHandler(handler)

    # 3. 读取变化检测结果
    prg_sender.send({'progress': 5, 'runningStatus': 'running', 'runningInfo': '读取变化检测结果SHP'})

    if not os.path.exists(mask_shp):
        logger.info("变化检测结果文件不存在，跳过分类")
        if output_dataset is not None:
            ds = DatasetBuilder(output_dataset)
            ds.add("result", dst_path, "vector")
            ds.set_render(["result"])
            ds.save()
        _send_completed('变化检测结果不存在，跳过分类')
        return

    gdf = gpd.read_file(mask_shp)
    logger.info(f'变化地块数量：{len(gdf)}')

    # 4. 如果无变化区域，不创建空输出，直接返回
    if len(gdf) == 0:
        logger.info("mask为空，没有变化区域可分类")
        if output_dataset is not None:
            ds = DatasetBuilder(output_dataset)
            ds.add("result", dst_path, "vector")
            ds.set_render(["result"])
            ds.save()
        _send_completed('无变化区域可分类')
        return

    gdf["uid"] = range(len(gdf))
    origin_crs = gdf.crs
    gdf["pre_code"] = 0
    gdf["curr_code"] = 0

    # 5. 准备 clie_new 推理环境（容器内 /app/module 为只读，所有输出重定向到可写目录）
    work_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "clie_new")
    run_py = os.path.join(work_dir, "run.py")
    model_file = os.path.join(work_dir, "tools", "fullclass_chinaall_3b.pth")

    tmprun_dir = tempfile.mkdtemp(prefix="clie_tmp_")
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
    post_stem = Path(post_image).stem
    class_output_dir = os.path.join(tmprun_dir, "class_output")
    os.makedirs(class_output_dir, exist_ok=True)

    # CLIE 子进程进度映射到父进程的 15%→35% 区间
    clie_env['CLIE_PROGRESS_MIN'] = '15'
    clie_env['CLIE_PROGRESS_MAX'] = '35'
    subprocess.run([
        sys.executable, run_py, "run",
        "--image_input_file", post_image,
        "--image_output_dir", class_output_dir,
        "--model_file", model_file,
        "--error_log", os.path.join(tmprun_dir, "error.log"),
        "--debug_file", os.path.join(tmprun_dir, "debug.txt"),
    ], check=True, cwd=tmprun_dir, env=clie_env)
    tif_file2 = os.path.join(class_output_dir, f"{post_stem}.tif")

    if not os.path.exists(tif_file2):
        _send_completed('后时相分类结果缺失')
        shutil.rmtree(tmprun_dir, ignore_errors=True)
        return

    # 7. 前时相影像分类
    prg_sender.send({'progress': 35, 'runningStatus': 'running', 'runningInfo': '前时相影像分类预测中'})
    pre_stem = Path(pre_image).stem
    class_output_dir1 = os.path.join(tmprun_dir, "class_output", "time1")
    os.makedirs(class_output_dir1, exist_ok=True)

    # CLIE 子进程进度映射到父进程的 35%→55% 区间
    clie_env['CLIE_PROGRESS_MIN'] = '35'
    clie_env['CLIE_PROGRESS_MAX'] = '55'
    subprocess.run([
        sys.executable, run_py, "run",
        "--image_input_file", pre_image,
        "--image_output_dir", class_output_dir1,
        "--model_file", model_file,
        "--error_log", os.path.join(tmprun_dir, "error.log"),
        "--debug_file", os.path.join(tmprun_dir, "debug.txt"),
    ], check=True, cwd=tmprun_dir, env=clie_env)
    tif_file1 = os.path.join(class_output_dir1, f"{pre_stem}.tif")

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
    out_shp_file = os.path.join(dst_path, f'{pre_stem}_{post_stem}_classified.shp')
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

    # 11. 报告完成
    _send_completed('类别识别算法完成')
    shutil.rmtree(tmprun_dir, ignore_errors=True)

# ========== 批量类别识别（文件夹模式）==========

def classification_folder(pre_folder, post_folder, mask_folder, model_path, dst_path, path_working, output_dataset):
    """批量类别识别：遍历前时相文件夹中所有影像文件，在后时相文件夹和掩码文件夹中匹配同名文件进行类别识别"""
    global prg_sender

    from pathlib import Path
    import rasterio
    import geopandas as gpd
    import numpy as np
    import subprocess
    import logging
    import multiprocessing as mp

    pre_path = Path(pre_folder)
    post_path = Path(post_folder)
    mask_path = Path(mask_folder)
    out_path = Path(dst_path)

    # 1. 报告启动
    prg_sender.send({'progress': 0, 'runningStatus': 'running', 'runningInfo': '批量类别识别算法启动'})

    # 2. 检查输入文件夹
    if not pre_path.exists():
        _send_completed(f'前时相文件夹不存在: {pre_folder}')
        return
    if not post_path.exists():
        _send_completed(f'后时相文件夹不存在: {post_folder}')
        return
    if not mask_path.exists():
        _send_completed(f'掩码文件夹不存在: {mask_folder}')
        return

    # 3. 获取前时相影像文件列表
    pre_files = sorted(list(pre_path.glob("*.tif")) + list(pre_path.glob("*.tiff")))

    if not pre_files:
        _send_completed(f'前时相文件夹中无影像文件: {pre_folder}')
        return

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
        _send_completed('未找到任何匹配的影像-掩码三元组')
        return

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

    # 6. 设置日志
    logger = logging.getLogger('classification_batch')
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s: %(message)s'))
        logger.addHandler(handler)

    # 7. 准备 clie_new 推理环境
    work_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "clie_new")
    run_py = os.path.join(work_dir, "run.py")
    model_file = os.path.join(work_dir, "tools", "fullclass_chinaall_3b.pth")

    tmprun_dir = tempfile.mkdtemp(prefix="clie_tmp_")
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
            # --- 单组类别识别核心逻辑 ---

            # a. 读取变化检测结果
            gdf = gpd.read_file(str(mask_file))
            logger.info(f'  变化地块数量：{len(gdf)}')

            # b. 如果无变化区域，保留空 SHP 文件供后续使用
            if len(gdf) == 0:
                logger.info(f'  mask为空，没有变化区域，保留空分类结果: {stem}')
                gdf.to_file(output_shp, encoding="utf-8")
                output_shp_list.append(output_shp)
                continue

            gdf["uid"] = range(len(gdf))
            origin_crs = gdf.crs
            gdf["pre_code"] = 0
            gdf["curr_code"] = 0

            # c. 后时相影像分类
            logger.info(f'  后时相影像分类预测中...')
            class_output_dir = os.path.join(png_dir, f"{stem}_post")
            os.makedirs(class_output_dir, exist_ok=True)

            # CLIE 子进程进度映射到当前 pair 的前半段
            _pair_start = prg_sender.calc_progress_value(idx, total, 5, 95)
            _pair_end = prg_sender.calc_progress_value(idx + 1, total, 5, 95)
            _pair_mid = _pair_start + (_pair_end - _pair_start) // 2
            clie_env['CLIE_PROGRESS_MIN'] = str(int(_pair_start))
            clie_env['CLIE_PROGRESS_MAX'] = str(int(_pair_mid))
            clie_env['CLIE_PREVIEW_BASE'] = f"/png_cls/{stem}_post/out_preview"
            subprocess.run([
                sys.executable, run_py, "run",
                "--image_input_file", str(post_file),
                "--image_output_dir", class_output_dir,
                "--model_file", model_file,
                "--error_log", os.path.join(tmprun_dir, f"error_{stem}.log"),
                "--debug_file", os.path.join(tmprun_dir, f"debug_{stem}.txt"),
            ], check=True, cwd=tmprun_dir, env=clie_env)
            tif_file2 = os.path.join(class_output_dir, f"{post_stem}.tif")

            if not os.path.exists(tif_file2):
                raise FileNotFoundError(f'后时相分类结果缺失: {tif_file2}')

            # d. 前时相影像分类
            logger.info(f'  前时相影像分类预测中...')
            class_output_dir1 = os.path.join(png_dir, f"{stem}_pre")
            os.makedirs(class_output_dir1, exist_ok=True)

            # CLIE 子进程进度映射到当前 pair 的后半段
            clie_env['CLIE_PROGRESS_MIN'] = str(int(_pair_mid))
            clie_env['CLIE_PROGRESS_MAX'] = str(int(_pair_end))
            clie_env['CLIE_PREVIEW_BASE'] = f"/png_cls/{stem}_pre/out_preview"
            subprocess.run([
                sys.executable, run_py, "run",
                "--image_input_file", str(pre_file),
                "--image_output_dir", class_output_dir1,
                "--model_file", model_file,
                "--error_log", os.path.join(tmprun_dir, f"error_{stem}.log"),
                "--debug_file", os.path.join(tmprun_dir, f"debug_{stem}.txt"),
            ], check=True, cwd=tmprun_dir, env=clie_env)
            tif_file1 = os.path.join(class_output_dir1, f"{stem}.tif")

            if not os.path.exists(tif_file1):
                raise FileNotFoundError(f'前时相分类结果缺失: {tif_file1}')

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
            os.makedirs(os.path.dirname(output_shp), exist_ok=True)
            gdf.to_file(output_shp, encoding="utf-8")

            logger.info(f'  处理完成，结果保存至：{output_shp}')
            output_shp_list.append(output_shp)

        except Exception as e:
            logger.error(f'处理 {stem} 失败: {e}', exc_info=True)
            failed_list.append({'file': stem, 'error': str(e)})

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
        merge_shp(output_shp_list, merged_shp)

    # 9. 输出 Swap 变量
    result_shp = merged_shp if merged_shp else shp_dir
    swap_write('output_shp', result_shp)
    swap_write('output_png', png_dir)
    swap_write('output_shp_list', output_shp_list)
    swap_write('merged_shp', merged_shp)
    swap_write('processed_count', len(output_shp_list))
    if failed_list:
        swap_write('failed_list', failed_list)

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

    # 11. 报告完成
    result_msg = f'批量类别识别完成，成功处理 {len(output_shp_list)}/{total} 组数据'
    if failed_list:
        result_msg += f'，{len(failed_list)} 组失败'
    _send_completed(result_msg)
    shutil.rmtree(tmprun_dir, ignore_errors=True)


# ========== 入口函数 ==========

def entry(pre_image, post_image, mask_shp, model_path, dst_path, path_working, output_dataset, step_id, step_name,
          kafka_server_ip_port, kafka_topic, kafka_task_id):
    init_progress_message_sender(kafka_server_ip_port, kafka_topic, kafka_task_id)
    init_progress_message_title(step_id, step_name)
    init_progress_message_source()

    # 自动检测：如果输入为文件夹（目录），则进入批量处理模式
    if os.path.isdir(pre_image) and os.path.isdir(post_image) and os.path.isdir(mask_shp):
        classification_folder(pre_image, post_image, mask_shp, model_path, dst_path, path_working, output_dataset)
    else:
        classification(pre_image, post_image, mask_shp, model_path, dst_path, output_dataset)
