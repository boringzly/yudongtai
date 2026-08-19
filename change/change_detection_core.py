import os
import sys
import json
import yaml

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
            self.prg_sender.send(message_dict)
        else:
            print(f'[SIMULATED SEND] {message_dict}')  # 模拟发送

    def calc_progress_value(self, index, total, min_value=0, max_value=100):
        if self.prg_sender is not None:
            return self.prg_sender.calc_progress_value(index, total, min_value, max_value)
        if total > 0:
            return min_value + (max_value - min_value) * (index / total)
        return min_value

def init_progress_message_sender(kafka_server_ip_port, kafka_topic, kafka_task_id):
    global prg_sender
    prg_sender = ProgressMessageSenderWrap(kafka_server_ip_port, kafka_topic, kafka_task_id)

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

# ========== DatasetBuilder ==========

from DatasetBuilder import DatasetBuilder

# ========== 算法主体 ==========

def change_detection(pre_image, post_image, model_path, dst_path, output_dataset):
    global prg_sender

    # 1. 报告启动
    prg_sender.send({'progress': 0, 'runningStatus': 'running', 'runningInfo': '变化检测算法启动'})

    # 2. 检查输入
    if not os.path.exists(pre_image):
        prg_sender.send({'progress': 100, 'runningStatus': 'completed', 'runningInfo': f'前时相影像不存在: {pre_image}'})
        return
    if not os.path.exists(post_image):
        prg_sender.send({'progress': 100, 'runningStatus': 'completed', 'runningInfo': f'后时相影像不存在: {post_image}'})
        return

    # 3. 创建输出目录
    os.makedirs(dst_path, exist_ok=True)

    # 4. 确定输出文件名
    from pathlib import Path
    pre_stem = Path(pre_image).stem
    post_stem = Path(post_image).stem
    output_shp = os.path.join(dst_path, f'{pre_stem}_{post_stem}_change.shp')

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

    # 6. 检查结果是否生成（无变化时不创建文件）
    if not os.path.exists(output_shp):
        prg_sender.send({'progress': 100, 'runningStatus': 'completed', 'runningInfo': '未检测到变化区域'})
        return

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
        prg_sender.send({'progress': 100, 'runningStatus': 'completed', 'runningInfo': f'前时相文件夹不存在: {pre_folder}'})
        return
    if not post_folder.exists():
        prg_sender.send({'progress': 100, 'runningStatus': 'completed', 'runningInfo': f'后时相文件夹不存在: {post_folder}'})
        return

    # 3. 获取前时相文件夹中所有影像文件
    pre_files = sorted(list(pre_folder.glob("*.tif")) + list(pre_folder.glob("*.tiff")))

    if not pre_files:
        prg_sender.send({'progress': 100, 'runningStatus': 'completed', 'runningInfo': f'前时相文件夹中无影像文件: {pre_folder}'})
        return

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
        prg_sender.send({'progress': 100, 'runningStatus': 'completed', 'runningInfo': '未找到任何匹配的前后时相影像对'})
        return

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

        progress_pct = prg_sender.calc_progress_value(idx, total, 5, 95)
        prg_sender.send({
            'progress': progress_pct,
            'runningStatus': 'running',
            'runningInfo': f'变化检测中 ({idx+1}/{total}): {stem}'
        })

        logger.info(f'Processing ({idx+1}/{total}): pre={pre_path}, post={post_path} -> {output_shp}')

        def _make_batch_cb(_stem, _idx):
            def _cb(current, total_patches):
                inner_pct = current / max(total_patches, 1)
                pct = prg_sender.calc_progress_value(_idx + inner_pct, total, 5, 95)
                prg_sender.send({
                    'progress': pct,
                    'runningStatus': 'running',
                    'runningInfo': f'变化检测中 ({_idx+1}/{total}): {_stem} ({current}/{total_patches} patches)'
                })
            return _cb

        try:
            test_lib_big_memeff(
                pre_img_path=str(pre_path),
                post_img_path=str(post_path),
                output_path=output_shp,
                logger=logger,
                callback_url=None,
                job_id=None,
                temp_dir_suffix="tmp",
                progress_callback=_make_batch_cb(stem, idx)
            )
            if os.path.exists(output_shp):
                output_shp_list.append(output_shp)
            else:
                logger.info(f'处理 {stem} 未检测到变化区域')
            # 将推理产生的 tif 中间结果移动到 tif 目录
            tif_src = os.path.join(shp_dir, f'{stem}.tif')
            if os.path.exists(tif_src):
                import shutil as _shutil
                _shutil.move(tif_src, os.path.join(tif_dir, f'{stem}.tif'))
        except Exception as e:
            logger.error(f'处理 {stem} 失败: {e}', exc_info=True)
            failed_list.append({'file': stem, 'error': str(e)})

    # 7. 不合并结果，保留单个 SHP 文件供后续分类使用
    prg_sender.send({'progress': 95, 'runningStatus': 'running', 'runningInfo': '跳过合并，保留单个变化检测结果'})

    # 8. 输出 Swap 变量
    swap_write('output_shp', shp_dir)    # 下游分类通过 glob 按 stem 匹配单个 SHP
    swap_write('output_shp_list', output_shp_list)
    swap_write('output_tif', tif_dir)
    swap_write('processed_count', len(output_shp_list))
    if failed_list:
        swap_write('failed_list', failed_list)

    # 9. 输出 Dataset
    prg_sender.send({'progress': 99, 'runningStatus': 'running', 'runningInfo': '创建输出数据集'})

    if output_dataset is not None:
        result_files = sorted(p.name for p in Path(shp_dir).glob("*.shp") if p.is_file())
        ds = DatasetBuilder(output_dataset)
        ds.add("result", shp_dir, "vector", result_files)
        ds.set_render(["result"])
        ds.save()

    # 10. 报告完成
    result_msg = f'批量变化检测完成，成功处理 {len(output_shp_list)}/{total} 对影像'
    if failed_list:
        result_msg += f'，{len(failed_list)} 对失败'
    prg_sender.send({'progress': 100, 'runningStatus': 'completed', 'runningInfo': result_msg})


# ========== 入口函数 ==========

def entry(pre_image, post_image, model_path, dst_path, output_dataset, step_id, step_name,
          kafka_server_ip_port, kafka_topic, kafka_task_id):
    init_progress_message_sender(kafka_server_ip_port, kafka_topic, kafka_task_id)
    init_progress_message_title(step_id, step_name)
    init_progress_message_source()

    # 自动检测：如果输入为文件夹（目录），则进入批量处理模式
    if os.path.isdir(pre_image) and os.path.isdir(post_image):
        change_detection_folder(pre_image, post_image, model_path, dst_path, output_dataset)
    else:
        change_detection(pre_image, post_image, model_path, dst_path, output_dataset)
