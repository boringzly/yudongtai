"""Cross-Pod, per-pair checkpoints. No CUDA imports; old results stay read-only.

Legacy import is explicit: logs cannot prove historical input/model contents.
New checkpoints use input path/size/mtime, model/code hashes and sidecar hashes.
"""
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile


SIDECARS = ('.shp', '.shx', '.dbf', '.prj', '.cpg', '.qix', '.sbn', '.sbx')
REQUIRED = ('.shp', '.shx', '.dbf', '.prj')


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def file_identity(path):
    path = Path(path).resolve()
    stat = path.stat()
    return {'path': str(path), 'size': stat.st_size, 'mtime_ns': stat.st_mtime_ns}


def run_identity(model_path, use_two_models, second_model_path):
    root = Path(__file__).resolve().parent
    # Hash inference dependencies, not orchestration/logging, so progress fixes
    # don't invalidate predictions. Any algorithm/config change must invalidate.
    sources = [root / 'test_lib_batch_memeff_single_image_nomp.py']
    for directory in ('networks', 'utils'):
        sources.extend(sorted((root / directory).rglob('*.py')))
        sources.extend(sorted((root / directory).rglob('*.so')))
    for name in ('mean_value.txt', 'std_value.txt', 'config.py'):
        if (root / name).is_file():
            sources.append(root / name)
    sources.extend(sorted(root.glob('config*.so')))
    code = hashlib.sha256()
    for path in sorted(sources):
        code.update(str(path.relative_to(root)).encode('utf-8'))
        code.update(digest(path).encode('ascii'))
    models = [digest(resolve_model(model_path, 'FBCD_test_select0207_best_acc.pth'))]
    if use_two_models:
        models.append(digest(resolve_model(second_model_path, 'FBCD_test_Levir_CD_best_acc.pth')))
    from inference_metrics import change_batch_size
    return {'schema': 1, 'models': models, 'dual': bool(use_two_models),
            'algorithm': code.hexdigest(), 'batch_size': change_batch_size()}


def resolve_model(requested, fallback):
    root = Path(__file__).resolve().parent
    candidates = [Path(requested), root / requested] if requested else []
    candidates.append(root / fallback)
    for path in candidates:
        if path.is_file():
            return path.resolve()
    raise FileNotFoundError('模型文件不存在: %s' % requested)


def pair_identity(pre_path, post_path, run):
    return {'pre': file_identity(pre_path), 'post': file_identity(post_path), 'run': run}


def record_path(directory, stem):
    # Names can contain arbitrary Unicode, but cannot escape the manifest dir.
    key = hashlib.sha256(stem.encode('utf-8')).hexdigest()
    return Path(directory) / (key + '.json')


def atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix='.checkpoint-', dir=str(path.parent))
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def sidecar_hashes(shp):
    shp = Path(shp)
    for ext in REQUIRED:
        path = shp.with_suffix(ext)
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError('SHP 配套文件缺失或为零字节: %s' % path)
    return {ext: digest(shp.with_suffix(ext)) for ext in SIDECARS
            if shp.with_suffix(ext).is_file()}


def validate_shp(shp):
    from osgeo import ogr
    ds = ogr.Open(str(shp), 0)
    if ds is None:
        raise ValueError('SHP 无法读取: %s' % shp)
    layer = ds.GetLayer()
    if layer is None or layer.GetSpatialRef() is None:
        raise ValueError('SHP 缺少图层或坐标系: %s' % shp)
    expected = layer.GetFeatureCount()
    count = 0
    for feature in layer:
        if feature.GetGeometryRef() is None:
            raise ValueError('SHP 存在缺失几何的记录: %s' % shp)
        count += 1
    if count != expected:
        raise ValueError('SHP 记录数不一致: %s' % shp)
    return count  # Valid empty shapefiles are successful predictions too.


def save_checkpoint(directory, stem, identity, shp, count, provenance='computed'):
    for key in ('pre', 'post'):
        if file_identity(identity[key]['path']) != identity[key]:
            raise ValueError('处理期间输入影像发生变化，不能保存完成清单')
    payload = {'status': 'completed', 'stem': stem, 'identity': identity,
               'feature_count': count, 'sidecars': sidecar_hashes(shp),
               'provenance': provenance}
    atomic_json(record_path(directory, stem), payload)
    return payload


def invalidate(directory, stem):
    record_path(directory, stem).unlink(missing_ok=True)


def resolve_source(value, dst_path):
    """Accept task id in the same output container, task root, or step root."""
    if not value:
        return None
    # Explicit environ={} avoids accidentally resolving OLD diagnostics using
    # the NEW Pod's PATH_OUTPUT/CD_ARTIFACTS_ROOT.
    from task_artifacts import _task_output_root, diagnostic_root
    if str(value).isdigit():
        current = _task_output_root(dst_path, {})
        source = current.parent / str(value)
    else:
        source = Path(value).expanduser()
        if not source.is_absolute():
            raise ValueError('resume_from 必须是任务 ID 或绝对路径')
    if (source / 'working' / 'change_detection').is_dir():
        source = source / 'working' / 'change_detection'
    source = source.resolve()
    if not (source / 'shp').is_dir():
        raise ValueError('续跑源不存在 shp 目录: %s' % source)
    logs = diagnostic_root(source, environ={}) / 'logs'
    return source, logs


def legacy_completions(log_dir):
    """Use the latest attempt per stem; an incomplete rerun invalidates success."""
    attempts = {}
    for path in sorted(Path(log_dir).glob('change_detection_gpu_*.log')) + list(Path(log_dir).glob('change_detection_cpu.log')):
        current = None
        with path.open(encoding='utf-8', errors='replace') as stream:
            for line in stream:
                match = re.search(r'Processing .*?: pre=(.*), post=(.*) -> (.*)\.shp\s*$', line)
                if match:
                    pre, post, output = match.groups()
                    current = {'pre': pre, 'post': post, 'stem': Path(output).name,
                               'time': line[:23], 'completed': False, 'models': [], 'dual': None}
                    prior = attempts.get(current['stem'])
                    if prior is None or current['time'] >= prior['time']:
                        attempts[current['stem']] = current
                elif current is not None:
                    if '变化检测主模型: ' in line:
                        current['models'] = [Path(line.split('变化检测主模型: ', 1)[1].strip()).name]
                    if '变化检测第二模型: ' in line:
                        current['models'].append(Path(line.split('变化检测第二模型: ', 1)[1].split('；', 1)[0].strip()).name)
                        current['dual'] = True
                    if '变化检测使用单模型模式' in line:
                        current['dual'] = False
                    if '变化检测使用双模型' in line:
                        current['dual'] = True
                    if ('处理 %s 完成：' % current['stem']) in line:
                        current['completed'] = True
    return attempts


def restore(task, source_step, source_records, legacy, allow_legacy, copy_shp,
            validator=validate_shp):
    """Return recovered result or raise; never mutate source outputs/records."""
    stem = task['stem']
    source = Path(source_step) / 'shp' / (stem + '.shp')
    record = record_path(source_records, stem)
    provenance = 'checkpoint'
    if record.exists():
        payload = json.loads(record.read_text(encoding='utf-8'))
        if not isinstance(payload, dict):
            raise ValueError('完成清单格式无效')
        if payload.get('status') != 'completed' or payload.get('stem') != stem:
            raise ValueError('完成清单无效')
        if payload.get('identity') != task['checkpoint_identity']:
            raise ValueError('输入、模型或推理代码指纹不匹配')
        if payload.get('sidecars') != sidecar_hashes(source):
            raise ValueError('结果文件指纹不匹配')
        provenance = payload.get('provenance', 'checkpoint')
    else:
        if not allow_legacy:
            raise ValueError('缺少完成清单；旧日志导入未启用')
        old = legacy.get(stem, {})
        models = [resolve_model(task['model_path'], 'FBCD_test_select0207_best_acc.pth').name]
        if task['use_two_models']:
            models.append(resolve_model(task['second_model_path'], 'FBCD_test_Levir_CD_best_acc.pth').name)
        if (not old.get('completed') or old.get('pre') != task['pre_path']
                or old.get('post') != task['post_path'] or old.get('models') != models
                or old.get('dual') != task['use_two_models']):
            raise ValueError('旧任务日志缺少匹配的成功记录、输入或模型模式')
        sidecar_hashes(source)
        payload = None
        provenance = 'legacy_assumed_unchanged'
    count = validator(source)
    if payload is not None and count != payload.get('feature_count'):
        raise ValueError('图斑数与完成清单不一致')
    before = sidecar_hashes(source)
    if payload is not None and before != payload['sidecars']:
        raise ValueError('校验期间源结果发生变化')
    destination = Path(task['output_shp'])
    if source.resolve() != destination.resolve():
        copy_shp(str(source), str(destination))
    if sidecar_hashes(destination) != before or validator(destination) != count:
        raise ValueError('结果复制后校验失败')
    save_checkpoint(task['checkpoint_dir'], stem, task['checkpoint_identity'],
                    destination, count, provenance)
    return {'type': 'result', 'status': 'completed', 'idx': task['idx'], 'stem': stem,
            'worker': 'resume', 'output_shp': str(destination), 'feature_count': count,
            'duration': 0, 'resumed': True, 'provenance': provenance}
