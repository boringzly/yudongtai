"""Per-pair classification checkpoints, independent of change-detection resume."""
import hashlib
import json
import os
from pathlib import Path
import re
import shutil

from change.resume_checkpoint import (SIDECARS, atomic_json, digest, file_identity,
                                     invalidate, record_path, sidecar_hashes, validate_shp)
from task_artifacts import _task_output_root, diagnostic_root


def parse_bool(value):
    text = str(value).strip().lower()
    if text in ('true', '1', 'yes', 'on'):
        return True
    if text in ('false', '0', 'no', 'off', ''):
        return False
    raise ValueError('无法解析布尔值: %s' % value)


def resolve_options(source=None, legacy=None, environ=None):
    env = os.environ if environ is None else environ
    workflow = json.loads(env.get('WF_CONFIG') or '{}')
    if not isinstance(workflow, dict) or not isinstance(workflow.get('config', {}), dict):
        raise ValueError('WF_CONFIG 及其 config 必须为对象')
    config = workflow.get('config', {})
    if source is None:
        source = env.get('classification_resume_from', env.get(
            'CLASSIFICATION_RESUME_FROM', config.get('classification_resume_from', '')))
    if legacy is None:
        legacy = env.get('classification_resume_allow_legacy', env.get(
            'CLASSIFICATION_RESUME_ALLOW_LEGACY', config.get('classification_resume_allow_legacy', False)))
    if isinstance(source, bool) or not isinstance(source, (str, int)):
        raise ValueError('classification_resume_from 必须是任务 ID 或绝对路径')
    return str(source).strip(), parse_bool(legacy)


def run_identity(model_file, environ=None):
    env = os.environ if environ is None else environ
    root = Path(__file__).resolve().parent
    model = Path(model_file).resolve()
    models = [model] if model.is_file() else sorted(p for p in model.rglob('*') if p.is_file())
    if not models:
        raise ValueError('分类模型为空')
    code = hashlib.sha256()
    sources = [root / 'classification_core.py', root / 'classification_schema.py']
    sources += sorted((root / 'clie_new').rglob('*.py'))
    sources += sorted((root / 'clie_new').rglob('*.yaml'))
    sources += sorted((root / 'clie_new').rglob('*.yml'))
    for path in sources:
        code.update(str(path.relative_to(root)).encode())
        code.update(digest(path).encode())
    try:
        factor = int(env.get('CLIE_INFER_DOWNSAMPLE', '2'))
    except ValueError:
        factor = 2
    factor = factor if factor in (1, 2) else 2
    try:
        batch = max(1, min(16, int(env.get('CLIE_BATCH_PER_GPU', '8'))))
    except ValueError:
        batch = 8
    return {'schema': 1, 'algorithm': code.hexdigest(), 'model_name': model.name,
            'models': {str(p.relative_to(model)) if model.is_dir() else model.name: digest(p) for p in models},
            'downsample': factor, 'batch_per_gpu': batch,
            'vector_wkt': env.get('VECTOR_WKT', ''), 'filter_class_id': env.get('FILTER_CLASS_ID', '')}


def pair_identity(pre, post, mask, run):
    return {'pre': file_identity(pre), 'post': file_identity(post),
            'mask': sidecar_hashes(mask), 'run': run}


def validate_result(path):
    count = validate_shp(path)
    from osgeo import ogr
    ds = ogr.Open(str(path), 0)
    layer = ds.GetLayer()
    definition = layer.GetLayerDefn()
    for field in ('pre_code', 'curr_code', 'pre_name', 'curr_name'):
        if definition.GetFieldIndex(field) < 0:
            raise ValueError('分类结果缺少字段: %s' % field)
    for feature in layer:
        for field in ('pre_code', 'curr_code'):
            if feature.GetField(field) not in range(7):
                raise ValueError('分类代码不在 0～6: %s' % path)
    return count


def legacy_completions(log_path):
    attempts, current, model = {}, None, None
    if not Path(log_path).is_file():
        return attempts
    with open(log_path, encoding='utf-8', errors='replace') as stream:
        for line in stream:
            if '分类模型: ' in line:
                model = Path(line.split('分类模型: ', 1)[1].strip()).name
            match = re.search(r'Processing .*?: pre=(.*), post=(.*), mask=(.*) -> (.*)\.shp\s*$', line)
            if match:
                pre, post, mask, output = match.groups()
                current = {'pre': pre, 'post': post, 'mask': mask, 'output': output + '.shp',
                           'model': model, 'complete': False, 'empty': False, 'factors': []}
                attempts[Path(output).name] = current
            elif current is not None:
                factor = re.search(r'下采样倍数=(\d+)', line)
                if factor:
                    current['factors'].append(int(factor.group(1)))
                if 'mask为空，没有变化区域，保留有效空分类结果:' in line:
                    current.update(complete=True, empty=True)
                if '处理完成，结果保存至：' in line:
                    output = line.split('处理完成，结果保存至：', 1)[1].strip()
                    current['complete'] = output == current['output']
                if '失败，已跳过' in line:
                    current['complete'] = False
    return attempts


def resolve_source(value, dst):
    if not value:
        return None
    source = (_task_output_root(dst, {}).parent / str(value)
              if str(value).isdigit() else Path(value).expanduser())
    if not source.is_absolute():
        raise ValueError('分类续跑源必须是任务 ID 或绝对目录')
    source = source.resolve()
    if (source / 'working' / 'change_classification').is_dir():
        source = source / 'working' / 'change_classification'
    if not (source / 'shp_cls').is_dir():
        raise ValueError('分类续跑源不存在 shp_cls: %s' % source)
    return source, diagnostic_root(source, environ={}) / 'logs'


class Checkpoints:
    def __init__(self, dst, shp_dir, log_dir, run, source='', allow_legacy=False):
        self.directory = Path(log_dir) / 'classification_checkpoints'
        self.shp_dir = Path(shp_dir)
        self.run = run
        self.source = resolve_source(source, dst)
        if self.source and self.source[0].resolve() == self.shp_dir.parent.resolve():
            raise ValueError('跨任务续跑源不能是当前分类工作目录；同目录清单会自动检查')
        self.legacy = legacy_completions(self.source[1] / 'classification.log') if self.source and allow_legacy else {}
        self.resumed = 0

    def identity(self, pre, post, mask):
        return pair_identity(pre, post, mask, self.run)

    def save(self, stem, identity, shp, mask, provenance='computed'):
        for key in ('pre', 'post'):
            if file_identity(identity[key]['path']) != identity[key]:
                raise ValueError('分类期间输入影像已改变')
        if sidecar_hashes(mask) != identity['mask']:
            raise ValueError('分类期间变化图斑已改变')
        count = validate_result(shp)
        payload = {'status': 'completed', 'stem': stem, 'identity': identity,
                   'feature_count': count, 'sidecars': sidecar_hashes(shp), 'provenance': provenance}
        atomic_json(record_path(self.directory, stem), payload)
        return count

    def _check(self, directory, shp, stem, identity, legacy=False):
        path = record_path(directory, stem)
        if path.is_file():
            record = json.loads(path.read_text(encoding='utf-8'))
            if (not isinstance(record, dict) or record.get('status') != 'completed'
                    or record.get('stem') != stem or record.get('identity') != identity):
                raise ValueError('完成清单身份不匹配')
            if record.get('sidecars') != sidecar_hashes(shp) or record.get('feature_count') != validate_result(shp):
                raise ValueError('分类 SHP 内容或图斑数与清单不符')
            return record['feature_count'], record.get('provenance', 'computed')
        old = self.legacy.get(stem) if legacy else None
        if not old or not old['complete']:
            raise ValueError('无已验证完成清单/授权的旧版成功记录')
        if Path(old['output']).resolve() != Path(shp).resolve():
            raise ValueError('旧日志输出路径不匹配')
        if any(Path(old[key]).resolve() != Path(identity[key]['path']) for key in ('pre', 'post')):
            raise ValueError('旧日志影像路径不匹配')
        if sidecar_hashes(old['mask']) != identity['mask']:
            raise ValueError('旧任务变化图斑与本次不同')
        if old['model'] != self.run['model_name']:
            raise ValueError('旧日志分类模型不匹配')
        if self.run['vector_wkt'] or self.run['filter_class_id']:
            raise ValueError('旧日志不能验证当前过滤参数')
        count = validate_result(shp)
        if old['empty']:
            if count != 0 or validate_shp(old['mask']) != 0:
                raise ValueError('旧空结果与图斑数不符')
        elif len(old['factors']) != 2 or any(f != self.run['downsample'] for f in old['factors']):
            raise ValueError('旧日志无法确认前后两次推理的下采样配置')
        return count, 'legacy_assumed_unchanged'

    def restore(self, stem, identity, output, mask, logger):
        candidates = [(self.directory, Path(output), False)]
        if self.source:
            candidates.append((self.source[1] / 'classification_checkpoints',
                               self.source[0] / 'shp_cls' / (stem + '.shp'), True))
        for directory, shp, legacy in candidates:
            # Missing local results are normal on a fresh Pod/task.
            if not shp.is_file():
                continue
            try:
                count, provenance = self._check(directory, shp, stem, identity, legacy)
                hashes = sidecar_hashes(shp)
            except (ValueError, OSError, KeyError) as exc:
                logger.info('[分类续跑] 不复用 %s: %s', shp, exc)
                continue
            # Copy only validated datasets; never merge a path inside the old task.
            if shp.resolve() != Path(output).resolve():
                invalidate(self.directory, stem)
                for ext in SIDECARS:
                    Path(output).with_suffix(ext).unlink(missing_ok=True)
                for ext in hashes:
                    shutil.copy2(shp.with_suffix(ext), Path(output).with_suffix(ext))
            if hashes != sidecar_hashes(output):
                raise ValueError('分类续跑复制期间结果发生变化')
            self.save(stem, identity, output, mask, provenance)
            self.resumed += 1
            logger.info('[分类续跑] 已恢复 %s，图斑=%s，累计=%s，来源=%s', stem, count, self.resumed, provenance)
            return count
        return None
