"""Durable raw-vector recovery; imports no model, Torch or CUDA."""
import json
from pathlib import Path
import shutil
import tempfile
import uuid

from resume_checkpoint import (SIDECARS, atomic_json, file_identity, record_path,
                               sidecar_hashes, validate_shp)
from resume_compat import identities_match


def _input_unchanged(identity):
    for key in ('pre', 'post'):
        if file_identity(identity[key]['path']) != identity[key]:
            raise ValueError('输入影像发生变化，不能使用矢量后处理恢复点')


def copy_vector(source, destination):
    source, destination = Path(source), Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    hashes = sidecar_hashes(source)
    for ext in SIDECARS:
        target = destination.with_suffix(ext)
        if ext in hashes:
            shutil.copy2(source.with_suffix(ext), target)
        elif target.exists():
            target.unlink()
    if sidecar_hashes(destination) != hashes:
        raise ValueError('矢量恢复点复制校验失败')


def save(directory, stem, identity, raw_shp, roi_wkt, projection, validator=validate_shp):
    """Publish the record last; incomplete copies never become resumable."""
    _input_unchanged(identity)
    count = validator(raw_shp)
    root = Path(directory).resolve()
    version = root / (record_path(root, stem).stem + '-' + uuid.uuid4().hex)
    version.mkdir(parents=True)
    try:
        saved = version / 'raw.shp'
        copy_vector(raw_shp, saved)
        if validator(saved) != count:
            raise ValueError('矢量恢复点图斑数不匹配')
        _input_unchanged(identity)
        payload = {'schema': 1, 'stage': 'polygonized', 'stem': stem,
                   'identity': identity, 'raw': str(saved.relative_to(root)),
                   'feature_count': count, 'sidecars': sidecar_hashes(saved),
                   'roi_wkt': roi_wkt, 'projection': projection}
        atomic_json(record_path(root, stem), payload)
        return payload
    except BaseException:
        shutil.rmtree(version, ignore_errors=True)
        raise


def load(directories, stem, identity, logger=None, validator=validate_shp):
    _input_unchanged(identity)
    for directory in directories:
        root = Path(directory).resolve()
        record = record_path(root, stem)
        if not record.is_file():
            continue
        try:
            data = json.loads(record.read_text(encoding='utf-8'))
            if (data.get('schema') != 1 or data.get('stage') != 'polygonized'
                    or data.get('stem') != stem or not identities_match(data.get('identity'), identity)):
                raise ValueError('恢复点阶段、输入、模型、范围或代码不匹配')
            path = (root / data['raw']).resolve()
            if root not in path.parents or path.suffix != '.shp':
                raise ValueError('恢复点路径越界')
            if (not isinstance(data.get('projection'), str) or not data['projection']
                    or data.get('roi_wkt') is not None and not isinstance(data['roi_wkt'], str)):
                raise ValueError('恢复点坐标系或范围记录无效')
            if sidecar_hashes(path) != data.get('sidecars') or validator(path) != data.get('feature_count'):
                raise ValueError('恢复点结果文件或图斑数不匹配')
            _input_unchanged(identity)
            return dict(data, raw_path=str(path))
        except (OSError, ValueError, TypeError, AttributeError, KeyError, RuntimeError) as exc:
            if logger:
                logger.warning('[后处理续跑] 不复用 %s: %s', stem, exc)
    return None


def finish(raw_shp, output_shp, roi_wkt, projection, fill_hole, smooth, clip,
           logger=None, validator=validate_shp):
    """Always work on copies; preserve the immutable recovery point on failure."""
    output_shp = Path(output_shp)
    output_shp.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.change-postprocess-', dir=str(output_shp.parent)) as folder:
        folder = Path(folder)
        raw, filled, processed = (folder / name for name in ('raw.shp', 'filled.shp', 'processed.shp'))
        copy_vector(raw_shp, raw)
        if validator(raw) == 0:
            copy_vector(raw, processed)
        else:
            fill_hole(str(raw), str(filled))
            smooth(str(filled), str(processed))
        if roi_wkt:
            clip(processed, roi_wkt, projection, logger=logger)
        count = validator(processed)
        copy_vector(processed, output_shp)
        if validator(output_shp) != count:
            raise ValueError('后处理结果复制校验失败')
        return count
