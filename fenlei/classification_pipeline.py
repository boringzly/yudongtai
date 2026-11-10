"""Bounded CPU-only classification loading, telemetry and reviewed update compatibility."""
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import re
import threading
import time

RELEASE = 'classification-prefetch-v1'
COMPAT_FILE = 'classification_pipeline_compat.json'


def _integer(env, name, default, minimum, maximum):
    value = str(env.get(name, default)).strip()
    if not re.fullmatch(r'[0-9]+', value) or not minimum <= int(value) <= maximum:
        raise ValueError('%s must be %s..%s' % (name, minimum, maximum))
    return int(value)


def configure_cpu_pipeline(torch, available_cpus, hami=False, environ=None):
    env = os.environ if environ is None else environ
    count = max(1, int(available_cpus))
    threads = min(count, _integer(env, 'CLIE_TORCH_THREADS', 4, 1, 256))
    prefetch = min(count, _integer(env, 'CLIE_PREFETCH_THREADS', 2 if hami else 0, 0, 8))
    torch.set_num_threads(threads)
    return prefetch, threads


class ThreadedBatchLoader:
    """Ordered samples with private readers; at most one batch ahead, no CUDA."""
    def __init__(self, factory, length, batch_size, workers, collate):
        if length < 0 or batch_size < 1 or workers < 1:
            raise ValueError('Invalid loader dimensions')
        self.factory, self.length = factory, length
        self.batch_size, self.workers, self.collate = batch_size, workers, collate
        self.iterator = None

    def __len__(self):
        return (self.length + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        self.iterator = self._batches()
        return self.iterator

    def close(self):
        if self.iterator is not None:
            self.iterator.close()

    def _batches(self):
        local = threading.local()
        datasets, pending = [], deque()
        lock = threading.Lock()
        def load(index):
            if not hasattr(local, 'dataset'):
                local.dataset = self.factory()
                with lock:
                    datasets.append(local.dataset)
            return local.dataset[index]
        pool = ThreadPoolExecutor(max_workers=self.workers, thread_name_prefix='classification-cpu')
        next_index = 0
        try:
            while next_index < min(self.batch_size, self.length):
                pending.append(pool.submit(load, next_index))
                next_index += 1
            while pending:
                samples = []
                for _ in range(min(self.batch_size, len(pending))):
                    samples.append(pending.popleft().result())
                    if next_index < self.length:
                        pending.append(pool.submit(load, next_index))
                        next_index += 1
                yield self.collate(samples)
        finally:
            for future in pending:
                future.cancel()
            pool.shutdown(wait=True)
            first_error = None
            for dataset in datasets:
                try:
                    dataset.close()
                except Exception as error:
                    first_error = first_error or error
            if first_error is not None:
                raise first_error


@contextmanager
def managed_batches(loader):
    try:
        yield loader
    finally:
        close = getattr(loader, 'close', None)
        if close is not None:
            close()


def timed_batches(loader, metrics):
    iterator = iter(loader)
    while True:
        started = time.perf_counter()
        try:
            batch = next(iterator)
        except StopIteration:
            return
        metrics.wait += time.perf_counter() - started
        yield batch


class ClassificationMetrics:
    def __init__(self, filename, batch_size):
        self.filename, self.batch_size = filename, batch_size
        self.started = self.last = time.perf_counter()
        self.wait = self.prepare = self.infer = self.write = 0.0
        self.samples = self.valid = self.batches = 0
        self.output = self.raster = self.message = 0.0
        self.mask = self.exception = self.log = 0.0

    def record(self, count, valid, logger):
        self.samples += count
        self.valid += valid
        self.batches += 1
        if self.batches == 1 or time.perf_counter() - self.last >= 30:
            self.report(logger)

    def report(self, logger, final=False):
        self.last = time.perf_counter()
        elapsed = self.last - self.started
        logger.info('[分类效率] %s | %s batch=%s, 批次=%s, 切片=%s, 有效=%s, '
                    '累计=%.2fs, 平均=%.2f切片/s；等数据=%.2fs, '
                    'CPU整理及缩放=%.2fs, 预测含传输及恢复=%.2fs, 写出及消息=%.2fs；'
                    '细分: 结果整理及日志=%.2fs, TIFF读写及刷盘=%.2fs, 消息发送=%.2fs；'
                    '整理细分: 掩膜及类别处理=%.2fs, 无效像素检查=%.2fs, 日志及进度整理=%.2fs' % (
                        self.filename, '结束' if final else '进度', self.batch_size,
                        self.batches, self.samples, self.valid, elapsed,
                        self.samples / max(elapsed, 1e-9), self.wait,
                        self.prepare, self.infer, self.write,
                        self.output, self.raster, self.message,
                        self.mask, self.exception, self.log))



def cpu_mask_array(value):
    """Convert CPU masks once per batch; avoid NumPy indexing via Torch masks."""
    import numpy as np
    return value.numpy() if hasattr(value, 'numpy') else np.asarray(value)


def prepare_output_exception_mask(reader, raw, read_origin, row, column,
                                  tile_height, tile_width, overlap, value):
    """Exact original write-window mask, computed before RGB transforms.

    Reuse the raw input window only when it contains the entire write window.
    Otherwise read that exact window with the caller's private GDAL reader.
    Return a fixed-size mask so edge/tail samples can use default_collate.
    """
    import numpy as np
    crop_height, crop_width = tile_height - overlap, tile_width - overlap
    x, y = int(column) * crop_width, int(row) * crop_height
    width = min(crop_width, reader.width - x)
    height = min(crop_height, reader.height - y)
    if width <= 0 or height <= 0:
        raise RuntimeError('Invalid classification output window')
    rx, ry = read_origin
    if (raw is not None and raw.ndim == 3 and x >= rx and y >= ry
            and x + width <= rx + raw.shape[1] and y + height <= ry + raw.shape[0]):
        pixels = raw[y - ry:y - ry + height, x - rx:x - rx + width]
    else:
        pixels = reader.read_image(read_range=(x, y, width, height))
    if (pixels is None or pixels.ndim != 3 or pixels.shape[:2] != (height, width)
            or pixels.shape[2] != len(reader.band_list)):
        raise RuntimeError('Cannot read classification exception-mask window')
    # Same all-bands equality rule, without an H x W x C promoted temporary.
    invalid = pixels[:, :, 0] == value
    for band in range(1, pixels.shape[2]):
        invalid &= pixels[:, :, band] == value
    result = np.zeros((tile_height, tile_width), dtype=np.uint8)
    result[:height, :width] = invalid
    return result


def code_fingerprint(root, include_pipeline=True):
    """Same v1 ordering; the new helper is also covered by the new fingerprint."""
    root = Path(root)
    sources = [root / 'classification_core.py', root / 'classification_schema.py']
    sources += sorted((root / 'clie_new').rglob('*.py'))
    sources += sorted((root / 'clie_new').rglob('*.yaml'))
    sources += sorted((root / 'clie_new').rglob('*.yml'))
    digest = hashlib.sha256()
    for path in sources:
        digest.update(str(path.relative_to(root)).encode())
        digest.update(hashlib.sha256(path.read_bytes()).hexdigest().encode())
    helper = root.parent / 'change' / 'roi_scope.py'
    if helper.is_file():
        digest.update(b'change/roi_scope.py')
        digest.update(hashlib.sha256(helper.read_bytes()).hexdigest().encode())
    if include_pipeline:
        digest.update(b'classification_pipeline.py')
        digest.update(hashlib.sha256((root / 'classification_pipeline.py').read_bytes()).hexdigest().encode())
    return digest.hexdigest()


def previous_algorithms(root, current):
    try:
        value = json.loads((Path(root) / COMPAT_FILE).read_text(encoding='utf-8'))
        if value.get('release') != RELEASE or value.get('current_algorithm') != current:
            return []
        hashes = value['previous_algorithms']
        if not isinstance(hashes, list) or any(not isinstance(h, str) or not re.fullmatch('[a-f0-9]{64}', h) for h in hashes):
            return []
        return hashes
    except (OSError, ValueError, TypeError, AttributeError, KeyError):
        return []


def identities_match(saved, current, root):
    if saved == current:
        return True
    if not isinstance(saved, dict) or not isinstance(current, dict):
        return False
    old_run, new_run = saved.get('run'), current.get('run')
    if not isinstance(old_run, dict) or not isinstance(new_run, dict):
        return False
    # Only a reviewed updater-produced code transition may differ. Input/model,
    # mask, batch, downsample and ROI content must still match exactly.
    if old_run.get('algorithm') not in previous_algorithms(root, new_run.get('algorithm')):
        return False
    equivalent = dict(saved)
    equivalent['run'] = dict(old_run, algorithm=new_run['algorithm'])
    return equivalent == current
