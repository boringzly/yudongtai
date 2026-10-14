"""Lightweight wall-clock telemetry; no CUDA synchronization or extra worker threads."""
import os
import time


def change_batch_size():
    try:
        value = int(os.environ.get('CHANGE_DETECTION_BATCH_SIZE', '4'))
        return value if 1 <= value <= 32 else 4
    except (TypeError, ValueError):
        return 4


class InferenceMetrics:
    def __init__(self, logger, name, batch_size, cuda=None, clock=time.perf_counter):
        self.logger, self.name, self.batch_size = logger, name, batch_size
        self.cuda, self.clock = cuda, clock
        self.started = clock()
        self.loop_started = None
        self.last_log = self.started
        self.last_processed = 0
        self.processed = self.valid = self.batches = 0
        self.total = 0
        self.seconds = dict(wait=0.0, prepare=0.0, inference=0.0, write=0.0)
        if cuda is not None:
            try:
                cuda.reset_peak_memory_stats()
            except Exception:
                # Optional telemetry must never abort a valid task (including HAMi).
                self.cuda = None

    def info(self, message, *args):
        if self.logger is not None:
            self.logger.info('[效率] %s | ' + message, self.name, *args)

    def start_loop(self, total):
        self.total = total
        self.loop_started = self.last_log = self.clock()
        self.info('开始切片循环 batch=%s, 总切片=%s, 模型加载及栅格准备=%.2fs；等待首批数据',
                  self.batch_size, total, self.loop_started - self.started)

    def iter_batches(self, loader):
        # Include worker startup and teardown in exposed data waiting time.
        start = self.clock()
        iterator = iter(loader)
        while True:
            try:
                batch = next(iterator)
            except StopIteration:
                self.seconds['wait'] += self.clock() - start
                return
            self.seconds['wait'] += self.clock() - start
            yield batch
            start = self.clock()

    def add(self, stage, started):
        self.seconds[stage] += self.clock() - started

    def complete_batch(self, processed, valid):
        self.processed += processed
        self.valid += valid
        self.batches += 1
        if self.batches == 1 or self.clock() - self.last_log >= 30:
            self.report('切片进度')

    def memory(self):
        if self.cuda is None:
            return '峰值显存=N/A'
        try:
            allocated = self.cuda.max_memory_allocated() / 1024 ** 3
            reserved = self.cuda.max_memory_reserved() / 1024 ** 3
            return 'PyTorch峰值显存: allocated=%.2fGiB, reserved=%.2fGiB' % (allocated, reserved)
        except Exception:
            self.cuda = None
            return '峰值显存=N/A'

    def report(self, label):
        now = self.clock()
        elapsed = max(now - self.loop_started, 1e-9)
        interval = max(now - self.last_log, 1e-9)
        self.info('%s batch=%s, 已处理=%s/%s, 有效推理=%s, 跳过=%s, '
                  '累计=%.2fs, 平均=%.2f切片/s, 有效=%.2f切片/s, 近期=%.2f切片/s；'
                  '累计阶段: 等数据=%.2fs, CPU整理=%.2fs, 推理含传输=%.2fs, '
                  '写TIFF含掩膜裁切=%.2fs；%s',
                  label, self.batch_size, self.processed, self.total, self.valid,
                  self.processed - self.valid, elapsed, self.processed / elapsed,
                  self.valid / elapsed, (self.processed - self.last_processed) / interval,
                  self.seconds['wait'], self.seconds['prepare'], self.seconds['inference'],
                  self.seconds['write'], self.memory())
        self.last_log, self.last_processed = now, self.processed

    def phase_done(self, phase, started):
        self.info('%s耗时=%.2fs', phase, self.clock() - started)

    def finish(self):
        self.info('单对影像算法完成=%.2fs（含模型加载、切片和矢量后处理；不含本地暂存及结果回传）；%s',
                  self.clock() - self.started, self.memory())
