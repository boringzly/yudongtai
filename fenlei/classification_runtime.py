"""Bounded CLIE execution. Never remove HAMi's injected resource limits."""
import os
import queue
import signal
import subprocess
import threading
import time


class ClassificationRuntimeError(RuntimeError):
    """Infrastructure failure: stop the batch, retaining completed checkpoints."""


def under_hami(env):
    preload = env.get('LD_PRELOAD', '').lower()
    return ('libvgpu' in preload or 'hami' in preload
            or any(env.get(key) for key in (
                'CUDA_DEVICE_MEMORY_LIMIT', 'CUDA_DEVICE_SM_LIMIT',
                'CUDA_DEVICE_MEMORY_SHARED_CACHE', 'HAMI_VISIBLE_DEVICES'))
            or os.path.exists('/tmp/cudevshr.cache'))


def _seconds(env, key, default):
    try:
        value = float(env.get(key, default))
    except (ValueError, TypeError) as exc:
        raise ClassificationRuntimeError('%s 必须是有限正数' % key) from exc
    if not 0 < value < float('inf'):
        raise ClassificationRuntimeError('%s 必须是有限正数' % key)
    return value


def _stop_owned_processes(process):
    # Each CLIE invocation has its own session on Linux. Include DataLoader
    # descendants even if their direct parent has already exited.
    if os.name == 'posix':
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        try:
            process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            pass
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    elif process.poll() is None:
        process.kill()
    try:
        process.wait(timeout=3)
    except subprocess.TimeoutExpired:
        pass  # Kernel uninterruptible I/O cannot be repaired here.


def run_logged(command, cwd, env, logger):
    child_env = dict(env)
    child_env['PYTHONUNBUFFERED'] = '1'
    if under_hami(child_env):
        child_env['CLIE_DATA_WORKERS'] = '0'
        logger.info('[HAMi保护] 分类 data_workers=0；保留 GPU 数量和 batch 配置')
    stall = _seconds(child_env, 'CLIE_STALL_TIMEOUT', 1800)
    maximum = _seconds(child_env, 'CLIE_PROCESS_TIMEOUT', 7200)
    logger.info('启动分类子进程: %s；无输出超时=%ss，总超时=%ss', command, stall, maximum)
    process = subprocess.Popen(
        command, cwd=cwd, env=child_env, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='replace',
        bufsize=1, start_new_session=(os.name == 'posix'))
    lines = queue.Queue(maxsize=1024)
    stopped = threading.Event()

    def offer(value):
        while not stopped.is_set():
            try:
                lines.put(value, timeout=.2)
                return
            except queue.Full:
                pass

    def reader():
        try:
            for line in process.stdout:
                offer(line)
        finally:
            offer(None)
            process.stdout.close()

    thread = threading.Thread(target=reader, daemon=True)
    thread.start()
    start = last = time.monotonic()
    eof = False
    try:
        while True:
            now = time.monotonic()
            if now - start > maximum or now - last > stall:
                raise ClassificationRuntimeError(
                    '分类子进程超时（总耗时 %.0fs，无输出 %.0fs）；已完成组可续跑' %
                    (now - start, now - last))
            try:
                line = lines.get(timeout=.2)
            except queue.Empty:
                line = ''
            if line is None:
                eof = True
            elif line:
                last = time.monotonic()
                logger.info('[CLIE] %s', line.rstrip())
                lower = line.lower()
                fatal = (('hami-core' in lower and 'error' in lower)
                         or ('dataloader worker' in lower and ('exited' in lower or 'killed' in lower))
                         or 'dataloader timed out' in lower
                         or 'cuda out of memory' in lower)
                if fatal:
                    raise ClassificationRuntimeError('分类运行保护触发: ' + line.strip())
            code = process.poll()
            if code is not None and code != 0:
                # Drain the bounded pipe before reporting, to retain traceback.
                if eof:
                    raise ClassificationRuntimeError('分类子进程异常退出，返回码=%s；请查看前面的 CLIE 日志' % code)
            if eof and code == 0:
                return 0
    finally:
        stopped.set()
        _stop_owned_processes(process)
        thread.join(timeout=1)
