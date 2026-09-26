"""Locate auxiliary task files OUTSIDE result trees scanned by ingestion.

Shared by change detection, spawned GPU workers, classification and previews.
No dependency on GDAL/PyTorch and no mutation of previously generated results.
"""

import hashlib
import os
from pathlib import Path


def _within(path, root):
    return path == root or root in path.parents


def _task_output_root(dst_path, environ):
    # The workflow's task-level output root takes precedence over a step's dst.
    for key in ('PATH_OUTPUT', 'path_output', 'DATA_OUTPUT_DIR'):
        value = environ.get(key, '').strip()
        if value:
            path = Path(value).expanduser()
            if not path.is_absolute():
                raise ValueError('%s 必须是绝对路径，以免不同子进程定位到不同目录' % key)
            return path.resolve()

    dst = Path(dst_path).resolve()
    for candidate in (dst, *dst.parents):
        if candidate.name.lower() == 'working':
            return candidate.parent
    for container in dst.parents:
        if container.name.lower() in ('output', 'outputs'):
            return container / dst.relative_to(container).parts[0]
    if dst.name.lower() in ('out', 'result', 'results'):
        return dst.parent
    return dst


def diagnostic_root(dst_path, environ=None):
    """Resolve without creating directories; overrides cannot target output trees.

    /project/output/123/out -> /project/task_diagnostics/123
    /project/123/working/change_detection -> /project/task_diagnostics/123
    Explicit CD_ARTIFACTS_ROOT uses <task-name>-<path-hash> to isolate projects.
    """
    environ = os.environ if environ is None else environ
    dst = Path(dst_path).resolve()
    task_root = _task_output_root(dst, environ)
    output_containers = [candidate for candidate in (task_root, *task_root.parents, dst, *dst.parents)
                         if candidate.name.lower() in ('output', 'outputs')]
    configured = environ.get('CD_ARTIFACTS_ROOT', '').strip()
    if configured:
        base = Path(configured).expanduser()
        if not base.is_absolute():
            raise ValueError('CD_ARTIFACTS_ROOT 必须是绝对路径')
        digest = hashlib.sha256(str(task_root).encode('utf-8')).hexdigest()[:10]
        root = base / ('%s-%s' % (task_root.name or 'task', digest))
    else:
        container = next((candidate for candidate in (task_root, *task_root.parents)
                          if candidate.name.lower() in ('output', 'outputs')), None)
        if container is not None:
            relative = task_root.relative_to(container)
            root = container.parent / 'task_diagnostics' / (relative if relative.parts else 'root')
        else:
            root = task_root.parent / 'task_diagnostics' / (task_root.name or 'root')
    root = root.resolve()
    # resolve() also detects an existing diagnostics symlink pointing into output.
    if any(_within(root, forbidden) for forbidden in (task_root, dst, *output_containers)):
        raise ValueError('日志和预览目录不能位于输出目录内: %s；请将 CD_ARTIFACTS_ROOT 指向独立可写目录' % root)
    return root


def ensure_log_dir(dst_path):
    root = diagnostic_root(dst_path)
    path = (root / 'logs').resolve()
    if path.parent != root:
        raise ValueError('logs 不能通过符号链接指向诊断目录以外: %s' % path)
    path.mkdir(parents=True, exist_ok=True)
    return str(path)
