"""Resolve per-GPU batch settings before CUDA workers/CLIE are started.

Engine 6.1 requires env vars for input.parameters, so these optional controls
belong in WF_CONFIG.config, not in declared external inputs.
"""
import json
import os
import re


SETTINGS = {
    'change': ('change_batch_size', 'CHANGE_DETECTION_BATCH_SIZE', 4, 32),
    'classification': ('classification_batch_per_gpu', 'CLIE_BATCH_PER_GPU', 8, 64),
}

CHANGE_PIPELINE_SETTINGS = {
    'change_prefetch_threads': ('CHANGE_DETECTION_PREFETCH_THREADS', 0, 8),
    'change_torch_threads': ('CHANGE_DETECTION_TORCH_THREADS', 1, 256),
}


CLASSIFICATION_PIPELINE_SETTINGS = {
    'classification_prefetch_threads': ('CLIE_PREFETCH_THREADS', 0, 8),
    'classification_torch_threads': ('CLIE_TORCH_THREADS', 1, 256),
}


def _apply_pipeline_config(config, env, settings):
    # Preserve inference defaults when optional fields are absent. Existing env
    # overrides workflow values, matching batch configuration precedence.
    for key, (env_key, minimum, maximum) in settings.items():
        if env_key in env:
            value, source = env[env_key], env_key
        elif key in config:
            value, source = config[key], 'WF_CONFIG.config'
        else:
            continue
        if (isinstance(value, bool) or not isinstance(value, (int, str))
                or not re.fullmatch(r'[0-9]+', str(value).strip())):
            raise ValueError('%s 必须是 %s～%s 的整数，收到 %r' % (key, minimum, maximum, value))
        threads = int(value)
        if not minimum <= threads <= maximum:
            raise ValueError('%s 必须在 %s～%s 之间，收到 %s' % (key, minimum, maximum, threads))
        env[env_key] = str(threads)
        print('[数据供给配置] %s=%s（来源=%s）' % (key, threads, source), flush=True)


def apply_batch_config(stage, explicit=None, environ=None):
    """CLI > existing environment > workflow config > backward-compatible default.

Write the resolved value back to the inherited environment so both inference
and checkpoint signatures use exactly the same batch. No CUDA imports needed.
"""
    env = os.environ if environ is None else environ
    key, env_key, default, maximum = SETTINGS[stage]
    try:
        workflow = json.loads(env.get('WF_CONFIG') or '{}')
    except (ValueError, TypeError) as exc:
        raise ValueError('无法读取 WF_CONFIG batch 配置: %s' % exc) from exc
    if not isinstance(workflow, dict) or not isinstance(workflow.get('config', {}), dict):
        raise ValueError('WF_CONFIG 及其 config 必须是 JSON 对象')
    config = workflow.get('config', {})
    if explicit is not None:
        value, source = explicit, 'CLI'
    elif env_key in env:
        value, source = env[env_key], env_key
    elif key in config:
        value, source = config[key], 'WF_CONFIG.config'
    else:
        value, source = default, 'default'
    # Do not silently clamp a requested batch: logs must reflect the request.
    if isinstance(value, bool) or not isinstance(value, (int, str)) or not re.fullmatch(r'[0-9]+', str(value).strip()):
        raise ValueError('%s 必须是 1～%s 的整数，收到 %r' % (key, maximum, value))
    batch = int(value)
    if not 1 <= batch <= maximum:
        raise ValueError('%s 必须在 1～%s 之间，收到 %s' % (key, maximum, batch))
    env[env_key] = str(batch)
    if stage == 'change':
        _apply_pipeline_config(config, env, CHANGE_PIPELINE_SETTINGS)
    elif stage == 'classification':
        _apply_pipeline_config(config, env, CLASSIFICATION_PIPELINE_SETTINGS)
    print('[Batch配置] %s=%s（每张 GPU 的切片数，来源=%s）' % (key, batch, source), flush=True)
    return batch
