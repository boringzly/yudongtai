"""Exact, installer-recorded compatibility for already completed change results."""
import copy
import hashlib
import json
from pathlib import Path
import re

RELEASE = 'change-roi-recovery-v1'
COMPAT_FILE = 'change_resume_compat.json'


def code_fingerprint(root):
    """Preserve the original ordering; new recovery dependencies join the hash."""
    root = Path(root)
    sources = [root / 'test_lib_batch_memeff_single_image_nomp.py']
    for name in ('inference_pipeline.py', 'roi_scope.py', 'resume_compat.py', 'postprocess_checkpoint.py'):
        if (root / name).is_file():
            sources.append(root / name)
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
        code.update(hashlib.sha256(path.read_bytes()).hexdigest().encode('ascii'))
    return code.hexdigest()


def previous_algorithms(root, current):
    try:
        data = json.loads((Path(root) / COMPAT_FILE).read_text(encoding='utf-8'))
        values = data.get('previous_algorithms')
        if (data.get('release') != RELEASE or data.get('current_algorithm') != current
                or not isinstance(values, list)
                or not all(isinstance(v, str) and re.fullmatch('[0-9a-f]{64}', v) for v in values)):
            return []
        return values
    except (OSError, ValueError, TypeError, AttributeError):
        return []


def identities_match(saved, current, root=None):
    if saved == current:
        return True
    root = Path(root) if root is not None else Path(__file__).resolve().parent
    if not isinstance(saved, dict) or not isinstance(current, dict):
        return False
    before, after = saved.get('run'), current.get('run')
    if not isinstance(before, dict) or not isinstance(after, dict):
        return False
    actual = code_fingerprint(root)
    if after.get('algorithm') != actual or before.get('algorithm') not in previous_algorithms(root, actual):
        return False
    candidate = copy.deepcopy(saved)
    candidate['run']['algorithm'] = after['algorithm']
    return candidate == current
