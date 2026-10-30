import ast
import json
import logging
import os
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch, MagicMock

ROOT = Path(__file__).resolve().parents[1]
tree = ast.parse((ROOT / 'change/change_detection_core.py').read_text(encoding='utf-8'))
names = {'_configure_change_memory', '_change_memory_snapshot', '_log_change_memory'}
functions = ast.Module(body=[node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names], type_ignores=[])
ns = {'os': os, 'json': json, 'Path': Path}
exec(compile(functions, 'actual_memory_guard', 'exec'), ns)


class MemoryGuardTests(unittest.TestCase):
    def setUp(self):
        self.cache = 64 * 1024 ** 3
        self.gdal = types.SimpleNamespace(GetCacheMax=lambda: self.cache, SetCacheMax=self.set_cache, GetCacheUsed=lambda: 0)
        self.logger = MagicMock()
        self.modules = patch.dict('sys.modules', {'osgeo': types.SimpleNamespace(gdal=self.gdal)})
        self.modules.start()
        self.env = patch.dict(os.environ, {}, clear=True); self.env.start()

    def tearDown(self):
        self.env.stop(); self.modules.stop()

    def set_cache(self, value): self.cache = value

    def test_default_overrides_host_sized_cache(self):
        ns['_configure_change_memory'](self.logger)
        self.assertEqual(self.cache, 512 * 1024 ** 2)

    def test_workflow_override(self):
        os.environ['WF_CONFIG'] = json.dumps({'config': {'change_gdal_cache_mb': 1024}})
        ns['_configure_change_memory'](self.logger)
        self.assertEqual(self.cache, 1024 * 1024 ** 2)

    def test_explicit_environment_precedes_workflow(self):
        os.environ.update(WF_CONFIG=json.dumps({'config': {'change_gdal_cache_mb': 1024}}), CHANGE_DETECTION_GDAL_CACHE_MB='256')
        ns['_configure_change_memory'](self.logger)
        self.assertEqual(self.cache, 256 * 1024 ** 2)

    def test_invalid_cache_rejected_before_changing_gdal(self):
        for value in [True, 0, -1, 4097, '1.5', 'oops']:
            with self.subTest(value=value):
                os.environ['WF_CONFIG'] = json.dumps({'config': {'change_gdal_cache_mb': value}})
                with self.assertRaises(ValueError): ns['_configure_change_memory'](self.logger)
                self.assertEqual(self.cache, 64 * 1024 ** 3)

    def test_v2_ram_is_separate_from_peak(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); (root / 'self').mkdir()
            (root / 'self/status').write_text('VmRSS: 1024 kB\nVmHWM: 2048 kB\n')
            (root / 'memory.current').write_text('1234'); (root / 'memory.max').write_text(str(64 * 1024 ** 3))
            snap = ns['_change_memory_snapshot'](root, root)
            self.assertEqual(snap, {'VmRSS': 1024 ** 2, 'VmHWM': 2 * 1024 ** 2, 'current': 1234, 'limit': 64 * 1024 ** 3})

    def test_v1_and_unlimited(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); (root / 'memory').mkdir()
            (root / 'memory/memory.usage_in_bytes').write_text('5678')
            (root / 'memory/memory.limit_in_bytes').write_text(str(2 ** 63 - 4096))
            self.assertEqual(ns['_change_memory_snapshot'](root, root), {'current': 5678})

    def test_missing_proc_files_are_nonfatal(self):
        self.assertEqual(ns['_change_memory_snapshot'](ROOT/'__missing_memory_test__', ROOT/'__missing_memory_test__'), {})

    def test_telemetry_failure_does_not_abort_inference(self):
        self.gdal.GetCacheUsed = MagicMock(side_effect=RuntimeError('unavailable'))
        ns['_log_change_memory'](self.logger, 'test')
        self.logger.debug.assert_called_once()

    def test_cap_set_before_importing_inference_in_worker(self):
        worker = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_process_change_pair')
        calls = [n for n in ast.walk(worker) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == '_configure_change_memory']
        imports = [n for n in ast.walk(worker) if isinstance(n, ast.ImportFrom) and n.module == 'test_lib_batch_memeff_single_image_nomp']
        self.assertEqual(len(calls), 1)
        self.assertLess(calls[0].lineno, imports[0].lineno)

    def test_single_image_path_also_capped(self):
        single = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'change_detection')
        self.assertTrue(any(isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == '_configure_change_memory' for n in ast.walk(single)))


if __name__ == '__main__': unittest.main()
