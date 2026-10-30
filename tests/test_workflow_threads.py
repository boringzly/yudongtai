import contextlib
import io
import json
import os
from pathlib import Path
import runpy
import sys
import types
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from workflow_batch_config import apply_batch_config, CHANGE_PIPELINE_SETTINGS


class WorkflowThreadTests(unittest.TestCase):
    def apply(self, config, overrides=None, stage='change'):
        env = {'WF_CONFIG': json.dumps({'config': config})}
        env.update(overrides or {})
        with contextlib.redirect_stdout(io.StringIO()):
            apply_batch_config(stage, environ=env)
        return env

    def test_workflow_values_are_exported_for_algorithm_children(self):
        env = self.apply({'change_batch_size': 16, 'change_prefetch_threads': 4, 'change_torch_threads': 4})
        self.assertEqual(env['CHANGE_DETECTION_BATCH_SIZE'], '16')
        self.assertEqual(env['CHANGE_DETECTION_PREFETCH_THREADS'], '4')
        self.assertEqual(env['CHANGE_DETECTION_TORCH_THREADS'], '4')

    def test_missing_fields_keep_inference_defaults(self):
        env = self.apply({})
        self.assertNotIn('CHANGE_DETECTION_PREFETCH_THREADS', env)
        self.assertNotIn('CHANGE_DETECTION_TORCH_THREADS', env)

    def test_environment_override_and_disable(self):
        env = self.apply({'change_prefetch_threads': 4, 'change_torch_threads': 4},
                         {'CHANGE_DETECTION_PREFETCH_THREADS': '0', 'CHANGE_DETECTION_TORCH_THREADS': '2'})
        self.assertEqual(env['CHANGE_DETECTION_PREFETCH_THREADS'], '0')
        self.assertEqual(env['CHANGE_DETECTION_TORCH_THREADS'], '2')

    def test_boundaries_and_invalid_values(self):
        for key, (env_key, minimum, maximum) in CHANGE_PIPELINE_SETTINGS.items():
            for value in (minimum, maximum, ' 4 '):
                self.assertEqual(self.apply({key: value})[env_key], str(int(value)))
            for value in (-1, maximum + 1, True, 1.5, '4.0', None, [], {}):
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    self.apply({key: value})
        with self.assertRaises(ValueError):
            self.apply({'change_torch_threads': 0})

    def test_classification_does_not_apply_change_settings(self):
        env = self.apply({'change_prefetch_threads': 'invalid'}, stage='classification')
        self.assertNotIn('CHANGE_DETECTION_PREFETCH_THREADS', env)

    def test_real_change_entry_exports_settings_before_calling_core(self):
        core = types.ModuleType('change_detection_core')
        captured = {}
        core.entry = lambda *args, **kwargs: captured.update(os.environ)
        argv = ['main.py', '--pre_image', '/pre', '--post_image', '/post', '--dst_path', '/out']
        env = {'WF_CONFIG': '{"config":{"change_batch_size":16,"change_prefetch_threads":4,"change_torch_threads":2}}'}
        with mock.patch.dict(sys.modules, {'change_detection_core': core}), \
                mock.patch.dict(os.environ, env, clear=True), \
                mock.patch.object(sys, 'argv', argv), contextlib.redirect_stdout(io.StringIO()):
            runpy.run_path(str(ROOT / 'change/main.py'), run_name='__main__')
        self.assertEqual(captured['CHANGE_DETECTION_PREFETCH_THREADS'], '4')
        self.assertEqual(captured['CHANGE_DETECTION_TORCH_THREADS'], '2')


if __name__ == '__main__':
    unittest.main()
