import ast
import contextlib
import io
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from workflow_batch_config import apply_batch_config, SETTINGS
from fenlei.classification_resume import run_identity


class WorkflowBatchTests(unittest.TestCase):
    def apply(self, stage, explicit=None, env=None):
        with contextlib.redirect_stdout(io.StringIO()):
            return apply_batch_config(stage, explicit, env)

    def test_defaults_remain_four_and_eight(self):
        env = {}
        self.assertEqual(self.apply('change', env=env), 4)
        self.assertEqual(self.apply('classification', env=env), 8)
        self.assertEqual(env, {'CHANGE_DETECTION_BATCH_SIZE': '4', 'CLIE_BATCH_PER_GPU': '8'})

    def test_workflow_values_set_child_environment(self):
        env = {'WF_CONFIG': json.dumps({'config': {'change_batch_size': 8, 'classification_batch_per_gpu': 16}})}
        self.assertEqual(self.apply('change', env=env), 8)
        self.assertEqual(self.apply('classification', env=env), 16)
        self.assertEqual(env['CHANGE_DETECTION_BATCH_SIZE'], '8')
        self.assertEqual(env['CLIE_BATCH_PER_GPU'], '16')

    def test_priority_cli_then_environment_then_workflow(self):
        for stage, (key, env_key, _, _) in SETTINGS.items():
            env = {'WF_CONFIG': json.dumps({'config': {key: 8}}), env_key: '2'}
            self.assertEqual(self.apply(stage, env=env), 2)
            self.assertEqual(self.apply(stage, explicit='4', env=env), 4)
            self.assertEqual(env[env_key], '4')

    def test_invalid_explicit_value_never_silently_falls_back(self):
        for stage, (key, env_key, _, maximum) in SETTINGS.items():
            for value in (0, -1, maximum + 1, True, 1.5, '4.0', '', None, [], {}):
                with self.subTest(stage=stage, value=value):
                    env = {'WF_CONFIG': json.dumps({'config': {key: value}})}
                    with self.assertRaises(ValueError):
                        self.apply(stage, env=env)
                    self.assertNotIn(env_key, env)

    def test_valid_boundaries_and_numeric_strings(self):
        for stage, (_, env_key, _, maximum) in SETTINGS.items():
            for value in (1, maximum, ' 8 '):
                env = {}
                self.assertEqual(self.apply(stage, explicit=value, env=env), int(value))
                self.assertEqual(env[env_key], str(int(value)))

    def test_malformed_workflow_fails(self):
        for raw in ('bad', '[]', '{"config":null}', '{"config":[]}'):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                self.apply('change', env={'WF_CONFIG': raw})

    def test_launch_log_reports_value_and_source(self):
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            apply_batch_config('classification', environ={'WF_CONFIG': '{"config":{"classification_batch_per_gpu":32}}'})
        self.assertIn('classification_batch_per_gpu=32', stream.getvalue())
        self.assertIn('WF_CONFIG.config', stream.getvalue())

    def test_both_real_cli_entries_apply_config_before_entry(self):
        for stage, main_path, core_name, args, env_key, expected in (
            ('change', 'change/main.py', 'change_detection_core', [], 'CHANGE_DETECTION_BATCH_SIZE', '8'),
            ('classification', 'fenlei/main.py', 'classification_core', ['--mask_shp', '/mask'], 'CLIE_BATCH_PER_GPU', '16'),
        ):
            core = types.ModuleType(core_name)
            observed = []
            core.entry = lambda *a, **kw: observed.append(os.environ.get(env_key))
            argv = ['main.py', '--pre_image', '/pre', '--post_image', '/post', '--dst_path', '/out'] + args
            env = {'WF_CONFIG': '{"config":{"change_batch_size":8,"classification_batch_per_gpu":16}}'}
            with self.subTest(stage=stage), mock.patch.dict(sys.modules, {core_name: core}), \
                    mock.patch.dict(os.environ, env, clear=True), mock.patch.object(sys, 'argv', argv), \
                    contextlib.redirect_stdout(io.StringIO()):
                runpy.run_path(str(ROOT / main_path), run_name='__main__')
            self.assertEqual(observed, [expected])

    def test_invalid_cli_fails_before_entry(self):
        for main_path, core_name, args in (
            ('change/main.py', 'change_detection_core', ['--change_batch_size', '0']),
            ('fenlei/main.py', 'classification_core', ['--mask_shp', '/mask', '--classification_batch_per_gpu', '65']),
        ):
            core = types.ModuleType(core_name)
            core.entry = mock.Mock()
            argv = ['main.py', '--pre_image', '/pre', '--post_image', '/post', '--dst_path', '/out'] + args
            with mock.patch.dict(sys.modules, {core_name: core}), mock.patch.dict(os.environ, {}, clear=True), \
                    mock.patch.object(sys, 'argv', argv), self.assertRaises(ValueError):
                runpy.run_path(str(ROOT / main_path), run_name='__main__')
            core.entry.assert_not_called()

    def test_inherited_environment_reaches_new_process(self):
        env = dict(os.environ)
        env.pop('CHANGE_DETECTION_BATCH_SIZE', None)
        env.pop('CLIE_BATCH_PER_GPU', None)
        env['WF_CONFIG'] = '{"config":{"change_batch_size":8,"classification_batch_per_gpu":32}}'
        self.apply('change', env=env)
        self.apply('classification', env=env)
        child = subprocess.run([sys.executable, '-B', '-c',
                                'import os; print(os.environ["CHANGE_DETECTION_BATCH_SIZE"], os.environ["CLIE_BATCH_PER_GPU"])'],
                               env=env, capture_output=True, text=True, timeout=10, check=True)
        self.assertEqual(child.stdout.strip(), '8 32')

    def test_classification_inference_and_resume_use_same_larger_batch(self):
        path = ROOT / 'fenlei/clie_new/CLInferEngine/clie_lib/run_lib.py'
        node = next(n for n in ast.parse(path.read_text(encoding='utf-8')).body
                    if isinstance(n, ast.FunctionDef) and n.name == '_classification_batch_per_gpu')
        namespace = {'os': os}
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), namespace)
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / 'model.pth'
            model.write_bytes(b'test weights')
            for batch in (8, 16, 32, 64):
                env = {'WF_CONFIG': json.dumps({'config': {'classification_batch_per_gpu': batch}})}
                with mock.patch.dict(os.environ, env, clear=True):
                    self.apply('classification')
                    self.assertEqual(namespace['_classification_batch_per_gpu'](), batch)
                    self.assertEqual(run_identity(model)['batch_per_gpu'], batch)

    def test_change_inference_uses_workflow_batch(self):
        metrics = runpy.run_path(str(ROOT / 'change/inference_metrics.py'))
        with mock.patch.dict(os.environ, {'WF_CONFIG': '{"config":{"change_batch_size":8}}'}, clear=True):
            self.apply('change')
            self.assertEqual(metrics['change_batch_size'](), 8)

    def test_templates_are_complete_and_do_not_declare_batch_as_external_inputs(self):
        for filename, expected in (
            ('change_detection_wf.json', (4, 8)),
            ('change_detection_wf_resume_7052.json', (4, 8)),
            ('change_detection_wf_single_gpu_80g.json', (8, 16)),
        ):
            wf = json.loads((ROOT / filename).read_text(encoding='utf-8'))
            self.assertEqual((wf['config']['change_batch_size'], wf['config']['classification_batch_per_gpu']), expected)
            for stage, (key, _, _, _) in SETTINGS.items():
                self.assertNotIn(key, wf['input']['parameters'])
                for step in wf['steps'].values():
                    self.assertNotIn(key, step['parameters']['input'])
                    self.assertEqual(step['process'], 'single')
            if '80g' in filename:
                self.assertEqual(wf['config']['resume_from'], '')
                self.assertEqual(wf['config']['classification_resume_from'], '')
                self.assertFalse(wf['config']['resume_allow_legacy'])
                self.assertFalse(wf['config']['classification_resume_allow_legacy'])


if __name__ == '__main__':
    unittest.main()
