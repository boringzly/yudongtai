import ast
import contextlib
import copy
import io
import json
import logging
import os
from pathlib import Path
import runpy
import subprocess
import sys
import tempfile
import time
import types
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fenlei import classification_resume as cr
from fenlei import classification_runtime as rt


def fake_shp(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    for ext in ('.shp', '.shx', '.dbf', '.prj', '.cpg'):
        path.with_suffix(ext).write_bytes(('test-' + ext).encode())


class ClassificationConfigTests(unittest.TestCase):
    def test_options_are_independent_and_defaults_disabled(self):
        self.assertEqual(cr.resolve_options(environ={}), ('', False))
        env = {'WF_CONFIG': json.dumps({'config': {'resume_from': '7052', 'resume_allow_legacy': True}})}
        self.assertEqual(cr.resolve_options(environ=env), ('', False))
        env['WF_CONFIG'] = json.dumps({'config': {
            'classification_resume_from': '7057', 'classification_resume_allow_legacy': True}})
        self.assertEqual(cr.resolve_options(environ=env), ('7057', True))
        self.assertEqual(cr.resolve_options('', False, env), ('', False))
        env.update(CLASSIFICATION_RESUME_FROM='99', CLASSIFICATION_RESUME_ALLOW_LEGACY='false')
        self.assertEqual(cr.resolve_options(environ=env), ('99', False))

    def test_invalid_config_rejected(self):
        for raw in ('[]', '{"config":[]}', 'bad', '{"config":{"classification_resume_from":true}}',
                    '{"config":{"classification_resume_allow_legacy":"wrong"}}'):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                cr.resolve_options(environ={'WF_CONFIG': raw})

    def test_cli_passes_config_without_new_external_env_fields(self):
        core = types.ModuleType('classification_core')
        core.entry = mock.Mock()
        env = {'WF_CONFIG': '{"config":{"classification_resume_from":"7057","classification_resume_allow_legacy":true}}'}
        argv = ['main.py', '--pre_image', '/pre', '--post_image', '/post', '--mask_shp', '/mask', '--dst_path', '/new']
        with mock.patch.dict(sys.modules, {'classification_core': core}), mock.patch.dict(os.environ, env, clear=True), \
                mock.patch.object(sys, 'argv', argv), contextlib.redirect_stdout(io.StringIO()):
            runpy.run_path(str(ROOT / 'fenlei/main.py'), run_name='__main__')
        self.assertEqual(core.entry.call_args.args[-2:], ('7057', True))
        for filename in ('change_detection_wf.json', 'change_detection_wf_resume_7052.json'):
            wf = json.loads((ROOT / filename).read_text(encoding='utf-8'))
            for key in ('classification_resume_from', 'classification_resume_allow_legacy'):
                self.assertIn(key, wf['config'])
                self.assertNotIn(key, wf['input']['parameters'])
                self.assertNotIn(key, wf['steps']['change_classification']['parameters']['input'])


class ClassificationResumeTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.old = self.root / 'output/7057/working/change_classification'
        self.new = self.root / 'output/8000/working/change_classification'
        self.old_logs = self.root / 'task_diagnostics/7057/logs'
        self.logs = self.root / 'task_diagnostics/8000/logs'
        self.shp = self.old / 'shp_cls/pair.shp'
        self.output = self.new / 'shp_cls/pair.shp'
        fake_shp(self.shp)
        self.output.parent.mkdir(parents=True)
        self.mask = self.root / 'mask/pair.shp'
        fake_shp(self.mask)
        self.pre = self.root / 'pre/pair.tif'
        self.post = self.root / 'post/pair.tif'
        for path in (self.pre, self.post):
            path.parent.mkdir()
            path.write_bytes(b'input')
        self.run = {'models': {'model.pth': 'sha'}, 'algorithm': 'v1', 'downsample': 2,
                    'model_name': 'model.pth', 'batch_per_gpu': 8, 'vector_wkt': '', 'filter_class_id': ''}
        self.identity = cr.pair_identity(self.pre, self.post, self.mask, self.run)
        patch = mock.patch.object(cr, 'validate_result', return_value=3)
        self.validate = patch.start()
        self.addCleanup(patch.stop)
        self.logger = logging.getLogger('classification-resume-test')

    def manager(self, source='7057', legacy=False):
        return cr.Checkpoints(self.new, self.output.parent, self.logs, self.run, source, legacy)

    def save_old(self, provenance='computed'):
        old = cr.Checkpoints(self.old, self.shp.parent, self.old_logs, self.run)
        old.save('pair', self.identity, self.shp, self.mask, provenance)

    def restore(self, manager=None, identity=None):
        return (manager or self.manager()).restore('pair', identity or self.identity, self.output, self.mask, self.logger)

    def test_cross_task_copies_validated_shp_without_tiffs_and_preserves_old(self):
        self.save_old()
        original = cr.sidecar_hashes(self.shp)
        manager = self.manager()
        self.assertEqual(self.restore(manager), 3)
        self.assertEqual(manager.resumed, 1)
        self.assertEqual(cr.sidecar_hashes(self.output), original)
        self.assertEqual(cr.sidecar_hashes(self.shp), original)
        self.assertTrue(cr.record_path(manager.directory, 'pair').is_file())
        self.assertEqual(self.restore(self.manager(source='')), 3)

    def test_valid_empty_is_a_completed_pair(self):
        self.validate.return_value = 0
        self.save_old()
        self.assertEqual(self.restore(), 0)

    def test_identity_changes_reject_even_when_legacy_enabled(self):
        self.save_old()
        for key, value in (('models', {'x': 'new'}), ('downsample', 1), ('batch_per_gpu', 4), ('algorithm', 'v2')):
            identity = copy.deepcopy(self.identity)
            identity['run'][key] = value
            with self.subTest(key=key):
                self.assertIsNone(self.restore(self.manager(legacy=True), identity))
        identity = copy.deepcopy(self.identity)
        identity['mask']['.dbf'] = 'changed'
        self.assertIsNone(self.restore(identity=identity))
        self.pre.write_bytes(b'new input')
        self.assertIsNone(self.restore(identity=cr.pair_identity(self.pre, self.post, self.mask, self.run)))

    def test_corrupt_results_and_uncommitted_files_not_reused(self):
        self.assertIsNone(self.restore())
        self.save_old()
        self.shp.with_suffix('.dbf').write_bytes(b'corrupt')
        self.assertIsNone(self.restore())
        self.shp.with_suffix('.prj').unlink()
        self.assertIsNone(self.restore())

    def test_mask_change_during_inference_cannot_commit(self):
        self.mask.with_suffix('.dbf').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, '图斑已改变'):
            self.save_old()
        self.assertFalse(cr.record_path(self.old_logs / 'classification_checkpoints', 'pair').exists())

    def test_corrupt_manifest_is_recomputed_not_fatal(self):
        self.save_old()
        cr.record_path(self.old_logs / 'classification_checkpoints', 'pair').write_text('[]')
        self.assertIsNone(self.restore())

    def test_failed_copy_does_not_commit(self):
        self.save_old()
        with mock.patch.object(cr.shutil, 'copy2', side_effect=OSError('disk full')), self.assertRaises(OSError):
            self.restore()
        self.assertFalse(cr.record_path(self.logs / 'classification_checkpoints', 'pair').exists())

    def legacy_log(self, empty=False, interrupted=False, factor=2):
        self.old_logs.mkdir(parents=True, exist_ok=True)
        processing = f'Processing (1/1): pre={self.pre}, post={self.post}, mask={self.mask} -> {self.shp}\n'
        content = '分类模型: /models/model.pth\n' + processing
        content += (f'mask为空，没有变化区域，保留有效空分类结果: pair\n' if empty else
                    f'下采样倍数={factor}\n下采样倍数={factor}\n处理完成，结果保存至：{self.shp}\n')
        if interrupted:
            content += processing
        (self.old_logs / 'classification.log').write_text(content, encoding='utf-8')

    def test_legacy_requires_explicit_permission_and_keeps_provenance(self):
        self.legacy_log()
        self.assertIsNone(self.restore())
        self.assertEqual(self.restore(self.manager(legacy=True)), 3)
        record = json.loads(cr.record_path(self.logs / 'classification_checkpoints', 'pair').read_text())
        self.assertEqual(record['provenance'], 'legacy_assumed_unchanged')
        self.restore(self.manager(source=''))
        self.assertEqual(json.loads(cr.record_path(self.logs / 'classification_checkpoints', 'pair').read_text())['provenance'],
                         'legacy_assumed_unchanged')

    def test_legacy_latest_incomplete_attempt_and_wrong_downsample_rejected(self):
        self.legacy_log(interrupted=True)
        self.assertIsNone(self.restore(self.manager(legacy=True)))
        self.legacy_log(factor=1)
        self.assertIsNone(self.restore(self.manager(legacy=True)))

    def test_legacy_empty_requires_empty_input_mask(self):
        self.legacy_log(empty=True)
        self.validate.return_value = 0
        with mock.patch.object(cr, 'validate_shp', return_value=2):
            self.assertIsNone(self.restore(self.manager(legacy=True)))
        with mock.patch.object(cr, 'validate_shp', return_value=0):
            self.assertEqual(self.restore(self.manager(legacy=True)), 0)

    def test_sources_and_invalid_source_fail_explicitly(self):
        for source in (str(self.old), str(self.old.parents[1])):
            self.assertEqual(cr.resolve_source(source, self.new)[0], self.old)
        with self.assertRaises(ValueError):
            self.manager(source='does-not-exist')
        with self.assertRaises(ValueError):
            self.manager(source=str(self.new))

    def test_all_restored_batch_skips_inference_and_merges_only_new_paths(self):
        self.save_old()
        source = (ROOT / 'fenlei/classification_core.py').read_text(encoding='utf-8')
        tree = ast.parse(source)
        node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'classification_folder')
        sender = mock.Mock()
        sender.calc_progress_value.side_effect = lambda i, total, low, high: low + i * (high-low) / total
        namespace = dict(os=os, sys=sys, Path=Path, prg_sender=sender, __file__=str(ROOT / 'fenlei/classification_core.py'))
        for name in ('swap_write', '_write_diagnostic_report', '_cleanup_clie_working_rasters',
                     '_cleanup_classification_temp_dir', '_send_completed', 'assign_province_names',
                     '_run_subprocess_logged', '_remove_shapefile_dataset', '_generate_result_previews'):
            namespace[name] = mock.Mock()
        namespace.update(_configure_persistent_logger=lambda *args: (self.logger, str(self.logs / 'classification.log')),
                         _ensure_log_dir=lambda *args: str(self.logs),
                         _resolve_classification_model=lambda *args: '/model.pth',
                         _create_classification_temp_dir=lambda: str(self.root / 'tmp'),
                         _cleanup_change_mask_tifs=lambda *args: {'removed_count': 0, 'failed_files': []})
        exec(compile(ast.Module(body=[node], type_ignores=[]), '<batch>', 'exec'), namespace)
        merged = self.root / 'output/8000/out/classification_result.shp'
        merge = types.ModuleType('merge')
        merge.merge_shp = mock.Mock(side_effect=lambda paths, output: (fake_shp(output), output)[1])
        gpd = types.ModuleType('geopandas')
        gpd.read_file = mock.Mock(side_effect=AssertionError('recovered pairs must not be inferred'))
        with mock.patch.dict(sys.modules, {'merge': merge, 'geopandas': gpd,
                                          'rasterio': types.ModuleType('rasterio'), 'numpy': types.ModuleType('numpy')}), \
                mock.patch.object(cr, 'run_identity', return_value=self.run):
            namespace['classification_folder'](self.pre.parent, self.post.parent, self.mask.parent,
                                               '/model.pth', str(merged.parent), str(self.new), None, '7057', False)
        namespace['_run_subprocess_logged'].assert_not_called()
        namespace['_remove_shapefile_dataset'].assert_not_called()
        self.assertEqual(merge.merge_shp.call_args.args[0], [str(self.output)])
        self.assertTrue(merged.exists())


class ClassificationRuntimeTests(unittest.TestCase):
    def run_child(self, script, env=None):
        environment = dict(os.environ)
        environment.update(CLIE_STALL_TIMEOUT='3', CLIE_PROCESS_TIMEOUT='5')
        environment.update(env or {})
        return rt.run_logged([sys.executable, '-u', '-c', script], str(ROOT), environment,
                             logging.getLogger('classification-runtime-test'))

    def test_success_and_nonzero_exit(self):
        self.assertEqual(self.run_child("print('ordinary warning')"), 0)
        with self.assertRaisesRegex(rt.ClassificationRuntimeError, '返回码=7'):
            self.run_child("print('traceback detail'); raise SystemExit(7)")

    def test_worker_hami_and_oom_messages_fail_even_if_child_keeps_running(self):
        for error in ('DataLoader worker (pid 123) exited unexpectedly',
                      '[HAMI-core ERROR] lock timeout', 'CUDA out of memory', 'DataLoader timed out after 120 seconds'):
            with self.subTest(error=error), self.assertRaises(rt.ClassificationRuntimeError):
                self.run_child(f'import time; print({error!r}); time.sleep(30)')

    def test_silent_hang_and_noisy_hang_bounded(self):
        start = time.monotonic()
        with self.assertRaises(rt.ClassificationRuntimeError):
            self.run_child('import time; time.sleep(30)', {'CLIE_STALL_TIMEOUT': '.3'})
        with self.assertRaises(rt.ClassificationRuntimeError):
            self.run_child('import time\nwhile True:\n print("alive"); time.sleep(.05)', {'CLIE_PROCESS_TIMEOUT': '.4'})
        self.assertLess(time.monotonic() - start, 6)

    def test_hami_forces_zero_workers_without_removing_preload(self):
        # Inspect child env without trying to preload Linux libraries on Windows.
        with mock.patch.object(rt, 'under_hami', return_value=True), \
                mock.patch.object(rt.subprocess, 'Popen', wraps=subprocess.Popen) as popen:
            self.run_child('import os; assert os.environ["CLIE_DATA_WORKERS"] == "0"', {'CLIE_DATA_WORKERS': '8'})
        self.assertEqual(popen.call_args.kwargs['env'].get('LD_PRELOAD'), os.environ.get('LD_PRELOAD'))

    def test_direct_engine_worker_selection_hami_and_nonhami(self):
        path = ROOT / 'fenlei/clie_new/CLInferEngine/clie_lib/run_lib.py'
        node = next(n for n in ast.parse(path.read_text(encoding='utf-8')).body
                    if isinstance(n, ast.FunctionDef) and n.name == '_recommended_data_workers')
        namespace = {'os': os, '_available_cpu_count': lambda: 60}
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), namespace)
        with mock.patch.dict(os.environ, {}, clear=True), mock.patch.object(os.path, 'exists', return_value=False):
            self.assertEqual(namespace['_recommended_data_workers'](2), 8)
            with mock.patch.dict(os.environ, {'LD_PRELOAD': '/usr/lib/libvgpu.so', 'CLIE_DATA_WORKERS': '8'}):
                self.assertEqual(namespace['_recommended_data_workers'](2), 0)

    def test_hami_zonal_stats_does_not_spawn_pool(self):
        tree = ast.parse((ROOT / 'fenlei/classification_core.py').read_text(encoding='utf-8'))
        node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_map_zonal_chunks')
        namespace = {'os': os, '_process_chunk': lambda item: item * 2}
        exec(compile(ast.Module(body=[node], type_ignores=[]), '<zonal>', 'exec'), namespace)
        with mock.patch.object(rt, 'under_hami', return_value=True):
            self.assertEqual(namespace['_map_zonal_chunks']([1, 2], 2), [2, 4])

    def test_bad_timeouts_fail_before_launch(self):
        for value in ('nan', '-1', 'no', 'inf'):
            with self.subTest(value=value), self.assertRaises(rt.ClassificationRuntimeError), \
                    mock.patch.object(rt.subprocess, 'Popen') as launch:
                self.run_child('print(1)', {'CLIE_STALL_TIMEOUT': value})
            launch.assert_not_called()

    def test_linux_cleanup_targets_only_the_owned_process_group(self):
        process = mock.Mock(pid=12345)
        with mock.patch.object(rt.os, 'name', 'posix'), \
                mock.patch.object(rt.os, 'killpg', create=True) as kill, \
                mock.patch.object(rt.signal, 'SIGKILL', 9, create=True):
            rt._stop_owned_processes(process)
        self.assertEqual([call.args[0] for call in kill.call_args_list], [12345, 12345])

    def test_infrastructure_error_sends_failed_and_is_reraised(self):
        tree = ast.parse((ROOT / 'fenlei/classification_core.py').read_text(encoding='utf-8'))
        names = {'entry', '_is_nonfatal_warning', 'NonFatalTaskWarning'}
        nodes = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names]
        namespace = {'os': os, 'traceback': __import__('traceback')}
        for name in ('_write_diagnostic_report', '_send_failed', '_send_completed',
                     'swap_write', '_cleanup_all_classification_temp_dirs', 'init_progress_message_sender',
                     'init_progress_message_title', 'init_progress_message_source', 'prg_sender'):
            namespace[name] = mock.Mock()
        namespace['_configure_persistent_logger'] = lambda *args: (mock.Mock(), 'test.log')
        namespace['classification_folder'] = mock.Mock(side_effect=rt.ClassificationRuntimeError('HAMi ERROR bigtiff'))
        exec(compile(ast.Module(body=nodes, type_ignores=[]), '<entry>', 'exec'), namespace)
        with mock.patch.object(os.path, 'isdir', return_value=True), contextlib.redirect_stdout(io.StringIO()), \
                self.assertRaises(rt.ClassificationRuntimeError):
            namespace['entry']('/pre', '/post', '/mask', '/model', '/dst', '/work', None, None, None, '', '', '')
        namespace['_send_failed'].assert_called_once()
        namespace['_send_completed'].assert_not_called()
        namespace['_cleanup_all_classification_temp_dirs'].assert_called_once()
        namespace['prg_sender'].close.assert_called_once()


if __name__ == '__main__':
    unittest.main()
