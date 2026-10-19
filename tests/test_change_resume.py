import ast
import copy
import importlib.util
import json
import logging
import os
from pathlib import Path
import queue
import shutil
import sys
import tempfile
import time
import types
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'change'))
sys.path.insert(0, str(ROOT))
import resume_checkpoint as rc


class ResumeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.old = self.root / 'output' / '7052' / 'working' / 'change_detection'
        self.new = self.root / 'output' / '9000' / 'working' / 'change_detection'
        self.old_records = self.root / 'task_diagnostics' / '7052' / 'logs' / 'change_detection_checkpoints'
        self.new_records = self.root / 'task_diagnostics' / '9000' / 'logs' / 'change_detection_checkpoints'
        self.shp = self.old / 'shp' / '影像.shp'
        self.shp.parent.mkdir(parents=True)
        for ext in rc.REQUIRED + ('.cpg',):
            self.shp.with_suffix(ext).write_bytes(('test' + ext).encode())
        self.pre = self.root / 'pre' / '影像.tif'
        self.post = self.root / 'post' / '影像.tif'
        for path in (self.pre, self.post):
            path.parent.mkdir()
            path.write_bytes(b'input')
        self.model = self.root / 'primary.pth'
        self.model.write_bytes(b'model')
        self.identity = rc.pair_identity(self.pre, self.post, {'model': 'sha', 'dual': False})
        self.task = dict(idx=0, stem='影像', pre_path=str(self.pre), post_path=str(self.post),
                         model_path=str(self.model), second_model_path=None, use_two_models=False,
                         output_shp=str(self.new / 'shp' / self.shp.name),
                         checkpoint_dir=str(self.new_records), checkpoint_identity=self.identity)

    def copy_shp(self, source, target):
        Path(target).parent.mkdir(parents=True, exist_ok=True)
        for ext in rc.SIDECARS:
            path = Path(source).with_suffix(ext)
            if path.is_file():
                shutil.copy2(path, Path(target).with_suffix(ext))

    def save(self, count=0):
        rc.save_checkpoint(self.old_records, '影像', self.identity, self.shp, count)

    def restore(self, legacy=None, allow=False, count=0, copier=None):
        return rc.restore(self.task, self.old, self.old_records, legacy or {}, allow,
                          copier or self.copy_shp, validator=lambda _: count)

    def test_cross_task_restore_valid_empty_shp_without_mask_tiff(self):
        self.save()
        original = rc.sidecar_hashes(self.shp)
        result = self.restore()
        self.assertTrue(result['resumed'])
        self.assertEqual(result['feature_count'], 0)
        self.assertEqual(rc.sidecar_hashes(self.shp), original)
        self.assertEqual(rc.sidecar_hashes(self.task['output_shp']), original)
        self.assertTrue(rc.record_path(self.new_records, '影像').is_file())

    def test_nonempty_and_feature_count_mismatch(self):
        self.save(3)
        with self.assertRaisesRegex(ValueError, '图斑数'):
            self.restore(count=0)
        self.assertEqual(self.restore(count=3)['feature_count'], 3)

    def test_input_or_model_or_code_change_rejects(self):
        self.save()
        self.task['checkpoint_identity'] = copy.deepcopy(self.identity)
        self.task['checkpoint_identity']['run']['model'] = 'new model'
        with self.assertRaisesRegex(ValueError, '指纹不匹配'):
            self.restore()
        self.pre.write_bytes(b'changed input')
        self.task['checkpoint_identity'] = rc.pair_identity(self.pre, self.post, self.identity['run'])
        with self.assertRaisesRegex(ValueError, '指纹不匹配'):
            self.restore()

    def test_input_changed_during_processing_cannot_commit(self):
        self.pre.write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, '输入影像发生变化'):
            self.save()
        self.assertFalse(rc.record_path(self.old_records, '影像').exists())

    def test_corrupt_missing_and_truncated_sidecar_rejected(self):
        self.save()
        self.shp.with_suffix('.dbf').write_bytes(b'corrupt')
        with self.assertRaisesRegex(ValueError, '指纹不匹配'):
            self.restore()
        self.shp.with_suffix('.shx').unlink()
        with self.assertRaisesRegex(ValueError, '缺失'):
            self.restore()
        self.shp.with_suffix('.shx').write_bytes(b'')
        with self.assertRaisesRegex(ValueError, '零字节'):
            self.restore()

    def test_interrupted_copy_does_not_publish_checkpoint(self):
        self.save()
        def partial(source, target):
            Path(target).parent.mkdir(parents=True)
            shutil.copy2(source, target)
            raise OSError('disk full')
        with self.assertRaises(OSError):
            self.restore(copier=partial)
        self.assertFalse(rc.record_path(self.new_records, '影像').exists())

    def test_same_directory_restore_does_not_copy_onto_source(self):
        self.save()
        self.task['output_shp'] = str(self.shp)
        self.task['checkpoint_dir'] = str(self.old_records)
        copier = mock.Mock(side_effect=AssertionError('must not copy'))
        self.restore(copier=copier)
        copier.assert_not_called()

    def test_invalid_manifest_never_falls_back_to_legacy(self):
        self.old_records.mkdir(parents=True)
        rc.record_path(self.old_records, '影像').write_text('{broken', encoding='utf-8')
        with self.assertRaises(ValueError):
            self.restore(allow=True)

    def legacy_log(self, suffix=''):
        log = self.old_records.parent / 'change_detection_gpu_1.log'
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text(
            f'2026-10-16 01:00:00,000 - Processing (1/2) on GPU 1: pre={self.pre}, post={self.post} -> {self.shp}\n'
            f'2026-10-16 01:00:01,000 - 变化检测主模型: {self.model}\n'
            '2026-10-16 01:00:01,001 - 变化检测使用单模型模式\n'
            '2026-10-16 01:00:03,000 - 处理 影像 完成：未检测到变化图斑，输出有效空 SHP\n'
            + suffix, encoding='utf-8')
        return rc.legacy_completions(log.parent)

    def test_legacy_requires_opt_in_success_input_and_model_match(self):
        legacy = self.legacy_log()
        with self.assertRaisesRegex(ValueError, '未启用'):
            self.restore(legacy)
        wrong = copy.deepcopy(legacy)
        wrong['影像']['pre'] = 'another input'
        with self.assertRaisesRegex(ValueError, '旧任务日志'):
            self.restore(wrong, allow=True)
        result = self.restore(legacy, allow=True)
        self.assertEqual(result['provenance'], 'legacy_assumed_unchanged')

    def test_later_incomplete_attempt_invalidates_legacy_success(self):
        legacy = self.legacy_log(
            f'2026-10-17 01:00:00,000 - Processing (1/2) on GPU 1: pre={self.pre}, post={self.post} -> {self.shp}\n')
        self.assertFalse(legacy['影像']['completed'])
        with self.assertRaises(ValueError):
            self.restore(legacy, allow=True)

    def test_source_resolution_ignores_new_task_environment(self):
        with mock.patch.dict(os.environ, {'PATH_OUTPUT': str(self.root / 'output' / '9000')}):
            for value in ('7052', str(self.old), str(self.root / 'output' / '7052')):
                step, logs = rc.resolve_source(value, self.new)
                self.assertEqual(step, self.old.resolve())
                self.assertEqual(logs, self.old_records.parent)
        with self.assertRaises(ValueError):
            rc.resolve_source('not-an-absolute-path', self.new)

    def test_invalidate_only_exact_checkpoint(self):
        self.save()
        rc.invalidate(self.old_records, '影像')
        self.assertFalse(rc.record_path(self.old_records, '影像').exists())
        self.assertTrue(self.shp.exists())

    def test_run_fingerprint_tracks_weights_code_config_and_batch(self):
        module = self.root / 'module'
        module.mkdir()
        (module / 'test_lib_batch_memeff_single_image_nomp.py').write_text('algorithm', encoding='utf-8')
        config = module / 'config.cpython-39.so'
        config.write_bytes(b'config1')
        with mock.patch.object(rc, '__file__', str(module / 'resume_checkpoint.py')), mock.patch.dict(
                os.environ, {'CHANGE_DETECTION_BATCH_SIZE': '4'}):
            first = rc.run_identity(str(self.model), False, None)
            config.write_bytes(b'config2')
            second = rc.run_identity(str(self.model), False, None)
            self.assertNotEqual(first['algorithm'], second['algorithm'])
            self.model.write_bytes(b'new model')
            third = rc.run_identity(str(self.model), False, None)
            self.assertNotEqual(second['models'], third['models'])
            self.assertEqual(third['batch_size'], 4)

    def batch_namespace(self):
        source = (ROOT / 'change' / 'change_detection_core.py').read_text(encoding='utf-8')
        node = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef)
                    and n.name == 'change_detection_folder')
        sender = mock.Mock()
        sender.calc_progress_value.side_effect = lambda n, total, lo, hi: lo + (hi-lo)*n/total
        context = mock.Mock()
        reports = {}
        namespace = dict(os=os, Path=Path, time=time, queue=queue, prg_sender=sender,
                         logging=logging, multiprocessing=mock.Mock(get_context=lambda _: context),
                         _configure_persistent_logger=lambda *a: (logging.getLogger('test'), 'test.log'),
                         _ensure_log_dir=lambda _: str(self.new_records.parent),
                         _write_diagnostic_report=lambda dst, name, payload: reports.update({name: payload}),
                         _detect_change_gpu_count=mock.Mock(side_effect=AssertionError('no GPU needed')),
                         _copy_shapefile_dataset=self.copy_shp, swap_write=mock.Mock(),
                         _format_duration=str, _format_file_size=str,
                         _change_pair_worker=mock.Mock())
        exec(compile(ast.Module(body=[node], type_ignores=[]), '<batch>', 'exec'), namespace)
        return namespace, reports, context

    def test_batch_all_restored_never_starts_gpu_and_reports_current_output(self):
        namespace, reports, context = self.batch_namespace()
        restored = dict(type='result', status='completed', idx=0, stem='影像',
                        output_shp=self.task['output_shp'], feature_count=0, resumed=True)
        with mock.patch.object(rc, 'run_identity', return_value={}), mock.patch.object(rc, 'restore', return_value=restored):
            namespace['change_detection_folder'](str(self.pre.parent), str(self.post.parent),
                                                 str(self.model), str(self.new), None, resume_from='7052')
        context.Process.assert_not_called()
        self.assertEqual(reports['change_detection_resume.json']['resumed_count'], 1)
        self.assertEqual(reports['change_detection_summary.json']['newly_computed_count'], 0)
        namespace['swap_write'].assert_any_call('output_shp', str(self.new / 'shp'))

    def test_mixed_resume_only_queues_missing_pair(self):
        for directory in (self.pre.parent, self.post.parent):
            (directory / 'z.tif').write_bytes(b'next input')
        namespace, reports, context = self.batch_namespace()
        namespace['_detect_change_gpu_count'] = lambda: 2
        tasks, events = mock.Mock(), mock.Mock()
        context.Queue.side_effect = [tasks, events]
        process = mock.Mock(exitcode=None)
        process.is_alive.return_value = True
        context.Process.return_value = process
        restored = dict(type='result', status='completed', idx=0, stem='z',
                        output_shp=str(self.new / 'shp' / 'z.shp'), feature_count=0)
        events.get.return_value = dict(type='result', status='completed', idx=1,
                                       stem='影像', worker='GPU 1',
                                       output_shp=self.task['output_shp'], feature_count=2)
        # Sorted Unicode order: ASCII z precedes the Chinese image stem.
        with mock.patch.object(rc, 'run_identity', return_value={}), mock.patch.object(
                rc, 'restore', side_effect=[restored, ValueError('missing')]):
            namespace['change_detection_folder'](str(self.pre.parent), str(self.post.parent),
                                                 str(self.model), str(self.new), None, resume_from='7052')
        submitted = [call.args[0] for call in tasks.put.call_args_list if call.args[0] is not None]
        self.assertEqual([task['stem'] for task in submitted], ['影像'])
        self.assertEqual(context.Process.call_count, 1)
        summary = reports['change_detection_summary.json']
        self.assertEqual((summary['resumed_count'], summary['newly_computed_count']), (1, 1))

    def test_abnormal_worker_exit_fails_batch_instead_of_completed(self):
        namespace, reports, context = self.batch_namespace()
        namespace['_detect_change_gpu_count'] = lambda: 1
        process = mock.Mock(exitcode=70)
        process.is_alive.return_value = False
        context.Process.return_value = process
        with mock.patch.object(rc, 'run_identity', return_value={}):
            with self.assertRaisesRegex(RuntimeError, 'exitcode=70'):
                namespace['change_detection_folder'](str(self.pre.parent), str(self.post.parent),
                                                     str(self.model), str(self.new), None)
        self.assertNotIn('change_detection_summary.json', reports)
        context.Queue.return_value.cancel_join_thread.assert_called()

    def test_live_worker_no_progress_times_out(self):
        namespace, reports, context = self.batch_namespace()
        namespace['_detect_change_gpu_count'] = lambda: 1
        namespace['time'] = types.SimpleNamespace(monotonic=mock.Mock(side_effect=[0, 1801]))
        process = mock.Mock(exitcode=None)
        process.is_alive.return_value = True
        context.Process.return_value = process
        with mock.patch.object(rc, 'run_identity', return_value={}), mock.patch.dict(
                os.environ, {'CHANGE_DETECTION_STALL_TIMEOUT': '1800'}):
            with self.assertRaisesRegex(RuntimeError, '无进度'):
                namespace['change_detection_folder'](str(self.pre.parent), str(self.post.parent),
                                                     str(self.model), str(self.new), None)
        process.terminate.assert_called_once()


if __name__ == '__main__':
    unittest.main()
