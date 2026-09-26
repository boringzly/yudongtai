import ast
import json
import logging
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from task_artifacts import diagnostic_root, ensure_log_dir
sys.path.pop(0)


class TaskArtifactTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.project = Path(self.temp.name) / 'project'
        self.output = self.project / 'output' / '6978'
        self.expected = self.project / 'task_diagnostics' / '6978'
        self.env = mock.patch.dict(os.environ, {}, clear=True)
        self.env.start()
        self.addCleanup(self.env.stop)

    def test_both_steps_share_one_root_outside_whole_output_tree(self):
        for dst in (self.output / 'working' / 'change_detection', self.output / 'out', self.output,
                    self.output / 'out' / 'result_step'):
            with self.subTest(dst=dst):
                self.assertEqual(diagnostic_root(dst), self.expected)
                self.assertEqual(ensure_log_dir(dst), str(self.expected / 'logs'))
                self.assertFalse(self.output.exists())

    def test_workflow_environment_has_priority_for_custom_step_names(self):
        for key in ('PATH_OUTPUT', 'path_output', 'DATA_OUTPUT_DIR'):
            with self.subTest(key=key), mock.patch.dict(os.environ, {key: str(self.output)}):
                self.assertEqual(diagnostic_root(self.output / 'custom_name'), self.expected)
                self.assertEqual(diagnostic_root(self.project / 'scratch' / 'change'), self.expected)

    def test_standalone_and_working_paths_without_named_output_container(self):
        root = self.project / 'task-abc'
        expected = self.project / 'task_diagnostics' / 'task-abc'
        for dst in (root, root / 'out', root / 'working' / 'step'):
            self.assertEqual(diagnostic_root(dst), expected)

    def test_output_root_itself_is_not_used_for_diagnostics(self):
        output_container = self.project / 'output'
        with mock.patch.dict(os.environ, {'PATH_OUTPUT': str(output_container)}):
            self.assertEqual(diagnostic_root(output_container / 'result'), self.project / 'task_diagnostics' / 'root')

    def test_override_isolated_by_task_and_project(self):
        external = self.project / 'diagnostic_archive'
        env = {'CD_ARTIFACTS_ROOT': str(external), 'PATH_OUTPUT': str(self.output)}
        first = diagnostic_root(self.output / 'out', env)
        second = diagnostic_root(self.output / 'working' / 'change', env)
        self.assertEqual(first, second)
        self.assertEqual(first.parent, external)
        self.assertTrue(first.name.startswith('6978-'))
        other = diagnostic_root(self.output, dict(env, PATH_OUTPUT=str(self.project / 'other' / 'output' / '6978')))
        self.assertNotEqual(first, other)

    def test_bad_overrides_cannot_pollute_output(self):
        for bad in (self.output, self.output / 'logs', self.project / 'output' / 'auxiliary'):
            with self.subTest(bad=bad), self.assertRaisesRegex(ValueError, '不能位于输出目录内'):
                diagnostic_root(self.output / 'out', {'PATH_OUTPUT': str(self.output), 'CD_ARTIFACTS_ROOT': str(bad)})
        with self.assertRaisesRegex(ValueError, '绝对路径'):
            diagnostic_root(self.output, {'CD_ARTIFACTS_ROOT': 'relative'})

    @staticmethod
    def load_helpers(path):
        names = {'_ensure_log_dir', '_configure_persistent_logger', '_write_diagnostic_report'}
        tree = ast.parse(path.read_text(encoding='utf-8'))
        functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
        scope = {'__file__': str(path), 'sys': sys, 'Path': Path, 'os': os, 'json': json, 'logging': logging}
        exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), 'exec'), scope)
        return scope

    def test_actual_core_logs_workers_reports_leave_results_untouched(self):
        self.output.mkdir(parents=True)
        shp = self.output / 'result.shp'
        shp.write_bytes(b'existing result')
        with mock.patch.dict(os.environ, {'PATH_OUTPUT': str(self.output)}):
            for module, filename in [('change/change_detection_core.py', 'change_detection.log'),
                                     ('change/change_detection_core.py', 'change_detection_gpu_2.log'),
                                     ('fenlei/classification_core.py', 'classification.log')]:
                helpers = self.load_helpers(ROOT / module)
                logger, log_path = helpers['_configure_persistent_logger']('test.' + filename, self.output, filename)
                try:
                    logger.warning('persistent outside output')
                    self.assertEqual(Path(log_path).parent, self.expected / 'logs')
                    report_path = helpers['_write_diagnostic_report'](self.output, filename + '.json', {'ok': True})
                    self.assertEqual(Path(report_path).parent, self.expected / 'logs')
                    self.assertEqual(json.loads(Path(report_path).read_text()), {'ok': True})
                finally:
                    for handler in list(logger.handlers):
                        logger.removeHandler(handler)
                        handler.close()
        self.assertEqual(list(self.output.iterdir()), [shp])
        self.assertEqual(shp.read_bytes(), b'existing result')

    def test_old_logs_are_not_deleted_or_silently_moved(self):
        old = self.output / 'out' / 'logs'
        old.mkdir(parents=True)
        (old / 'classification.log').write_text('historical evidence')
        ensure_log_dir(self.output / 'out')
        self.assertEqual((old / 'classification.log').read_text(), 'historical evidence')


if __name__ == '__main__':
    unittest.main()
