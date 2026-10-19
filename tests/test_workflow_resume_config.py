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
MAIN = ROOT / 'change' / 'main.py'


class WorkflowResumeConfigTests(unittest.TestCase):
    def setUp(self):
        self.entry = mock.Mock()
        self.core = types.ModuleType('change_detection_core')
        self.core.entry = self.entry
        with mock.patch.dict(sys.modules, {'change_detection_core': self.core}):
            self.resolve = runpy.run_path(str(MAIN))['resolve_resume_options']

    def test_workflow_config_alone_no_new_environment_variables(self):
        env = {'WF_CONFIG': json.dumps({'config': {'resume_from': '7052', 'resume_allow_legacy': True}})}
        self.assertEqual(self.resolve(environ=env), ('7052', True))

    def test_missing_and_disabled_default(self):
        for env in ({}, {'WF_CONFIG': '{}'}, {'WF_CONFIG': '{"config":{"fast_run":false}}'}):
            self.assertEqual(self.resolve(environ=env), ('', False))

    def test_explicit_empty_and_false_cli_override_config(self):
        env = {'WF_CONFIG': '{"config":{"resume_from":"7052","resume_allow_legacy":true}}'}
        self.assertEqual(self.resolve('', False, env), ('', False))

    def test_environment_precedence_and_string_bool(self):
        env = {'WF_CONFIG': '{"config":{"resume_from":"7052","resume_allow_legacy":true}}',
               'CHANGE_DETECTION_RESUME_FROM': '9999', 'resume_allow_legacy': 'false'}
        self.assertEqual(self.resolve(environ=env), ('9999', False))

    def test_bad_values_fail_instead_of_silent_full_recompute(self):
        for raw in ('broken', '[]', '{"config":[]}',
                    '{"config":{"resume_from":true}}',
                    '{"config":{"resume_allow_legacy":"invalid"}}'):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                self.resolve(environ={'WF_CONFIG': raw})

    def test_templates_do_not_declare_external_resume_inputs(self):
        for filename, expected in (('change_detection_wf.json', ('', False)),
                                   ('change_detection_wf_resume_7052.json', ('7052', True))):
            wf = json.loads((ROOT / filename).read_text(encoding='utf-8'))
            for name in ('resume_from', 'resume_allow_legacy'):
                self.assertNotIn(name, wf['input']['parameters'])
                self.assertNotIn(name, wf['steps']['change_detection']['parameters']['input'])
            self.assertEqual(self.resolve(environ={'WF_CONFIG': json.dumps(wf)}), expected)

    def test_actual_cli_passes_workflow_values_to_entry_without_resume_args(self):
        wf = (ROOT / 'change_detection_wf_resume_7052.json').read_text(encoding='utf-8')
        argv = ['main.py', '--pre_image', '/pre', '--post_image', '/post', '--dst_path', '/output/new']
        with mock.patch.dict(sys.modules, {'change_detection_core': self.core}), \
                mock.patch.dict(os.environ, {'WF_CONFIG': wf}, clear=True), \
                mock.patch.object(sys, 'argv', argv), contextlib.redirect_stdout(io.StringIO()) as output:
            runpy.run_path(str(MAIN), run_name='__main__')
        self.assertEqual(self.entry.call_args.kwargs['resume_from'], '7052')
        self.assertIs(self.entry.call_args.kwargs['resume_allow_legacy'], True)
        self.assertIn("resume_from='7052'", output.getvalue())


if __name__ == '__main__':
    unittest.main()
