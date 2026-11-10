import ast
import copy
from contextlib import nullcontext
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import threading
import types
import unittest
from unittest import mock

ROOT = Path(os.environ.get('CLASSIFICATION_TEST_ROOT', Path(__file__).resolve().parents[1]))
spec = importlib.util.spec_from_file_location('tested_classification_pipeline', ROOT / 'fenlei/classification_pipeline.py')
p = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p)


class PipelineTests(unittest.TestCase):
    def test_cpu_settings_defaults_limits_and_invalid_values(self):
        torch = mock.Mock()
        self.assertEqual(p.configure_cpu_pipeline(torch, 64, True, {}), (2, 4))
        torch.set_num_threads.assert_called_with(4)
        self.assertEqual(p.configure_cpu_pipeline(torch, 2, True, {'CLIE_PREFETCH_THREADS': '4', 'CLIE_TORCH_THREADS': '4'}), (2, 2))
        self.assertEqual(p.configure_cpu_pipeline(torch, 64, False, {}), (0, 4))
        for name in ('CLIE_PREFETCH_THREADS', 'CLIE_TORCH_THREADS'):
            for value in ('-1', 'invalid', '1.5', 'True', '300'):
                with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                    p.configure_cpu_pipeline(torch, 64, True, {name: value})
        self.assertEqual(p.configure_cpu_pipeline(torch, 64, True, {'CLIE_PREFETCH_THREADS': '0'}), (0, 4))

    def test_order_tail_private_readers_main_thread_collation_and_cleanup(self):
        main = threading.get_ident()
        instances = []
        class Dataset:
            def __init__(self):
                self.owner = threading.get_ident()
                self.closed = False
                instances.append(self)
            def __getitem__(self, i):
                self_test.assertEqual(self.owner, threading.get_ident())
                return i
            def close(self):
                self.closed = True
        def collate(samples):
            self.assertEqual(main, threading.get_ident())
            return samples
        self_test = self
        loader = p.ThreadedBatchLoader(Dataset, 19, 4, 4, collate)
        with p.managed_batches(loader):
            self.assertEqual(list(loader), [list(range(i, min(i + 4, 19))) for i in range(0, 19, 4)])
        self.assertEqual(len(loader), 5)
        self.assertTrue(instances)
        self.assertTrue(all(d.closed for d in instances))
        self.assertLessEqual(len(instances), 4)

    def test_prefetch_overlaps_consumer_and_is_bounded_on_early_exit(self):
        barrier = threading.Barrier(2)
        prefetched = threading.Event()
        visited, closed = [], []
        class Dataset:
            def __getitem__(self, i):
                visited.append(i)
                if i < 2:
                    barrier.wait(timeout=5)
                else:
                    prefetched.set()
                return i
            def close(self):
                closed.append(True)
        loader = p.ThreadedBatchLoader(Dataset, 10000, 2, 2, list)
        with p.managed_batches(loader):
            self.assertEqual(next(iter(loader)), [0, 1])
            self.assertTrue(prefetched.wait(timeout=5))
        self.assertLessEqual(len(visited), 4)
        self.assertTrue(closed)

    def test_read_and_collation_errors_close_all_readers(self):
        for failure in ('read', 'collate'):
            closed = []
            class Dataset:
                def __getitem__(self, i):
                    if failure == 'read' and i == 2:
                        raise RuntimeError('read failed')
                    return i
                def close(self):
                    closed.append(True)
            def collate(items):
                if failure == 'collate':
                    raise RuntimeError('collation failed')
                return items
            with self.subTest(failure=failure), self.assertRaises(RuntimeError):
                with p.managed_batches(p.ThreadedBatchLoader(Dataset, 8, 4, 2, collate)) as loader:
                    list(loader)
            self.assertTrue(closed)

    def test_empty_input_and_metrics(self):
        logger = mock.Mock()
        metrics = p.ClassificationMetrics('a.tif', 16)
        with p.managed_batches(p.ThreadedBatchLoader(lambda: None, 0, 16, 2, list)) as loader:
            self.assertEqual(list(p.timed_batches(loader, metrics)), [])
        metrics.record(16, 12, logger)
        metrics.report(logger, final=True)
        self.assertEqual((metrics.samples, metrics.valid), (16, 12))
        self.assertIn('写出及消息=', logger.info.call_args.args[0])

    def test_real_dataset_clone_owns_image_mask_roi_and_close(self):
        tree = ast.parse((ROOT / 'fenlei/clie_new/CLInferEngine/clie_lib/dataset.py').read_text(encoding='utf-8'))
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'InferDataMgr')
        cls.body = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in ('clone_for_thread', 'reopen_readers', 'close')]
        cls.bases = []
        readers = []
        class Reader:
            def __init__(self, filename):
                self.filename, self.closed = filename, False
                readers.append(self)
            def set_band_list(self, bands): pass
            def close(self): self.closed = True
        scope = {'ImageReader': Reader}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[])), '<dataset-clone>', 'exec'), scope)
        original = scope['InferDataMgr']()
        original.filename, original.requested_band_list, original.mask_filename = 'image', [1, 2, 3], 'mask'
        original.image_reader, original.mask_reader = Reader('original'), Reader('original-mask')
        original.roi_wkt, original.roi = 'polygon', object()
        original._open_roi = types.MethodType(lambda self: setattr(self, 'roi', object()), original)
        # Use an ordinary method so the cloned object receives the clone as self.
        scope['InferDataMgr']._open_roi = lambda self: setattr(self, 'roi', object())
        del original._open_roi
        clone = original.clone_for_thread()
        self.assertIsNot(original.image_reader, clone.image_reader)
        self.assertIsNot(original.mask_reader, clone.mask_reader)
        self.assertIsNot(original.roi, clone.roi)
        clone.close()
        self.assertFalse(original.image_reader.closed)
        self.assertFalse(original.mask_reader.closed)


class CompatibilityTests(unittest.TestCase):
    def test_only_exact_reviewed_transition_is_accepted(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            old, new = '1' * 64, '2' * 64
            saved = {'pre': {'size': 8}, 'post': {'mtime': 9}, 'mask': {'.shp': 'mask'},
                     'run': {'algorithm': old, 'models': {'m': 'sha'}, 'batch_per_gpu': 16,
                             'downsample': 2, 'roi': {'sha': 'scope'}}}
            current = copy.deepcopy(saved)
            current['run']['algorithm'] = new
            self.assertFalse(p.identities_match(saved, current, root))
            (root / p.COMPAT_FILE).write_text(json.dumps({'release': p.RELEASE,
                'current_algorithm': new, 'previous_algorithms': [old]}), encoding='utf-8')
            self.assertTrue(p.identities_match(saved, current, root))
            for key, value in [('models', {'m': 'different'}), ('batch_per_gpu', 32), ('downsample', 1), ('roi', {'sha': 'different'})]:
                changed = copy.deepcopy(current)
                changed['run'][key] = value
                self.assertFalse(p.identities_match(saved, changed, root))
            for key in ('pre', 'post', 'mask'):
                changed = copy.deepcopy(current)
                changed[key] = {'different': True}
                self.assertFalse(p.identities_match(saved, changed, root))
            changed = copy.deepcopy(current)
            changed['run']['algorithm'] = '3' * 64
            self.assertFalse(p.identities_match(saved, changed, root))

    def test_bad_metadata_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for value in ('[]', '{', '{}', '{"previous_algorithms": "anything"}'):
                (root / p.COMPAT_FILE).write_text(value, encoding='utf-8')
                self.assertEqual(p.previous_algorithms(root, 'x'), [])


if __name__ == '__main__':
    unittest.main()
