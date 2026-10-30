import importlib.util
import ast
from contextlib import nullcontext
from pathlib import Path
import threading
import unittest
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('inference_pipeline', ROOT / 'change/inference_pipeline.py')
pipeline = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pipeline)


class PipelineTests(unittest.TestCase):
    def test_pack_matches_original_for_all_mixed_empty_and_tail_batches(self):
        rng = np.random.default_rng(45)
        for count in (1, 4, 16):
            for mode in ('valid', 'mixed', 'empty'):
                with self.subTest(count=count, mode=mode):
                    pre = rng.random((count, 3, 7, 9), dtype=np.float32)
                    post = rng.random(pre.shape, dtype=np.float32)
                    mask = rng.integers(0, 2, (count, 7, 9), dtype=np.uint8)
                    if mode == 'mixed':
                        mask[::2] = 0
                    elif mode == 'empty':
                        mask[:] = 0
                    indices = [np.arange(count) + 10 * i for i in range(4)]
                    old_valid = np.abs(mask).reshape(count, -1).sum(axis=1) != 0
                    expected = np.concatenate((pre[old_valid], post[old_valid]), axis=1)
                    actual, masks, offsets = pipeline.pack_valid_batch(pre, post, mask, indices)
                    np.testing.assert_array_equal(actual, expected)
                    np.testing.assert_array_equal(masks, mask[old_valid])
                    for got, original in zip(offsets, indices):
                        np.testing.assert_array_equal(got, original[old_valid])
                    self.assertTrue(actual.flags.c_contiguous)

    def test_loader_orders_samples_and_owns_readers_per_thread(self):
        datasets = []

        class Dataset:
            def __init__(self):
                self.owner, self.closed = threading.get_ident(), False
                datasets.append(self)

            def __getitem__(self, index):
                self_test.assertEqual(threading.get_ident(), self.owner)
                return index

            def close(self):
                self.closed = True

        self_test = self
        with pipeline.ThreadedBatchLoader(Dataset, 19, 4, 2, list) as loader:
            self.assertEqual(len(loader), 5)
            self.assertEqual(list(loader), [list(range(i, min(i + 4, 19))) for i in range(0, 19, 4)])
        self.assertTrue(all(dataset.closed for dataset in datasets))
        self.assertLessEqual(len(datasets), 2)

    def test_early_exit_cleans_up_and_bounds_prefetch(self):
        visited, closed = [], []

        class Dataset:
            def __getitem__(self, index):
                visited.append(index)
                return index

            def close(self):
                closed.append(True)

        with pipeline.ThreadedBatchLoader(Dataset, 10000, 4, 2, list) as loader:
            self.assertEqual(next(iter(loader)), [0, 1, 2, 3])
        self.assertLessEqual(len(visited), 8)
        self.assertTrue(closed)

    def test_cpu_work_is_concurrent_and_next_batch_runs_during_consumption(self):
        barrier = threading.Barrier(2)
        prefetched = threading.Event()

        class Dataset:
            def __getitem__(self, index):
                if index < 2:
                    barrier.wait(timeout=5)
                else:
                    prefetched.set()
                return index

        with pipeline.ThreadedBatchLoader(Dataset, 4, 2, 2, list) as loader:
            iterator = iter(loader)
            self.assertEqual(next(iterator), [0, 1])
            # Do not request another batch: loading must already run in background.
            self.assertTrue(prefetched.wait(timeout=5))
            self.assertEqual(next(iterator), [2, 3])

    def test_worker_error_propagates_and_closes_readers(self):
        closed = []

        class Dataset:
            def __getitem__(self, index):
                if index == 2:
                    raise ValueError('read failed')
                return index

            def close(self):
                closed.append(True)

        with self.assertRaisesRegex(ValueError, 'read failed'):
            with pipeline.ThreadedBatchLoader(Dataset, 10, 4, 2, list) as loader:
                list(loader)
        self.assertTrue(closed)

    def test_empty_loader(self):
        with pipeline.ThreadedBatchLoader(lambda: None, 0, 4, 2, list) as loader:
            self.assertEqual(list(loader), [])

    def test_actual_inference_loop_preserves_masks_offsets_tail_and_progress(self):
        source = ROOT / 'change/test_lib_batch_memeff_single_image_nomp.py'
        tree = ast.parse(source.read_text(encoding='utf-8'))
        block = next(node for node in ast.walk(tree) if isinstance(node, ast.With)
                     and isinstance(node.items[0].context_expr, ast.IfExp)
                     and isinstance(node.items[0].context_expr.test, ast.Name)
                     and node.items[0].context_expr.test.id == 'thread_workers')

        class CpuBatch:
            def __init__(self, values):
                self.values = values

            def size(self, dimension):
                return self.values.shape[dimension]

            def __array__(self, dtype=None, copy=None):
                return np.asarray(self.values, dtype=dtype)

        def batch(mask_values, base):
            mask = np.asarray(mask_values, dtype=np.uint8)
            images = np.arange(mask.shape[0] * 3 * 4).reshape(-1, 3, 2, 2).astype(np.float32)
            offsets = [np.arange(len(mask)) + base, np.arange(len(mask)) + 30,
                       np.zeros(len(mask), dtype=int), np.zeros(len(mask), dtype=int)]
            return CpuBatch(images), CpuBatch(images + 100), CpuBatch(mask), offsets

        batches = [batch([[[0, 0], [0, 0]]], 0),
                   batch([[[1, 0], [0, 1]], [[0, 0], [0, 0]]], 10),
                   batch([[[1, 1], [1, 1]]], 20)]
        meter = mock.Mock()
        meter.iter_batches.side_effect = iter
        writes, seen_inputs = [], []
        band = mock.Mock()
        band.WriteArray.side_effect = lambda pixels, x, y: writes.append((pixels.copy(), x, y))
        band.FlushCache.return_value = None
        gdal = mock.Mock(CE_Failure=3)
        gdal.GetLastErrorType.return_value = 0
        output = mock.Mock()
        output.GetRasterBand.return_value = band

        def infer(_net, images, _classes, _device):
            seen_inputs.append(images.copy())
            return np.ones((len(images), 2, 2), dtype=np.uint8)

        callback = mock.Mock()
        namespace = dict(thread_workers=0, nullcontext=nullcontext,
                         data_loader_test=batches, efficiency=meter,
                         pack_valid_batch=pipeline.pack_valid_batch, perf_counter=lambda: 0,
                         processed_patches=0, total_patches=4, local_rank=0,
                         pbar=mock.Mock(), progress_callback=callback, np=np,
                         cfg=mock.Mock(WITH_TTA=False, MODEL_NUM_CLASSES=1, TEST_PIXEL_OVERLAP=0),
                         test_normal=infer, net=None, net2=None, device='cpu',
                         out_data=output, gdal=gdal)
        exec(compile(ast.Module(body=[block], type_ignores=[]), str(source), 'exec'), namespace)
        self.assertEqual(namespace['processed_patches'], 4)
        self.assertEqual([call.args for call in callback.call_args_list], [(1, 4), (3, 4), (4, 4)])
        self.assertEqual([(x, y) for _, x, y in writes], [(10, 30), (20, 30)])
        np.testing.assert_array_equal(writes[0][0], [[1, 0], [0, 1]])
        np.testing.assert_array_equal(writes[1][0], np.ones((2, 2), dtype=np.uint8))
        self.assertEqual(len(seen_inputs), 2)


if __name__ == '__main__':
    unittest.main()
