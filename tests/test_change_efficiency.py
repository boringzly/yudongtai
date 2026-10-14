import ast
import importlib.util
import os
from pathlib import Path
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('change_metrics_test', ROOT / 'change/inference_metrics.py')
metrics = importlib.util.module_from_spec(spec)
spec.loader.exec_module(metrics)


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


class EfficiencyTests(unittest.TestCase):
    def test_batch_default_override_and_invalid_values(self):
        for value, expected in [(None, 4), ('1', 1), ('2', 2), ('4', 4), ('8', 8),
                                ('0', 4), ('-1', 4), ('33', 4), ('bad', 4)]:
            with self.subTest(value=value), mock.patch.dict(os.environ, {}, clear=True):
                if value is not None:
                    os.environ['CHANGE_DETECTION_BATCH_SIZE'] = value
                self.assertEqual(metrics.change_batch_size(), expected)

    def test_wait_includes_startup_but_excludes_processing(self):
        clock, logger = Clock(), mock.Mock()
        meter = metrics.InferenceMetrics(logger, 'a.tif', 4, clock=clock)
        meter.start_loop(6)

        class Loader:
            def __iter__(self):
                clock.now += 2  # iterator startup
                return self

            def __next__(self):
                clock.now += 3
                if clock.now > 30:
                    raise StopIteration
                return 'batch'

        iterator = meter.iter_batches(Loader())
        self.assertEqual(next(iterator), 'batch')
        self.assertEqual(meter.seconds['wait'], 5)
        start = clock()
        clock.now += 10
        meter.add('inference', start)
        meter.complete_batch(4, 3)
        self.assertEqual(next(iterator), 'batch')
        self.assertEqual(meter.seconds['wait'], 8)
        clock.now += 20
        meter.complete_batch(2, 0)  # short tail batch, entirely invalid
        self.assertEqual(list(iterator), [])
        self.assertEqual(meter.seconds['wait'], 11)
        self.assertEqual(meter.seconds['inference'], 10)
        self.assertEqual((meter.processed, meter.valid), (6, 3))
        meter.report('切片循环结束')
        text = logger.info.call_args.args[0] % logger.info.call_args.args[1:]
        self.assertIn('已处理=6/6', text)
        self.assertIn('跳过=3', text)
        self.assertIn('峰值显存=N/A', text)

    def test_first_batch_periodic_and_final_reports(self):
        clock, logger = Clock(), mock.Mock()
        meter = metrics.InferenceMetrics(logger, 'a.tif', 4, clock=clock)
        meter.start_loop(12)
        logger.reset_mock()
        clock.now = 3
        meter.complete_batch(4, 4)
        self.assertEqual(logger.info.call_count, 1)
        clock.now = 10
        meter.complete_batch(4, 4)
        self.assertEqual(logger.info.call_count, 1)
        clock.now = 33
        meter.complete_batch(4, 2)
        self.assertEqual(logger.info.call_count, 2)
        meter.report('切片循环结束')
        self.assertEqual(logger.info.call_count, 3)

    def test_gpu_peaks_reset_once_and_failure_is_optional(self):
        cuda = mock.Mock()
        cuda.max_memory_allocated.return_value = 3 * 1024 ** 3
        cuda.max_memory_reserved.return_value = 4 * 1024 ** 3
        meter = metrics.InferenceMetrics(None, 'a', 4, cuda)
        self.assertIn('allocated=3.00GiB, reserved=4.00GiB', meter.memory())
        cuda.reset_peak_memory_stats.assert_called_once()
        cuda.synchronize.assert_not_called()
        cuda.max_memory_allocated.side_effect = RuntimeError('HAMi stats unavailable')
        self.assertEqual(meter.memory(), '峰值显存=N/A')
        cuda.reset_peak_memory_stats.side_effect = RuntimeError('not supported')
        self.assertEqual(metrics.InferenceMetrics(None, 'a', 4, cuda).memory(), '峰值显存=N/A')

    def test_empty_loop_and_postprocessing(self):
        meter = metrics.InferenceMetrics(None, 'empty', 4, clock=Clock())
        meter.start_loop(0)
        self.assertEqual(list(meter.iter_batches([])), [])
        meter.report('empty')
        meter.phase_done('vector', 0)
        meter.finish()

    def test_loader_exception_not_swallowed(self):
        def broken():
            raise RuntimeError('DataLoader worker exited unexpectedly')
            yield
        meter = metrics.InferenceMetrics(None, 'a', 4)
        meter.start_loop(4)
        with self.assertRaisesRegex(RuntimeError, 'DataLoader worker'):
            list(meter.iter_batches(broken()))

    def test_real_loop_wiring_and_valid_index_filter(self):
        source = (ROOT / 'change/test_lib_batch_memeff_single_image_nomp.py').read_text(encoding='utf-8')
        ast.parse(source)
        self.assertIn("'TEST_BATCHES': change_batch_size()", source)
        self.assertIn('enumerate(efficiency.iter_batches(data_loader_test))', source)
        self.assertIn('efficiency.complete_batch(batch_patches, 0)', source)
        self.assertIn('efficiency.complete_batch(batch_patches, int(output.shape[0]))', source)
        self.assertIn('valid_indexs = [component[non_zero_mask] for component in indexs]', source)
        self.assertIn('valid_nodata_mask = nodata_mask[non_zero_mask]', source)


if __name__ == '__main__':
    unittest.main()
