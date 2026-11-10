import ast
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import types
import unittest
from unittest import mock
import numpy as np

ROOT = Path(os.environ.get('CLASSIFICATION_TEST_ROOT', Path(__file__).resolve().parents[1]))

class FakeGdal:
    GDT_Byte, GDT_UInt16, GDT_Int16, GDT_UInt32, GDT_Int32, GDT_Float32, GDT_Float64 = range(1, 8)
    CE_Failure = 3
    def __init__(self):
        self.error_type, self.message = 0, ''
    def GetDriverByName(self, driver): return self
    def Create(self, filename, width, height, nbands, dtype, options=None):
        self.dataset = FakeDataset(self, width, height, nbands)
        return self.dataset
    def ErrorReset(self): self.error_type, self.message = 0, ''
    def GetLastErrorType(self): return self.error_type
    def GetLastErrorMsg(self): return self.message

class FakeDataset:
    def __init__(self, gdal, width, height, nbands):
        self.gdal, self.fail_flush, self.silent_flush_error = gdal, False, False
        self.fail_write, self.flushes, self.band_flushes = False, 0, 0
        self.pixels = np.zeros((height, width, nbands), dtype=np.uint8)
    def GetRasterBand(self, index):
        dataset = self
        class Band:
            def WriteArray(self, array, x, y):
                if dataset.fail_write:
                    dataset.gdal.error_type = 3
                    dataset.gdal.message = 'Disk quota exceeded'
                    return 1
                dataset.pixels[y:y + array.shape[0], x:x + array.shape[1], index - 1] = array
                return 0
            def FlushCache(self):
                dataset.band_flushes += 1
                return 0
        return Band()
    def FlushCache(self):
        self.flushes += 1
        if self.fail_flush or self.silent_flush_error:
            self.gdal.error_type = 3
            self.gdal.message = 'Disk quota exceeded'
            return None if self.silent_flush_error else 1
        return None  # Also exercise older GDAL Python bindings.

def writer_class():
    tree = ast.parse((ROOT / 'fenlei/clie_new/CLInferEngine/clie_lib/imageio.py').read_text(encoding='utf-8'))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'ImageWriter')
    fake = FakeGdal()
    namespace = {'gdal': fake}
    exec(compile(ast.Module(body=[cls], type_ignores=[]), '<real-image-writer>', 'exec'), namespace)
    return namespace['ImageWriter'], fake

class WriteBatchingTests(unittest.TestCase):
    def test_default_callers_retain_per_tile_band_flush(self):
        Writer, gdal = writer_class()
        writer = Writer('a.tif', 10, 4, 3)
        writer.write_image(np.zeros((4, 5, 3), dtype=np.uint8))
        self.assertEqual(gdal.dataset.band_flushes, 3)
        self.assertEqual(gdal.dataset.flushes, 0)

    def test_batch_mode_defers_explicit_flush_until_batch_boundary(self):
        Writer, gdal = writer_class()
        writer = Writer('a.tif', 32, 2, 1, flush_each_write=False)
        for batch in range(2):
            for i in range(16):
                writer.write_image(np.full((2, 1, 1), batch * 16 + i, dtype=np.uint8), (batch * 16 + i, 0))
            self.assertEqual(gdal.dataset.flushes, batch)
            writer.flush()
        self.assertEqual(gdal.dataset.band_flushes, 0)
        self.assertEqual(gdal.dataset.flushes, 2)
        dataset = gdal.dataset
        writer.close()
        self.assertEqual(dataset.flushes, 3)
        writer.close()
        self.assertEqual(dataset.flushes, 3)

    def test_rgb_band_mapping_overlap_and_tail_pixels_are_unchanged(self):
        outputs = []
        for per_tile in (True, False):
            Writer, gdal = writer_class()
            writer = Writer('a.tif', 9, 5, 3, compress='LZW', flush_each_write=per_tile)
            for i, (x, width) in enumerate(((0, 4), (2, 4), (6, 3))):
                image = np.arange(5 * width * 3, dtype=np.uint8).reshape(5, width, 3) + i
                writer.write_image(image, (x, 0))
            writer.flush()
            outputs.append(gdal.dataset.pixels.copy())
            writer.close()
        np.testing.assert_array_equal(*outputs)

    def test_write_failure_still_fails_immediately_in_batch_mode(self):
        Writer, gdal = writer_class()
        writer = Writer('a.tif', 4, 4, 1, flush_each_write=False)
        gdal.dataset.fail_write = True
        with self.assertRaisesRegex(RuntimeError, 'Disk quota exceeded'):
            writer.write_image(np.zeros((4, 4, 1), dtype=np.uint8))
        self.assertEqual(gdal.dataset.flushes, 0)

    def test_batch_and_final_flush_errors_are_not_swallowed(self):
        for final in (False, True):
            for silent in (False, True):
                Writer, gdal = writer_class()
                writer = Writer('a.tif', 4, 4, 1, flush_each_write=False)
                gdal.dataset.fail_flush = not silent
                gdal.dataset.silent_flush_error = silent
                with self.subTest(final=final, silent=silent), self.assertRaisesRegex(RuntimeError, 'Disk quota exceeded'):
                    writer.close() if final else writer.flush()
                self.assertIsNotNone(writer.dataset)

    def test_split_metrics_keep_aggregate_output_timing(self):
        spec = importlib.util.spec_from_file_location('_write_metrics', ROOT / 'fenlei/classification_pipeline.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        metrics = module.ClassificationMetrics('a.tif', 16)
        metrics.write, metrics.output, metrics.raster, metrics.message = 10, 2, 7, 1
        logger = mock.Mock()
        metrics.report(logger)
        value = logger.info.call_args.args[0]
        for item in ('写出及消息=10.00s', '结果整理及日志=2.00s', 'TIFF读写及刷盘=7.00s', '消息发送=1.00s'):
            self.assertIn(item, value)

    def test_all_classification_writers_use_batch_mode_and_flush_before_progress(self):
        tree = ast.parse((ROOT / 'fenlei/clie_new/CLInferEngine/clie_lib/run_lib.py').read_text(encoding='utf-8'))
        calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == 'ImageWriter']
        self.assertEqual(len(calls), 3)
        for call in calls:
            self.assertTrue(any(k.arg == 'flush_each_write' and isinstance(k.value, ast.Constant) and k.value.value is False for k in call.keywords))
        loop = next(n for n in ast.walk(tree) if isinstance(n, ast.For) and isinstance(n.target, ast.Tuple) and isinstance(n.target.elts[0], ast.Name) and n.target.elts[0].id == 'idx_x')
        flush = next(n for n in loop.body if isinstance(n, ast.For) and isinstance(n.target, ast.Name) and n.target.id == 'writer')
        progress = next(n for n in loop.body if isinstance(n, ast.Expr) and isinstance(n.value, ast.Call) and isinstance(n.value.func, ast.Attribute) and n.value.func.attr == 'send')
        self.assertLess(flush.lineno, progress.lineno)

if __name__ == '__main__':
    unittest.main()
