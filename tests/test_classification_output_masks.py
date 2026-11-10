import ast
import importlib.util
import os
from pathlib import Path
import sys
import types
import unittest
from unittest import mock

import numpy as np

ROOT = Path(os.environ.get('CLASSIFICATION_TEST_ROOT', Path(__file__).resolve().parents[1]))
spec = importlib.util.spec_from_file_location('_output_masks', ROOT / 'fenlei/classification_pipeline.py')
p = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p)


class Reader:
    def __init__(self, image):
        self.image = image
        self.height, self.width, bands = image.shape
        self.band_list = list(range(1, bands + 1))
        self.calls = []

    def read_image(self, read_range):
        self.calls.append(read_range)
        x, y, w, h = read_range
        return self.image[y:y+h, x:x+w].copy()


class OutputMaskTests(unittest.TestCase):
    def test_reuses_original_pixels_for_every_boundary_window(self):
        for dtype in (np.uint8, np.uint16, np.float32):
            image = np.arange(19*23*3).reshape(19, 23, 3).astype(dtype)
            image[::3, ::4] = 0
            image[2::3, 1::4] = 17
            reader = Reader(image)
            height = width = 8
            overlap = 2
            for row in range(4):
                for col in range(4):
                    # Same read/pad window selection as the production dataset.
                    x, y = col*6-1, row*6-1
                    rx = max(0, x) if x+width <= reader.width else reader.width-width
                    ry = max(0, y) if y+height <= reader.height else reader.height-height
                    raw = image[ry:ry+height, rx:rx+width]
                    for value in (0, 17, 256):
                        actual = p.prepare_output_exception_mask(reader, raw, (rx, ry), row, col,
                                                                 height, width, overlap, value)
                        pixels = image[row*6:min(row*6+6, 19), col*6:min(col*6+6, 23)]
                        expected = np.zeros((height, width), dtype=np.uint8)
                        expected[:pixels.shape[0], :pixels.shape[1]] = np.all(pixels == [value]*3, axis=2)
                        np.testing.assert_array_equal(actual, expected)
            self.assertEqual(reader.calls, [])

    def test_noncontained_raw_falls_back_to_exact_write_window(self):
        reader = Reader(np.zeros((11, 13, 4), dtype=np.uint16))
        result = p.prepare_output_exception_mask(reader, reader.image[:2, :2], (0, 0),
                                                1, 2, 6, 6, 2, 0)
        self.assertEqual(reader.calls, [(8, 4, 4, 4)])
        self.assertEqual(result.sum(), 16)

    def test_missing_raw_uses_same_all_band_rule_including_nan(self):
        reader = Reader(np.array([[[0, 0, 0], [0, 1, 0]], [[np.nan]*3, [2]*3]], dtype=np.float32))
        for value in (0, 2, np.nan):
            actual = p.prepare_output_exception_mask(reader, None, (0, 0), 0, 0, 2, 2, 0, value)
            np.testing.assert_array_equal(actual, np.all(reader.image == [value]*3, axis=2))

    def test_failed_fallback_read_is_not_a_silent_valid_mask(self):
        reader = Reader(np.zeros((8, 8, 3), dtype=np.uint8))
        reader.read_image = lambda **_: None
        with self.assertRaisesRegex(RuntimeError, 'Cannot read'):
            p.prepare_output_exception_mask(reader, None, (0, 0), 0, 0, 8, 8, 0, 0)

    def test_cpu_tensor_converted_before_numpy_boolean_indexing(self):
        expected = np.array([[0, 1], [1, 0]], dtype=np.uint8)
        class Tensor:
            def numpy(self): return expected
            def __eq__(self, _): raise AssertionError('Torch comparison must not run')
        actual = np.full((2, 2), 7, dtype=np.uint8)
        actual[p.cpu_mask_array(Tensor()) == 0] = 0
        np.testing.assert_array_equal(actual, [[0, 7], [7, 0]])

    def test_dataset_masks_use_raw_pixels_before_clahe_and_keep_model_inputs(self):
        source = ROOT / 'fenlei/clie_new/CLInferEngine/clie_lib/dataset.py'
        node = next(n for n in ast.parse(source.read_text(encoding='utf-8')).body
                    if isinstance(n, ast.ClassDef) and n.name == 'InferDataMgr')
        image = np.ones((14, 14, 3), dtype=np.uint8)
        image[::3, ::3] = 0
        class ImageReader(Reader):
            proj, coord = 'test', (0, 1, 0, 14, 0, -1)
            def __init__(self, _): super().__init__(image)
            def set_band_list(self, _): pass
            def close(self): pass
        namespace = dict(np=np, os=os, data=types.SimpleNamespace(Dataset=object), ImageReader=ImageReader,
            torch=types.SimpleNamespace(from_numpy=lambda x: x),
            T=types.SimpleNamespace(Normalize=lambda **_: None, ToTensor=lambda: None,
                                    Compose=lambda _: lambda x: x.copy()),
            clahe_process=lambda x, bands: np.full_like(x, 99))
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), 'exec'), namespace)
        Dataset = namespace['InferDataMgr']
        helper = types.ModuleType('fenlei.classification_pipeline')
        helper.prepare_output_exception_mask = p.prepare_output_exception_mask
        with mock.patch.dict(os.environ, {}, clear=True), mock.patch.dict(sys.modules, {'fenlei.classification_pipeline': helper}):
            old = Dataset('/input.tif', [1, 2, 3], 8, 8, 2, [0]*3, [1]*3, with_clahe=True)
            new = Dataset('/input.tif', [1, 2, 3], 8, 8, 2, [0]*3, [1]*3, with_clahe=True)
            new.output_exception_value = 0
            for index, (row, col) in enumerate(old.index_list):
                before, after = old[index], new[index]
                np.testing.assert_array_equal(before[0], after[0])
                np.testing.assert_array_equal(before[1], after[1]['valid'])
                self.assertEqual(before[2:], after[2:])
                pixels = image[row*6:min(row*6+6, 14), col*6:min(col*6+6, 14)]
                np.testing.assert_array_equal(after[1]['exception'][:pixels.shape[0], :pixels.shape[1]],
                                              np.all(pixels == [0]*3, axis=2))
            self.assertEqual(len(new.image_reader.calls), len(new))

    def test_metrics_split_result_processing(self):
        metrics = p.ClassificationMetrics('input', 16)
        metrics.mask, metrics.exception, metrics.log = 1, 2, 3
        logger = mock.Mock()
        metrics.report(logger)
        message = logger.info.call_args.args[0]
        for item in ('掩膜及类别处理=1.00s', '无效像素检查=2.00s', '日志及进度整理=3.00s'):
            self.assertIn(item, message)


if __name__ == '__main__':
    unittest.main()
