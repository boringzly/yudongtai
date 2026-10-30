import ast
import os
from pathlib import Path
import types
import unittest
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


class ClassificationRoiTests(unittest.TestCase):
    def setUp(self):
        path = ROOT / 'fenlei/clie_new/CLInferEngine/clie_lib/dataset.py'
        tree = ast.parse(path.read_text(encoding='utf-8'))
        node = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'InferDataMgr')

        class Reader:
            width, height, proj, coord = 16, 16, 'test CRS', (0, 1, 0, 16, 0, -1)

            def __init__(self, _):
                self.calls = []

            def set_band_list(self, _):
                pass

            def read_image(self, read_range):
                self.calls.append(read_range)
                return np.arange(8 * 8 * 3, dtype=np.uint8).reshape(8, 8, 3)

            def close(self):
                pass

        namespace = {'os': os, 'np': np, 'data': types.SimpleNamespace(Dataset=object),
                     'torch': types.SimpleNamespace(from_numpy=lambda array: array.copy()),
                     'ImageReader': Reader,
                     'T': types.SimpleNamespace(Normalize=lambda **_: None, ToTensor=lambda: None,
                                                Compose=lambda _: lambda image: image.copy())}
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), namespace)
        self.Dataset = namespace['InferDataMgr']

        class Scope:
            def intersects(self, x, y, width, height):
                return x < 6 and y < 6 and x + width > 0 and y + height > 0

            def mask(self, x, y, width, height):
                rows, cols = np.arange(height)[:, None] + y, np.arange(width)[None, :] + x
                return ((rows >= 0) & (rows < 6) & (cols >= 0) & (cols < 6)).astype(np.uint8)

        def open_roi(dataset, initial=False):
            dataset.roi_wkt = 'test footprint'
            dataset.roi = Scope()

        self.Dataset._open_roi = open_roi

    def dataset(self):
        return self.Dataset('/image.tif', [1, 2, 3], 8, 8, 2, [0, 0, 0], [1, 1, 1])

    def test_filters_before_read_and_keeps_original_grid_offsets(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            full = self.dataset()
        with mock.patch.dict(os.environ, {'CHANGE_DETECTION_ROI_WKT_FILE': '/roi.wkt'}, clear=True):
            limited = self.dataset()
        self.assertEqual(limited.roi_candidates, len(full))
        self.assertEqual(limited.index_list, [(0, 0)])
        self.assertEqual(limited.image_reader.calls, [])

    def test_boundary_mask_does_not_change_rgb_context(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            original = self.dataset()[0]
        with mock.patch.dict(os.environ, {'CHANGE_DETECTION_ROI_WKT_FILE': '/roi.wkt'}, clear=True):
            scoped = self.dataset()[0]
        np.testing.assert_array_equal(original[0], scoped[0])
        expected = np.zeros((8, 8), dtype=np.uint8)
        expected[1:7, 1:7] = 1
        np.testing.assert_array_equal(scoped[1], expected)
        self.assertEqual(scoped[2:], ('0', '0', 'good'))

    def test_worker_pickle_reopens_own_geometry_and_readers(self):
        with mock.patch.dict(os.environ, {'CHANGE_DETECTION_ROI_WKT_FILE': '/roi.wkt'}, clear=True):
            dataset = self.dataset()
        state = dataset.__getstate__()
        self.assertIsNone(state['roi'])
        self.assertIsNone(state['image_reader'])
        reopened = self.Dataset.__new__(self.Dataset)
        reopened.__setstate__(state)
        self.assertIsNot(reopened.roi, dataset.roi)
        self.assertIsNot(reopened.image_reader, dataset.image_reader)
        self.assertEqual(reopened.index_list, dataset.index_list)

    def test_invalid_coverage_is_not_swallowed_by_image_retry(self):
        with mock.patch.object(self.Dataset, '_open_roi', side_effect=ValueError('invalid ROI')):
            with mock.patch.dict(os.environ, {'CHANGE_DETECTION_ROI_VECTOR': '/roi.shp'}, clear=True):
                with self.assertRaisesRegex(ValueError, 'invalid ROI'):
                    self.dataset()


if __name__ == '__main__':
    unittest.main()
