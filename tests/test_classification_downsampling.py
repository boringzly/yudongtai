"""Load the actual inference loop via AST, without importing CUDA/GDAL/model weights."""
import ast
import importlib.util
import os
from pathlib import Path
import time
import types
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
LIB = ROOT / 'fenlei/clie_new/CLInferEngine/clie_lib'
SOURCE = LIB / 'run_lib.py'
TREE = ast.parse(SOURCE.read_text(encoding='utf-8'))
HELPERS = {'_classification_downsample_factor', '_inference_input_size', '_configure_inference_downsample',
           '_downsample_classification_input', '_restore_classification_prediction'}

try:
    import numpy as np
    import cv2
    HAS_ARRAYS = True
except ImportError:
    HAS_ARRAYS = False
try:
    import torch
    HAS_TORCH = hasattr(torch, 'Tensor')
except ImportError:
    HAS_TORCH = False


def scope():
    namespace = {'os': os}
    if HAS_ARRAYS:
        namespace.update(np=np, cv2=cv2)
    if HAS_TORCH:
        namespace.update(t=torch, torch=torch, Variable=torch.autograd.Variable)
    functions = [node for node in TREE.body if isinstance(node, ast.FunctionDef) and node.name in HELPERS]
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(SOURCE), 'exec'), namespace)
    return namespace


class DownsampleConfigTests(unittest.TestCase):
    def test_default_and_explicit_legacy_mode(self):
        functions = scope()
        for value, expected in [(None, 2), ('1', 1), ('2', 2), ('4', 2), ('bad', 2), ('0', 2)]:
            with self.subTest(value=value), mock.patch.dict(os.environ, {}, clear=True):
                if value is not None:
                    os.environ['CLIE_INFER_DOWNSAMPLE'] = value
                self.assertEqual(functions['_classification_downsample_factor'](), expected)

    def test_legacy_upsample_cannot_cancel_default_downsample(self):
        for original in (False, True):
            config = types.SimpleNamespace(IMG_SIZE=1024, UPSAMPLE=original)
            log = mock.Mock()
            with mock.patch.dict(os.environ, {'CLIE_INFER_DOWNSAMPLE': '2'}):
                self.assertEqual(scope()['_configure_inference_downsample'](config, log), 2)
            self.assertFalse(config.UPSAMPLE)
            self.assertIn('模型输入=512x512', log.info.call_args.args[0])

    def test_disable_preserves_existing_upsample_setting(self):
        config = types.SimpleNamespace(IMG_SIZE=1024, UPSAMPLE=True)
        with mock.patch.dict(os.environ, {'CLIE_INFER_DOWNSAMPLE': '1'}):
            self.assertEqual(scope()['_configure_inference_downsample'](config, mock.Mock()), 1)
        self.assertTrue(config.UPSAMPLE)
        self.assertEqual(scope()['_inference_input_size'](1024, 1024, 1, True), (2048, 2048))

    def test_input_uses_bilinear_and_keeps_original_reference(self):
        functions = scope()
        tensor = types.SimpleNamespace(shape=(3, 3, 1024, 1024))
        interpolation = mock.Mock(return_value='half-size tensor')
        functions['t'] = types.SimpleNamespace(nn=types.SimpleNamespace(functional=types.SimpleNamespace(interpolate=interpolation)))
        self.assertEqual(functions['_downsample_classification_input'](tensor, 2), 'half-size tensor')
        interpolation.assert_called_once_with(tensor, size=(512, 512), mode='bilinear', align_corners=True)
        self.assertIs(functions['_downsample_classification_input'](tensor, 1), tensor)
        self.assertEqual(tensor.shape, (3, 3, 1024, 1024))

    def test_scale_configured_after_last_cli_override(self):
        source = SOURCE.read_text(encoding='utf-8')
        self.assertGreater(source.index('inference_downsample = _configure_inference_downsample'), source.rindex('cfg.set_parse(kwargs)'))
        self.assertIn('_inference_input_size(height, width, inference_downsample, cfg.UPSAMPLE)', source)
        self.assertIn('input_data = torch.rand((1, cfg.BAND_NUM, infer_height, infer_width)', source)


@unittest.skipUnless(HAS_ARRAYS, 'numpy and OpenCV required')
class PredictionRestoreTests(unittest.TestCase):
    def test_nearest_restores_class_ids_without_new_values(self):
        classes = np.array([[0, 4], [5, 13]], dtype=np.int64)
        actual = scope()['_restore_classification_prediction'](classes, 4, 4)
        np.testing.assert_array_equal(actual, classes.repeat(2, 0).repeat(2, 1))
        self.assertEqual(actual.dtype, np.uint8)
        self.assertEqual(set(np.unique(actual)), {0, 4, 5, 13})

    def test_restore_nonsquare_and_original_mask_before_overlap_crop(self):
        restored = scope()['_restore_classification_prediction'](np.full((3, 5), 5), 7, 11)
        mask = np.ones((7, 11), dtype=np.uint8)
        mask[2:5, 3:7] = 0
        restored[mask == 0] = 0
        cropped = restored[1:-1, 1:-1]
        self.assertEqual(cropped.shape, (5, 9))
        self.assertTrue(np.all(cropped[1:4, 2:6] == 0))
        self.assertEqual(cropped[0, 0], 5)


@unittest.skipUnless(HAS_ARRAYS and HAS_TORCH, 'CPU PyTorch, numpy and OpenCV required')
class RealInferenceLoopTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Execute the existing pred allocation / good_indices / inference branches
        # verbatim, rather than duplicating their control flow in tests.
        loop = next(node for node in ast.walk(TREE) if isinstance(node, ast.For)
                    and isinstance(node.target, ast.Tuple)
                    and ast.unparse(node.target) == '(idx_x, (img, img_mask, idx_i, idx_j, status))')
        nodes = []
        for node in loop.body:
            if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id in ('pred', 'good_indices') for target in node.targets):
                nodes.append(node)
            elif isinstance(node, ast.If) and isinstance(node.test, ast.Name) and node.test.id == 'good_indices':
                nodes.append(node)
        cls.inference_code = compile(ast.Module(body=nodes, type_ignores=[]), str(SOURCE), 'exec')
        spec = importlib.util.spec_from_file_location('downsample_test_tta', LIB / 'tta.py')
        cls.tta = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.tta)

    def run_loop(self, tta=False, agriculture=False, binary=False, factor=2, size=16):
        functions = scope()
        img = torch.zeros((3, 3, size, size))
        img[0, 0, :, size // 2:] = 1
        img[2, 0] = 1
        original = img.clone()
        calls = []
        def model(data):
            calls.append(tuple(data.shape))
            values = data[:, 0, 0, 0] if agriculture else data[:, 0]
            if binary:
                return (values * 20 - 10).unsqueeze(1)
            return torch.stack([1 - values, values], dim=1) * 10
        config = types.SimpleNamespace(WITH_TTA=tta, IS_AGRICULTURE_INFER=agriculture,
                    USE_GPU=False, NON_BLOCKING=False, MODEL_NUM_CLASSES=1 if binary else 2,
                    USE_BINARY=True, THRESHOLD=.5, TTACH=False, FOREGROUND_IDX=0,
                    AGRICULTURE_WINDOW_SIZE=1, TEST_BATCHES=1, MODEL_NAME='Agriculture')
        status = ['good', 'bad', 'good'] if not agriculture else ['good']
        if agriculture:
            img = img[:1]
            original = original[:1]
        functions.update(img=img, status=status, height=size, width=size, cfg=config,
                         inference_downsample=factor, model=model, time=time, _USE_DIST=False,
                         tqdm=mock.Mock(),
                         # A CPU point extractor: real branch grouping/reassembly is tested.
                         img2points_group=lambda data, _: torch.from_numpy(data[0].transpose(1, 2, 0)).unsqueeze(-1).unsqueeze(-1),
                         tta_encoder=self.tta.TTAEncoder({'tta_flip': True, 'tta_rotate': True}),
                         tta_decoder=self.tta.TTADecoder({'tta_flip': True, 'tta_rotate': True}))
        exec(self.inference_code, functions)
        torch.testing.assert_close(img, original)
        self.assertEqual(functions['pred'].shape, (len(status), size, size))
        self.assertEqual(set(np.unique(functions['pred'][0])), {0, 1})
        if not agriculture:
            self.assertTrue(np.all(functions['pred'][1] == 0))
            self.assertTrue(np.all(functions['pred'][2] == 1))
            self.assertTrue(all(shape[-2:] == (size // factor, size // factor) for shape in calls))
        return functions, calls

    def test_default_1024_tile_runs_model_at_512(self):
        self.run_loop(size=1024)

    def test_flip_rotate_tta_and_binary_branches(self):
        self.run_loop(tta=True)
        self.run_loop(binary=True)
        self.run_loop(tta=True, binary=True)

    def test_disabled_downsample_and_agriculture_reassembly(self):
        self.run_loop(factor=1)
        self.run_loop(agriculture=True, size=10)  # 5 rows exercises the remainder path

    def test_all_bad_tiles_do_not_call_model(self):
        functions, _ = self.run_loop()
        functions['status'] = ['bad'] * 3
        functions['model'] = mock.Mock(side_effect=AssertionError('must not infer invalid tiles'))
        exec(self.inference_code, functions)
        self.assertFalse(np.any(functions['pred']))
        functions['model'].assert_not_called()


if __name__ == '__main__':
    unittest.main()
