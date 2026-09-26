"""Run with unittest; real GIS integration checks require rasterio, fiona, Pillow."""
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
sys.path.insert(0, str(ROOT / 'fenlei'))
import result_previews as preview
sys.path.pop(0)

try:
    import numpy as np
    import fiona
    import rasterio
    from rasterio.features import rasterize
    from PIL import Image
    HAS_GIS = True
except ImportError:
    HAS_GIS = False


def actual_mapping():
    path = ROOT / 'fenlei' / 'classification_core.py'
    tree = ast.parse(path.read_text(encoding='utf-8'))
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == '_map_class_code')
    namespace = {}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), 'exec'), namespace)
    return {i: namespace['_map_class_code'](i) for i in range(14)}


def make_fixture(folder, name='scene', count=8):
    """Synthetic imagery only, including a diagonal polygon with a real hole."""
    from rasterio.features import rasterize
    from rasterio.transform import from_origin
    from rasterio.warp import transform_geom
    from rasterio.enums import ColorInterp

    folder = Path(folder)
    transform = from_origin(500000, 4000000, 2, 2)
    crs = 'EPSG:32650'
    yy, xx = np.mgrid[:512, :512]
    rng = np.random.default_rng(7)
    terrain = np.full((512, 512), 1, dtype='uint8')
    terrain[(xx < 130) & (yy > 280)] = 4
    terrain[(xx > 320) & (yy < 170)] = 5
    terrain[(yy > 400) & (xx < 250)] = 11
    rgb = np.asarray([[0, 0, 0], [125, 145, 70], [125, 145, 70], [52, 91, 60],
                      [45, 78, 51], [102, 129, 76], [177, 170, 151], [160, 163, 162],
                      [210, 215, 213], [162, 158, 153], [142, 107, 77], [45, 89, 119]], dtype='float32')[terrain]
    rgb += rng.normal(0, 7, (*terrain.shape, 1))
    rgb += (((xx // 34 + yy // 47) % 2) * 12)[:, :, None]
    rgb = np.clip(rgb, 0, 255).astype('uint8')
    rings = [[(120, 80), (450, 410), (410, 450), (80, 120), (120, 80)],
             [(248, 240), (240, 248), (267, 275), (275, 267), (248, 240)]]
    geometry = {'type': 'Polygon', 'coordinates': [[transform * point for point in ring] for ring in rings]}
    mask = rasterize([(geometry, 1)], out_shape=(512, 512), transform=transform).astype(bool)
    rgb_after = rgb.copy()
    rgb_after[mask] = (166, 170, 168)
    # Road markings and roofs make the synthetic QC example easier to inspect.
    rgb_after[mask & (((xx + yy) % 35) < 3)] = (224, 220, 205)
    class_after = terrain.copy()
    class_after[mask] = 7
    source = {'shp': str(folder / (name + '.shp'))}
    for key, data in [('pre_image', rgb.transpose(2, 0, 1)), ('post_image', rgb_after.transpose(2, 0, 1)),
                      ('pre_class', terrain[None]), ('post_class', class_after[None])]:
        source[key] = str(folder / (name + '_' + key + '.tif'))
        with rasterio.open(source[key], 'w', driver='GTiff', width=512, height=512,
                           count=data.shape[0], dtype='uint8', transform=transform, crs=crs) as ds:
            ds.write(data)
            if data.shape[0] == 3:
                ds.colorinterp = (ColorInterp.red, ColorInterp.green, ColorInterp.blue)
    schema = {'geometry': 'Polygon', 'properties': {'uid': 'int', 'pre_code': 'int', 'curr_code': 'int'}}
    # Vector deliberately differs from the raster CRS.
    with fiona.open(source['shp'], 'w', driver='ESRI Shapefile', schema=schema, crs='EPSG:4326', encoding='UTF-8') as ds:
        for i in range(count):
            ds.write({'geometry': transform_geom(crs, 'EPSG:4326', geometry),
                      'properties': {'uid': i, 'pre_code': 1, 'curr_code': 5 if i % 2 == 0 else 4}})
    return source


class PreviewUnitTests(unittest.TestCase):
    def setUp(self):
        self.log = logging.getLogger('preview_tests')
        self.log.addHandler(logging.NullHandler())

    def test_config_is_bounded(self):
        self.assertEqual(preview.preview_count({}), 30)
        self.assertEqual(preview.preview_count({'CLASSIFICATION_PREVIEW_COUNT': '9999'}), 60)
        self.assertEqual(preview.preview_count({'CLASSIFICATION_PREVIEW_COUNT': '-1'}), 0)
        self.assertEqual(preview.preview_count({'CLASSIFICATION_PREVIEW_COUNT': 'bad'}), 30)

    def test_disabled_does_not_load_gis_or_render(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(os.environ, {'CLASSIFICATION_PREVIEW_COUNT': '0'}), \
                mock.patch.object(preview, 'sample_features') as sample:
            report = preview.generate_previews([], tmp, {}, self.log)
            self.assertEqual(report['status'], 'disabled')
            self.assertFalse((Path(tmp) / 'previews').exists())
            sample.assert_not_called()

    def test_cleanup_only_owned_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'sample_001.jpg').write_bytes(b'old')
            (root / 'user.jpg').write_bytes(b'keep')
            (root / 'manifest.json').write_text(json.dumps({'generator': preview.GENERATOR,
                  'samples': [{'file': 'sample_001.jpg'}, {'file': '../user.jpg'}, {'file': 'user.jpg'}]}))
            preview._prepare_directory(root)
            self.assertFalse((root / 'sample_001.jpg').exists())
            self.assertEqual((root / 'user.jpg').read_bytes(), b'keep')

    def test_nonowned_directory_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'previews'
            root.mkdir()
            (root / 'index.html').write_text('user file')
            result = preview.generate_previews([], tmp, {}, self.log)
            self.assertEqual(result['status'], 'failed')
            self.assertEqual((root / 'index.html').read_text(), 'user file')

    def test_render_failure_is_nonfatal_and_reported(self):
        candidate = {'source': {'shp': 'missing.shp'}, 'fid': '0'}
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(preview, 'sample_features', return_value=[candidate]), \
                mock.patch.object(preview, 'render_sheet', side_effect=RuntimeError('missing raster')):
            report = preview.generate_previews([], tmp, {}, self.log)
            self.assertEqual(report['status'], 'completed_with_warnings')
            self.assertEqual(report['samples'], [])
            self.assertIn('missing raster', report['warnings'][0])
            self.assertTrue((Path(tmp) / 'previews' / 'index.html').exists())

    def test_budget_enforced_before_write(self):
        candidate = {'source': {'shp': 'test.shp'}, 'fid': '0'}
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(preview, 'MAX_BYTES', 1), \
                mock.patch.object(preview, 'sample_features', return_value=[candidate]), \
                mock.patch.object(preview, 'render_sheet', return_value=(b'large', {})):
            result = preview.generate_previews([], tmp, {}, self.log)
            self.assertEqual(result['samples'], [])
            self.assertEqual(len(list((Path(tmp) / 'previews').glob('*.jpg'))), 0)

    def test_core_hook_uses_log_sibling_and_keeps_mapping(self):
        core = ROOT / 'fenlei' / 'classification_core.py'
        tree = ast.parse(core.read_text(encoding='utf-8'))
        functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in
                     ('_generate_result_previews', '_ensure_log_dir', '_map_class_code')]
        namespace = {'Path': Path, 'sys': sys, '__file__': str(core),
                     'prg_sender': mock.Mock(), 'swap_write': mock.Mock()}
        exec(compile(ast.Module(body=functions, type_ignores=[]), str(core), 'exec'), namespace)
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(preview, 'generate_previews', return_value={'samples': []}) as generate:
            task = Path(tmp) / 'task'
            dst = task / 'working' / 'classification'
            namespace['_generate_result_previews']([], str(dst), self.log)
            self.assertEqual(generate.call_args.args[1], str(Path(tmp) / 'task_diagnostics' / 'task'))
            self.assertEqual(generate.call_args.args[2], actual_mapping())
        with mock.patch.object(preview, 'generate_previews', side_effect=RuntimeError('optional renderer')):
            with tempfile.TemporaryDirectory() as tmp:
                self.assertEqual(namespace['_generate_result_previews']([], str(Path(tmp) / 'task'), self.log)['status'], 'failed')


@unittest.skipUnless(HAS_GIS, 'Real preview tests need rasterio/fiona/numpy/Pillow')
class PreviewGISTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.log = logging.getLogger('preview_gis_tests')
        self.log.addHandler(logging.NullHandler())
        self.source = make_fixture(self.root)

    def tearDown(self):
        self.temp.cleanup()

    def test_sampling_repeatable_and_pair_transition_coverage(self):
        other = make_fixture(self.root, 'other', 12)
        samples = preview.sample_features([self.source, other], 6, self.fail, lambda _: None)
        self.assertEqual(samples, preview.sample_features([self.source, other], 6, self.fail, lambda _: None))
        self.assertEqual(len(samples), 6)
        self.assertEqual(len({sample['source']['shp'] for sample in samples[:2]}), 2)
        self.assertEqual(len({(sample['source']['shp'], sample['fid']) for sample in samples}), 6)
        self.assertEqual({int(sample['fid']) % 2 for sample in samples[:4]}, {0, 1})

    def test_actual_jpegs_gallery_manifest_and_rerun_cleanup(self):
        with mock.patch.dict(os.environ, {'CLASSIFICATION_PREVIEW_COUNT': '3'}):
            report = preview.generate_previews([self.source], str(self.root / 'result'), actual_mapping(), self.log)
        self.assertEqual(report['warnings'], [])
        self.assertEqual(len(report['samples']), 3)
        directory = Path(report['directory'])
        for item in report['samples']:
            with Image.open(directory / item['file']) as img:
                self.assertEqual(img.size, (1504, 1762))
                self.assertEqual(img.format, 'JPEG')
            self.assertGreater(min(item['valid_image_fraction']), .8)
        self.assertEqual(len(list(directory.glob('*.jpg'))), 3)
        self.assertEqual(json.loads((directory / 'manifest.json').read_text(encoding='utf-8'))['bytes'], report['bytes'])
        with mock.patch.dict(os.environ, {'CLASSIFICATION_PREVIEW_COUNT': '1'}):
            preview.generate_previews([self.source], str(self.root / 'result'), actual_mapping(), self.log)
        self.assertEqual(len(list(directory.glob('*.jpg'))), 1)
        with mock.patch.dict(os.environ, {'CLASSIFICATION_PREVIEW_COUNT': '0'}):
            preview.generate_previews([self.source], str(self.root / 'result'), actual_mapping(), self.log)
        self.assertEqual(len(list(directory.glob('*.jpg'))), 0)
        self.assertEqual(json.loads((directory / 'manifest.json').read_text(encoding='utf-8'))['status'], 'disabled')

    def test_missing_chinese_font_has_readable_fallback(self):
        with mock.patch.object(preview, '_font_path', return_value=None):
            data, report = preview.render_sheet({'source': self.source, 'fid': '0'}, 1, actual_mapping())
        self.assertFalse(report['chinese_font'])
        self.assertGreater(len(data), 1000)

    def test_rgb_stretch_is_shared_and_byte_imagery_is_preserved(self):
        data = np.arange(12, dtype='uint8').reshape(3, 2, 2)
        valid = np.ones((2, 2), dtype=bool)
        first, second = preview._rgb_pair((data, valid), (data, valid))
        np.testing.assert_array_equal(first, data.transpose(1, 2, 0))
        np.testing.assert_array_equal(first, second)
        high = data.astype('uint16') * 500
        first, second = preview._rgb_pair((high, valid), (high, valid))
        np.testing.assert_array_equal(first, second)

    def test_empty_results_and_missing_files_are_explained(self):
        empty = make_fixture(self.root, 'empty', 0)
        report = preview.generate_previews([empty], self.root / 'empty_result', actual_mapping(), self.log)
        self.assertEqual(report['samples'], [])
        self.assertIn('无可抽样', report['warnings'][0])
        broken = dict(self.source, pre_class=str(self.root / 'missing.tif'))
        with mock.patch.dict(os.environ, {'CLASSIFICATION_PREVIEW_COUNT': '1'}):
            report = preview.generate_previews([broken], self.root / 'broken', actual_mapping(), self.log)
        self.assertEqual(report['samples'], [])
        self.assertTrue(report['warnings'])

    def test_nodata_and_different_raster_crs_share_the_same_grid(self):
        from rasterio.transform import from_bounds
        from rasterio.warp import transform_bounds
        with rasterio.open(self.source['pre_class']) as src:
            bbox = transform_bounds(src.crs, 'EPSG:4326', *src.bounds)
        grid = from_bounds(*bbox, 128, 128)
        values, valid = preview._read_grid(self.source['pre_class'], 'EPSG:4326', grid, 128)
        self.assertGreater(np.mean(valid), .95)
        self.assertTrue(set(np.unique(values)).issubset({0, 1, 4, 5, 11}))
        outside = from_bounds(0, 0, 1, 1, 128, 128)
        _, outside_valid = preview._read_grid(self.source['pre_image'], 'EPSG:4326', outside, 128, True)
        self.assertFalse(np.any(outside_valid))

    def test_six_class_colors_and_polygon_holes(self):
        from rasterio.features import rasterize
        from rasterio.transform import from_origin
        raw = np.arange(14, dtype='uint8')[None]
        mapped = preview._coarse_grid(raw, np.ones_like(raw, dtype=bool), actual_mapping())
        self.assertEqual(mapped.tolist(), [[0, 1, 1, 2, 2, 3, 5, 5, 5, 5, 6, 4, 6, 6]])
        geometry = {'type': 'Polygon', 'coordinates': [[(1, 1), (8, 8), (9, 8), (2, 1), (1, 1)],
                                                      [(4, 4), (5, 5), (5.5, 5), (4.5, 4), (4, 4)]]}
        transform = from_origin(0, 10, .1, .1)
        mask = rasterize([(geometry, 1)], out_shape=(100, 100), transform=transform).astype(bool)
        rgb = np.zeros((100, 100, 3), dtype='uint8')
        colored = preview._blend(rgb, mask, (255, 0, 0), 1)
        self.assertFalse(colored[20, 20].any())  # in bbox, outside the diagonal
        self.assertFalse(colored[54, 48].any())  # interior hole
        self.assertTrue(colored[79, 26].any())   # actual diagonal polygon


if __name__ == '__main__':
    unittest.main()
