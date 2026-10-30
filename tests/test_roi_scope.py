import importlib.util
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('roi_scope_test', ROOT / 'change/roi_scope.py')
roi = importlib.util.module_from_spec(spec)
spec.loader.exec_module(roi)
try:
    from osgeo import ogr, osr
except ImportError:
    ogr = osr = None


class RoiSourceTests(unittest.TestCase):
    def test_unconfigured_coverage_preserves_full_image_mode(self):
        self.assertEqual(roi.source({}), (None, None))
        self.assertIsNone(roi.fingerprint({}))
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertIsNone(roi.projected_wkt(''))

    def test_ambiguous_or_missing_coverage_fails(self):
        with self.assertRaises(ValueError):
            roi.source({'CHANGE_DETECTION_ROI_VECTOR': '/a', 'CHANGE_DETECTION_ROI_WKT_FILE': '/b'})
        with self.assertRaises(FileNotFoundError):
            roi.fingerprint({'CHANGE_DETECTION_ROI_VECTOR': '/missing/roi.shp'})

    def test_geometry_content_changes_checkpoint_fingerprint(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'roi.wkt'
            env = {'CHANGE_DETECTION_ROI_WKT_FILE': str(path)}
            path.write_text('first footprint', encoding='utf-8')
            first = roi.fingerprint(env)
            path.write_text('second footprint', encoding='utf-8')
            self.assertNotEqual(first, roi.fingerprint(env))

    def test_shapefile_requires_and_hashes_crs_and_geometry_sidecars(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'roi.shp'
            path.write_bytes(b'geometry')
            env = {'CHANGE_DETECTION_ROI_VECTOR': str(path)}
            with self.assertRaises(FileNotFoundError):
                roi.fingerprint(env)
            for suffix in ('.shx', '.dbf', '.prj'):
                path.with_suffix(suffix).write_bytes(b'sidecar')
            first = roi.fingerprint(env)
            path.with_suffix('.prj').write_bytes(b'new CRS')
            self.assertNotEqual(first, roi.fingerprint(env))

    def test_window_transform_preserves_rotation_and_pixel_origin(self):
        self.assertEqual(roi.window_transform((100, 2, .5, 200, .25, -2), 3, 4),
                         (108, 2, .5, 192.75, .25, -2))


@unittest.skipIf(ogr is None, 'GDAL/OGR unavailable; run these GIS tests in the algorithm environment')
class RoiGeometryTests(unittest.TestCase):
    WKT = 'POLYGON((0 0,10 0,10 10,0 10,0 0),(4 4,4 6,6 6,6 4,4 4))'
    GT = (0, 1, 0, 10, 0, -1)

    def test_touching_lines_do_not_discard_valid_polygon_components(self):
        collection = ogr.CreateGeometryFromWkt('GEOMETRYCOLLECTION(POLYGON((0 0,2 0,2 2,0 2,0 0)),LINESTRING(3 3,4 4))')
        self.assertAlmostEqual(roi.polygonal_part(collection).GetArea(), 4)

    def test_holes_boundary_and_outside_windows(self):
        scope = roi.RoiScope(self.WKT, self.GT)
        mask = scope.mask(0, 0, 10, 10)
        expected = np.ones((10, 10), dtype=np.uint8)
        expected[4:6, 4:6] = 0
        np.testing.assert_array_equal(mask, expected)
        self.assertFalse(scope.intersects(20, 20, 2, 2))
        self.assertTrue(scope.intersects(-1, -1, 2, 2))
        self.assertFalse(scope.mask(20, 20, 2, 2).any())

    def test_planning_removes_outside_windows_and_preserves_original_offsets(self):
        indices = [[0, 0, 0, 0], [20, 20, 0, 0]]
        self.assertEqual(roi.filter_indices(indices, self.WKT, self.GT, 2), indices[:1])

    def test_multipart_coverage_is_not_replaced_by_envelope(self):
        scope = roi.RoiScope('MULTIPOLYGON(((0 0,2 0,2 2,0 2,0 0)),((8 8,10 8,10 10,8 10,8 8)))', self.GT)
        self.assertTrue(scope.intersects(0, 8, 2, 2))
        self.assertTrue(scope.intersects(8, 0, 2, 2))
        self.assertFalse(scope.intersects(4, 4, 2, 2))

    def test_wkt_coverage_is_reprojected_from_wgs84(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'roi.wkt'
            path.write_text('POLYGON((0 0,1 0,1 1,0 1,0 0))', encoding='utf-8')
            srs = osr.SpatialReference()
            srs.ImportFromEPSG(3857)
            with mock.patch.dict(os.environ, {'CHANGE_DETECTION_ROI_WKT_FILE': str(path)}, clear=True):
                geometry = ogr.CreateGeometryFromWkt(roi.projected_wkt(srs.ExportToWkt()))
            envelope = geometry.GetEnvelope()
            self.assertAlmostEqual(envelope[1], 111319.4908, places=3)
            self.assertAlmostEqual(envelope[3], 111325.1429, places=3)

    def test_invalid_polygon_and_missing_raster_crs_fail_closed(self):
        with self.assertRaises(ValueError):
            roi.RoiScope('POLYGON((0 0,2 2,0 2,2 0,0 0))', self.GT)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'roi.wkt'
            path.write_text(self.WKT, encoding='utf-8')
            with mock.patch.dict(os.environ, {'CHANGE_DETECTION_ROI_WKT_FILE': str(path)}, clear=True):
                with self.assertRaises(ValueError):
                    roi.projected_wkt('')

    def test_final_vector_clip_preserves_holes_and_attributes_after_smoothing(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'result.shp'
            srs = osr.SpatialReference()
            srs.ImportFromEPSG(4326)
            dataset = ogr.GetDriverByName('ESRI Shapefile').CreateDataSource(str(path))
            layer = dataset.CreateLayer('result', srs=srs, geom_type=ogr.wkbPolygon)
            layer.CreateField(ogr.FieldDefn('code', ogr.OFTInteger))
            for wkt in ('POLYGON((-1 -1,11 -1,11 11,-1 11,-1 -1))',
                        'POLYGON((20 20,22 20,22 22,20 22,20 20))'):
                feature = ogr.Feature(layer.GetLayerDefn())
                feature.SetField('code', 7)
                feature.SetGeometry(ogr.CreateGeometryFromWkt(wkt))
                layer.CreateFeature(feature)
            feature, layer, dataset = None, None, None
            roi.clip_shapefile(path, self.WKT, srs.ExportToWkt())
            dataset = ogr.Open(str(path), 0)
            layer = dataset.GetLayer(0)
            self.assertEqual(layer.GetFeatureCount(), 1)
            feature = layer.GetNextFeature()
            self.assertEqual(feature.GetField('code'), 7)
            self.assertAlmostEqual(feature.GetGeometryRef().GetArea(), 96.0)
            feature, layer, dataset = None, None, None


if __name__ == '__main__':
    unittest.main()
