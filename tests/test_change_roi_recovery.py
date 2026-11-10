import ast
import copy
import importlib.util
import json
import logging
import os
from pathlib import Path
import queue
import shutil
import sys
import tempfile
import time
import types
import unittest
from unittest import mock

ROOT = Path(os.environ.get('CHANGE_RECOVERY_TEST_ROOT', Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(ROOT/'change'))
sys.path.insert(0, str(ROOT))
import postprocess_checkpoint as pc
import resume_checkpoint as rc
import resume_compat as compat
spec = importlib.util.spec_from_file_location('_recovery_roi', ROOT/'change/roi_scope.py')
roi = importlib.util.module_from_spec(spec)
spec.loader.exec_module(roi)


class DurableStageTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.pre, self.post = self.root/'pre.tif', self.root/'post.tif'
        for path in (self.pre, self.post): path.write_bytes(b'image')
        self.identity = rc.pair_identity(self.pre, self.post,
            {'algorithm': 'a'*64, 'batch_size': 16, 'roi': 'coverage', 'models': ['weights']})
        self.raw = self.root/'raw.shp'
        for ext in rc.REQUIRED: self.raw.with_suffix(ext).write_bytes(('raw'+ext).encode())
        self.stages = self.root/'stages'
        self.count = lambda _: 3

    def save(self):
        return pc.save(self.stages, 'image', self.identity, self.raw, 'POLYGON(test)', 'test CRS', validator=self.count)

    def load(self, identity=None):
        return pc.load([self.stages], 'image', identity or self.identity, validator=self.count)

    def test_cross_task_stage_survives_scratch_removal(self):
        self.save()
        for ext in rc.REQUIRED: self.raw.with_suffix(ext).unlink()
        result = self.load()
        self.assertEqual(result['stage'], 'polygonized')
        self.assertEqual(result['feature_count'], 3)
        self.assertEqual(rc.sidecar_hashes(result['raw_path']), result['sidecars'])

    def test_inputs_models_roi_batch_are_still_strict(self):
        self.save()
        for key, value in [('models', ['other']), ('roi', 'other'), ('batch_size', 32), ('algorithm', 'b'*64)]:
            identity = copy.deepcopy(self.identity); identity['run'][key] = value
            self.assertIsNone(self.load(identity))
        self.pre.write_bytes(b'new content')
        with self.assertRaisesRegex(ValueError, '输入影像发生变化'): self.load()

    def test_changed_source_during_save_never_publishes_record(self):
        self.pre.write_bytes(b'changed')
        with self.assertRaises(ValueError): self.save()
        self.assertFalse(rc.record_path(self.stages, 'image').exists())

    def test_corrupt_stage_and_path_escape_are_rejected(self):
        data = self.save()
        saved = self.stages/data['raw']
        saved.with_suffix('.dbf').write_bytes(b'corrupt')
        self.assertIsNone(self.load())
        data['raw'] = '../raw.shp'
        rc.atomic_json(rc.record_path(self.stages, 'image'), data)
        self.assertIsNone(self.load())

    def test_failed_copy_never_publishes_record(self):
        with mock.patch.object(pc, 'copy_vector', side_effect=OSError('disk full')):
            with self.assertRaises(OSError): self.save()
        self.assertFalse(rc.record_path(self.stages, 'image').exists())
        self.assertEqual(list(self.stages.iterdir()), [])

    def test_postprocessing_failure_preserves_raw_stage_and_no_final_output(self):
        data = self.save()
        raw = self.stages/data['raw']
        before = rc.sidecar_hashes(raw)
        def fill(source, target):
            pc.copy_vector(source, target)
            Path(source).with_suffix('.dbf').write_bytes(b'only mutate working copy')
        clip = mock.Mock(side_effect=RuntimeError('intersection failed'))
        with self.assertRaisesRegex(RuntimeError, 'intersection failed'):
            pc.finish(raw, self.root/'out/result.shp', 'ROI', 'CRS', fill, pc.copy_vector,
                      clip, validator=self.count)
        self.assertEqual(rc.sidecar_hashes(raw), before)
        self.assertIsNotNone(self.load())
        self.assertFalse((self.root/'out/result.shp').exists())

    def test_empty_stage_skips_smoothing_but_preserves_valid_empty_result(self):
        never = mock.Mock(side_effect=AssertionError('empty result must not smooth'))
        output = self.root/'empty/result.shp'
        result = pc.finish(self.raw, output, None, 'CRS', never, never, never, validator=lambda _: 0)
        self.assertEqual(result, 0)
        self.assertEqual(rc.sidecar_hashes(output), rc.sidecar_hashes(self.raw))

    def test_worker_verified_stage_never_stages_inputs_or_calls_model(self):
        source = ROOT/'change/change_detection_core.py'
        node = next(n for n in ast.parse(source.read_text(encoding='utf-8')).body
                    if isinstance(n, ast.FunctionDef) and n.name == '_process_change_pair')
        task = dict(idx=0, total=1, stem='image', pre_path=str(self.pre), post_path=str(self.post),
                    output_shp=str(self.root/'output/shp/image.shp'), shp_dir=str(self.root/'output/shp'),
                    tif_dir=str(self.root/'output/tif'), dst_path=str(self.root/'output'),
                    checkpoint_dir=str(self.root/'logs/change_detection_checkpoints'), checkpoint_identity=self.identity)
        events = queue.Queue(); logger = mock.Mock()
        namespace = dict(Path=Path, os=os, time=time, shutil=shutil, traceback=__import__('traceback'),
            _configure_persistent_logger=lambda *a, **k: (logger, 'worker.log'),
            _configure_change_memory=mock.Mock(), _log_change_memory=mock.Mock(),
            _remove_shapefile_dataset=mock.Mock(), _remove_file_if_exists=mock.Mock(),
            _shapefile_feature_count=lambda _: 3,
            _create_local_pair_scratch=mock.Mock(side_effect=AssertionError('must not stage inputs')))
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), 'exec'), namespace)
        model = mock.Mock(side_effect=AssertionError('model must not run'))
        def finish(raw, output, *args):
            Path(output).parent.mkdir(parents=True)
            Path(output).write_bytes(b'result')
        fake = types.SimpleNamespace(test_lib_big_memeff=model, finish_change_vectors=finish)
        with mock.patch.dict(sys.modules, {'test_lib_batch_memeff_single_image_nomp': fake,
                'torch': types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: False))}), \
             mock.patch.object(pc, 'load', return_value={'raw_path': str(self.raw), 'roi_wkt': 'ROI', 'projection': 'CRS'}), \
             mock.patch.object(pc, 'save') as carry_stage, \
             mock.patch.object(rc, 'save_checkpoint') as save:
            namespace['_process_change_pair'](task, events, mock.Mock(), 'GPU 1')
        model.assert_not_called()
        namespace['_create_local_pair_scratch'].assert_not_called()
        save.assert_called_once()
        carry_stage.assert_called_once()
        self.assertEqual(events.get()['type'], 'started')
        self.assertEqual(events.get()['status'], 'completed')


class CompatibilityTests(unittest.TestCase):
    def test_only_recorded_algorithm_change_can_be_accepted(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root/'test_lib_batch_memeff_single_image_nomp.py').write_text('current')
            current = compat.code_fingerprint(root)
            before = 'b'*64
            rc.atomic_json(root/compat.COMPAT_FILE, {'release': compat.RELEASE,
                'current_algorithm': current, 'previous_algorithms': [before]})
            new = {'pre': {'path': 'pre'}, 'run': {'algorithm': current, 'roi': 'same', 'batch_size': 16}}
            old = copy.deepcopy(new); old['run']['algorithm'] = before
            self.assertTrue(compat.identities_match(old, new, root))
            for key, value in [('roi', 'changed'), ('batch_size', 32), ('algorithm', 'c'*64)]:
                altered = copy.deepcopy(old); altered['run'][key] = value
                self.assertFalse(compat.identities_match(altered, new, root))
            (root/'test_lib_batch_memeff_single_image_nomp.py').write_text('unreviewed change')
            self.assertFalse(compat.identities_match(old, new, root))


class GeometryFlowTests(unittest.TestCase):
    def setUp(self):
        self.repair_calls = 0
        class Geometry:
            def __init__(self, valid=True, empty=False, result=None, repairable=True):
                self.valid, self.empty, self.result, self.repairable = valid, empty, result, repairable
            def IsValid(self): return self.valid
            def IsEmpty(self): return self.empty
            def FlattenTo2D(self): pass
            def GetGeometryType(self): return 3
            def MakeValid(geom):
                self.repair_calls += 1
                return Geometry() if geom.repairable else None
            def Intersection(geom, other):
                if not geom.valid or not other.valid: raise RuntimeError('TopologyException')
                return self.intersection
        self.Geometry = Geometry
        self.original = Geometry(valid=False)
        self.intersection = Geometry()
        self.feature = types.SimpleNamespace(GetGeometryRef=lambda: self.original,
            SetGeometry=mock.Mock(return_value=0), GetFID=lambda: 5)
        srs = types.SimpleNamespace(Clone=lambda: srs, SetAxisMappingStrategy=lambda _: None,
                                    ImportFromWkt=lambda _: 0, IsSame=lambda _: True)
        self.layer = mock.Mock()
        self.layer.__iter__ = lambda _: iter([self.feature])
        self.layer.GetFeature.return_value = self.feature
        self.layer.GetSpatialRef.return_value = srs
        self.layer.SetFeature.return_value = self.layer.DeleteFeature.return_value = self.layer.SyncToDisk.return_value = 0
        ogr = types.SimpleNamespace(Open=lambda *a: types.SimpleNamespace(GetLayer=lambda _: self.layer),
            CreateGeometryFromWkt=lambda _: Geometry(), GT_Flatten=lambda n: n,
            wkbPolygon=3, wkbMultiPolygon=6, wkbGeometryCollection=7)
        osr = types.SimpleNamespace(SpatialReference=lambda: srs, OAMS_TRADITIONAL_GIS_ORDER=0)
        patcher = mock.patch.dict(sys.modules, {'osgeo': types.SimpleNamespace(ogr=ogr, osr=osr)})
        patcher.start(); self.addCleanup(patcher.stop)

    def test_invalid_output_repaired_before_intersection_and_attributes_retained(self):
        stats = roi.clip_shapefile('result.shp', 'ROI', 'CRS')
        self.assertEqual(self.repair_calls, 1)
        self.assertEqual(stats['written'], 1)
        self.assertEqual(stats['repairs'], 1)
        self.assertIs(self.layer.SetFeature.call_args.args[0], self.feature)
        self.layer.DeleteFeature.assert_not_called()

    def test_unrepairable_input_is_failure_not_silently_removed(self):
        self.original.repairable = False
        with self.assertRaisesRegex(RuntimeError, 'MakeValid failed'):
            roi.clip_shapefile('result.shp', 'ROI', 'CRS')
        self.layer.DeleteFeature.assert_not_called()

    def test_intersection_failure_remains_failure_with_feature_id(self):
        self.intersection = None
        with self.assertRaisesRegex(RuntimeError, 'FID=5'):
            roi.clip_shapefile('result.shp', 'ROI', 'CRS')
        self.layer.SetFeature.assert_not_called()

    def test_valid_empty_intersection_is_removed(self):
        self.intersection = self.Geometry(empty=True)
        stats = roi.clip_shapefile('result.shp', 'ROI', 'CRS')
        self.assertEqual(stats['removed'], 1)
        self.layer.DeleteFeature.assert_called_once_with(5)


try:
    from osgeo import ogr, osr
except ImportError:
    ogr = osr = None


@unittest.skipIf(ogr is None, 'Real GDAL/OGR unavailable; run inside algorithm environment')
class RealGeometryTests(unittest.TestCase):
    def test_self_intersection_repaired_clipped_and_attributes_preserved(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'result.shp'
            srs = osr.SpatialReference(); srs.ImportFromEPSG(4326)
            ds = ogr.GetDriverByName('ESRI Shapefile').CreateDataSource(str(path))
            layer = ds.CreateLayer('result', srs=srs, geom_type=ogr.wkbPolygon)
            layer.CreateField(ogr.FieldDefn('code', ogr.OFTInteger))
            f = ogr.Feature(layer.GetLayerDefn()); f.SetField('code', 7)
            f.SetGeometry(ogr.CreateGeometryFromWkt('POLYGON((0 0,2 2,0 2,2 0,0 0))'))
            self.assertEqual(layer.CreateFeature(f), 0)
            f = layer = ds = None
            bounds = 'POLYGON((-1 -1,3 -1,3 3,-1 3,-1 -1))'
            stats = roi.clip_shapefile(path, bounds, srs.ExportToWkt())
            self.assertGreater(stats['repairs'], 0)
            ds = ogr.Open(str(path)); layer = ds.GetLayer(); f = layer.GetNextFeature()
            geometry = f.GetGeometryRef()
            self.assertTrue(geometry.IsValid())
            self.assertAlmostEqual(geometry.GetArea(), 2)
            self.assertEqual(f.GetField('code'), 7)
            self.assertTrue(geometry.Difference(ogr.CreateGeometryFromWkt(bounds)).IsEmpty())
            f = layer = ds = None


if __name__ == '__main__': unittest.main()
