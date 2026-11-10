"""Optional per-task coverage; GIS imports are lazy for checkpoint handling."""
import hashlib
import os
from pathlib import Path

import numpy as np


def source(environ=None):
    env = os.environ if environ is None else environ
    vector = env.get('CHANGE_DETECTION_ROI_VECTOR', '').strip()
    wkt_file = env.get('CHANGE_DETECTION_ROI_WKT_FILE', '').strip()
    if vector and wkt_file:
        raise ValueError('Only one ROI vector or WKT file may be configured')
    return ('vector', vector) if vector else ('wkt', wkt_file) if wkt_file else (None, None)


def fingerprint(environ=None):
    kind, name = source(environ)
    if not name:
        return None
    path = Path(name)
    if not path.is_file():
        raise FileNotFoundError('ROI file does not exist: ' + name)
    paths = [path]
    if path.suffix.lower() == '.shp':
        # Sidecars define both geometry and CRS; incomplete shapefiles fail closed.
        for suffix in ('.shx', '.dbf', '.prj'):
            sidecar = path.with_suffix(suffix)
            if not sidecar.is_file():
                raise FileNotFoundError('Missing ROI sidecar: ' + str(sidecar))
            paths.append(sidecar)
        if path.with_suffix('.cpg').is_file():
            paths.append(path.with_suffix('.cpg'))
    digest = hashlib.sha256(kind.encode('ascii'))
    for item in paths:
        digest.update(item.suffix.lower().encode('ascii'))
        with item.open('rb') as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(block)
    return digest.hexdigest()


def _polygon(geometry):
    from osgeo import ogr
    if (geometry is None or geometry.IsEmpty() or not geometry.IsValid()
            or ogr.GT_Flatten(geometry.GetGeometryType()) not in (ogr.wkbPolygon, ogr.wkbMultiPolygon)):
        raise ValueError('ROI must contain valid nonempty Polygon/MultiPolygon geometry')
    geometry.FlattenTo2D()
    return geometry


def projected_wkt(projection):
    """Load exact coverage and transform it into the working raster CRS."""
    kind, name = source()
    if not name:
        return None
    fingerprint()  # Validate required files before GIS reads.
    from osgeo import ogr, osr
    if not projection:
        raise ValueError('Raster CRS is missing; cannot apply ROI')
    target = osr.SpatialReference()
    if target.ImportFromWkt(projection) != 0:
        raise ValueError('Invalid raster CRS')
    target.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    if kind == 'wkt':
        geometry = _polygon(ogr.CreateGeometryFromWkt(Path(name).read_text(encoding='utf-8').strip()))
        origin = osr.SpatialReference()
        origin.ImportFromEPSG(4326)
    else:
        dataset = ogr.Open(name, 0)
        if dataset is None or dataset.GetLayerCount() != 1:
            raise ValueError('ROI vector must have exactly one readable layer')
        layer = dataset.GetLayer(0)
        origin = layer.GetSpatialRef()
        if origin is None:
            raise ValueError('ROI vector CRS is missing')
        origin = origin.Clone()
        geometry = None
        for feature in layer:
            polygon = _polygon(feature.GetGeometryRef().Clone() if feature.GetGeometryRef() else None)
            geometry = polygon if geometry is None else geometry.Union(polygon)
        geometry = _polygon(geometry)
        dataset = None
    origin.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    if not origin.IsSame(target):
        if geometry.Transform(osr.CoordinateTransformation(origin, target)) != 0:
            raise ValueError('Failed to transform ROI into raster CRS')
    return _polygon(geometry).ExportToWkt()


def window_transform(transform, x, y):
    return (transform[0] + x * transform[1] + y * transform[2], transform[1], transform[2],
            transform[3] + x * transform[4] + y * transform[5], transform[4], transform[5])


class RoiScope:
    def __init__(self, wkt, transform, projection=''):
        from osgeo import ogr
        self.geometry = _polygon(ogr.CreateGeometryFromWkt(wkt))
        self.transform = tuple(transform)
        self.projection = projection

    def intersects(self, x, y, width, height):
        from osgeo import ogr
        ring = ogr.Geometry(ogr.wkbLinearRing)
        for column, row in ((x, y), (x + width, y), (x + width, y + height), (x, y + height), (x, y)):
            gt = window_transform(self.transform, column, row)
            ring.AddPoint_2D(gt[0], gt[3])
        polygon = ogr.Geometry(ogr.wkbPolygon)
        polygon.AddGeometry(ring)
        return bool(self.geometry.Intersects(polygon))

    def mask(self, x, y, width, height):
        from osgeo import gdal, ogr, osr
        vector = ogr.GetDriverByName('Memory').CreateDataSource('')
        srs = None
        if self.projection:
            srs = osr.SpatialReference()
            srs.ImportFromWkt(self.projection)
            srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
        layer = vector.CreateLayer('roi', srs=srs, geom_type=ogr.wkbUnknown)
        feature = ogr.Feature(layer.GetLayerDefn())
        feature.SetGeometry(self.geometry)
        if layer.CreateFeature(feature) != 0:
            raise RuntimeError('Failed to create ROI mask feature')
        raster = gdal.GetDriverByName('MEM').Create('', int(width), int(height), 1, gdal.GDT_Byte)
        raster.SetGeoTransform(window_transform(self.transform, x, y))
        if self.projection:
            raster.SetProjection(self.projection)
        raster.GetRasterBand(1).Fill(0)
        # Geometry already uses the raster CRS. Both MEM layers have no assigned
        # SRS to avoid implicit reprojection; use the same pixel-center convention.
        if gdal.RasterizeLayer(raster, [1], layer, burn_values=[1]) != 0:
            raise RuntimeError('Failed to rasterize ROI window')
        result = raster.GetRasterBand(1).ReadAsArray().astype(np.uint8, copy=False)
        raster, feature, layer, vector = None, None, None, None
        return result


def filter_indices(indices, wkt, transform, size, overlap=0):
    scope = RoiScope(wkt, transform)
    half = overlap // 2
    return [index for index in indices if scope.intersects(index[0] - half, index[1] - half, size, size)]


def polygonal_part(geometry):
    """Keep polygon parts when an intersection also contains touching lines."""
    from osgeo import ogr
    kind = ogr.GT_Flatten(geometry.GetGeometryType())
    if kind in (ogr.wkbPolygon, ogr.wkbMultiPolygon):
        return geometry
    if kind != ogr.wkbGeometryCollection:
        return None
    result = ogr.Geometry(ogr.wkbMultiPolygon)
    for index in range(geometry.GetGeometryCount()):
        part = polygonal_part(geometry.GetGeometryRef(index))
        if part is None or part.IsEmpty():
            continue
        if ogr.GT_Flatten(part.GetGeometryType()) == ogr.wkbPolygon:
            result.AddGeometry(part)
        else:
            for child in range(part.GetGeometryCount()):
                result.AddGeometry(part.GetGeometryRef(child))
    return result if result.GetGeometryCount() else None


def _valid_polygonal(geometry, context):
    """Repair invalid polygon topology; never silently drop an unrepairable area."""
    if geometry is None or geometry.IsEmpty():
        raise RuntimeError('Missing or empty geometry: ' + context)
    repaired = not geometry.IsValid()
    if repaired:
        if not hasattr(geometry, 'MakeValid'):
            raise RuntimeError('GDAL MakeValid unavailable: ' + context)
        geometry = geometry.MakeValid()
        if geometry is None:
            raise RuntimeError('Geometry MakeValid failed: ' + context)
    geometry = polygonal_part(geometry)
    if geometry is None or geometry.IsEmpty() or not geometry.IsValid():
        raise RuntimeError('No valid polygon after geometry repair: ' + context)
    return geometry, repaired


def clip_shapefile(path, wkt, projection, logger=None):
    """Clip repaired output polygons to exact coverage, preserving attributes."""
    from osgeo import ogr, osr
    dataset = ogr.Open(str(path), 1)
    if dataset is None:
        raise RuntimeError('Cannot open output for ROI clipping: ' + str(path))
    layer = feature = None
    repaired = removed = written = 0
    try:
        layer = dataset.GetLayer(0)
        target = layer.GetSpatialRef() if layer is not None else None
        if target is None:
            raise ValueError('Output vector CRS missing during ROI clipping')
        target = target.Clone()
        target.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
        origin = osr.SpatialReference()
        if origin.ImportFromWkt(projection) != 0:
            raise ValueError('Invalid raster CRS during ROI clipping')
        origin.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
        geometry = _polygon(ogr.CreateGeometryFromWkt(wkt))
        if not origin.IsSame(target):
            if geometry.Transform(osr.CoordinateTransformation(origin, target)) != 0:
                raise RuntimeError('Cannot transform ROI to output CRS')
        geometry, roi_repaired = _valid_polygonal(geometry, 'ROI after CRS transformation')
        identifiers = [item.GetFID() for item in layer]
        for identifier in identifiers:
            feature = layer.GetFeature(identifier)
            original = feature.GetGeometryRef() if feature is not None else None
            original, changed = _valid_polygonal(original, 'output FID=%s' % identifier)
            repaired += int(changed)
            try:
                clipped = original.Intersection(geometry)
            except RuntimeError as exc:
                raise RuntimeError('Output geometry intersection failed at FID=%s: %s' % (identifier, exc)) from exc
            if clipped is None:
                raise RuntimeError('Output geometry intersection failed at FID=%s' % identifier)
            clipped = polygonal_part(clipped)
            if clipped is None or clipped.IsEmpty():
                feature = None
                if layer.DeleteFeature(identifier) != 0:
                    raise RuntimeError('Cannot remove output outside ROI at FID=%s' % identifier)
                removed += 1
            else:
                clipped, changed = _valid_polygonal(clipped, 'clipped FID=%s' % identifier)
                repaired += int(changed)
                if feature.SetGeometry(clipped) not in (None, 0) or layer.SetFeature(feature) != 0:
                    raise RuntimeError('Cannot write ROI-clipped output at FID=%s' % identifier)
                written += 1
                feature = None
        if layer.SyncToDisk() != 0:
            raise RuntimeError('Cannot flush ROI-clipped output')
        if logger:
            logger.info('[范围修复] %s | 几何修复次数=%s, 范围修复=%s, 保留=%s, 范围外移除=%s',
                        Path(path).name, repaired, int(roi_repaired), written, removed)
        return {'repairs': repaired, 'roi_repaired': roi_repaired, 'written': written, 'removed': removed}
    finally:
        feature, layer, dataset = None, None, None
