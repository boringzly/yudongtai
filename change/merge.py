from pathlib import Path

from osgeo import ogr, osr


def _traditional_axis_order(spatial_ref):
    if spatial_ref is not None and hasattr(spatial_ref, 'SetAxisMappingStrategy'):
        spatial_ref.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    return spatial_ref


def merge_shp(shp_files, save_shp_path):
    """统一到首个文件的 CRS 后合并；全空输入也生成带字段的 UTF-8 SHP。"""
    if not shp_files:
        raise ValueError('没有可合并的 SHP 文件')

    driver = ogr.GetDriverByName('ESRI Shapefile')
    if driver is None:
        raise RuntimeError('OGR ESRI Shapefile 驱动不可用')

    input_paths = [str(path) for path in shp_files]
    template_ds = driver.Open(input_paths[0], 0)
    if template_ds is None:
        raise RuntimeError(f'无法打开待合并 SHP: {input_paths[0]}')
    template_layer = template_ds.GetLayer()
    if template_layer is None:
        template_ds = None
        raise RuntimeError(f'待合并 SHP 没有有效图层: {input_paths[0]}')

    target_srs = template_layer.GetSpatialRef()
    target_srs = _traditional_axis_order(target_srs.Clone()) if target_srs is not None else None
    geometry_type = template_layer.GetGeomType()
    template_definition = template_layer.GetLayerDefn()

    output_path = Path(save_shp_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        driver.DeleteDataSource(str(output_path))
    output_stem = str(output_path.with_suffix(''))
    for extension in ('.shp', '.shx', '.dbf', '.prj', '.cpg', '.qix', '.sbn', '.sbx', '.fix', '.shp.xml'):
        sidecar = Path(output_stem + extension)
        if sidecar.is_file():
            sidecar.unlink()

    output_ds = driver.CreateDataSource(str(output_path))
    if output_ds is None:
        template_ds = None
        raise RuntimeError(f'无法创建合并 SHP: {output_path}')
    output_layer = output_ds.CreateLayer(
        output_path.stem,
        srs=target_srs,
        geom_type=geometry_type,
        options=['ENCODING=UTF-8'],
    )
    if output_layer is None:
        output_ds = None
        template_ds = None
        raise RuntimeError(f'无法创建合并 SHP 图层: {output_path}')

    for index in range(template_definition.GetFieldCount()):
        if output_layer.CreateField(template_definition.GetFieldDefn(index)) != 0:
            output_ds = None
            template_ds = None
            raise RuntimeError(f'无法创建合并 SHP 字段: {output_path}')
    template_ds = None

    output_definition = output_layer.GetLayerDefn()
    output_field_names = [
        output_definition.GetFieldDefn(index).GetNameRef()
        for index in range(output_definition.GetFieldCount())
    ]

    for input_path in input_paths:
        input_ds = driver.Open(input_path, 0)
        if input_ds is None:
            output_ds = None
            raise RuntimeError(f'无法打开待合并 SHP: {input_path}')
        input_layer = input_ds.GetLayer()
        if input_layer is None:
            input_ds = None
            output_ds = None
            raise RuntimeError(f'待合并 SHP 没有有效图层: {input_path}')
        source_srs = input_layer.GetSpatialRef()
        source_srs = _traditional_axis_order(source_srs.Clone()) if source_srs is not None else None

        transform = None
        if target_srs is not None and source_srs is not None and not source_srs.IsSame(target_srs):
            transform = osr.CoordinateTransformation(source_srs, target_srs)
        elif (target_srs is None) != (source_srs is None):
            input_ds = None
            output_ds = None
            raise RuntimeError(f'待合并 SHP 坐标系缺失或不一致: {input_path}')

        for input_feature in input_layer:
            output_feature = ogr.Feature(output_definition)
            for field_name in output_field_names:
                field_index = input_feature.GetFieldIndex(field_name)
                field_value = input_feature.GetField(field_index) if field_index >= 0 else None
                if field_value is not None:
                    output_feature.SetField(field_name, field_value)

            geometry = input_feature.GetGeometryRef()
            if geometry is not None:
                geometry = geometry.Clone()
                if transform is not None and geometry.Transform(transform) != 0:
                    input_ds = None
                    output_ds = None
                    raise RuntimeError(f'合并 SHP 坐标转换失败: {input_path}')
                output_feature.SetGeometry(geometry)
            if output_layer.CreateFeature(output_feature) != 0:
                input_ds = None
                output_ds = None
                raise RuntimeError(f'写入合并 SHP 失败: {output_path}')
            output_feature = None
        input_ds = None

    output_ds = None
    if not output_path.exists():
        raise RuntimeError(f'合并 SHP 未生成: {output_path}')
    return str(output_path)
