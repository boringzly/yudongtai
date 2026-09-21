"""Assign province names to final classification polygons."""

from pathlib import Path

from osgeo import ogr, osr


PROVINCE_FIELD = "province"
UNKNOWN_PROVINCE = "未知"
DEFAULT_PROVINCE_DATA = Path(__file__).resolve().parent / "assets" / "china_provinces.geojson"


def _traditional_axis_order(spatial_ref):
    if spatial_ref is not None and hasattr(spatial_ref, "SetAxisMappingStrategy"):
        spatial_ref.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    return spatial_ref


def _envelopes_intersect(left, right):
    return not (
        left[1] < right[0]
        or right[1] < left[0]
        or left[3] < right[2]
        or right[3] < left[2]
    )


def _make_valid(geometry):
    if geometry is None or geometry.IsEmpty() or geometry.IsValid():
        return geometry
    if hasattr(geometry, "MakeValid"):
        valid_geometry = geometry.MakeValid()
    else:
        valid_geometry = geometry.Buffer(0)
    if valid_geometry is None or valid_geometry.IsEmpty():
        return geometry
    return valid_geometry


def _load_provinces(province_data_path):
    province_data_path = Path(province_data_path)
    if not province_data_path.is_file():
        raise FileNotFoundError(f"省界数据不存在: {province_data_path}")

    data_source = ogr.Open(str(province_data_path), 0)
    if data_source is None:
        raise RuntimeError(f"无法打开省界数据: {province_data_path}")
    layer = data_source.GetLayer()
    if layer is None:
        data_source = None
        raise RuntimeError(f"省界数据没有有效图层: {province_data_path}")

    spatial_ref = layer.GetSpatialRef()
    if spatial_ref is None:
        data_source = None
        raise RuntimeError(f"省界数据缺少坐标系: {province_data_path}")
    spatial_ref = _traditional_axis_order(spatial_ref.Clone())

    provinces = []
    for feature in layer:
        name = feature.GetField("name")
        national_code = feature.GetField("gb")
        geometry_ref = feature.GetGeometryRef()
        if not name or not national_code or geometry_ref is None:
            continue
        if "POLYGON" not in geometry_ref.GetGeometryName().upper():
            continue
        geometry = _make_valid(geometry_ref.Clone())
        if geometry is None or geometry.IsEmpty():
            continue
        provinces.append((str(name), geometry.GetEnvelope(), geometry))
    data_source = None

    if not provinces:
        raise RuntimeError(f"省界数据中没有有效省级面要素: {province_data_path}")
    return spatial_ref, provinces


def _province_for_geometry(geometry, provinces):
    if geometry is None or geometry.IsEmpty():
        return UNKNOWN_PROVINCE

    geometry = _make_valid(geometry)
    target_envelope = geometry.GetEnvelope()
    best_name = UNKNOWN_PROVINCE
    best_area = 0.0

    for province_name, province_envelope, province_geometry in provinces:
        if not _envelopes_intersect(target_envelope, province_envelope):
            continue
        try:
            if province_geometry.Contains(geometry):
                return province_name
            if not province_geometry.Intersects(geometry):
                continue
            intersection = province_geometry.Intersection(geometry)
            if intersection is None or intersection.IsEmpty():
                continue
            intersection_area = intersection.GetArea()
            if intersection_area > best_area:
                best_area = intersection_area
                best_name = province_name
        except RuntimeError:
            # 单个无效交集不应阻止其他省份继续参与匹配。
            continue
    return best_name


def assign_province_names(
    shp_path,
    province_data_path=DEFAULT_PROVINCE_DATA,
    field_name=PROVINCE_FIELD,
    logger=None,
):
    """Add/update a province-name field using the largest polygon intersection."""
    shp_path = Path(shp_path)
    if not shp_path.is_file():
        raise FileNotFoundError(f"待赋省名的 SHP 不存在: {shp_path}")

    province_srs, provinces = _load_provinces(province_data_path)
    driver = ogr.GetDriverByName("ESRI Shapefile")
    if driver is None:
        raise RuntimeError("OGR ESRI Shapefile 驱动不可用")
    data_source = driver.Open(str(shp_path), 1)
    if data_source is None:
        raise RuntimeError(f"无法以更新模式打开 SHP: {shp_path}")
    layer = data_source.GetLayer()
    if layer is None:
        data_source = None
        raise RuntimeError(f"SHP 没有有效图层: {shp_path}")

    target_srs = layer.GetSpatialRef()
    if target_srs is None:
        data_source = None
        raise RuntimeError(f"SHP 缺少坐标系，无法赋省名: {shp_path}")
    target_srs = _traditional_axis_order(target_srs.Clone())
    transform = None
    if not target_srs.IsSame(province_srs):
        transform = osr.CoordinateTransformation(target_srs, province_srs)

    assignments = []
    unmatched_count = 0
    for feature in layer:
        geometry_ref = feature.GetGeometryRef()
        geometry = geometry_ref.Clone() if geometry_ref is not None else None
        if geometry is not None and transform is not None and geometry.Transform(transform) != 0:
            data_source = None
            raise RuntimeError(f"图斑坐标转换失败，FID={feature.GetFID()}: {shp_path}")
        province_name = _province_for_geometry(geometry, provinces)
        if province_name == UNKNOWN_PROVINCE:
            unmatched_count += 1
        assignments.append((feature.GetFID(), province_name))

    field_index = layer.GetLayerDefn().GetFieldIndex(field_name)
    if field_index < 0:
        field_definition = ogr.FieldDefn(field_name, ogr.OFTString)
        field_definition.SetWidth(40)
        if layer.CreateField(field_definition) != 0:
            data_source = None
            raise RuntimeError(f"无法创建省名称字段 {field_name}: {shp_path}")
        field_index = layer.GetLayerDefn().GetFieldIndex(field_name)
    if field_index < 0:
        data_source = None
        raise RuntimeError(f"创建后仍未找到省名称字段 {field_name}: {shp_path}")

    actual_field_name = layer.GetLayerDefn().GetFieldDefn(field_index).GetNameRef()
    for feature_id, province_name in assignments:
        feature = layer.GetFeature(feature_id)
        if feature is None:
            data_source = None
            raise RuntimeError(f"写入省名时无法读取图斑 FID={feature_id}: {shp_path}")
        feature.SetField(actual_field_name, province_name)
        if layer.SetFeature(feature) != 0:
            data_source = None
            raise RuntimeError(f"写入省名失败，FID={feature_id}: {shp_path}")
        feature = None
    data_source = None

    result = {
        "feature_count": len(assignments),
        "matched_count": len(assignments) - unmatched_count,
        "unmatched_count": unmatched_count,
        "field_name": actual_field_name,
    }
    if logger is not None:
        logger.info(
            "省名称赋值完成: 总数=%s, 已匹配=%s, 未匹配=%s, 字段=%s",
            result["feature_count"],
            result["matched_count"],
            result["unmatched_count"],
            result["field_name"],
        )
    return result
