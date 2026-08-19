from osgeo import gdal, ogr, osr


def get_intersecting_raster_names_and_counties(shp_path, raster_path):
    """
    获取与栅格相交的矢量要素信息

    参数:
    - shp_path: 矢量文件路径
    - raster_path: 栅格文件路径

    返回:
    - intersecting_names: 相交的影像名称列表（去重）
    - county_ids: 相交的区县ID列表（去重）
    """
    # 获取栅格影像的范围
    raster_ds = gdal.Open(raster_path)
    if raster_ds is None:
        raise Exception(f"无法打开栅格文件: {raster_path}")

    geotransform = raster_ds.GetGeoTransform()
    cols = raster_ds.RasterXSize
    rows = raster_ds.RasterYSize

    # 计算边界坐标
    left = geotransform[0]
    top = geotransform[3]
    right = left + geotransform[1] * cols
    bottom = top + geotransform[5] * rows

    raster_ds = None

    # 创建栅格范围的多边形几何体
    ring = ogr.Geometry(ogr.wkbLinearRing)
    ring.AddPoint(left, bottom)
    ring.AddPoint(right, bottom)
    ring.AddPoint(right, top)
    ring.AddPoint(left, top)
    ring.AddPoint(left, bottom)

    raster_polygon = ogr.Geometry(ogr.wkbPolygon)
    raster_polygon.AddGeometry(ring)

    # 打开矢量文件
    driver = ogr.GetDriverByName('ESRI Shapefile')
    vector_ds = driver.Open(shp_path, 0)
    if vector_ds is None:
        raise Exception(f"无法打开矢量文件: {shp_path}")

    vector_layer = vector_ds.GetLayer()

    # 设置空间过滤器
    vector_layer.SetSpatialFilter(raster_polygon)

    # 使用集合避免重复
    intersecting_names_set = set()
    county_ids_set = set()

    feature = vector_layer.GetNextFeature()

    while feature:
        geometry = feature.GetGeometryRef()
        if geometry and geometry.Intersect(raster_polygon):
            # 获取影像名称
            name = feature.GetField('Name')
            if name:
                intersecting_names_set.add(name)

            # 获取区县ID
            county_id = feature.GetField('Countyid')
            if county_id:
                county_ids_set.add(county_id)

        feature = vector_layer.GetNextFeature()

    vector_ds = None

    # 转换为列表
    intersecting_names = list(intersecting_names_set)
    county_ids = list(county_ids_set)

    print(f"相交的影像名称: {intersecting_names}")
    print(f"相交的区县ID: {county_ids}")

    return intersecting_names, county_ids


def get_intersecting_raster_names(shp_path, raster_path):
    """
    保持原有函数的兼容性，只返回影像名称列表
    """
    intersecting_names, _ = get_intersecting_raster_names_and_counties(shp_path, raster_path)
    return intersecting_names


def get_intersecting_county_ids(shp_path, raster_path):
    """
    只返回区县ID列表的便捷函数
    """
    _, county_ids = get_intersecting_raster_names_and_counties(shp_path, raster_path)
    return county_ids


if __name__ == '__main__':
    shp_path = r'E:\project\四维\2020ditu_with_pac\2020ditu_with_pac.shp'
    raster_path = r'D:\private\jiangbc\bistu_2nd_term\tea_map\GF2_selected\S_852_421_3411_1685.tif'

    # 方法1：获取两个列表
    image_names, county_ids = get_intersecting_raster_names_and_counties(shp_path, raster_path)

    # 方法2：只获取影像名称（保持兼容性）
    # image_names = get_intersecting_raster_names(shp_path, raster_path)

    # 方法3：只获取区县ID
    # county_ids = get_intersecting_county_ids(shp_path, raster_path)