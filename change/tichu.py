import geopandas as gpd


def filter_shp_by_area(input_shp, output_shp, min_area=400):
    """
    过滤shp图斑，去除面积小于指定值的图斑

    参数:
    input_shp: 输入shp文件路径
    output_shp: 输出shp文件路径
    min_area: 最小面积阈值（平方米），默认400
    """

    # 读取shp文件
    gdf = gpd.read_file(input_shp)

    # 如果已经是投影坐标系，直接计算面积
    if gdf.crs.is_projected:
        print("当前为投影坐标系，直接计算面积...")
        gdf['area_m2'] = gdf.geometry.area
    else:
        # 地理坐标系需要转换为投影坐标系计算面积
        print("当前为地理坐标系，转换为投影坐标系计算面积...")

        # 使用适用于中国的投影坐标系（UTM或Albers）
        # 这里使用CGCS2000 3 Degree GK Zone 40 (EPSG:4539)
        target_crs = 'EPSG:4539'  # 适用于中国大部分地区

        # 转换为投影坐标系
        gdf_projected = gdf.to_crs(target_crs)
        gdf['area_m2'] = gdf_projected.geometry.area

    # 过滤图斑
    gdf_filtered = gdf[gdf['area_m2'] >= min_area]

    print(f"\n过滤结果:")
    print(f"过滤前图斑数量: {len(gdf)}")
    print(f"过滤后图斑数量: {len(gdf_filtered)}")
    print(f"去除图斑数量: {len(gdf) - len(gdf_filtered)}")
    print(input_shp)
    # 保存结果
    gdf_filtered.to_file(output_shp, encoding='utf-8')
    print(f"\n结果已保存至: {output_shp}")
