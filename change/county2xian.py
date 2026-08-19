import geopandas as gpd

def assign_field_by_max_overlap(gdf_target, gdf_source, source_field, target_field):
    """
    根据最大相交面积，将source的字段赋值给target

    参数:
        gdf_target: 目标GeoDataFrame（需要被赋值的）
        gdf_source: 源GeoDataFrame（提供字段值的）
        source_field: 源字段名（如'county'）
        target_field: 目标字段名（如'xian'）
    """
    # 统一坐标系
    if gdf_target.crs != gdf_source.crs:
        gdf_source = gdf_source.to_crs(gdf_target.crs)
    # 检查是否存在名为'index'的字段
    if 'index' in gdf_source.columns:
        # 删除index字段
        print('删除index字段')
        gdf_source = gdf_source.drop(columns=['index'])
    else:
        gdf_source = gdf_source.reset_index(drop=True)

    # 添加临时ID
    gdf_target['temp_uid'] = range(len(gdf_target))

    # 相交并计算面积
    overlay = gpd.overlay(gdf_target, gdf_source, how='intersection')
    # 删除overlay中可能产生的index列
    overlay = overlay.drop(columns=[col for col in overlay.columns if 'index' in col.lower()], errors='ignore')
    if 'index' in overlay.columns:  # 双重保险
        overlay = overlay.drop(columns=['index'])
    overlay['area'] = overlay.geometry.area

    # 按最大面积匹配
    max_area_idx = overlay.groupby("temp_uid")["area"].idxmax()
    classified = overlay.loc[max_area_idx, ["temp_uid", source_field]]

    # 合并赋值
    gdf_target = gdf_target.merge(classified, on='temp_uid', how='left')
    gdf_target[target_field] = gdf_target[source_field]

    # 清理临时字段
    gdf_target = gdf_target.drop(columns=['temp_uid', source_field])

    return gdf_target
