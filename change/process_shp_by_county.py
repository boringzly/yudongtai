import geopandas as gpd
import pandas as pd
import os
from county2xian import *


def process_prediction_with_division(prediction_shp_path, gdb_path, division_shp_path=None, county_ids=None):
    """
    处理预测结果图斑：
    从gdb对应图层的ld2023字段赋值给预测结果shp的pre_code字段

    参数:
    - prediction_shp_path: 预测结果shp文件路径
    - gdb_path: gdb文件路径
    - division_shp_path: 分幅shp文件路径（包含Countyid字段），当county_ids为None时必需
    - county_ids: 可选，手动指定的区县ID列表，如果提供则不需要division_shp_path
    """

    print("=== 开始处理预测结果图斑 ===")

    # 1. 读取预测结果shp
    print("正在读取预测结果shapefile...")
    prediction_gdf_tmp = gpd.read_file(prediction_shp_path)
    origin_crs = prediction_gdf_tmp.crs
    print(f"预测结果数据量: {len(prediction_gdf_tmp)}")

    # 添加pre_code字段（如果不存在）
    if 'pre_code' not in prediction_gdf_tmp.columns:
        prediction_gdf_tmp['pre_code'] = None
        print("已添加pre_code字段")

    if 'xian' not in prediction_gdf_tmp.columns:
        prediction_gdf_tmp['xian'] = '0'
        #prediction_gdf_tmp['xian'].fillna('0').astype(str)
        #prediction_gdf_tmp['xian'] = prediction_gdf_tmp['xian'].astype(str)
        print("已添加xian字段")
    else:
        prediction_gdf_tmp['xian'] = prediction_gdf_tmp['xian'].fillna('0').astype(str)

    #fenfushp = gpd.read_file("./assets/2020ditu_with_pac/2020ditu_with_pac.shp")
    #prediction_gdf  = assign_field_by_max_overlap(prediction_gdf_tmp,  fenfushp, 'County', 'xian')
    countyshp = gpd.read_file("./assets/countyshp/xian201912.shp")
    countyshp['PAC'] = countyshp['PAC'].astype(str)
    prediction_gdf  = assign_field_by_max_overlap(prediction_gdf_tmp,  countyshp, 'PAC', 'xian')
    prediction_gdf['xian'] = prediction_gdf['xian'].astype(str)
    
    prediction_gdf = prediction_gdf.drop(columns=[col for col in prediction_gdf.columns if 'index' in col.lower()], errors='ignore')
    if 'index' in prediction_gdf.columns:
        prediction_gdf = prediction_gdf.drop(columns=['index'])
        print("已删除prediction_gdf中的index列")
    # result.to_file(prediction_shp_path)

    # 为每行添加唯一标识
    prediction_gdf['temp_uid'] = range(len(prediction_gdf))

    # 2. 获取区县ID列表
    if county_ids is not None:
        print(f"使用手动提供的区县列表: {county_ids}")
        print("跳过分幅shp处理")
    else:
        if division_shp_path is None:
            raise ValueError("当county_ids为None时，必须提供division_shp_path")

        # 读取分幅shp
        print("正在读取分幅shapefile...")
        division_gdf = gpd.read_file(division_shp_path)
        print(f"分幅数据量: {len(division_gdf)}")

        # 检查Countyid字段
        if 'Countyid' not in division_gdf.columns:
            print("错误: 分幅shp中没有Countyid字段")
            print(f"可用字段: {list(division_gdf.columns)}")
            return prediction_gdf

        # 坐标系统一
        if prediction_gdf.crs != division_gdf.crs:
            print("统一坐标系...")
            division_gdf = division_gdf.to_crs(prediction_gdf.crs)

        # 空间相交获取Countyid列表
        print("正在进行空间相交分析...")
        intersection = gpd.overlay(prediction_gdf, division_gdf, how="intersection")
        print(f"相交结果数量: {len(intersection)}")

        if len(intersection) == 0:
            print("警告: 预测结果与分幅没有相交部分")
            return prediction_gdf

        # 获取所有相交的Countyid值
        county_ids = intersection['Countyid'].dropna().unique().tolist()

    print(f"最终处理的区县图层: {county_ids}")

    # 3. 检查gdb文件
    if not os.path.exists(gdb_path):
        print(f"错误: gdb文件不存在: {gdb_path}")
        return prediction_gdf

    # 统计初始状态
    initial_null_count = prediction_gdf['pre_code'].isna().sum()
    print(f"初始pre_code为空的记录数: {initial_null_count}")

    processed_count = 0
    error_count = 0

    # 4. 循环处理每个区县图层
    for county_id in county_ids:
        print(f"\n正在处理区县图层: {county_id}")

        try:
            # 读取gdb中的特定图层
            landuse_gdf = gpd.read_file(gdb_path, layer=county_id)
            print(f"图层数据量: {len(landuse_gdf)}")
            ld = landuse_gdf.columns[0]
            # 检查gdb中的ld2023字段
            """
            if 'ld2024' not in landuse_gdf.columns:
                print("警告: gdb图层中没有ld2024字段")
                print(f"可用字段: {list(landuse_gdf.columns)}")
                continue
            """
            # 显示ld2023字段的信息
            non_null_ld = landuse_gdf[ld].dropna()
            print(f"{ld}非空数量: {len(non_null_ld)}")
            if len(non_null_ld) > 0:
                unique_ld = non_null_ld.unique()
                print(f"{ld}唯一值数量: {len(unique_ld)}")
                print(f"{ld}示例值: {unique_ld[:5]}")

            # 复制预测结果数据用于处理
            temp_prediction_gdf = prediction_gdf.copy()

            # 坐标系统一 - 将预测结果转换到gdb的坐标系
            if temp_prediction_gdf.crs != landuse_gdf.crs:
                print("转换预测结果坐标系...")
                temp_prediction_gdf = temp_prediction_gdf.to_crs(landuse_gdf.crs)

            print("执行空间相交操作...")
            # 求相交
            overlay = gpd.overlay(temp_prediction_gdf, landuse_gdf, how="intersection")
            print(f"相交结果数量: {len(overlay)}")
            overlay = overlay.drop(columns=[col for col in overlay.columns if 'index' in col.lower()], errors='ignore')
            # 删除可能产生的index列
            if 'index' in overlay.columns:
                overlay = overlay.drop(columns=['index'])
                print("已删除overlay中的index列")

            if len(overlay) == 0:
                print(f"图层 {county_id} 与预测结果无相交，跳过")
                continue

            # 计算相交面积
            overlay["area"] = overlay.geometry.area

            # 按temp_uid分组，找到面积最大的相交部分，获取对应的ld2023值
            max_area_idx = overlay.groupby("temp_uid")["area"].idxmax()
            classified = overlay.loc[max_area_idx, ["temp_uid", ld]].reset_index(drop=True)

            print(f"可分类的图斑数量: {len(classified)}")

            # 显示要赋值的ld2023统计
            if len(classified) > 0:
                ld_counts = classified[ld].value_counts()
                print(f"本次要赋值的ld统计:")
                print(ld_counts.head(10))

            # 更新pre_code值
            updated_this_round = 0
            for idx, row in classified.iterrows():
                temp_uid = row['temp_uid']
                ld_value = row[ld]

                # 转换数据类型
                if pd.notna(ld_value):
                    ld_value = int(ld_value)

                # 找到对应的行
                uid_mask = (prediction_gdf['temp_uid'] == temp_uid)
                if not uid_mask.any():
                    continue

                current_pre_code = prediction_gdf.loc[uid_mask, 'pre_code'].iloc[0]

                # 检查是否为0或空值
                is_zero = (current_pre_code == 0) or (current_pre_code == '0') or (str(current_pre_code) == '0')
                is_null = pd.isna(current_pre_code)

                if is_zero or is_null:
                    # 应用映射规则
                    mapping_dict = {
                        111: 11, 112: 11, 113: 11, 114: 11, 115: 11,
                        121: 12, 122: 12, 123: 12, 124: 12, 125: 12
                    }
                    mapped_value = mapping_dict.get(ld_value, ld_value)
                    prediction_gdf.loc[uid_mask, 'pre_code'] = mapped_value
                    # prediction_gdf.loc[uid_mask, 'xian'] = county_id
                    updated_this_round += 1
                    processed_count += 1

            print(f"本轮更新记录数: {updated_this_round}")

        except Exception as e:
            print(f"处理图层 {county_id} 时出错: {e}")
            import traceback
            traceback.print_exc()
            error_count += 1
            continue

    # 5. 数据类型处理和结果统计
    try:
        # 将pre_code转换为整数类型（处理空值）
        prediction_gdf["pre_code"] = pd.to_numeric(prediction_gdf["pre_code"], errors='coerce')
        prediction_gdf["pre_code"] = prediction_gdf["pre_code"].astype('Int64')  # 支持空值的整数类型
    except Exception as e:
        print(f"数据类型转换警告: {e}")

    # 统计最终结果
    final_null_count = prediction_gdf['pre_code'].isna().sum()
    final_filled_count = len(prediction_gdf) - final_null_count

    print(f"\n=== 处理结果统计 ===")
    print(f"总记录数: {len(prediction_gdf)}")
    print(f"成功赋值记录数: {processed_count}")
    print(f"处理失败: {error_count} 个图层")
    print(f"最终有pre_code值的记录数: {final_filled_count}")
    print(f"最终pre_code为空的记录数: {final_null_count}")

    # 显示pre_code的分布统计
    if final_filled_count > 0:
        pre_code_final_counts = prediction_gdf['pre_code'].value_counts()
        print(f"最终pre_code分布:")
        print(pre_code_final_counts.head(10))

    # 6. 清理临时列
    prediction_gdf.drop(columns=['temp_uid'], inplace=True)
    prediction_gdf = prediction_gdf.drop(columns=[col for col in prediction_gdf.columns if 'index' in col.lower()], errors='ignore')
    if 'index' in prediction_gdf.columns:
        prediction_gdf = prediction_gdf.drop(columns=['index'])
        print("已删除最终结果中的index列")

    # 恢复原始坐标系
    if origin_crs and origin_crs != prediction_gdf.crs:
        print("恢复原始坐标系...")
        prediction_gdf = prediction_gdf.to_crs(origin_crs)

    # 7. 保存结果
    prediction_gdf.to_file(prediction_shp_path)
    print(f"结果已更新到原文件: {prediction_shp_path}")

    return prediction_gdf


if __name__ == "__main__":
    # 使用示例
    prediction_shp_path = r'E:\project\四维\output0921\output_ecomn_2024_13_153_F49D008007_ld440902,ld441721,ld440904.shp'
    gdb_path = r'E:\project\四维\2023年广东省分县现状.gdb'

    # 方法1：手动提供区县ID列表（推荐，不需要分幅shp）
    county_ids = ['ld440902', 'ld441721', 'ld440904']
    result = process_prediction_with_division(prediction_shp_path, gdb_path, county_ids=county_ids)
