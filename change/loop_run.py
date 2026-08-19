import os
import geopandas as gpd
import pandas as pd
from get_intersecting_raster_names import get_intersecting_raster_names
from process_shp_by_county import process_shp_by_county
from test_lib_batch_memeff_single_image import test_lib_big_memeff, run_parallel


def merge_shapefiles(shapefile_paths, output_path):
    """
    使用GeoPandas合并Shapefile文件
    """
    print(f"开始合并 {len(shapefile_paths)} 个Shapefile...")

    gdfs = []
    # 读取所有文件
    for i, path in enumerate(shapefile_paths, 1):
        print(f"读取文件 {i}: {os.path.basename(path)}")

        # 尝试不同编码
        for encoding in ['utf-8', 'gbk', 'latin1']:
            # import pdb;pdb.set_trace()
            gdf = gpd.read_file(path, encoding=encoding)
            print(f"  ✓ 读取成功 ({len(gdf)} 要素，编码: {encoding})")
            gdfs.append(gdf)
            break
        # except Exception as e:
            #    if encoding == 'latin1':  # 最后一个编码
            #        print(f"  ✗ 读取失败: {e}")

    if not gdfs:
        print("❌ 没有成功读取任何文件")
        return False

    # 合并数据
    print("合并数据...")
    merged_gdf = pd.concat(gdfs, ignore_index=True)
    print(f"合并后总要素数: {len(merged_gdf)}")

    # 保存文件
    print(f"保存到: {output_path}")
    merged_gdf.to_file(output_path, encoding='utf-8')

    # 创建编码文件确保中文正常显示
    cpg_path = output_path.replace('.shp', '.cpg')
    with open(cpg_path, 'w') as f:
        f.write('UTF-8')

    print("✅ 合并完成!")
    return True


def prediction_function(post_img, pre_img, county_id, shp_path, logger, callback_url, job_id):
    """
    预测函数 - 处理单个shapefile的预测分类
    参数:
    post_img: str - 后时相影像路径
    pre_img: str - 前时相影像路径
    county_id: str - 区县ID
    shp_path: str - 输出shp路径
    返回:
    str - 生成的shp文件路径
    """
    gdb_path = "./assets/2023年广东省分县现状.gdb"
    # import pdb;pdb.set_trace()
    run_parallel([pre_img], [post_img], shp_path, 4, logger, callback_url, job_id)
    # test_lib_big_memeff(pre_img, post_img, shp_path, logger, callback_url, job_id)
    process_shp_by_county(shp_path, gdb_path, county_id)

    return shp_path


def process_images_sequential(image_dict, output_dir, process_func, output_dir_final, logger, callback_url, job_id):
    """
    顺序处理所有影像对并合并结果
    参数:
    image_dict: dict - 影像字典 {后时相: {前时相: 县区ID}}
    output_dir: str - 输出目录
    process_func: function - 处理函数
    返回:
    str - 合并后的Shapefile路径
    """
    if not image_dict:
        print("❌ 影像字典为空")
        return None

    result_shps = []
    total_tasks = sum(len(pre_dict) for pre_dict in image_dict.values())
    current_task = 0

    print(f"总共需要处理 {total_tasks} 个影像对")

    # 循环处理每个影像对
    for post_img, pre_dict in image_dict.items():
        for pre_img, county_id in pre_dict.items():
            current_task += 1

            try:
                # import pdb;pdb.set_trace()
                # 生成输出文件名
                post_name = os.path.basename(post_img).split('.')[0]
                pre_name = os.path.basename(pre_img).split('.')[0]
                output_filename = f"output_{post_name}_{pre_name}_{county_id}.shp"
                output_shp_path = os.path.join(output_dir, output_filename)

                print(f"[{current_task}/{total_tasks}] 处理: 后时相={post_name}, 前时相={pre_name}, 县区={county_id}")

                # 调用处理函数
                # import pdb;pdb.set_trace()
                output_shp = process_func(post_img, pre_img, county_id, output_shp_path, logger, callback_url, job_id)

                if output_shp and os.path.exists(output_shp):
                    result_shps.append(output_shp)
                    print(f"  ✓ 完成")
                else:
                    print(f"  ✗ 处理失败或文件未生成")

            except Exception as e:
                print(f"  ✗ 处理出错: {e}")
                continue

    # 合并所有生成的shp文件
    if result_shps:
        print(f"\n成功处理了 {len(result_shps)} 个文件，开始合并...")
        merged_output = output_dir_final
        # merged_output = os.path.join(output_dir_final, "merged_result.shp")

        if merge_shapefiles(result_shps, merged_output):
            # 可选：删除临时文件
            print("清理临时文件...")
            for shp_file in result_shps:
                try:
                    # 删除shp及相关文件
                    base_name = os.path.splitext(shp_file)[0]
                    for ext in ['.shp', '.shx', '.dbf', '.prj', '.cpg']:
                        file_to_delete = base_name + ext
                        if os.path.exists(file_to_delete):
                            os.remove(file_to_delete)
                except:
                    pass

            print(f"✅ 所有处理完成！最终结果: {merged_output}")
            return merged_output
        else:
            print("❌ 合并失败")
            return None
    else:
        print("❌ 没有成功处理任何文件")
        return None


if __name__ == "__main__":
    # 配置路径
    shp_path = os.getenv("PATH_SHP")  # 分幅矢量
    raster_path = os.getenv("PATH_POST_IMG")  # 后时相影像
    output_dir = os.getenv("PATH_RESULT")  # 最终输出图斑目录

    image_dict = get_intersecting_raster_names(shp_path, raster_path)

    # 顺序处理所有影像对
    result_shp = process_images_sequential(
        image_dict=image_dict,
        output_dir=output_dir,
        process_func=prediction_function
    )
