import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from test_lib_batch_memeff_single_image_nomp import test_lib_big_memeff
from pathlib import Path
import geopandas as gpd

def add_required_fields(shp_path, inplace=True):
    gdf = gpd.read_file(shp_path)
    # 需要添加的字段及默认值
    fields_defaults = {
        'pre_code': 0,
        'curr_code': 0,
    }
    
    # 添加不存在的字段
    for field, default_val in fields_defaults.items():
        if field not in gdf.columns:
            gdf[field] = default_val
            # 设置正确的数据类型
            if isinstance(default_val, int):
                gdf[field] = gdf[field].astype(int)
            else:
                gdf[field] = gdf[field].astype(str)
    
    gdf.to_file(shp_path, encoding='utf-8')
    print(f"字段添加完成，保存至：{shp_path}")

if __name__ == "__main__":
    """
    prefile = "/mnt/prod/k3s/working/cd_seg/testdata/pre/R_417_203_1668_815.tif"
    postfile = "/mnt/prod/k3s/working/cd_seg/testdata/post/R_417_203_1668_815.tif"
    output_file = '/mnt/prod/k3s/working/cd_seg/output/R_417_203_1668_815.shp'
    """
    prefile = "/share/test/pre/107445_107671_24095_24317.tif"
    postfile = "/share/test/post/107445_107671_24095_24317.tif"
    output_file = '/output/R_test.shp'
    test_lib_big_memeff(prefile,postfile,output_file)
    # add_required_fields(output_file)
