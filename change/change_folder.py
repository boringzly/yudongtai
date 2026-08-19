import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from test_lib_batch_memeff_single_image_nomp import test_lib_big_memeff
from pathlib import Path
import geopandas as gpd

def process_folder(pre_folder, post_folder, output_folder):
    pre_folder = Path(pre_folder)
    post_folder = Path(post_folder)
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    
    # 获取pre_folder中所有tif/tiff文件
    pre_files = list(pre_folder.glob("*.tif")) + list(pre_folder.glob("*.tiff"))
    for pre_path in pre_files:
        stem = pre_path.stem  # 不含扩展名的基础名
        # 在post_folder中查找同名文件，支持tif/tiff扩展名
        post_candidates = list(post_folder.glob(f"{stem}.tif")) + list(post_folder.glob(f"{stem}.tiff"))
        if not post_candidates:
            print(f"Warning: No matching post file found for {pre_path.name}. Skipping.")
            continue
        # 如果找到多个，取第一个
        post_path = post_candidates[0]
        output_shp = output_folder / f"{stem}.shp"
        print(f"Processing: pre={pre_path}, post={post_path}, output={output_shp}")
        test_lib_big_memeff(str(pre_path), str(post_path), str(output_shp))

if __name__ == "__main__":
    pre_folder = '/share/test/pre'       # 前时相影像文件夹
    post_folder = '/share/test/post'     # 后时相影像文件夹
    output_folder = '/output'        # 结果输出文件夹
    process_folder(pre_folder, post_folder, output_folder)
