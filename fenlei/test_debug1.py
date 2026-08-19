import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import sys
from pathlib import Path
from QWEN_CD import main_classification

pre_folder_path = "/cresdashare/data2/2020/guangdong/"
pre_path_list = Path(pre_folder_path).glob(f'*.tif')
pre_path_list = list(pre_path_list)
post_path_list = ["/data2/inner_data/13/171/ecomn_2024_13_171.tif"]
save_folder = "./output"
import logging
logger = logging.getLogger('mylog')
logger.setLevel(logging.INFO)
pair_path_list = []
for pre_path in pre_path_list:
    for post_path in post_path_list:
        pair_path_list.append([str(pre_path), post_path])

shp_path = "output/output_path544154917.shp"
save_shp = "output/output_path544154917_class_0920_1.shp"

main_classification(pre_folder_path, post_path_list[0], shp_path, save_shp, logger=logger)
# import pdb;pdb.set_trace()
'''
origin_shp, to_shp = shp_path, ""
for num, pair_path in enumerate(pair_path_list):
    print(f"当前正在处理: {num} / {len(pair_path_list)}")
    print(f"{pair_path[0]}")
    print(f"{pair_path[1]}")
    if num != (len(pair_path_list) - 1):
        to_shp = str(Path(save_folder) / (Path(pair_path[0]).stem + ".shp"))
    else:
        to_shp=  save_shp
    main_classification(pair_path[0], pair_path[1], origin_shp, to_shp, logger=logger)
    origin_shp = to_shp
'''
