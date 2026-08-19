import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import sys
from pathlib import Path

from QWEN_CD import main_classification


def run_class_prediction():
    # os.environ['PATH_IMAGE_FILE_1'] = 'asset/H49D003012.tif'
    # os.environ['PATH_IMAGE_FILE_2'] = 'asset/H49D003012.tif'
    # os.environ['PATH_SHAPE_FILE'] = 'asset/Export_Output_2.shp'
    # os.environ['PATH_OUTPUT'] = 'asset1'

    input_path1 = os.environ['PATH_IMAGE_FILE_1']
    input_path2 = os.environ['PATH_IMAGE_FILE_2']
    mask_path = os.environ['PATH_SHAPE_FILE']
    output_path = os.environ['PATH_OUTPUT']
    Path(output_path).mkdir(parents=True, exist_ok=True)
    output_path = str(Path(output_path) / Path(mask_path).name)
    print(f'前时相影像路径：{input_path1}')
    print(f'后时相影像路径：{input_path2}')
    print(f'变化掩码路径：{mask_path}')
    print(f'类别掩码路径：{output_path}')
    main_classification(input_path1, input_path2, mask_path, output_path)


if __name__ == '__main__':
    run_class_prediction()
