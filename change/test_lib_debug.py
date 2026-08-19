import os 
from test_lib_batch_memeff_single_image_nomp import test_lib_big_memeff



if __name__ == "__main__":
    input_path1 = "/cresdashare/data2/2024/上海/H51D004004.tif"
    input_path2 = "/cresdashare/data2/2020/shanghai/H51D004004.tif"
    output = "./output/test.shp"
    test_lib_big_memeff(input_path1,input_path2,output)
