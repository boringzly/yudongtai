import numpy as np
import sys
from osgeo import gdal

# 定义标签映射规则
label_mapping = {
    0: 0,
    1: 1, 2: 1,
    3: 2, 4: 2,
    5: 3,
    11: 4,
    6: 5, 7: 5, 8: 5, 9: 5,
    10: 6, 12: 6, 13: 6
}

def process_tif(input_path, output_path):
    """
    参数:
    input_path: 输入TIFF文件路径
    output_path: 输出TIFF文件路径
    """
    # 打开输入数据集
    src_ds = gdal.Open(input_path, gdal.GA_ReadOnly)
    if src_ds is None:
        print(f"无法打开文件: {input_path}")
        return
    
    # 获取波段数据
    band = src_ds.GetRasterBand(1)
    data = band.ReadAsArray()
    
    
    # 创建输出数组
    output_data = np.zeros_like(data)
    
    # 应用映射规则
    for old_val, new_val in label_mapping.items():
        output_data[data == old_val] = new_val
    
    
    # 获取驱动器和创建输出数据集
    driver = gdal.GetDriverByName('GTiff')
    out_ds = driver.Create(
        output_path,
        src_ds.RasterXSize,
        src_ds.RasterYSize,
        1,  # 波段数
        band.DataType,  # 数据类型
        options=['COMPRESS=LZW', 'TILED=YES', 'BIGTIFF=YES']
    )
    
    # 设置地理参考信息
    out_ds.SetGeoTransform(src_ds.GetGeoTransform())
    out_ds.SetProjection(src_ds.GetProjection())
    
    # 写入数据
    out_band = out_ds.GetRasterBand(1)
    out_band.WriteArray(output_data)
    
    # 设置无数据值（如果有）
    if band.GetNoDataValue() is not None:
        out_band.SetNoDataValue(band.GetNoDataValue())
    
    # 关闭数据集
    out_band.FlushCache()
    out_ds = None
    src_ds = None
    
    print(f"处理完成！输出文件已保存至: {output_path}")

# 示例使用
if __name__ == "__main__":
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    process_tif(input_file, output_file)
