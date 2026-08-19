# gdal包安装 
# conda install gdal
from osgeo import gdal, ogr, osr
import os, argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_shp",
                        type=str,
                        required=True,
                        help="path to the input shapefile")
    parser.add_argument("--reference_tif",
                        type=str,
                        required=True,
                        help="path to the reference tif file")
    parser.add_argument("--output_mask",
                        type=str,
                        required=True,
                        help="path to save the output mask")
    args = parser.parse_args()
    return args

def convert_shp_to_mask(vector_fn, reference_tif_fn, output_mask_fn):
    """
    Convert shapefile to binary mask (0/255) with same dimensions and resolution as reference tif
    
    Args:
        vector_fn: path to input shapefile
        reference_tif_fn: path to reference tif file
        output_mask_fn: path to output mask file
    """
    # Open reference tif to get spatial parameters
    reference_ds = gdal.Open(reference_tif_fn)
    if reference_ds is None:
        print(f'[FATAL] Cannot open reference tif: {reference_tif_fn}')
        return False
    
    # Get reference parameters
    geo_transform = reference_ds.GetGeoTransform()
    projection = reference_ds.GetProjection()
    x_size = reference_ds.RasterXSize
    y_size = reference_ds.RasterYSize
    
    print(f'Reference tif info:')
    print(f'  Size: {x_size} x {y_size}')
    print(f'  Pixel size: {geo_transform[1]} x {-geo_transform[5]}')
    
    # Open shapefile
    source_ds = ogr.Open(vector_fn)
    if source_ds is None:
        print(f'[FATAL] Cannot open shapefile: {vector_fn}')
        reference_ds = None
        return False
    
    source_layer = source_ds.GetLayer()
    feature_count = source_layer.GetFeatureCount()
    print(f'Shapefile features: {feature_count}')
    
    # Create output raster with same parameters as reference
    driver = gdal.GetDriverByName('GTiff')
    target_ds = driver.Create(
        output_mask_fn, 
        x_size, 
        y_size, 
        1, 
        gdal.GDT_Byte,  # Use Byte type for 0/255 mask
        options=['COMPRESS=LZW', 'BIGTIFF=YES']
    )
    
    if target_ds is None:
        print(f'[FATAL] Cannot create output file: {output_mask_fn}')
        source_ds = None
        reference_ds = None
        return False
    
    # Set same geotransform and projection as reference
    target_ds.SetGeoTransform(geo_transform)
    target_ds.SetProjection(projection)
    
    band = target_ds.GetRasterBand(1)
    #band.SetNoDataValue(0)
    band.Fill(0)  # Initialize with 0
    
    # Rasterize - burn value 255 for all features
    if feature_count > 0:
        options = ["BURN_VALUE=255", "ALL_TOUCHED=TRUE"]
        err = gdal.RasterizeLayer(target_ds, [1], source_layer, options=options)
        if err != 0:
            print(f'[WARNING] RasterizeLayer returned error code: {err}')
        else:
            print(f'Successfully burned {feature_count} features with value 255')
    else:
        print('[WARNING] Source layer is empty')
    
    # Build overviews for faster display
    target_ds.BuildOverviews('NEAREST', overviewlist=[2, 4, 8, 16, 32, 64, 128])
    target_ds.FlushCache()
    
    # Clean up
    target_ds = None
    source_ds = None
    reference_ds = None
    
    print(f'Successfully created mask: {output_mask_fn}')
    return True

if __name__ == "__main__":
    args = parse_args()
    
    input_shp = args.input_shp
    reference_tif = args.reference_tif
    output_mask = args.output_mask
    
    # Create output directory if needed
    output_dir = os.path.dirname(output_mask)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Convert shapefile to mask
    convert_shp_to_mask(input_shp, reference_tif, output_mask)
    '''
    python shp2mask_v1.py --input_shp /nfs/project/netdisk/192.168.100.189/d/zanly/1037/test/ecomn_2024_27_263.shp --reference_tif /nfs/project/netdisk/192.168.100.189/d/zanly/1037/ecomn_2024_27_263.tif --output_mask /nfs/project/netdisk/192.168.100.189/d/zanly/1037/test/mask.tif
    '''