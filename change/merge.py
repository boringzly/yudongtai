from osgeo import gdal, ogr, osr
from pathlib import Path


def merge_shp(shp_files, save_shp_path):
    print(f'mergeing')
    driver = ogr.GetDriverByName("ESRI Shapefile")
    
    # 过滤：只要有图斑的面类型
    valid_shp_files = []
    for shp_file in shp_files:
        ds = driver.Open(str(shp_file), 0)
        layer = ds.GetLayer()
        if layer.GetFeatureCount() > 0 and layer.GetGeomType() in [3, 6]:  # Polygon, MultiPolygon
            valid_shp_files.append(shp_file)
        ds.Destroy()
    if Path(save_shp_path).exists():
        driver.DeleteDataSource(str(save_shp_path))
    if not valid_shp_files:
        print("No valid SHP files with features to merge, skipping")
        return
    out_datasource = driver.CreateDataSource(str(save_shp_path))
    out_layer = None
    
    for shp_file in valid_shp_files:
        in_datasource = driver.Open(str(shp_file), 0)
        in_layer = in_datasource.GetLayer()
        if out_layer == None:
            proj_shp = in_layer.GetSpatialRef()
            out_layer = out_datasource.CreateLayer('label', proj_shp, ogr.wkbPolygon)
            in_layer_defn = in_layer.GetLayerDefn()
            for i in range(in_layer_defn.GetFieldCount()):
                field_defn = in_layer_defn.GetFieldDefn(i)
                if 'index' not in field_defn.GetName().lower():
                    out_layer.CreateField(field_defn)
        for index, feature in enumerate(in_layer):
            out_layer.CreateFeature(feature)
        in_datasource.Destroy()
    out_datasource.Destroy()
