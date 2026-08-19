from .image2graph import *
from osgeo import gdalconst, gdal, ogr, osr



def imagexy2geo(dataset, col, row):
    trans = dataset.GetGeoTransform()
    px = trans[0] + col * trans[1] + row * trans[2]
    py = trans[3] + col * trans[4] + row * trans[5]
    return px, py


def raster2LineShp(img_path, strVectorFile):
    graph = generateGraph(img_path)
    dataset = gdal.Open(img_path)

    gdal.SetConfigOption("GDAL_FILENAME_IS_UTF8", "NO")
    gdal.SetConfigOption("SHAPE_ENCODING", "CP936")
    ogr.RegisterAll()
    strDriverName = "ESRI Shapefile"
    oDriver = ogr.GetDriverByName(strDriverName)

    oDS = oDriver.CreateDataSource(strVectorFile)

    srs = osr.SpatialReference(
        wkt=dataset.GetProjection())
    papszLCO = []
    oLayer = oDS.CreateLayer("TestPolygon", srs, ogr.wkbMultiLineString, papszLCO)

    oDefn = oLayer.GetLayerDefn()
    oFeatureTriangle = ogr.Feature(oDefn)
    for n, v in graph.items():

        for nei in v:
            line = ogr.Geometry(ogr.wkbLinearRing)
            nx, ny = n[1], n[0]
            nx, ny = imagexy2geo(dataset, nx, ny)
            line.AddPoint(nx, ny)

            neix, neiy = nei[1], nei[0]
            neix, neiy = imagexy2geo(dataset, neix, neiy)
            line.AddPoint(neix, neiy)

            oFeatureTriangle.SetGeometry(line)
            oLayer.CreateFeature(oFeatureTriangle)
    oDS.Destroy()

if __name__ == '__main__':
    rasterPath = '/nfs/project/netdisk/192.168.10.227/d/private/dongsj/road_test/pred/out_color/nanning.tif'
    shpPath = '/nfs/project/netdisk/192.168.10.227/d/private/dongsj/road_test/pred/out_color/nanning.shp'
    raster2LineShp(rasterPath, shpPath)
