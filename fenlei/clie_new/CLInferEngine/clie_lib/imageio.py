import numpy as np
from osgeo import gdal, ogr, osr


class ImageReader():

    def __init__(self, filename):
        self.dataset = None
        try:
            self.dataset = gdal.Open(filename)
        except:
            print('WARNING: [GDAL] can not open %s !' % filename)
        if self.dataset is None:
            print('WARNING: [GDAL] can not open %s !' % filename)

        self.width = self.dataset.RasterXSize
        self.height = self.dataset.RasterYSize
        self.nbands = self.dataset.RasterCount

        self.band_list = [i + 1 for i in range(self.nbands)]
        if self.nbands == 3 or self.nbands == 4:
            self.band_list[0] = 3
            self.band_list[2] = 1

        self.proj = self.dataset.GetProjection()
        self.coord = self.dataset.GetGeoTransform()

    def set_band_list(self, band_list=None):
        if band_list is not None:
            _band_list = []
            for item in band_list:
                _band_list.append(self.band_list[item - 1])
            self.band_list = _band_list

    def read_image(self, read_range=None):
        if read_range is None:
            read_range = (0, 0, self.width, self.height)

        img = None
        for band in self.band_list:
            _data = self.dataset.GetRasterBand(band)
            try:
            	_img = _data.ReadAsArray(*read_range)[:, :, None]
            except:
                return None
            if img is None:
                img = _img
            else:
                img = np.append(img, _img, axis=2)
        del _data; del _img
        return img

    def close(self):
        del self.dataset

    def build_overviews(self, overviewlist=[1, 2, 4, 8, 16]):
        self.dataset.BuildOverviews(overviewlist=overviewlist)


class ImageWriter():

    def __init__(self, filename, width, height, nbands, dtype='uint8', driver='GTiff', compress=None):
        type_dict = {
            'uint8': gdal.GDT_Byte,
            'uint16': gdal.GDT_UInt16,
            'int16': gdal.GDT_Int16,
            'uint32': gdal.GDT_UInt32,
            'int32': gdal.GDT_Int32,
            'float32': gdal.GDT_Float32,
            'float64': gdal.GDT_Float64
        }

        self.band_list = [i + 1 for i in range(nbands)]
        if nbands == 3 or nbands == 4:
            self.band_list[0] = 3
            self.band_list[2] = 1

        options = []
        if driver == 'GTiff':
            # 压缩后的最终大小无法可靠预估，IF_SAFER 仍可能在写入中途撞上 4GB 上限。
            # 分类结果统一使用 BigTIFF，彻底避免 TIFFAppendToStrip 超限。
            options.extend(['BIGTIFF=YES', 'TILED=YES'])
            if compress:
                options.append(f'COMPRESS={compress}')
        
        driver = gdal.GetDriverByName(driver)
        if driver is None:
            raise RuntimeError(f'GDAL driver is unavailable for output: {filename}')
        gdal.ErrorReset()
        self.dataset = driver.Create(filename, width, height, nbands, type_dict[dtype], options=options)
        if self.dataset is None or gdal.GetLastErrorType() >= gdal.CE_Failure:
            error_message = gdal.GetLastErrorMsg() or 'unknown GDAL error'
            raise RuntimeError(f'Failed to create raster {filename}: {error_message}')

        self.filename = filename
        self.width = width
        self.height = height
        self.nbands = nbands

    def set_proj(self, proj=None):
        if proj is not None:
            self.dataset.SetProjection(proj)

    def set_coord(self, coord=None):
        if coord is not None:
            self.dataset.SetGeoTransform(coord)

    def set_color_table(self, color_table=None):
        if color_table is not None:
            ct = gdal.ColorTable()
            for i, color in enumerate(color_table):
                ct.SetColorEntry(i, color)

            for item in self.band_list:
                _band = self.dataset.GetRasterBand(item)
                _band.SetRasterColorInterpretation(gdal.GCI_PaletteIndex)
                _band.SetColorTable(ct)

    def set_no_data_value(self, no_data_value=None):
        if no_data_value is not None:
            for item in self.band_list:
                _band = self.dataset.GetRasterBand(item)
                _band.SetNoDataValue(no_data_value)

    def set_band_list(self, band_list=None):
        if band_list is not None:
            _band_list = []
            for item in band_list:
                _band_list.append(self.band_list[item - 1])
            self.band_list = _band_list

    def write_image(self, img, write_offset=None):
        if write_offset is None:
            write_offset = (0, 0)

        for i, band in enumerate(self.band_list):
            _band = self.dataset.GetRasterBand(i + 1)
            gdal.ErrorReset()
            write_result = _band.WriteArray(img[:, :, band - 1], *write_offset)
            flush_result = _band.FlushCache()
            if (write_result not in (None, 0)
                    or flush_result not in (None, 0)
                    or gdal.GetLastErrorType() >= gdal.CE_Failure):
                error_message = gdal.GetLastErrorMsg() or 'unknown GDAL error'
                raise RuntimeError(f'Failed to write raster {self.filename}: {error_message}')

    def close(self):
        if self.dataset is None:
            return
        gdal.ErrorReset()
        flush_result = self.dataset.FlushCache()
        if flush_result not in (None, 0) or gdal.GetLastErrorType() >= gdal.CE_Failure:
            error_message = gdal.GetLastErrorMsg() or 'unknown GDAL error'
            raise RuntimeError(f'Failed to finalize raster {self.filename}: {error_message}')
        self.dataset = None

    def build_overviews(self, overviewlist=[2,4,8,16,32,64,128]):
        self.dataset.BuildOverviews('NEAREST', overviewlist=overviewlist)
        self.dataset.FlushCache()


class ImageUtils():

    @staticmethod
    def polygonize(raster_filename, shapefile_filename, pred_band):
        raster_dataset = gdal.Open(raster_filename)
        if raster_dataset is None:
            print('[FATAL] GDAL open file failed. [%s]'%raster_filename)
            exit(1)

        driver = ogr.GetDriverByName('ESRI Shapefile')
        if driver is None:
            print('[FATAL] OGR create driver failed. [%s]'%'ESRI Shapefile')
            exit(1)
        
        shape_dataset = driver.CreateDataSource(shapefile_filename)
        if shape_dataset is None:
            print('[FATAL] OGR create file failed. [%s]'%shapefile_filename)
            exit(1)

        proj_ref = raster_dataset.GetProjectionRef()
        proj_shp = osr.SpatialReference()
        proj_shp.ImportFromWkt(proj_ref)
        layer = shape_dataset.CreateLayer('pred', proj_shp, ogr.wkbPolygon)
        field_name = ogr.FieldDefn('objects', ogr.OFTInteger)
        layer.CreateField(field_name)
        band = raster_dataset.GetRasterBand(pred_band)
        gdal.Polygonize(band, band, layer, 0)
        del shape_dataset


    @staticmethod
    def polygonize_mem(pred, proj, coord, shapefile_filename, pred_band):
        mem_driver = gdal.GetDriverByName('MEM')
        raster_dataset = mem_driver.Create('', pred.shape[1], pred.shape[0], 1)
        raster_dataset.SetProjection(proj)
        raster_dataset.SetGeoTransform(coord)
        band = raster_dataset.GetRasterBand(pred_band)
        band.WriteArray(pred)

        driver = ogr.GetDriverByName('ESRI Shapefile')
        if driver is None:
            print('[FATAL] OGR create driver failed. [%s]'%'ESRI Shapefile')
            exit(1)

        shape_dataset = driver.CreateDataSource(shapefile_filename)
        if shape_dataset is None:
            print('[FATAL] OGR create file failed. [%s]'%shapefile_filename)
            exit(1)

        proj_ref = raster_dataset.GetProjectionRef()
        proj_shp = osr.SpatialReference()
        proj_shp.ImportFromWkt(proj_ref)
        layer = shape_dataset.CreateLayer('pred', proj_shp, ogr.wkbPolygon)
        field_name = ogr.FieldDefn('Shape', ogr.OFTInteger)
        layer.CreateField(field_name)
        #gdal.Polygonize(band, None, layer, 0)
        gdal.Polygonize(band, band, layer, 0)
        #layer.SyncToDisk()
        dict_json = ''
        feature_count = layer.GetFeatureCount()
        for i in range(feature_count):
            dict_json += layer.GetFeature(i).ExportToJson()
            dict_json += ','
        dict_json = dict_json[:-1]
        del shape_dataset
        return dict_json
