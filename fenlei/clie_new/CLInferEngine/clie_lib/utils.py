import os
import shutil
import math
import numpy as np

class Utils():

    @staticmethod
    def coloring3d(img, color_table=None):
        if color_table is None:
            return img

        img_color = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
        for i in range(len(color_table)):
            for j in range(3):
                img_color[:, :, 2 - j][img == i] = color_table[i][j]
        return img_color

    @staticmethod
    def check_path(pathname, reset=False):
        if not os.path.exists(pathname):
            os.makedirs(pathname)
            print(pathname + ' has been created!')
        else:
            if reset:
                shutil.rmtree(pathname)
                os.makedirs(pathname)
                print(pathname + ' has been reset!')

    @staticmethod
    def check_file(filename):
        if os.path.isfile(filename):
            os.remove(filename)
            print(filename + ' will be updated!')
        else:
            print(filename + ' will be created!')

    @staticmethod
    def generate_baselist(file_root, suffix):
        basename_list = []
        filename_list = []
        listfile = os.listdir(file_root)
        listfile.sort()
        for basename in listfile:
            file_suffix = '.' + basename.split('.')[-1]
            suffix_length = len(file_suffix)
            if file_suffix in suffix:
                basename_list.append(basename[:(-suffix_length)])
                filename_list.append(os.path.join(file_root, basename))
        return basename_list, filename_list

    @staticmethod
    def generate_list(file_root, basename_list, suffix):
        filename_list = []
        for basename in basename_list:
            filename = file_root + '/' + basename + suffix
            filename_list.append(filename)
        return filename_list

    @staticmethod
    def generate_filelist(file_path, data_path):
        basename_list = []
        filename_list = []
        with open(file_path, 'r') as f:
            for line in f:
                filename_list.append(os.path.join(data_path, line.strip()))
                basename_full = line.strip().split('/')[-1]
                basename_suffix = basename_full.split('.')[-1]
                basename = basename_full[:-(len(basename_suffix) + 1)]
                basename_list.append(basename)
        return basename_list, filename_list

    @staticmethod
    def generate_filelist_for_aiserver(filelist):
        basename_list = []
        filename_list = []
        for item in filelist:
            filename_list.append(item.strip())
            basename_full = os.path.basename(item.strip())
            basename_suffix = basename_full.split('.')[-1]
            basename = basename_full[:-(len(basename_suffix) + 1)]
            basename_list.append(basename)
        return basename_list, filename_list

    @staticmethod
    def split_filename_list_for_dist(basename_list, filename_list, split=1):
        basename_list_list = []
        filename_list_list = []
        for i in range(split):
            basename_list_list.append([])
            filename_list_list.append([])
        index_list = range(len(basename_list))
        for index, basename, filename in zip(index_list, basename_list, filename_list):
            basename_list_list[index % split].append(basename)
            filename_list_list[index % split].append(filename)
        return basename_list_list, filename_list_list

    @staticmethod
    def set_resume_list(basename_list_list, filename_list_list, resume=0):
        for basename_list in basename_list_list:
            basename_list = basename_list[resume:]
        for filename_list in filename_list_list:
            filename_list = filename_list[resume:]
        return basename_list_list, filename_list_list

    @staticmethod
    def load_mean_file(mean_file):
        mean_value = []
        with open(mean_file, 'r') as f:
            for line in f:
                line = line.strip()
                mean_value.append(float(line))
        print(' Mean Value: ' + str(mean_value))
        return mean_value

    @staticmethod
    def load_std_file(std_file):
        std_value = []
        with open(std_file, 'r') as f:
            for line in f:
                line = line.strip()
                std_value.append(float(line))
        print(' Stddev Value: ' + str(std_value))
        return std_value

    @staticmethod
    def load_band_list_file(band_list_file):
        band_list = []
        with open(band_list_file, 'r') as f:
            for line in f:
                band_list.append(int(line.strip()))
        return band_list

    @staticmethod
    def generate_suffix(driver_name):
        suffix = '.null'
        if driver_name == 'GTiff':
            suffix = '.tif'
        elif driver_name == 'ENVI':
            suffix = '.dat'
        elif driver_name == 'HFA':
            suffix = '.img'
        else:
            suffix = '.undefined'
        return suffix

    @staticmethod
    def generate_color_table(color_table_file):
        color_table = []
        with open(color_table_file, 'r') as f:
            for line in f:
                color_table.append(tuple([int(i) for i in line.strip().split('#')[0].split('/')]))
        return color_table

    @staticmethod
    def get_clip_geoinfo(im_geo, height_crop, width_crop, row, col):
        clip_geo = [0, 0, 0, 0, 0, 0]
        clip_geo[3] = row * height_crop * im_geo[5] + col * width_crop * im_geo[4] + im_geo[3]
        clip_geo[0] = col * width_crop * im_geo[1] + row * height_crop * im_geo[2] + im_geo[0]
        clip_geo[1:3] = im_geo[1:3]
        clip_geo[4:6] = im_geo[4:6]
        return clip_geo 

    @staticmethod
    def get_rect_geoinfo(im_geo, width, height):
        clip_geo = [im_geo[0], 0, 0, im_geo[3]]
        clip_geo[1] = width * im_geo[1] + height * im_geo[2] + im_geo[0]
        clip_geo[2] = width * im_geo[4] + height * im_geo[5] + im_geo[3]
        return clip_geo

    @staticmethod
    def get_environ_var(key):
        if key in os.environ:
            return os.environ[key]
        else:
            return None

    @staticmethod
    def webmercator_to_lonlat(x, y):
        _x = x / 20037508.342787001 * 180
        _y = y / 20037508.342787001 * 180
        lon = _x
        lat = (180.0 / math.pi) * 2.0 * (math.atan(math.exp(_y * (math.pi / 180.0))) - math.pi / 4.0)
        return lon, lat
