import cv2
import numpy as np
from .imageio import ImageReader, ImageWriter
import os
import tqdm
from skimage import morphology


# ct_list = [(0, 0, 0), (255, 255, 255)]
# suffix = '.tif'
# num_area = 0.5  # 保留多少个最大的连通域, 0表示不计算筛选，(0， 1)表示保留百分之多少的区域，>1表示保留多少个最大区域
# center = True  # 是否生成中心线
# contour = True  # 是都生成轮廓区域
# morphology_state = 2   #选择膨


def morphology_process(img, state=2):
    if state == 2:
        img1 = morphology.dilation(img, selem=None, out=None, shift_x=False, shift_y=False)
        img2 = morphology.erosion(img1, selem=None, out=None, shift_x=False, shift_y=False)
        return img2
    elif state == 1:
        img1 = morphology.dilation(img, selem=None, out=None, shift_x=False, shift_y=False)
        return img1
    elif state == 0:
        img2 = morphology.erosion(img, selem=None, out=None, shift_x=False, shift_y=False)
        return img2
    else:
        return img


def center_road(img):
    img_ = np.where(img > 0, 1, 0)
    skeleton_img = morphology.skeletonize(img_)
    skeleton_img = np.where(skeleton_img > 0, 7, 0)
    return skeleton_img


def contour_filter(img, road_thread):
    img = np.where(img == 255, 0, img)
    # import pudb;pu.db
    if road_thread == 0:
        contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=lambda y: y.shape[0], reverse=True)  # 按照检测到的连通域边界长度从大到小排列
        print('current contours count is {}'.format(len(contours)))
        # return img, contours
        return img
    elif road_thread > 1:
        img_temp = img.copy()
        contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=lambda y: y.shape[0], reverse=True)  # 按照检测到的连通域边界长度从大到小排列
        print('current contours count is {}'.format(len(contours)))
        if len(contours) > road_thread:
            need_contours = contours[:road_thread]  # 得到需要涂色边缘的连通域
            del_contours = contours[road_thread:]  # 表示从第road_thread开始，去掉小的连通域，保留前面的一共road_thread个连通域
            img_out = cv2.drawContours(img_temp, del_contours, -1, 0, thickness=-1)  # 去掉上面的contours的区域
        else:
            img_out = img_temp
        # return img_out, need_contours
        return img_out
    elif road_thread < 1 and road_thread > 0:  # 采用百分比计算road_thread
        img_temp = img.copy()
        contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=lambda y: y.shape[0], reverse=True)
        # print('current contours count is {}'.format(len(contours)))
        road_thread = int(len(contours) * road_thread)  # 保留百分之多少的大的连通域
        if len(contours) > road_thread:
            need_contours = contours[:road_thread]  # 得到需要涂色边缘的连通域
            del_contours = contours[road_thread:]
            img_out = cv2.drawContours(img_temp, del_contours, -1, 0, thickness=-1)
        else:
            img_out = img_temp
        # return img_out, need_contours
        return img_out


def process_road(file_path, save_path, center_save_path, road_thread, extract_center):
    # import pudb;pu.db
    ct_list = [(0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0),(255, 255, 255)]
    reader = ImageReader(file_path)
    img = reader.read_image()
    img = morphology_process(img)
    # print(img)
    # print(img_geo_info)
    img_out = contour_filter(img, road_thread)
    writer = ImageWriter(save_path, img_out.shape[1], img_out.shape[0], 1, dtype='uint8', driver='GTiff', compress='LZW')
    writer.set_proj(reader.proj)
    writer.set_coord(reader.coord)
    writer.set_color_table(ct_list)
    writer.write_image(img_out)
    if extract_center:
        skeleton_img = center_road(img_out)
        writer = ImageWriter(center_save_path, skeleton_img.shape[1], skeleton_img.shape[0], 1, dtype='uint8', driver='GTiff', compress='LZW')
        writer.set_proj(reader.proj)
        writer.set_coord(reader.coord)
        writer.write_image(skeleton_img)
        writer.set_color_table(ct_list)
    # if draw_contours:
    #     img_contours = cv2.drawContours(img_out, contours, -1, 2, 4)  # 采用灰度2表示边缘
    #     yimage.io.write_image(save_path.replace(suffix, '_contours' + suffix), img_contours, geo_info=img_geo_info,
    #                           color_table=ct_list)
    print('finished processing file_path is {}'.format(file_path))
    print('saved file_path is {}'.format(save_path))


# def folder_func(fin, fout):
    # imgs = os.listdir(fin)
    # try:
    #     for i in tqdm.tqdm(range(len(imgs))):
    #         if imgs[i][-4:] == suffix:
    #             process_road(os.path.join(fin, imgs[i]), os.path.join(fout, imgs[i]), 10, False, True)
    # except:
    #     pass

    # for i in tqdm.tqdm(range(len(imgs))):
    #     if imgs[i][-4:] == suffix:
    #         process_road(os.path.join(fin, imgs[i]), os.path.join(fout, imgs[i]), num_area, center, contour)


# def road_func(img_path, filter_rate=0.8):
#     img = img[0, :, :]
#     img = morphology_process(img, 2)
#     # print(img)
#     # print(img_geo_info)
#     img_filter, contours = contour_filter(img, filter_rate)
#     img_center = center_road(img_filter)
#     img_contour = cv2.drawContours(img_filter, contours, -1, 2, 4)  # 采用灰度2表示边缘
#     return img_filter, img_center, img_contour


# if __name__ == '__main__':
#     # _, ct_list = yimage.io.load_color_table_file('/opt/netdisk/192.168.0.31/d/private/dongsj/code_sj/color_table.txt')
#     ct_list = [(0, 0, 0), (255, 255, 255)]
#     suffix = '.tif'
#     num_area = 0.5  # 保留多少个最大的连通域, 0表示不计算筛选，(0， 1)表示保留百分之多少的区域，>1表示保留多少个最大区域
#     center = True  # 是否生成中心线
#     contour = True  # 是都生成轮廓区域
#     morphology_state = 2   #选择膨胀腐蚀参数，2表示先膨胀再腐蚀，1表示只做膨胀，0表示只做腐蚀, 其他数字表示什么都不做
#     pfolder = '/nfs/project/netdisk/192.168.10.227/d/change_detection/project/给空天院试验影像/000提交/xinxiang'
#     # base param
#
#     pfolder_out = pfolder + '_filter'  # 生成文件夹的名字
#     if os.path.exists(pfolder_out) is False:
#         os.mkdir(pfolder_out)
#     folder_func(pfolder, pfolder_out)

# if __name__ == '__main__':
    # pred_path = '/nfs/project/netdisk/192.168.10.227/d/private/dongsj/road_test/pred/out_color/nanning.tif'
    # filter_path = '/nfs/project/netdisk/192.168.10.227/d/private/dongsj/road_test/pred/out_color/nanning_filter.tif'
    # center_path = '/nfs/project/netdisk/192.168.10.227/d/private/dongsj/road_test/pred/out_color/nanning_center.tif'
    #
    # process_road(pred_path, pred_path, center_path, 0.4, True)
