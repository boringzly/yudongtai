import numpy as np
import cv2


class TTAEncoder():

    def __init__(self, params=None):
        self.utils = TTAUtils()

        self.tta_flip = False
        self.tta_rotate = False
        if params:
            if 'tta_flip' in params:
                self.tta_flip = params['tta_flip']
            if 'tta_rotate' in params:
                self.tta_rotate = params['tta_rotate']

        if self.tta_flip and self.tta_rotate:
            self.tta_univ = self._tta_flip_rotate
        elif self.tta_flip and (not self.tta_rotate):
            self.tta_univ = self._tta_flip
        elif (not self.tta_flip) and self.tta_rotate:
            self.tta_univ = self._tta_rotate
        else:
            raise Exception('Please set opt.use_tta = False')


    def tta(self, imgs):
        tta_imgs = []
        for i in range(imgs.shape[0]):
            tta_img = []
            tta_img.append(imgs[i])
            tta_img = self.tta_univ(tta_img, imgs[i])
            tta_imgs.append(tta_img)
        return self._tta_reshape(tta_imgs)


    def _tta_reshape(self, tta_imgs):
        tta_reshape_imgs = []
        for i in range(len(tta_imgs[0])):
            tta_reshape_img = tta_imgs[0][i][None, :, :, :]
            for j in range(1, len(tta_imgs)):
                tta_reshape_img = np.append(tta_reshape_img, tta_imgs[j][i][None, :, :, :], axis=0)
            tta_reshape_imgs.append(tta_reshape_img)
        return tta_reshape_imgs


    def _tta_flip_rotate(self, tta_img, img):
        tta_img.append(self.utils.HorizontalFlip(img))
        tta_img.append(self.utils.VerticalFlip(img))
        tta_img.append(self.utils.HorizontalVerticalFlip(img))
        tta_img.append(self.utils.Rotate90(img, 1))
        tta_img.append(self.utils.Rotate90(img, 3))
        tta_img.append(self.utils.VerticalFlip(self.utils.Rotate90(img, 1)))
        tta_img.append(self.utils.VerticalFlip(self.utils.Rotate90(img, 3)))
        return tta_img


    def _tta_flip(self, tta_img, img):
        tta_img.append(self.utils.HorizontalFlip(img))
        tta_img.append(self.utils.VerticalFlip(img))
        tta_img.append(self.utils.HorizontalVerticalFlip(img))
        return tta_img


    def _tta_rotate(self, tta_img, img):
        tta_img.append(self.utils.Rotate90(img, 1))
        tta_img.append(self.utils.Rotate90(img, 2))
        tta_img.append(self.utils.Rotate90(img, 3))
        return tta_img


class TTADecoder():

    def __init__(self, params=None):
        self.utils = TTAUtils()

        self.tta_flip = False
        self.tta_rotate = False
        if params:
            if 'tta_flip' in params:
                self.tta_flip = params['tta_flip']
            if 'tta_rotate' in params:
                self.tta_rotate = params['tta_rotate']

        if self.tta_flip and self.tta_rotate:
            self.tta_univ = self._tta_flip_rotate
        elif self.tta_flip and (not self.tta_rotate):
            self.tta_univ = self._tta_flip
        elif (not self.tta_flip) and self.tta_rotate:
            self.tta_univ = self._tta_rotate
        else:
            raise Exception('Please set opt.use_tta = False')


    def tta(self, imgs):
        tta_imgs = []
        for i in range(imgs[0].shape[0]):
            tta_img = []
            tta_img.append(imgs[0][i])
            tta_img = self.tta_univ(tta_img, imgs[1:], i)
            tta_imgs.append(tta_img)
        return self._tta_reshape(tta_imgs)


    def _tta_reshape(self, tta_imgs):
        tta_reshape_imgs = []
        for i in range(len(tta_imgs[0])):
            tta_reshape_img = tta_imgs[0][i][None, :, :, :]
            for j in range(1, len(tta_imgs)):
                tta_reshape_img = np.append(tta_reshape_img, tta_imgs[j][i][None, :, :, :], axis=0)
            tta_reshape_imgs.append(tta_reshape_img)
        return tta_reshape_imgs


    def _tta_flip_rotate(self, tta_img, img, batch_num):
        tta_img.append(self.utils.HorizontalFlip(img[0][batch_num]))
        tta_img.append(self.utils.VerticalFlip(img[1][batch_num]))
        tta_img.append(self.utils.HorizontalVerticalFlip(img[2][batch_num]))
        tta_img.append(self.utils.Rotate90(img[3][batch_num], 3))
        tta_img.append(self.utils.Rotate90(img[4][batch_num], 1))
        tta_img.append(self.utils.Rotate90(self.utils.VerticalFlip(img[5][batch_num]), 3))
        tta_img.append(self.utils.Rotate90(self.utils.VerticalFlip(img[6][batch_num]), 1))
        return tta_img


    def _tta_flip(self, tta_img, img, batch_num):
        tta_img.append(self.utils.HorizontalFlip(img[0][batch_num]))
        tta_img.append(self.utils.VerticalFlip(img[1][batch_num]))
        tta_img.append(self.utils.HorizontalVerticalFlip(img[2][batch_num]))
        return tta_img


    def _tta_rotate(self, tta_img, img, batch_num):
        tta_img.append(self.utils.Rotate90(img[0][batch_num], 3))
        tta_img.append(self.utils.Rotate90(img[1][batch_num], 2))
        tta_img.append(self.utils.Rotate90(img[2][batch_num], 1))
        return tta_img


class TTAUtils():

    def __init__(self):
        pass


    def HorizontalFlip(self, img):
        _img = cv2.flip(img, 1)
        return self._check_ndim_3(_img)


    def VerticalFlip(self, img):
        _img = cv2.flip(img, 0)
        return self._check_ndim_3(_img)


    def HorizontalVerticalFlip(self, img):
        _img = cv2.flip(img, -1)
        return self._check_ndim_3(_img)


    def Rotate90(self, img, times=1):
        _img = img
        for _ in range(times):
            _img = np.rot90(_img)
        _img = _img.copy()
        return self._check_ndim_3(_img)


    def _check_ndim_3(self, img):
        if img.ndim < 3:
            img = img[:, :, None]
        return img