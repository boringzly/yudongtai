import torch
import torch.nn as nn
from networks.utils.CostVolume import CostVolumeLayer2
bn_mom = 0.0003
"""Implemention of dense fusion module"""

class densecat_cat_add(nn.Module):
    def __init__(self, in_chn, out_chn):
        super(densecat_cat_add, self).__init__()

        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_chn, in_chn, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(in_chn, in_chn, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
        )
        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv2d(in_chn, in_chn, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
        )
        self.conv_out = torch.nn.Sequential(
            torch.nn.Conv2d(in_chn, out_chn, kernel_size=1, padding=0),
            nn.BatchNorm2d(out_chn, momentum=bn_mom),
            torch.nn.ReLU(inplace=True),
        )

    def forward(self, x, y):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2+x1)

        y1 = self.conv1(y)
        y2 = self.conv2(y1)
        y3 = self.conv3(y2+y1)

        return self.conv_out(x1 + x2 + x3 + y1 + y2 + y3)


class densecat_cat_diff(nn.Module):
    def __init__(self, in_chn, out_chn):
        super(densecat_cat_diff, self).__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_chn, in_chn, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(in_chn, in_chn, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
        )
        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv2d(in_chn, in_chn, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
        )
        self.conv_out = torch.nn.Sequential(
            torch.nn.Conv2d(in_chn, out_chn, kernel_size=1, padding=0),
            nn.BatchNorm2d(out_chn, momentum=bn_mom),
            torch.nn.ReLU(inplace=True),
        )

    def forward(self, x, y):

        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2+x1)

        y1 = self.conv1(y)
        y2 = self.conv2(y1)
        y3 = self.conv3(y2+y1)
        out = self.conv_out(torch.abs(x1 + x2 + x3 - y1 - y2 - y3))
        return out
        

class densecat_cat_add2(densecat_cat_add):
    def __init__(self, in_chn, out_chn):
        super(densecat_cat_add2, self).__init__(in_chn, out_chn)

    def forward(self, x, y):
        x1 = self.conv1(x)
        x2 = self.conv2(x1+x)
        x3 = self.conv3(x2+x1+x)

        y1 = self.conv1(y)
        y2 = self.conv2(y1+y)
        y3 = self.conv3(y2+y1+y)

        return self.conv_out(x + x1 + x2 + x3 + y + y1 + y2 + y3)


class densecat_cat_diff2(densecat_cat_diff):
    def __init__(self, in_chn, out_chn):
        super(densecat_cat_diff2, self).__init__(in_chn, out_chn)

    def forward(self, x, y):

        x1 = self.conv1(x)
        x2 = self.conv2(x1+x)
        x3 = self.conv3(x2+x1+x)

        y1 = self.conv1(y)
        y2 = self.conv2(y1+y)
        y3 = self.conv3(y2+y1+y)
        out = self.conv_out(torch.abs(x + x1 + x2 + x3 - y - y1 - y2 - y3))
        return out


class DF_Module(nn.Module):
    def __init__(self, dim_in, dim_out, reduction=True):
        super(DF_Module, self).__init__()
        if reduction:
            self.reduction = torch.nn.Sequential(
                torch.nn.Conv2d(dim_in, dim_in//2, kernel_size=1, padding=0),
                nn.BatchNorm2d(dim_in//2, momentum=bn_mom),
                torch.nn.ReLU(inplace=True),
            )
            dim_in = dim_in//2
        else:
            self.reduction = None
        self.cat1 = densecat_cat_add(dim_in, dim_out)
        self.cat2 = densecat_cat_diff(dim_in, dim_out)
        self.conv1 = nn.Sequential(
            nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )

    def forward(self, x1, x2):
        if self.reduction is not None:
            x1 = self.reduction(x1)
            x2 = self.reduction(x2)
        x_add = self.cat1(x1, x2)
        x_diff = self.cat2(x1, x2)
        y = self.conv1(x_diff) + x_add
        return y


class DF_Module_more_dense(DF_Module):
    def __init__(self, dim_in, dim_out, reduction=True):
        super(DF_Module_more_dense, self).__init__(dim_in, dim_out, reduction)
        if reduction:
            self.reduction = torch.nn.Sequential(
                torch.nn.Conv2d(dim_in, dim_in//2, kernel_size=1, padding=0),
                nn.BatchNorm2d(dim_in//2, momentum=bn_mom),
                torch.nn.ReLU(inplace=True),
            )
            dim_in = dim_in//2
        else:
            self.reduction = None
        self.cat1 = densecat_cat_add2(dim_in, dim_out)
        self.cat2 = densecat_cat_diff2(dim_in, dim_out)


class densecat_cat_diff3(densecat_cat_diff):
    def __init__(self, in_chn, out_chn):
        super(densecat_cat_diff3, self).__init__(in_chn, out_chn)
        self.cv_layer = CostVolumeLayer2(2)

    def forward(self, x, y):

        x1 = self.conv1(x)
        x2 = self.conv2(x1+x)
        x3 = self.conv3(x2+x1+x)

        y1 = self.conv1(y)
        y2 = self.conv2(y1+y)
        y3 = self.conv3(y2+y1+y)
        cv0_1 = self.cv_layer(x, y)
        cv1_1 = self.cv_layer(x1, y1)
        cv2_1 = self.cv_layer(x2, y1)
        cv3_1 = self.cv_layer(x3, y1)
        cv0_2 = self.cv_layer(y, x)
        cv1_2 = self.cv_layer(y1, x1)
        cv2_2 = self.cv_layer(y2, x1)
        cv3_2 = self.cv_layer(y3, x1)
        out = self.conv_out(torch.abs(x + x1 + x2 + x3 - y - y1 - y2 - y3))
        return out


__all__ = [
    "DF_Module",
    "DF_Module_more_dense",
]