import torch
import torch.nn as nn
import torch.nn.functional as F
from networks.utils.block import *
from .attentions import CAM

class CostVolumeLayer(nn.Module):

    def __init__(self, search_range=2):
        super(CostVolumeLayer, self).__init__()
        self.search_range = search_range

    def forward(self, x1, x2):

        shape = list(x1.size()); shape[1] = (self.search_range * 2 + 1) ** 2
        # 跟随当前 DataParallel replica 的输入设备，不能硬编码到 cuda:0。
        cv = x1.new_zeros(shape)

        for i in range(-self.search_range, self.search_range + 1):
            for j in range(-self.search_range, self.search_range + 1):
                if   i < 0: slice_h, slice_h_r = slice(None, i), slice(-i, None)
                elif i > 0: slice_h, slice_h_r = slice(i, None), slice(None, -i)
                else:       slice_h, slice_h_r = slice(None),    slice(None)

                if   j < 0: slice_w, slice_w_r = slice(None, j), slice(-j, None)
                elif j > 0: slice_w, slice_w_r = slice(j, None), slice(None, -j)
                else:       slice_w, slice_w_r = slice(None),    slice(None)

                cv[:, (self.search_range*2+1) * i + j, slice_h, slice_w] = (x1[:,:,slice_h, slice_w]  * x2[:,:,slice_h_r, slice_w_r]).sum(1)
    
        return cv / shape[1]


class CostVolumeLayer2(nn.Module):

    def __init__(self, search_range=2):
        super(CostVolumeLayer2, self).__init__()
        self.search_range = search_range

    def forward(self, x1, x2):
        x1 = torch.sigmoid(x1)
        x2 = torch.sigmoid(x2)
        shape = list(x1.size()); shape[1] = (self.search_range * 2 + 1) ** 2
        cv = x1[:,0:1,::].expand(shape)*0

        for i in range(-self.search_range, self.search_range + 1):
            for j in range(-self.search_range, self.search_range + 1):
                # if   i < 0: slice_h, slice_h_r = slice(None, i), slice(-i, None)
                # elif i > 0: slice_h, slice_h_r = slice(i, None), slice(None, -i)
                # else:       slice_h, slice_h_r = slice(None),    slice(None)

                # if   j < 0: slice_w, slice_w_r = slice(None, j), slice(-j, None)
                # elif j > 0: slice_w, slice_w_r = slice(j, None), slice(None, -j)
                # else:       slice_w, slice_w_r = slice(None),    slice(None)
                if i < 0:
                    if j < 0:
                        cv[:, (self.search_range*2+1) * i + j, 0:shape[2]+i, 0:shape[3]+j] = torch.mul(x1[:,:,0:shape[2]+i, 0:shape[3]+j], x2[:,:,-i:, -j:]).sum(1)
                    elif j == 0:
                        cv[:, (self.search_range*2+1) * i + j, 0:shape[2]+i, :] = torch.mul(x1[:,:,0:shape[2]+i, :], x2[:,:,-i:, :]).sum(1)
                    else:
                        cv[:, (self.search_range*2+1) * i + j, 0:shape[2]+i, j:] = torch.mul(x1[:,:,0:shape[2]+i, j:], x2[:,:,-i:, 0:shape[3]-j]).sum(1)
                elif i == 0:
                    if j < 0:
                        cv[:, j, :, 0:shape[3]+j] = torch.mul(x1[:,:,:, 0:shape[3]+j], x2[:,:,:, -j:]).sum(1)
                    elif j == 0:
                        cv[:, j, :, :] = torch.mul(x1[:,:,:, :], x2[:,:,:, :]).sum(1)
                    else:
                        cv[:, j, :, j:] = torch.mul(x1[:,:,:, j:], x2[:,:,:, 0:shape[3]-j]).sum(1)
                else:
                    if j < 0:
                        cv[:, (self.search_range*2+1) * i + j, i:, 0:shape[3]+j] = torch.mul(x1[:,:,i:, 0:shape[3]+j], x2[:,:,0:shape[2]-i, -j:]).sum(1)
                    elif j == 0:
                        cv[:, (self.search_range*2+1) * i + j, i:, :] = torch.mul(x1[:,:,i:, :], x2[:,:,0:shape[2]-i, :]).sum(1)
                    else:
                        cv[:, (self.search_range*2+1) * i + j, i:, j:] = torch.mul(x1[:,:,i:, j:], x2[:,:,0:shape[2]-i, 0:shape[3]-j]).sum(1)

        return cv / shape[1]

class CostVolumeLayer3(nn.Module):

    def __init__(self, search_range=2):
        super(CostVolumeLayer3, self).__init__()
        self.search_range = search_range
        self.sigmoid = nn.Sigmoid()

    def forward(self, x1, x2):
        shape = list(x1.shape)
        cv = x1[:,0:1,::]*0
        x1 =  F.pad(self.sigmoid(x1), (self.search_range, self.search_range, self.search_range, self.search_range), mode="constant")
        x2 =  F.pad(self.sigmoid(x2), (self.search_range, self.search_range, self.search_range, self.search_range), mode="constant")
        shape[1] = (self.search_range * 2 + 1) ** 2
        cv = cv.expand(shape)

        for i in range(-self.search_range, self.search_range + 1):
            for j in range(-self.search_range, self.search_range + 1):
                shift1 = [i, j]
                shift2 = [-i, -j]
                x1_new = torch.roll(x1, shifts=shift1, dims=[2,3])
                x2_new = torch.roll(x2, shifts=shift2, dims=[2,3])
                cv[:, (self.search_range*2+1) * i + j, ::] = torch.mul(x1_new, x2_new).sum(1)[:,self.search_range:shape[2]+self.search_range,self.search_range:shape[3]+self.search_range]
        return cv / shape[1]


class CV_cat(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(CV_cat, self).__init__()
        self.cv_layer = CostVolumeLayer()
        self.cat = cat(dim_in, dim_in, dim_in, False)
        self.conv1 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x1, x2):
        cv1  =self.cv_layer(x1, x2)
        cv2  =self.cv_layer(x2, x1)
        cv = cv1 + cv2
        cv = torch.mean(cv, 1)[:,None,::]
        cv = torch.sigmoid(cv)
        y = self.cat(x1, x2) * cv
        y = self.conv1(y)
        return y, cv


class CV4(nn.Module):
    def __init__(self, dim_in, dim_out, search_range=1):
        super(CV4, self).__init__()
        self.search_range = search_range
        self.cv_layer = CostVolumeLayer2(self.search_range)
        self.conv_1x1 = nn.Sequential(
            nn.Conv2d((self.search_range * 2 + 1) ** 2, self.search_range * 2 + 1, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(self.search_range * 2 + 1, self.search_range * 2 + 1, kernel_size=1, stride=1, padding=0),
        )
        self.cat = cat_conv(dim_in, dim_in, dim_out, False)
        self.conv_out = nn.Sequential(
            nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(dim_out),
            nn.ReLU(),
        )
        
    def forward(self, x1, x2):
        cv = 1 - self.cv_layer(x1, x2) / (self.search_range * 2 + 1) ** 2
        cv = self.conv_1x1(cv).mean(1).unsqueeze(1)
        cv = torch.sigmoid(cv)
        y = self.cat(x1, x2)
        y = y * cv + y
        y = self.conv_out(y)
        return y, cv

class CV5(nn.Module):
    def __init__(self, dim_in, dim_out, search_range=1):
        super(CV5, self).__init__()
        self.search_range = search_range
        self.cv_layer = CostVolumeLayer2(self.search_range)
        self.conv_1x1 = nn.Sequential(
            nn.Conv2d((self.search_range * 2 + 1) ** 2, self.search_range * 2 + 1, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.search_range * 2 + 1),
            nn.ReLU(),
            nn.Conv2d(self.search_range * 2 + 1, 1, kernel_size=1, stride=1, padding=0),
        )
        self.cat = cat_conv(dim_in, dim_in, dim_out, False)
        self.conv_out = nn.Sequential(
            nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(dim_out),
            nn.ReLU(),
        )
        
    def forward(self, x1, x2):
        cv = 1 - self.cv_layer(x1, x2) / (self.search_range * 2 + 1) ** 2
        cv = self.conv_1x1(cv)
        cv = torch.sigmoid(cv)
        y = self.cat(x1, x2)
        y = y * cv + y
        y = self.conv_out(y)
        return y, cv


class CDM(nn.Module):
    def __init__(self, channel):
        super(CDM, self).__init__()
        self.conv_1x1_1 = nn.Sequential(
            nn.Conv2d(channel*2, channel, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channel),
            nn.ReLU(),
        )
        self.conv_1x1_2 = nn.Sequential(
            nn.Conv2d(channel*2, channel, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channel),
            nn.ReLU(),
        )

    def forward(self, f1, f2):
        fd1 = f1 - f2
        f12 = torch.cat((f1, f2), 1)
        fd2 = self.conv_1x1_1(f12)
        fd = self.conv_1x1_2(torch.cat((fd1, fd2), 1))
        return fd
        

class CCAM(nn.Module):
    """ Change Channel attention module"""
    def __init__(self, in_dim):
        super(CCAM, self).__init__()
        self.conv_1x1 = nn.Conv2d(in_dim, in_dim*3, kernel_size=1, stride=1, padding=0)
        self.cdm = CDM(in_dim)
        self.softmax  = nn.Softmax(dim=-1)
        # self.bn = nn.BatchNorm2d(in_dim)
        # self.gamma1 = nn.Parameter(torch.zeros(1))
        # self.gamma2 = nn.Parameter(torch.zeros(1))

    def forward(self, x1, x2):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X C X C
        """
        m_batchsize, C, height, width = x1.size()
        diff = self.cdm(x1, x2)
        qkv = self.conv_1x1(diff).reshape(m_batchsize, C, 3, height, width).permute(2, 0, 1, 3, 4)
        proj_query, proj_key, proj_value = qkv[0], qkv[1], qkv[2]
        proj_query = proj_query.view(m_batchsize, C, -1)
        proj_key = proj_key.view(m_batchsize, C, -1).permute(0, 2, 1)
        proj_value = proj_value.view(m_batchsize, C, -1)

        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy)-energy

        attention =self.softmax(energy_new)

        proj_value1 = x1.view(m_batchsize, C, -1)
        proj_value2 = x2.view(m_batchsize, C, -1)

        out1 = torch.bmm(attention, proj_value1)
        out1 = out1.view(m_batchsize, C, height, width)
        out2 = torch.bmm(attention, proj_value2)
        out2 = out2.view(m_batchsize, C, height, width)
        out_diff = torch.bmm(attention, proj_value)
        out_diff = out_diff.view(m_batchsize, C, height, width)
        out1 = out1 + x1
        out2 = out2 + x2
        out_diff = out_diff + diff
        
        return out1, out2, out_diff


class CV6(nn.Module):
    def __init__(self, dim_in, dim_out, search_range=1):
        super(CV6, self).__init__()
        self.ccam = CCAM(dim_in)
        self.search_range = search_range
        self.cv_layer = CostVolumeLayer2(self.search_range)
        self.conv_1x1 = nn.Sequential(
            nn.Conv2d((self.search_range * 2 + 1) ** 2, self.search_range * 2 + 1, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.search_range * 2 + 1),
            nn.Conv2d(self.search_range * 2 + 1, 1, kernel_size=1, stride=1, padding=0),
        )
        # self.cat = cat_conv(dim_in, dim_in, dim_out, False)
        self.conv_out = nn.Sequential(
            nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(dim_out),
            nn.ReLU(),
        )
        
    def forward(self, x1, x2):
        x1, x2, diff = self.ccam(x1, x2)
        cv = 1 - self.cv_layer(x1, x2) / (self.search_range * 2 + 1) ** 2
        cv = self.conv_1x1(cv)
        cv = torch.sigmoid(cv)
        y = torch.mul(diff, cv) + diff
        y = self.conv_out(y)
        return y, cv


# class CCAM2(nn.Module):
#     """ Change Channel attention module"""
#     def __init__(self, in_dim):
#         super(CCAM2, self).__init__()
#         self.conv_1x1 = nn.Conv2d(in_dim, in_dim*3, kernel_size=1, stride=1, padding=0)
#         self.cdm = CDM(in_dim)
#         self.softmax  = nn.Softmax(dim=-1)

#     def forward(self, x1, x2):
#         """
#             inputs :
#                 x : input feature maps( B X C X H X W)
#             returns :
#                 out : attention value + input feature
#                 attention: B X C X C
#         """
#         m_batchsize, C, height, width = x1.size()
#         diff = abs(x1 - x2)
#         qkv = self.conv_1x1(diff).reshape(m_batchsize, C, 3, height, width).permute(2, 0, 1, 3, 4)
#         proj_query, proj_key, proj_value = qkv[0], qkv[1], qkv[2]
#         proj_query = proj_query.view(m_batchsize, C, -1)
#         proj_key = proj_key.view(m_batchsize, C, -1).permute(0, 2, 1)
#         proj_value = proj_value.view(m_batchsize, C, -1)

#         energy = torch.bmm(proj_query, proj_key)
#         energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy)-energy

#         attention =self.softmax(energy_new)

#         proj_value1 = x1.view(m_batchsize, C, -1)
#         proj_value2 = x2.view(m_batchsize, C, -1)

#         out1 = torch.bmm(attention, proj_value1)
#         out1 = out1.view(m_batchsize, C, height, width)
#         out2 = torch.bmm(attention, proj_value2)
#         out2 = out2.view(m_batchsize, C, height, width)
#         out_diff = torch.bmm(attention, proj_value)
#         out_diff = out_diff.view(m_batchsize, C, height, width)
#         out1 = out1 + x1
#         out2 = out2 + x2
#         out_diff = out_diff + diff
        
#         return out1, out2, out_diff


class CCAM2(nn.Module):
    """ Change Channel attention module"""
    def __init__(self, in_dim):
        super(CCAM2, self).__init__()
        self.conv_1x1 = nn.Conv2d(in_dim, in_dim*3, kernel_size=1, stride=1, padding=0)
        self.relu  = nn.ReLU()
        self.softmax  = nn.Softmax(dim=-1)

    def forward(self, x1, x2):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X C X C
        """
        m_batchsize, C, height, width = x1.size()
        diff = abs(x1 - x2)
        qkv = self.conv_1x1(diff).reshape(m_batchsize, C, 3, height, width).permute(2, 0, 1, 3, 4)
        proj_query, proj_key, proj_value = qkv[0], qkv[1], qkv[2]
        proj_query = proj_query.view(m_batchsize, C, -1)
        proj_key = proj_key.view(m_batchsize, C, -1).permute(0, 2, 1)
        proj_value = proj_value.view(m_batchsize, C, -1)

        energy = torch.bmm(proj_query, proj_key)
        """way 1"""
        # energy_new = (torch.max(energy, -1, keepdim=True)[0].expand_as(energy)-energy)
        # """way 2"""
        energy_new = energy / C ** 0.5
        # """way 3"""
        # energy_new = torch.sqrt(energy/C) #cause nan

        attention =self.softmax(energy_new)

        proj_value1 = x1.view(m_batchsize, C, -1)
        proj_value2 = x2.view(m_batchsize, C, -1)

        out1 = torch.bmm(attention, proj_value1)
        out1 = out1.view(m_batchsize, C, height, width)
        out2 = torch.bmm(attention, proj_value2)
        out2 = out2.view(m_batchsize, C, height, width)
        out_diff = torch.bmm(attention, proj_value)
        out_diff = out_diff.view(m_batchsize, C, height, width)
        out1 = out1 + x1
        out2 = out2 + x2
        out_diff = out_diff + diff
        
        return out1, out2, out_diff


class CV7(nn.Module):
    def __init__(self, dim_in, dim_out, search_range=1):
        super(CV7, self).__init__()
        self.ccam = CCAM2(dim_in)
        self.search_range = search_range
        self.cv_layer = CostVolumeLayer2(self.search_range)
        self.conv_1x1 = nn.Sequential(
            nn.Conv2d((self.search_range * 2 + 1) ** 2, self.search_range * 2 + 1, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.search_range * 2 + 1),
            nn.Conv2d(self.search_range * 2 + 1, 1, kernel_size=1, stride=1, padding=0),
        )
        self.cat = cat_conv(dim_in, dim_in, dim_in, False)
        self.conv_1x1_2 = nn.Sequential(
            nn.Conv2d(dim_in*2, dim_in, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(dim_in),
            nn.ReLU(),
        )
        self.conv_out = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(dim_out),
            nn.ReLU(),
        )
        
    def forward(self, x1, x2):
        x1, x2, diff = self.ccam(x1, x2)
        cv = 1 - self.cv_layer(x1, x2) / (self.search_range * 2 + 1) ** 2
        cv = self.conv_1x1(cv)
        cv = torch.sigmoid(cv)
        y = self.cat(x1, x2)
        y = self.conv_1x1_2(torch.cat((y, diff), 1))
        y = torch.mul(y, cv) + y
        y = self.conv_out(y)
        return y, cv


class CV8(CV7):
    def __init__(self, dim_in, dim_out, search_range=1):
        super(CV8, self).__init__( dim_in, dim_out, search_range)
        self.cat = cat(dim_in, dim_in, dim_in, False)


class CV9(nn.Module):
    def __init__(self, dim_in, dim_out, search_range=1):
        super(CV9, self).__init__()
        self.ccam = CCAM2(dim_in)
        self.search_range = search_range
        self.cv_layer = CostVolumeLayer2(self.search_range)
        self.conv_1x1 = nn.Sequential(
            nn.Conv2d((self.search_range * 2 + 1) ** 2, self.search_range * 2 + 1, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.search_range * 2 + 1),
            nn.Conv2d(self.search_range * 2 + 1, 1, kernel_size=1, stride=1, padding=0),
        )
        self.cat = cat_conv(dim_in, dim_in, dim_in, False)
        self.conv_1x1_2 = nn.Sequential(
            nn.Conv2d(dim_in*2, dim_in, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(dim_in),
            nn.ReLU(),
        )
        
    def forward(self, x1, x2):
        x1, x2, diff = self.ccam(x1, x2)
        cv = 1 - self.cv_layer(x1, x2) / (self.search_range * 2 + 1) ** 2
        cv = self.conv_1x1(cv)
        cv = torch.sigmoid(cv)
        y = self.cat(x1, x2)
        y = self.conv_1x1_2(torch.cat((y, diff), 1))
        # from clcore import ImageIO;io = ImageIO()
        # aa = y[0].permute(1,2,0)
        # io.write_image(f'./test1.tif', aa.cpu().numpy(), dtype='float32')
        # import ipdb;ipdb.set_trace()
        # import cv2;import numpy as np
        # aa = cv[0][0].cpu().numpy()*255
        # # cv2.imwrite('test.png', aa.astype(np.uint8))
        # aa = cv2.resize(aa, dsize=(512, 512), interpolation=cv2.INTER_LINEAR)
        # aa = aa.astype(np.uint8)
        # heatmap = cv2.applyColorMap(aa, cv2.COLORMAP_JET)
        # cv2.imwrite('test110_hm.png', heatmap)
        return y, cv


class CV10(CV9):
    def __init__(self, dim_in, dim_out, search_range=1):
        super(CV10, self).__init__(dim_in, dim_out, search_range)
        self.cv_layer = CostVolumeLayer3(self.search_range)
        
    def forward(self, x1, x2):
        x1, x2, diff = self.ccam(x1, x2)
        cv = 1 - self.cv_layer(x1, x2) / (self.search_range * 2 + 1) ** 2
        cv = self.conv_1x1(cv)
        cv = torch.sigmoid(cv)
        y = self.cat(x1, x2)
        y = self.conv_1x1_2(torch.cat((y, diff), 1))
        # from clcore import ImageIO;io = ImageIO()
        # aa = y[0].permute(1,2,0)
        # io.write_image(f'./test1.tif', aa.cpu().numpy(), dtype='float32')
        # import ipdb;ipdb.set_trace()
        # import cv2;import numpy as np
        # aa = cv[0][0].cpu().numpy()*255
        # # cv2.imwrite('test.png', aa.astype(np.uint8))
        # aa = cv2.resize(aa, dsize=(512, 512), interpolation=cv2.INTER_LINEAR)
        # aa = aa.astype(np.uint8)
        # heatmap = cv2.applyColorMap(aa, cv2.COLORMAP_JET)
        # cv2.imwrite('test110_hm.png', heatmap)
        # import ipdb;ipdb.set_trace()
        return y, cv


class CCAM3(nn.Module):
    """ Change Channel attention module"""
    def __init__(self, in_dim):
        super(CCAM3, self).__init__()
        self.cat = cat(in_dim, in_dim, in_dim)
        self.relu  = nn.ReLU()
        self.softmax  = nn.Softmax(dim=-1)
        self.cam = CAM(in_dim)

    def forward(self, x1, x2):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X C X C
        """
        m_batchsize, C, height, width = x1.size()
        diff = self.cat(x1, x2)
        attention = self.cam(diff)
        out1 = x1 + x1 * attention
        out2 = x2 + x2 * attention
        out_diff = diff + diff * attention
        
        return out1, out2, out_diff


class CV11(CV10):
    def __init__(self, dim_in, dim_out, search_range=1):
        super(CV10, self).__init__(dim_in, dim_out, search_range)
        self.ccam = CCAM3(dim_in)
        
    def forward(self, x1, x2):
        x1, x2, diff = self.ccam(x1, x2)
        cv = 1 - self.cv_layer(x1, x2) / (self.search_range * 2 + 1) ** 2
        cv = self.conv_1x1(cv)
        cv = torch.sigmoid(cv)
        y = self.cat(x1, x2)
        y = self.conv_1x1_2(torch.cat((y, diff), 1))
        return y, cv


__all__ = [
    "CostVolumeLayer",
    "CV_cat",
    "CV4",
    "CV5",
    "CV6",
    "CV7",
    "CV8",
    "CV9",
    "CV10",
    "CV11",
]
