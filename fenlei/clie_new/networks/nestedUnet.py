import torch
from torch import nn
import random
from .utils.se_basicblock import SEModule
from networks.backbone import Build_Backbone
import torch.nn.functional as F

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch,residual=True):
        super(DoubleConv, self).__init__()
        self.residual = residual
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.relu2 = nn.ReLU(inplace=True)


    def forward(self, input):
        x0 = self.conv1(input)
        x0 = self.bn1(x0)
        x0=self.relu1(x0)
        x = self.conv2(x0)
        x = self.bn2(x)
        x = self.relu2(x)
        if self.residual:
            x=x+x0
        return x


class SeDoubleConv(DoubleConv):
    def __init__(self, in_ch, out_ch,residual=True, use_se=True):
        super(SeDoubleConv, self).__init__(in_ch, out_ch,residual)
        self.se = SEModule(out_ch, out_ch//4) if use_se else None

    def forward(self, input):
        x0 = self.conv1(input)
        x0 = self.bn1(x0)
        x0=self.relu1(x0)
        x = self.conv2(x0)
        x = self.bn2(x)
        x = self.relu2(x)
        if self.se is not None:
            x = self.se(x)
        if self.residual:
            x=x+x0
        return x

class NestedUNet(nn.Module):
    def __init__(self, num_band=3, num_class=1, deep_supervise=False, mode="seg", residual=True, with_matrix_learning=False, **kwargs):
        super(NestedUNet,self).__init__()
        if mode == 'change':
            num_band = num_band*2
        self.deep_supervise = deep_supervise
        self.with_matrix_learning = with_matrix_learning

        self.nb_filter = [32, 64, 128, 256, 512]
        # nb_filter = [24, 48, 96, 192, 384]

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = DoubleConv(num_band, self.nb_filter[0],residual=residual)
        self.conv1_0 = DoubleConv(self.nb_filter[0], self.nb_filter[1],residual=residual)
        self.conv2_0 = DoubleConv(self.nb_filter[1], self.nb_filter[2],residual=residual)
        self.conv3_0 = DoubleConv(self.nb_filter[2], self.nb_filter[3],residual=residual)
        self.conv4_0 = DoubleConv(self.nb_filter[3], self.nb_filter[4],residual=residual)

        self.conv0_1 = DoubleConv(self.nb_filter[0]+self.nb_filter[1], self.nb_filter[0],residual=residual)
        self.conv1_1 = DoubleConv(self.nb_filter[1]+self.nb_filter[2], self.nb_filter[1],residual=residual)
        self.conv2_1 = DoubleConv(self.nb_filter[2]+self.nb_filter[3], self.nb_filter[2],residual=residual)
        self.conv3_1 = DoubleConv(self.nb_filter[3]+self.nb_filter[4], self.nb_filter[3],residual=residual)

        self.conv0_2 = DoubleConv(self.nb_filter[0]*2+self.nb_filter[1], self.nb_filter[0],residual=residual)
        self.conv1_2 = DoubleConv(self.nb_filter[1]*2+self.nb_filter[2], self.nb_filter[1],residual=residual)
        self.conv2_2 = DoubleConv(self.nb_filter[2]*2+self.nb_filter[3], self.nb_filter[2],residual=residual)

        self.conv0_3 = DoubleConv(self.nb_filter[0]*3+self.nb_filter[1], self.nb_filter[0],residual=residual)
        self.conv1_3 = DoubleConv(self.nb_filter[1]*3+self.nb_filter[2], self.nb_filter[1],residual=residual)

        self.conv0_4 = DoubleConv(self.nb_filter[0]*4+self.nb_filter[1], self.nb_filter[0],residual=residual)
        self.sigmoid = nn.Sigmoid()
        if self.deep_supervise:
            self.final1 = nn.Conv2d(self.nb_filter[0], num_class, kernel_size=1)
            self.final2 = nn.Conv2d(self.nb_filter[0], num_class, kernel_size=1)
            self.final3 = nn.Conv2d(self.nb_filter[0], num_class, kernel_size=1)
            self.final4 = nn.Conv2d(self.nb_filter[0], num_class, kernel_size=1)
        else:
            self.final = nn.Conv2d(self.nb_filter[0], num_class, kernel_size=1)
        self.final_operation = None

    def forward(self, x, gt=None):
        b, c, h, w = x.size()
        x0_0 = self.conv0_0(x)

        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))

        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))

        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))
        deep_features = []
        indexs = []
        if self.with_matrix_learning and gt != None:
            deep_features = []
            indexs = []
            for i in range(16):
                idx_h = random.randint(0, h//16-1)
                idx_w = random.randint(0, h//16-1)
                if gt[:,:,idx_h, idx_w].sum() == 0:
                    continue
                indexs.append([idx_h, idx_w])
                deep_features.append(x4_0[:,:,idx_h, idx_w])
            
        if self.deep_supervise:
            if self.final_operation != None:
                output1 = self.final1(self.final_operation(x0_1))
                output2 = self.final2(self.final_operation(x0_2))
                output3 = self.final3(self.final_operation(x0_3))
                output4 = self.final4(self.final_operation(x0_4))
            else:
                output1 = self.final1(x0_1)
                output2 = self.final2(x0_2)
                output3 = self.final3(x0_3)
                output4 = self.final4(x0_4)
            return [output1, output2, output3, output4]

        else:
            if self.final_operation != None:
                x0_4 = self.final_operation(x0_4)
            output = self.final(x0_4)
            if self.with_matrix_learning and gt != None:
                return [output, deep_features, indexs]
            else:
                return [output]


class Deep_NestedUNet(nn.Module):
    def __init__(self, backbone_name, pretrained, num_band, output_stride=16, num_class=1,  mode='seg', use_se=True, \
        deep_supervise=False, smp_encoders=False, pretrained_path='', with_matrix_learning=False, **kwargs):
        super(Deep_NestedUNet,self).__init__()
        if mode == 'change':
            num_band = num_band*2
        self.deep_supervise = deep_supervise
        self.with_matrix_learning = with_matrix_learning
        residual = True

        self.backbone, self.channels_blocks, do_upsample = Build_Backbone(backbone_name, pretrained, num_band, \
            output_stride, pretrained_path=pretrained_path)
    
        self.nb_filter = self.channels_blocks[::-1]
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_1 = DoubleConv(self.nb_filter[0]+self.nb_filter[1], self.nb_filter[0],residual=residual)
        self.conv1_1 = DoubleConv(self.nb_filter[1]+self.nb_filter[2], self.nb_filter[1],residual=residual)
        self.conv2_1 = DoubleConv(self.nb_filter[2]+self.nb_filter[3], self.nb_filter[2],residual=residual)
        self.conv3_1 = DoubleConv(self.nb_filter[3]+self.nb_filter[4], self.nb_filter[3],residual=residual)

        self.conv0_2 = DoubleConv(self.nb_filter[0]*2+self.nb_filter[1], self.nb_filter[0],residual=residual)
        self.conv1_2 = DoubleConv(self.nb_filter[1]*2+self.nb_filter[2], self.nb_filter[1],residual=residual)
        self.conv2_2 = DoubleConv(self.nb_filter[2]*2+self.nb_filter[3], self.nb_filter[2],residual=residual)

        self.conv0_3 = DoubleConv(self.nb_filter[0]*3+self.nb_filter[1], self.nb_filter[0],residual=residual)
        self.conv1_3 = DoubleConv(self.nb_filter[1]*3+self.nb_filter[2], self.nb_filter[1],residual=residual)

        self.conv0_4 = DoubleConv(self.nb_filter[0]*4+self.nb_filter[1], self.nb_filter[0],residual=residual)
        self.sigmoid = nn.Sigmoid()
        if self.deep_supervise:
            self.final1 = nn.Conv2d(self.nb_filter[0], num_class, kernel_size=3, padding=1)
            self.final2 = nn.Conv2d(self.nb_filter[0], num_class, kernel_size=3, padding=1)
            self.final3 = nn.Conv2d(self.nb_filter[0], num_class, kernel_size=3, padding=1)
            self.final4 = nn.Conv2d(self.nb_filter[0], num_class, kernel_size=3, padding=1)
        else:
            self.final = nn.Conv2d(self.nb_filter[0], num_class, kernel_size=3, padding=1)
        self.final_operation = None

    def forward(self, x, gt=None):
        b, c, h, w = x.size()
        layers = self.backbone(x)
        # x0_0 = self.conv0_0(x)

        # x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([layers[0], self.up(layers[1])], 1))

        # x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([layers[1], self.up(layers[2])], 1))
        x0_2 = self.conv0_2(torch.cat([layers[0], x0_1, self.up(x1_1)], 1))

        # x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([layers[2], self.up(layers[3])], 1))
        x1_2 = self.conv1_2(torch.cat([layers[1], x1_1, self.up(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([layers[0], x0_1, x0_2, self.up(x1_2)], 1))

        # x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.conv3_1(torch.cat([layers[3], self.up(layers[4])], 1))
        x2_2 = self.conv2_2(torch.cat([layers[2], x2_1, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([layers[1], x1_1, x1_2, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([layers[0], x0_1, x0_2, x0_3, self.up(x1_3)], 1))

        deep_features = []
        indexs = []
        if self.with_matrix_learning and gt != None:
            deep_features = []
            indexs = []
            for i in range(8):
                idx_h = random.randint(0, h//16-1)
                idx_w = random.randint(0, h//16-1)
                if gt[:,:,idx_h, idx_w].sum() == 0:
                    continue
                indexs.append([idx_h, idx_w])
                deep_features.append(x1_3[:,:,idx_h, idx_w])

        if self.deep_supervise:
            if self.final_operation != None:
                output1 = self.final1(self.final_operation(x0_1))
                output2 = self.final2(self.final_operation(x0_2))
                output3 = self.final3(self.final_operation(x0_3))
                output4 = self.final4(self.final_operation(x0_4))
            else:
                output1 = self.final1(x0_1)
                output2 = self.final2(x0_2)
                output3 = self.final3(x0_3)
                output4 = self.final4(x0_4)
            return [output1, output2, output3, output4]

        else:
            x0_4 = F.interpolate(x0_4, size=(h, w), mode='bilinear', align_corners=True)
            if self.final_operation != None:
                x0_4 = self.final_operation(x0_4)
            output = self.final(x0_4)
            if self.with_matrix_learning and gt != None:
                return [output, deep_features, indexs]
            else:
                return output


class NestedUNet_U2(NestedUNet):
    def __init__(self, num_band=3, num_class=1, deep_supervise=False, mode="seg", residual=True, with_matrix_learning=False, **kwargs):
        super(NestedUNet_U2, self).__init__(num_band, num_class, deep_supervise, mode, residual, with_matrix_learning, **kwargs)
        self.final_operation = nn.MaxPool2d(2, 2)


class Deep_NestedUNet_U2(Deep_NestedUNet):
    def __init__(self, backbone_name, pretrained, num_band, output_stride=16, num_class=1,  mode='seg', use_se=True, \
        deep_supervise=False, smp_encoders=False, pretrained_path='', with_matrix_learning=False, **kwargs):
        super(Deep_NestedUNet_U2, self).__init__(backbone_name, pretrained, num_band, output_stride, num_class,  mode, use_se, \
        deep_supervise, smp_encoders, pretrained_path, with_matrix_learning, **kwargs)
        self.final_operation = nn.MaxPool2d(2, 2)


class Se_NestedUNet(NestedUNet):
    def __init__(self, num_band=3, num_class=1, deep_supervise=False, residual=True, mode="seg", with_matrix_learning=False, use_se=True, **kwargs):
        super(Se_NestedUNet,self).__init__(num_band, num_class, deep_supervise, residual, mode, with_matrix_learning, **kwargs)
        if mode == 'change':
            num_band = num_band*2
        self.with_matrix_learning = with_matrix_learning

        self.nb_filter = [32, 64, 128, 256, 512]

        self.conv0_1 = SeDoubleConv(self.nb_filter[0]+self.nb_filter[1], self.nb_filter[0],residual=residual, use_se=use_se)
        self.conv1_1 = SeDoubleConv(self.nb_filter[1]+self.nb_filter[2], self.nb_filter[1],residual=residual, use_se=use_se)
        self.conv2_1 = SeDoubleConv(self.nb_filter[2]+self.nb_filter[3], self.nb_filter[2],residual=residual, use_se=use_se)
        self.conv3_1 = SeDoubleConv(self.nb_filter[3]+self.nb_filter[4], self.nb_filter[3],residual=residual, use_se=use_se)

        self.conv0_2 = SeDoubleConv(self.nb_filter[0]*2+self.nb_filter[1], self.nb_filter[0],residual=residual, use_se=use_se)
        self.conv1_2 = SeDoubleConv(self.nb_filter[1]*2+self.nb_filter[2], self.nb_filter[1],residual=residual, use_se=use_se)
        self.conv2_2 = SeDoubleConv(self.nb_filter[2]*2+self.nb_filter[3], self.nb_filter[2],residual=residual, use_se=use_se)

        self.conv0_3 = SeDoubleConv(self.nb_filter[0]*3+self.nb_filter[1], self.nb_filter[0],residual=residual, use_se=use_se)
        self.conv1_3 = SeDoubleConv(self.nb_filter[1]*3+self.nb_filter[2], self.nb_filter[1],residual=residual, use_se=use_se)

        self.conv0_4 = SeDoubleConv(self.nb_filter[0]*4+self.nb_filter[1], self.nb_filter[0],residual=residual, use_se=use_se)
        self.sigmoid = nn.Sigmoid()
        if self.deep_supervise:
            self.final1 = nn.Conv2d(self.nb_filter[0], num_class, kernel_size=3, padding=1)
            self.final2 = nn.Conv2d(self.nb_filter[0], num_class, kernel_size=3, padding=1)
            self.final3 = nn.Conv2d(self.nb_filter[0], num_class, kernel_size=3, padding=1)
            self.final4 = nn.Conv2d(self.nb_filter[0], num_class, kernel_size=3, padding=1)
        else:
            self.final = nn.Conv2d(self.nb_filter[0], num_class, kernel_size=3, padding=1)


class Se_NestedUNet_U2(Se_NestedUNet):
    def __init__(self, num_band=3, num_class=1, deep_supervise=False, residual=True, mode="seg", with_matrix_learning=False, **kwargs):
        super(Se_NestedUNet_U2,self).__init__(num_band, num_class, deep_supervise, residual, mode, with_matrix_learning, **kwargs)
        self.final_operation = nn.MaxPool2d(2, 2)