import torch.nn as nn
import torch
# import resnet
from .backbone import Build_Backbone
import torch.nn.functional as F
from networks.utils.ASPP import ASPPModule
from networks.utils.se_basicblock import SeDoubleConv

bn_mom = 0.0003

class double_conv(torch.nn.Module):
    def __init__(self,in_chn, out_chn):#params:in_chn(input channel of double conv),out_chn(output channel of double conv)
        super(double_conv,self).__init__() ##parent's init func

        self.conv=torch.nn.Sequential(
            torch.nn.Conv2d(in_chn,out_chn,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(out_chn, momentum=bn_mom),
            torch.nn.ReLU(),
            torch.nn.Conv2d(out_chn,out_chn,kernel_size=3,stride=1,padding=1),
        )
    
    def forward(self,x):
        x = self.conv(x)
        return x
    
class cat(torch.nn.Module):
    def __init__(self, in_chn_high, in_chn_low, out_chn, upsample = False):
        super(cat,self).__init__() ##parent's init func
        self.do_upsample = upsample
        self.upsample = torch.nn.Upsample(scale_factor=2,mode='bilinear',align_corners=True)
        self.conv2d=torch.nn.Sequential(
            torch.nn.Conv2d(in_chn_high + in_chn_low, out_chn, kernel_size=1,stride=1,padding=0),
            nn.BatchNorm2d(out_chn, momentum=bn_mom),
            torch.nn.ReLU(),
        )
    
    def forward(self,x,y):
        # import ipdb
        # ipdb.set_trace()
        if self.do_upsample:
            x = self.upsample(x)
        
        # diffY = y.size()[2] - x.size()[2]
        # diffX = y.size()[3] - x.size()[3]

        # x = F.pad(x, (diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2))

        x = torch.cat((x,y),1)#x,y shape(batch_sizxe,channel,w,h), concat at the dim of channel
        return self.conv2d(x)


class unet_encoder(torch.nn.Module):
    def __init__(self,num_band):
        super(unet_encoder, self).__init__() ##parent's init func
        self.conv1=torch.nn.Sequential(
            double_conv(num_band,32),
            nn.BatchNorm2d(32, momentum=bn_mom),
            torch.nn.ReLU()
        )
        self.conv2=torch.nn.Sequential(
            torch.nn.MaxPool2d(stride=2,kernel_size=2),
            double_conv(32, 64),
            nn.BatchNorm2d(64, momentum=bn_mom),
            torch.nn.ReLU()
        )
        self.conv3=torch.nn.Sequential(
            torch.nn.MaxPool2d(stride=2,kernel_size=2),
            double_conv(64,128),
            nn.BatchNorm2d(128, momentum=bn_mom),
            torch.nn.ReLU()
        )
        self.conv4=torch.nn.Sequential(
            torch.nn.MaxPool2d(stride=2,kernel_size=2),
            double_conv(128,256),
            nn.BatchNorm2d(256, momentum=bn_mom),
            torch.nn.ReLU()
        )
        self.conv5=torch.nn.Sequential(
            torch.nn.MaxPool2d(stride=2,kernel_size=2),
            double_conv(256,512),
            nn.BatchNorm2d(512, momentum=bn_mom),
            torch.nn.ReLU()
        )

    def forward(self,x):
        copies=[]#copies for upsample
        x=self.conv1(x)
        copies.append(x)    #X1
        x=self.conv2(x)
        copies.append(x)    #X2
        x=self.conv3(x)
        copies.append(x)    #X4
        x=self.conv4(x)
        copies.append(x)    #X8
        x=self.conv5(x)
        copies.append(x)    #X16
        # x=self.conv6(x)     #X16
        # copies.append(x)
        # x=self.conv7(x)
        return copies

class bridge(torch.nn.Module):  #X16 -> X32
    def __init__(self, in_chn, output_stride = 1):
        super(bridge, self).__init__() ##parent's init func
        if output_stride not in [1, 2]:
            raise ValueError('UNet.py: invalid output_stride')
        self.downsample = torch.nn.MaxPool2d(stride=2, kernel_size=2) if output_stride == 2 else None
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_chn, in_chn * 2, kernel_size=3,stride = output_stride, padding=1),
            nn.BatchNorm2d(in_chn * 2, momentum=bn_mom),
            torch.nn.ReLU(),
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(in_chn * 2, in_chn, kernel_size=1,stride = 1, padding=0),
            nn.BatchNorm2d(in_chn, momentum=bn_mom),
            torch.nn.ReLU(),
        )

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample:
            x = self.downsample(x)
        return out + x

class bridge2(torch.nn.Module):  #X16 -> X32
    def __init__(self, in_chn, out_chn, output_stride = 1):
        super(bridge2, self).__init__() ##parent's init func
        if output_stride not in [1, 2]:
            raise ValueError('UNet.py: invalid output_stride')

        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_chn, in_chn * 2, kernel_size=3,stride = output_stride, padding=1),
            nn.BatchNorm2d(in_chn * 2, momentum=bn_mom),
            torch.nn.ReLU(),
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(in_chn * 2, out_chn, kernel_size=1,stride = 1, padding=0),
            nn.BatchNorm2d(out_chn, momentum=bn_mom),
            torch.nn.ReLU(),
        )

    def forward(self, x):
        x = self.conv1(x)
        out = self.conv2(x)
        return out


class bridge_res(torch.nn.Module):  #X16 -> X32 -> X16
    def __init__(self, in_chn):
        super(bridge_res, self).__init__() ##parent's init func
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_chn, in_chn * 2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(in_chn * 2, momentum=bn_mom),
            torch.nn.ReLU(),
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(in_chn * 2, in_chn, kernel_size=1,stride=1,padding=0),
            nn.BatchNorm2d(in_chn, momentum=bn_mom),
            torch.nn.ReLU(),
        )
        self.upsample_2 = nn.Upsample(scale_factor=2,mode='bilinear',align_corners=True)

    def forward(self, x):
        y = self.conv1(x)
        y = self.conv2(y)
        y = self.upsample_2(y)
        return x + y

class Dblock_r25(nn.Module):
    def __init__(self, channel):
        super(Dblock_r25, self).__init__()
        self.dilate1 = nn.Conv2d(channel, channel, kernel_size=3, dilation=1, padding=1)
        self.dilate2 = nn.Conv2d(channel, channel, kernel_size=3, dilation=2, padding=2)
        self.dilate3 = nn.Conv2d(channel, channel, kernel_size=3, dilation=3, padding=3)
        self.dilate4 = nn.Conv2d(channel, channel, kernel_size=3, dilation=6, padding=6)
        #self.dilate5 = nn.Conv2d(channel, channel, kernel_size=3, dilation=16, padding=16)
        self.bn1 = nn.BatchNorm2d(channel)
        self.bn2 = nn.BatchNorm2d(channel)
        self.bn3 = nn.BatchNorm2d(channel)
        self.bn4 = nn.BatchNorm2d(channel)
        self.relu = nn.ReLU(inplace=True)
                    
    def forward(self, x):
        dilate1_out = self.relu(self.bn1(self.dilate1(x)))
        dilate2_out = self.relu(self.bn2(self.dilate2(dilate1_out)))
        dilate3_out = self.relu(self.bn3(self.dilate3(dilate2_out)))
        dilate4_out = self.relu(self.bn4(self.dilate4(dilate3_out)))
        #dilate5_out = nonlinearity(self.dilate5(dilate4_out))
        out = x + dilate1_out + dilate2_out + dilate3_out + dilate4_out# + dilate5_out
        return out

class unet_decoder(torch.nn.Module):
    def __init__(self, num_class):
        super(unet_decoder,self).__init__() ##parent's init func
        # self.cat1=cat(2048,2048, 2048, upsample = False)
        self.cat2=cat(512,512, 512, upsample = True)
        self.cat3=cat(256,256, 256, upsample = True)
        self.cat4=cat(128,128, 128, upsample = True)
        self.cat5=cat(64,64, 64, upsample = True)
        self.cat6=cat(32,32,32, upsample = True)

        # self.conv1=torch.nn.Sequential(
        #     double_conv(2048,1024),
        #     nn.BatchNorm2d(1024, momentum=bn_mom),
        #     torch.nn.ReLU(),
        #     # torch.nn.Upsamle(scale_factor=2,mode='bilinear',align_corners=True)
        # )
        self.conv2=torch.nn.Sequential(
            double_conv(512,256),
            nn.BatchNorm2d(256, momentum=bn_mom),
            torch.nn.ReLU(),
            # torch.nn.Upsample(scale_factor=2,mode='bilinear',align_corners=True)
        )
        self.conv3=torch.nn.Sequential(
            double_conv(256,128),
            nn.BatchNorm2d(128, momentum=bn_mom),
            torch.nn.ReLU(),
            # torch.nn.Upsample(scale_factor=2,mode='bilinear',align_corners=True)
        )
        self.conv4=torch.nn.Sequential(
            double_conv(128,64),
            nn.BatchNorm2d(64, momentum=bn_mom),
            torch.nn.ReLU(),
            # torch.nn.Upsample(scale_factor=2,mode='bilinear',align_corners=True)
        )
        self.conv5=torch.nn.Sequential(
            double_conv(64,32),
            nn.BatchNorm2d(32, momentum=bn_mom),
            torch.nn.ReLU(),
            # torch.nn.Upsample(scale_factor=2,mode='bilinear',align_corners=True)
        )
        self.conv6=torch.nn.Sequential(
            double_conv(32,16),
            nn.BatchNorm2d(16, momentum=bn_mom),
            torch.nn.ReLU(),
            # torch.nn.Upsample(scale_factor=2,mode='bilinear',align_corners=True)
        )
        self.conv7=torch.nn.Sequential(
            torch.nn.Conv2d(16,num_class,kernel_size=1,stride=1,padding=0)
        )

    def forward(self,x,copies):
        # import ipdb
        # ipdb.set_trace()
        # x=self.cat1(x,copies[5])
        # x=self.conv1(x)
        x=self.cat2(x,copies[4])
        x=self.conv2(x)
        x=self.cat3(x,copies[3])
        x=self.conv3(x)
        x=self.cat4(x,copies[2])
        x=self.conv4(x)
        x=self.cat5(x,copies[1])
        x=self.conv5(x)
        x=self.cat6(x,copies[0])
        x=self.conv6(x)
        y=self.conv7(x)
        return y

class unet(torch.nn.Module):
    def __init__(self, loss_func, num_band=3, num_class=10):
        super(unet,self).__init__() ##parent's init func
        self.num_class = num_class
        self.loss_func = loss_func
        self.encoder = unet_encoder(num_band)
        self.bridge = bridge(512, 2)
        self.decoder=unet_decoder(num_class)
    
    def forward(self,x):
        copies=self.encoder(x[0])
        x = self.bridge(copies[-1])
        y=self.decoder(x,copies)

        return y

class res_unet_old(torch.nn.Module):
    def __init__(self, backbone_name, pretrained, os, num_band=3, num_class=10):
        super(res_unet_old,self).__init__() ##parent's init func
        self.num_class = num_class
        self.backbone, channels_blocks, do_upsample = Build_Backbone(backbone_name, pretrained, num_band, os)
        # self.encoder=unet_encoder(num_band)
        self.bridge = bridge2(channels_blocks[0], channels_blocks[0], 1)  
        
        self.cat4=cat(channels_blocks[0],channels_blocks[0], channels_blocks[0], upsample = False)
        self.cat3=cat(channels_blocks[1],channels_blocks[1], channels_blocks[1], upsample = do_upsample[0])
        self.cat2=cat(channels_blocks[2],channels_blocks[2], channels_blocks[2], upsample = do_upsample[1])
        self.cat1=cat(channels_blocks[3],channels_blocks[3], channels_blocks[3], upsample = do_upsample[2])

        self.decoder_4=torch.nn.Sequential(
            double_conv(channels_blocks[0],channels_blocks[1]),
            nn.BatchNorm2d(channels_blocks[1], momentum=bn_mom),
            torch.nn.ReLU()
        )
        self.decoder_3=torch.nn.Sequential(
            double_conv(channels_blocks[1],channels_blocks[2]),
            nn.BatchNorm2d(channels_blocks[2], momentum=bn_mom),
            torch.nn.ReLU()
        )
        self.decoder_2=torch.nn.Sequential(
            double_conv(channels_blocks[2],channels_blocks[3]),
            nn.BatchNorm2d(channels_blocks[3], momentum=bn_mom),
            torch.nn.ReLU()
        )
        self.decoder_1=torch.nn.Sequential(
            double_conv(channels_blocks[3],channels_blocks[3]),
            nn.BatchNorm2d(channels_blocks[3], momentum=bn_mom),
            torch.nn.ReLU()
        )

        self.upsample_x4=nn.Sequential(
                        nn.Conv2d( channels_blocks[3],24,kernel_size=3, stride=1, padding=1),
                        nn.BatchNorm2d(24, momentum=bn_mom),
                        nn.ReLU(inplace=True),
                        nn.UpsamplingBilinear2d(scale_factor=2),
                        nn.Conv2d(24,16,kernel_size=3, stride=1, padding=1),
                        nn.BatchNorm2d(16, momentum=bn_mom),
                        nn.ReLU(inplace=True),
                        nn.UpsamplingBilinear2d(scale_factor=2)
                        )
        self.conv_out = torch.nn.Conv2d(16,num_class,kernel_size=1,stride=1,padding=0)

    
    def forward(self,x):
        x0_h, x0_w = x.size(2), x.size(3)
        layers = self.backbone(x)
        x = self.bridge(layers[3])
        x = self.cat4(x, layers[3])
        x=self.decoder_4(x)
        x = self.cat3(x, layers[2])
        x=self.decoder_3(x)
        x = self.cat2(x, layers[1])
        x=self.decoder_2(x)
        x = self.cat1(x, layers[0])
        x=self.decoder_1(x)

        x =self.upsample_x4(x)
        y = self.conv_out(x)

        return y

class res_unet(torch.nn.Module):
    def __init__(self, backbone_name, pretrained, output_stride=16, num_band=3, num_class=1, mode='seg', pretrained_path='', **kwargs):
        super(res_unet,self).__init__() ##parent's init func
        if mode == 'change':
            num_band = num_band*2
        self.backbone, self.channels_blocks, do_upsample = Build_Backbone(backbone_name, pretrained, num_band, output_stride, pretrained_path=pretrained_path)
        self.num_base_layers = len(self.channels_blocks)

        self.center = bridge2(self.channels_blocks[0], self.channels_blocks[0], 1)

        self.concat_blocks = nn.ModuleList([cat(self.channels_blocks[0],self.channels_blocks[0], self.channels_blocks[0], False)])
        self.decode_blocks = nn.ModuleList([double_conv(self.channels_blocks[0], self.channels_blocks[1])])
        for i in range(1, self.num_base_layers):
            self.concat_blocks.append(cat(self.channels_blocks[i],self.channels_blocks[i], self.channels_blocks[i], do_upsample[i]))
            if i < self.num_base_layers-1:
                self.decode_blocks.append(double_conv(self.channels_blocks[i], self.channels_blocks[i+1]))
            else:
                self.decode_blocks.append(double_conv(self.channels_blocks[i], self.channels_blocks[i]))

        self.segmentation_head = nn.Sequential(
                        nn.Conv2d(self.channels_blocks[-1], 16,kernel_size=3, stride=1, padding=1),
                        nn.BatchNorm2d(16, momentum=bn_mom),
                        nn.ReLU(inplace=True),
                        torch.nn.Conv2d(16,num_class,kernel_size=1,stride=1,padding=0)
                        )
    
    def forward(self, x):
        x0_h, x0_w = x.size(2), x.size(3)
        layers = self.backbone(x)
        y = self.center(layers[-1])
        for i in range(self.num_base_layers) :
            y = self.concat_blocks[i](y, layers[self.num_base_layers-i-1])
            y = self.decode_blocks[i](y)

        y = F.interpolate(y, size=(x0_h, x0_w), mode='bilinear', align_corners=True)
        y = self.segmentation_head(y)
        
        return y


class res_unet_aux(res_unet):
    def __init__(self, backbone_name, pretrained, output_stride=16, num_band=3, num_class=1, mode='seg', pretrained_path='', **kwargs):
        super(res_unet_aux,self).__init__(backbone_name, pretrained, output_stride, num_band, num_class, mode, pretrained_path, **kwargs) ##parent's init func
        
        self.segmentation_head_aux = nn.Sequential(
                        nn.Conv2d(self.channels_blocks[0], 16,kernel_size=3, stride=1, padding=1),
                        nn.BatchNorm2d(16, momentum=bn_mom),
                        nn.ReLU(inplace=True),
                        torch.nn.Conv2d(16,num_class,kernel_size=1,stride=1,padding=0)
                        )
    
    def forward(self, x):
        x0_h, x0_w = x.size(2), x.size(3)
        layers = self.backbone(x)
        y = self.center(layers[-1])
        y_aux = self.segmentation_head_aux(y)
        for i in range(self.num_base_layers) :
            y = self.concat_blocks[i](y, layers[self.num_base_layers-i-1])
            y = self.decode_blocks[i](y)

        y = F.interpolate(y, size=(x0_h, x0_w), mode='bilinear', align_corners=True)
        y = self.segmentation_head(y)
        return y

class SA_Aux(torch.nn.Module):
    def __init__(self, in_chn):
        super(SA_Aux, self).__init__()
        self.conv1x1_1 = torch.nn.Sequential(
            nn.Conv2d(in_chn, in_chn, kernel_size=1,stride=1,padding=0),
            nn.BatchNorm2d(in_chn, momentum=bn_mom),
            torch.nn.ReLU()
        )
        self.conv1x1_2 = nn.Conv2d(in_chn, 1, kernel_size=1,stride=1,padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        z = self.conv1x1_1(x)
        z = self.sigmoid(self.conv1x1_2(x))
        return [x*z, z]


class PA_Module(nn.Module):
    """ Position attention module"""
    #Ref from SAGAN
    def __init__(self, in_dim):
        super(PA_Module, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """
        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width*height).permute(0, 2, 1)#(B, W*H, C/8)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width*height)#(B, C/8, W*H)
        energy = torch.bmm(proj_query, proj_key)#(B, W*H, W*H)
        attention = self.softmax(energy)#(B, W*H, W*H)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width*height)#(B, C, W*H)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))#(B, W*H, C)
        out = out.view(m_batchsize, C, height, width)#(B, C, H, W)

        out = self.gamma*out + x
        return out, attention


class CA_Module(nn.Module):
    """ Channel attention module"""
    def __init__(self, in_dim):
        super(CA_Module, self).__init__()
        self.chanel_in = in_dim

        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax  = nn.Softmax(dim=-1)

    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X C X C
        """
        m_batchsize, C, height, width = x.size()
        proj_query = x.view(m_batchsize, C, -1)#(B,C,H*W)
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)#(B,H*W,C)
        energy = torch.bmm(proj_query, proj_key)#(B,C,C)
        """*******************"""
        energy_max = torch.max(energy, -1, keepdim=True)[0].expand_as(energy)#(B,C,C)
        # energy_mean = torch.mean(energy, -1, keepdim=True)[0].expand_as(energy)#(B,C,C)
        energy_new = energy_max  - energy#(B,C,C)
        """*******************"""
        attention = self.softmax(energy_new)#(B,C,C)
        proj_value = x.view(m_batchsize, C, -1)#(B,C,H*W)

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma*out + x
        return out


class CSA_Head(nn.Module):
    def __init__(self, in_chn):
        super(CSA_Head, self).__init__()
        self.cam = CA_Module(in_chn)
        self.pam = PA_Module(in_chn)
    
    def forward(self, x):
        y = self.cam(x)
        y, y_a = self.pam(y)
        return y, y_a

class res_unet_CSA_Aux(nn.Module):
    def __init__(self, loss_func, backbone_name, pretrained, os, num_band=3, num_class=10):
        super(res_unet_CSA_Aux, self).__init__()
        self.num_class = num_class
        self.loss_func = loss_func
        self.backbone, channels_blocks, do_upsample = Build_Backbone(backbone_name, pretrained, num_band, os)
        self.head = CSA_Head(channels_blocks[0])

        self.cat4=cat(channels_blocks[0],channels_blocks[0], channels_blocks[0], upsample = False)
        self.cat3=cat(channels_blocks[1],channels_blocks[1], channels_blocks[1], upsample = do_upsample[0])
        self.cat2=cat(channels_blocks[2],channels_blocks[2], channels_blocks[2], upsample = do_upsample[1])
        self.cat1=cat(channels_blocks[3],channels_blocks[3], channels_blocks[3], upsample = do_upsample[2])

        self.decoder_4=torch.nn.Sequential(
            double_conv(channels_blocks[0],channels_blocks[1]),
            nn.BatchNorm2d(channels_blocks[1], momentum=bn_mom),
            torch.nn.ReLU()
        )
        self.decoder_3=torch.nn.Sequential(
            double_conv(channels_blocks[1],channels_blocks[2]),
            nn.BatchNorm2d(channels_blocks[2], momentum=bn_mom),
            torch.nn.ReLU()
        )
        self.decoder_2=torch.nn.Sequential(
            double_conv(channels_blocks[2],channels_blocks[3]),
            nn.BatchNorm2d(channels_blocks[3], momentum=bn_mom),
            torch.nn.ReLU()
        )
        self.decoder_1=torch.nn.Sequential(
            double_conv(channels_blocks[3],channels_blocks[3]),
            nn.BatchNorm2d(channels_blocks[3], momentum=bn_mom),
            torch.nn.ReLU()
        )

        self.upsample_x4=nn.Sequential(
                        nn.Conv2d( channels_blocks[3],24,kernel_size=3, stride=1, padding=1),
                        nn.BatchNorm2d(24, momentum=bn_mom),
                        nn.ReLU(inplace=True),
                        nn.UpsamplingBilinear2d(scale_factor=2),
                        nn.Conv2d(24,16,kernel_size=3, stride=1, padding=1),
                        nn.BatchNorm2d(16, momentum=bn_mom),
                        nn.ReLU(inplace=True),
                        nn.UpsamplingBilinear2d(scale_factor=2)
                        )
        self.conv_out = torch.nn.Conv2d(16,num_class,kernel_size=1,stride=1,padding=0)

    
    def forward(self,x):
        layers = self.backbone(x[0])
        x, y_aux = self.head(layers[3])
        x = self.cat4(x, layers[3])
        x=self.decoder_4(x)
        x = self.cat3(x, layers[2])
        x=self.decoder_3(x)
        x = self.cat2(x, layers[1])
        x=self.decoder_2(x)
        x = self.cat1(x, layers[0])
        x=self.decoder_1(x)

        x =self.upsample_x4(x)
        y = self.conv_out(x)

        if self.num_class > 1:
            return [torch.softmax(y, dim = 1), y_aux]
        elif  'lovasz' not in self.loss_func:
            return [torch.sigmoid(y), y_aux]
        else:
            return [y, y_aux]

class res_unet_CSA(nn.Module):
    def __init__(self, loss_func, backbone_name, pretrained, os, num_band=3, num_class=10):
        super(res_unet_CSA, self).__init__()
        self.num_class = num_class
        self.loss_func = loss_func
        self.backbone, channels_blocks, do_upsample = Build_Backbone(backbone_name, pretrained, num_band, os)
        self.head = CSA_Head(channels_blocks[0])

        self.cat4=cat(channels_blocks[0],channels_blocks[0], channels_blocks[0], upsample = False)
        self.cat3=cat(channels_blocks[1],channels_blocks[1], channels_blocks[1], upsample = do_upsample[0])
        self.cat2=cat(channels_blocks[2],channels_blocks[2], channels_blocks[2], upsample = do_upsample[1])
        self.cat1=cat(channels_blocks[3],channels_blocks[3], channels_blocks[3], upsample = do_upsample[2])

        self.decoder_4=torch.nn.Sequential(
            double_conv(channels_blocks[0],channels_blocks[1]),
            nn.BatchNorm2d(channels_blocks[1], momentum=bn_mom),
            torch.nn.ReLU()
        )
        self.decoder_3=torch.nn.Sequential(
            double_conv(channels_blocks[1],channels_blocks[2]),
            nn.BatchNorm2d(channels_blocks[2], momentum=bn_mom),
            torch.nn.ReLU()
        )
        self.decoder_2=torch.nn.Sequential(
            double_conv(channels_blocks[2],channels_blocks[3]),
            nn.BatchNorm2d(channels_blocks[3], momentum=bn_mom),
            torch.nn.ReLU()
        )
        self.decoder_1=torch.nn.Sequential(
            double_conv(channels_blocks[3],channels_blocks[3]),
            nn.BatchNorm2d(channels_blocks[3], momentum=bn_mom),
            torch.nn.ReLU()
        )

        self.upsample_x4=nn.Sequential(
                        nn.Conv2d( channels_blocks[3],24,kernel_size=3, stride=1, padding=1),
                        nn.BatchNorm2d(24, momentum=bn_mom),
                        nn.ReLU(inplace=True),
                        nn.UpsamplingBilinear2d(scale_factor=2),
                        nn.Conv2d(24,16,kernel_size=3, stride=1, padding=1),
                        nn.BatchNorm2d(16, momentum=bn_mom),
                        nn.ReLU(inplace=True),
                        nn.UpsamplingBilinear2d(scale_factor=2)
                        )
        self.conv_out = torch.nn.Conv2d(16,num_class,kernel_size=1,stride=1,padding=0)

    
    def forward(self,x):
        layers = self.backbone(x[0])
        x, _ = self.head(layers[3])
        x = self.cat4(x, layers[3])
        x=self.decoder_4(x)
        x = self.cat3(x, layers[2])
        x=self.decoder_3(x)
        x = self.cat2(x, layers[1])
        x=self.decoder_2(x)
        x = self.cat1(x, layers[0])
        x=self.decoder_1(x)

        x =self.upsample_x4(x)
        y = self.conv_out(x)

        if self.num_class > 1:
            return [torch.softmax(y, dim = 1)]
        elif  'lovasz' not in self.loss_func:
            return [torch.sigmoid(y)]
        else:
            return [y]


class dense_connect_head(nn.Module):
    """ dense connect module"""
    def __init__(self, in_dim):
        super(dense_connect_head, self).__init__()
        self.chanel_in = in_dim
        self.conv_a = nn.Conv2d(in_dim, in_dim, kernel_size=3, padding=1)
        self.conv_b = nn.Conv2d(in_dim, in_dim, kernel_size=3, padding=1)
        self.conv_c = nn.Conv2d(in_dim, in_dim, kernel_size=3, padding=1)
        self.conv_d = nn.Conv2d(in_dim, in_dim, kernel_size=3, padding=1)
        self.conv_e = nn.Conv2d(in_dim, in_dim, kernel_size=3, padding=1)
        self.conv_f = nn.Conv2d(in_dim, in_dim, kernel_size=3, padding=1)
        self.conv_fuse = nn.Conv2d(in_dim, in_dim, kernel_size=3, padding=1)

        self.nonlinearity = nn.ReLU()


    def forward(self, x):
        a = self.nonlinearity(self.conv_a(x))
        c = self.nonlinearity(self.conv_c(a))
        b = self.nonlinearity(self.conv_b(a+c))
        f = self.nonlinearity(self.conv_f(a+c))
        e = self.nonlinearity(self.conv_e(f+c))
        d = self.nonlinearity(self.conv_d(a+b+e+f))
        y = self.nonlinearity(self.conv_fuse(x+d))
        return y
        

class res_unet_dense(nn.Module):
    def __init__(self, loss_func, backbone_name, pretrained, os, num_band=3, num_class=10):
        super(res_unet_dense, self).__init__()
        self.num_class = num_class
        self.loss_func = loss_func
        self.backbone, channels_blocks, do_upsample = Build_Backbone(backbone_name, pretrained, num_band, os)
        self.head = dense_connect_head(channels_blocks[0])

        self.cat4=cat(channels_blocks[0],channels_blocks[0], channels_blocks[0], upsample = False)
        self.cat3=cat(channels_blocks[1],channels_blocks[1], channels_blocks[1], upsample = do_upsample[0])
        self.cat2=cat(channels_blocks[2],channels_blocks[2], channels_blocks[2], upsample = do_upsample[1])
        self.cat1=cat(channels_blocks[3],channels_blocks[3], channels_blocks[3], upsample = do_upsample[2])

        self.decoder_4=torch.nn.Sequential(
            double_conv(channels_blocks[0],channels_blocks[1]),
            nn.BatchNorm2d(channels_blocks[1], momentum=bn_mom),
            torch.nn.ReLU()
        )
        self.decoder_3=torch.nn.Sequential(
            double_conv(channels_blocks[1],channels_blocks[2]),
            nn.BatchNorm2d(channels_blocks[2], momentum=bn_mom),
            torch.nn.ReLU()
        )
        self.decoder_2=torch.nn.Sequential(
            double_conv(channels_blocks[2],channels_blocks[3]),
            nn.BatchNorm2d(channels_blocks[3], momentum=bn_mom),
            torch.nn.ReLU()
        )
        self.decoder_1=torch.nn.Sequential(
            double_conv(channels_blocks[3],channels_blocks[3]),
            nn.BatchNorm2d(channels_blocks[3], momentum=bn_mom),
            torch.nn.ReLU()
        )

        self.upsample_x4=nn.Sequential(
                        nn.Conv2d( channels_blocks[3],24,kernel_size=3, stride=1, padding=1),
                        nn.BatchNorm2d(24, momentum=bn_mom),
                        nn.ReLU(inplace=True),
                        nn.UpsamplingBilinear2d(scale_factor=2),
                        nn.Conv2d(24,16,kernel_size=3, stride=1, padding=1),
                        nn.BatchNorm2d(16, momentum=bn_mom),
                        nn.ReLU(inplace=True),
                        nn.UpsamplingBilinear2d(scale_factor=2)
                        )
        self.conv_out = torch.nn.Conv2d(16,num_class,kernel_size=1,stride=1,padding=0)

    
    def forward(self,x):
        layers = self.backbone(x)
        x = self.head(layers[3])
        x = self.cat4(x, layers[3])
        x=self.decoder_4(x)
        x = self.cat3(x, layers[2])
        x=self.decoder_3(x)
        x = self.cat2(x, layers[1])
        x=self.decoder_2(x)
        x = self.cat1(x, layers[0])
        x=self.decoder_1(x)

        x =self.upsample_x4(x)
        y = self.conv_out(x)

        return y


class res_unet_dense_aux(nn.Module):
    def __init__(self, loss_func, backbone_name, pretrained, os, num_band=3, num_class=10):
        super(res_unet_dense_aux, self).__init__()
        self.num_class = num_class
        self.loss_func = loss_func
        self.backbone, channels_blocks, do_upsample = Build_Backbone(backbone_name, pretrained, num_band, os)
        self.head = dense_connect_head(channels_blocks[0])

        self.cat4=cat(channels_blocks[0],channels_blocks[0], channels_blocks[0], upsample = False)
        self.cat3=cat(channels_blocks[1],channels_blocks[1], channels_blocks[1], upsample = do_upsample[0])
        self.cat2=cat(channels_blocks[2],channels_blocks[2], channels_blocks[2], upsample = do_upsample[1])
        self.cat1=cat(channels_blocks[3],channels_blocks[3], channels_blocks[3], upsample = do_upsample[2])

        self.decoder_4=torch.nn.Sequential(
            double_conv(channels_blocks[0],channels_blocks[1]),
            nn.BatchNorm2d(channels_blocks[1], momentum=bn_mom),
            torch.nn.ReLU()
        )
        self.decoder_3=torch.nn.Sequential(
            double_conv(channels_blocks[1],channels_blocks[2]),
            nn.BatchNorm2d(channels_blocks[2], momentum=bn_mom),
            torch.nn.ReLU()
        )
        self.decoder_2=torch.nn.Sequential(
            double_conv(channels_blocks[2],channels_blocks[3]),
            nn.BatchNorm2d(channels_blocks[3], momentum=bn_mom),
            torch.nn.ReLU()
        )
        self.decoder_1=torch.nn.Sequential(
            double_conv(channels_blocks[3],channels_blocks[3]),
            nn.BatchNorm2d(channels_blocks[3], momentum=bn_mom),
            torch.nn.ReLU()
        )

        self.aux_brunch = nn.Sequential(
                nn.Conv2d(channels_blocks[0],64,kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64,32,kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                )

        self.upsample_x4=nn.Sequential(
                        nn.Conv2d( channels_blocks[3],24,kernel_size=3, stride=1, padding=1),
                        nn.BatchNorm2d(24, momentum=bn_mom),
                        nn.ReLU(inplace=True),
                        nn.UpsamplingBilinear2d(scale_factor=2),
                        nn.Conv2d(24,16,kernel_size=3, stride=1, padding=1),
                        nn.BatchNorm2d(16, momentum=bn_mom),
                        nn.ReLU(inplace=True),
                        nn.UpsamplingBilinear2d(scale_factor=2)
                        )
        self.conv_out = torch.nn.Conv2d(16,num_class,kernel_size=1,stride=1,padding=0)
        self.cls_conv_aux = nn.Conv2d(32, num_class, 1, 1, padding=0)

    
    def forward(self,x):
        layers = self.backbone(x[0])
        x = self.head(layers[3])
        y_aux = self.aux_brunch(x)
        x = self.cat4(x, layers[3])
        x = self.decoder_4(x)
        x = self.cat3(x, layers[2])
        x = self.decoder_3(x)
        x = self.cat2(x, layers[1])
        x = self.decoder_2(x)
        x = self.cat1(x, layers[0])
        x = self.decoder_1(x)

        x = self.upsample_x4(x)
        y = self.conv_out(x)
        y_aux = self.cls_conv_aux(y_aux)

        if self.num_class > 1:
            return [torch.softmax(y, dim = 1), torch.softmax(y_aux, dim = 1)]
        elif  'lovasz' not in self.loss_func:
            return [torch.sigmoid(y), torch.sigmoid(y_aux)]
        else:
            return [y, y_aux]


class SpatialAttention2d(nn.Module):
    def __init__(self, channel):
        super(SpatialAttention2d, self).__init__()
        self.squeeze = nn.Conv2d(channel, 1, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        z = self.squeeze(x)
        z = self.sigmoid(z)
        return x * z


class GAB(nn.Module):
    def __init__(self, input_dim, reduction=4):
        super(GAB, self).__init__()
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(input_dim, input_dim // reduction, kernel_size=1, stride=1)
        self.conv2 = nn.Conv2d(input_dim // reduction, input_dim, kernel_size=1, stride=1)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        z = self.global_avgpool(x)
        z = self.relu(self.conv1(z))
        z = self.sigmoid(self.conv2(z))
        return x * z

class SCse(nn.Module):
    def __init__(self, dim):
        super(SCse, self).__init__()
        self.satt = SpatialAttention2d(dim)
        self.catt = GAB(dim)

    def forward(self, x):
        return self.satt(x) + self.catt(x)


class dense_connect_head_v2(nn.Module):
    """ dense connect module"""
    def __init__(self, in_dim):
        super(dense_connect_head_v2, self).__init__()
        
        self.conv_a = nn.Conv2d(in_dim, in_dim//2, kernel_size=3, padding=1)
        self.conv_b = nn.Conv2d(in_dim//2, in_dim//2, kernel_size=3, padding=1)
        self.conv_c = nn.Conv2d(in_dim//2, in_dim//2, kernel_size=3, padding=1)
        self.conv_d = nn.Conv2d(in_dim//2, in_dim//2, kernel_size=3, padding=1)
        self.conv_e = nn.Conv2d(in_dim//2, in_dim//2, kernel_size=3, padding=1)
        self.conv_f = nn.Conv2d(in_dim//2, in_dim//2, kernel_size=3, padding=1)
        self.conv_fuse = double_conv(in_dim*4, in_dim)

        self.nonlinearity = nn.ReLU()

    def forward(self, x):
        a = self.nonlinearity(self.conv_a(x))
        c = self.nonlinearity(self.conv_c(a))
        b = self.nonlinearity(self.conv_b(a+c))
        f = self.nonlinearity(self.conv_f(a+c))
        e = self.nonlinearity(self.conv_e(f+c))
        d = self.nonlinearity(self.conv_d(a+b+e+f))
        y = torch.cat((x, a, b, c, d, e, f),1)
        y = self.nonlinearity(self.conv_fuse(y))
        return y


class res_unet_dense_scse_aux(nn.Module):
    def __init__(self, loss_func, backbone_name, pretrained, os, num_band=3, num_class=10):
        super(res_unet_dense_scse_aux, self).__init__()
        self.num_class = num_class
        self.loss_func = loss_func
        self.backbone, channels_blocks, do_upsample = Build_Backbone(backbone_name, pretrained, num_band, os)
        self.head = dense_connect_head_v2(channels_blocks[0])

        self.scse1 = SCse(channels_blocks[0])
        self.scse2 = SCse(channels_blocks[1])
        self.scse3 = SCse(channels_blocks[2])
        self.scse4 = SCse(channels_blocks[3])

        self.cat4=cat(channels_blocks[0],channels_blocks[0], channels_blocks[0], upsample = False)
        self.cat3=cat(channels_blocks[1],channels_blocks[1], channels_blocks[1], upsample = do_upsample[0])
        self.cat2=cat(channels_blocks[2],channels_blocks[2], channels_blocks[2], upsample = do_upsample[1])
        self.cat1=cat(channels_blocks[3],channels_blocks[3], channels_blocks[3], upsample = do_upsample[2])

        self.decoder_4=torch.nn.Sequential(
            double_conv(channels_blocks[0],channels_blocks[1]),
            nn.BatchNorm2d(channels_blocks[1], momentum=bn_mom),
            torch.nn.ReLU()
        )
        self.decoder_3=torch.nn.Sequential(
            double_conv(channels_blocks[1],channels_blocks[2]),
            nn.BatchNorm2d(channels_blocks[2], momentum=bn_mom),
            torch.nn.ReLU()
        )
        self.decoder_2=torch.nn.Sequential(
            double_conv(channels_blocks[2],channels_blocks[3]),
            nn.BatchNorm2d(channels_blocks[3], momentum=bn_mom),
            torch.nn.ReLU()
        )
        self.decoder_1=torch.nn.Sequential(
            double_conv(channels_blocks[3],channels_blocks[3]),
            nn.BatchNorm2d(channels_blocks[3], momentum=bn_mom),
            torch.nn.ReLU()
        )

        self.aux_brunch = nn.Sequential(
                nn.Conv2d(channels_blocks[0],64,kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64,32,kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                )

        self.upsample_x4=nn.Sequential(
                        nn.Conv2d( channels_blocks[3],24,kernel_size=3, stride=1, padding=1),
                        nn.BatchNorm2d(24, momentum=bn_mom),
                        nn.ReLU(inplace=True),
                        nn.UpsamplingBilinear2d(scale_factor=2),
                        nn.Conv2d(24,16,kernel_size=3, stride=1, padding=1),
                        nn.BatchNorm2d(16, momentum=bn_mom),
                        nn.ReLU(inplace=True),
                        nn.UpsamplingBilinear2d(scale_factor=2)
                        )
        self.conv_out = torch.nn.Conv2d(16,num_class,kernel_size=1,stride=1,padding=0)
        self.cls_conv_aux = nn.Conv2d(32, num_class, 1, 1, padding=0)

    def forward(self,x):
        layers = self.backbone(x[0])
        x = self.head(layers[3])
        y_aux = self.aux_brunch(x)
        x = self.cat4(x, self.scse1(layers[3]))
        x = self.decoder_4(x)
        x = self.cat3(x, self.scse2(layers[2]))
        x = self.decoder_3(x)
        x = self.cat2(x, self.scse3(layers[1]))
        x = self.decoder_2(x)
        x = self.cat1(x, self.scse4(layers[0]))
        x = self.decoder_1(x)

        x = self.upsample_x4(x)
        y = self.conv_out(x)
        y_aux = self.cls_conv_aux(y_aux)

        if self.num_class > 1:
            return [torch.softmax(y, dim = 1), torch.softmax(y_aux, dim = 1)]
        elif  'lovasz' not in self.loss_func:
            return [torch.sigmoid(y), torch.sigmoid(y_aux)]
        else:
            return [y, y_aux]


class ddense_connect_head(nn.Module):
    """ dense connect module"""
    def __init__(self, in_dim):
        super(ddense_connect_head, self).__init__()
        
        self.conv_a = nn.Conv2d(in_dim, in_dim//2, kernel_size=3, dilation=2,  padding=2)
        self.conv_b = nn.Conv2d(in_dim//2, in_dim//2, kernel_size=3, dilation=2,  padding=2)
        self.conv_c = nn.Conv2d(in_dim//2, in_dim//2, kernel_size=3, dilation=2,  padding=2)
        self.conv_d = nn.Conv2d(in_dim//2, in_dim//2, kernel_size=3, dilation=2,  padding=2)
        self.conv_e = nn.Conv2d(in_dim//2, in_dim//2, kernel_size=3, dilation=2,  padding=2)
        self.conv_f = nn.Conv2d(in_dim//2, in_dim//2, kernel_size=3, dilation=2,  padding=2)
        self.conv_fuse = double_conv(in_dim*4, in_dim)

        self.nonlinearity = nn.ReLU()

    def forward(self, x):
        a = self.nonlinearity(self.conv_a(x))
        c = self.nonlinearity(self.conv_c(a))
        b = self.nonlinearity(self.conv_b(a+c))
        f = self.nonlinearity(self.conv_f(a+c))
        e = self.nonlinearity(self.conv_e(f+c))
        d = self.nonlinearity(self.conv_d(a+b+e+f))
        y = torch.cat((x, a, b, c, d, e, f),1)
        y = self.nonlinearity(self.conv_fuse(y))
        return y


class res_unet_ddense_scse_aux(nn.Module):
    def __init__(self, loss_func, backbone_name, pretrained, os, num_band=3, num_class=10):
        super(res_unet_ddense_scse_aux, self).__init__()
        self.num_class = num_class
        self.loss_func = loss_func
        self.backbone, channels_blocks, do_upsample = Build_Backbone(backbone_name, pretrained, num_band, os)
        self.head = ddense_connect_head(channels_blocks[0])

        self.scse1 = SCse(channels_blocks[0])
        self.scse2 = SCse(channels_blocks[1])
        self.scse3 = SCse(channels_blocks[2])
        self.scse4 = SCse(channels_blocks[3])

        self.cat4=cat(channels_blocks[0],channels_blocks[0], channels_blocks[0], upsample = False)
        self.cat3=cat(channels_blocks[1],channels_blocks[1], channels_blocks[1], upsample = do_upsample[0])
        self.cat2=cat(channels_blocks[2],channels_blocks[2], channels_blocks[2], upsample = do_upsample[1])
        self.cat1=cat(channels_blocks[3],channels_blocks[3], channels_blocks[3], upsample = do_upsample[2])

        self.decoder_4=torch.nn.Sequential(
            double_conv(channels_blocks[0],channels_blocks[1]),
            nn.BatchNorm2d(channels_blocks[1], momentum=bn_mom),
            torch.nn.ReLU()
        )
        self.decoder_3=torch.nn.Sequential(
            double_conv(channels_blocks[1],channels_blocks[2]),
            nn.BatchNorm2d(channels_blocks[2], momentum=bn_mom),
            torch.nn.ReLU()
        )
        self.decoder_2=torch.nn.Sequential(
            double_conv(channels_blocks[2],channels_blocks[3]),
            nn.BatchNorm2d(channels_blocks[3], momentum=bn_mom),
            torch.nn.ReLU()
        )
        self.decoder_1=torch.nn.Sequential(
            double_conv(channels_blocks[3],channels_blocks[3]),
            nn.BatchNorm2d(channels_blocks[3], momentum=bn_mom),
            torch.nn.ReLU()
        )

        self.aux_brunch = nn.Sequential(
                nn.Conv2d(channels_blocks[0],64,kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64,32,kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                )

        self.upsample_x4=nn.Sequential(
                        nn.Conv2d( channels_blocks[3],24,kernel_size=3, stride=1, padding=1),
                        nn.BatchNorm2d(24, momentum=bn_mom),
                        nn.ReLU(inplace=True),
                        nn.UpsamplingBilinear2d(scale_factor=2),
                        nn.Conv2d(24,16,kernel_size=3, stride=1, padding=1),
                        nn.BatchNorm2d(16, momentum=bn_mom),
                        nn.ReLU(inplace=True),
                        nn.UpsamplingBilinear2d(scale_factor=2)
                        )
        self.conv_out = torch.nn.Conv2d(16,num_class,kernel_size=1,stride=1,padding=0)
        self.cls_conv_aux = nn.Conv2d(32, num_class, 1, 1, padding=0)

    def forward(self, x):
        layers = self.backbone(x)
        x = self.head(layers[3])
        y_aux = self.aux_brunch(x)
        x = self.cat4(x, self.scse1(layers[3]))
        x = self.decoder_4(x)
        x = self.cat3(x, self.scse2(layers[2]))
        x = self.decoder_3(x)
        x = self.cat2(x, self.scse3(layers[1]))
        x = self.decoder_2(x)
        x = self.cat1(x, self.scse4(layers[0]))
        x = self.decoder_1(x)

        x = self.upsample_x4(x)
        y = self.conv_out(x)
        y_aux = self.cls_conv_aux(y_aux)
        return y


class SERes_UNet(torch.nn.Module):
    def __init__(self, backbone_name, pretrained, output_stride=16, num_band=3, num_class=1, mode='seg', pretrained_path='', **kwargs):
        super(SERes_UNet,self).__init__() ##parent's init func
        if mode == 'change':
            num_band = num_band*2
        self.backbone, self.channels_blocks, do_upsample = Build_Backbone(backbone_name, pretrained, num_band, output_stride, pretrained_path=pretrained_path)
        self.num_base_layers = len(self.channels_blocks)

        self.center = bridge(self.channels_blocks[0], 1)

        self.concat_blocks = nn.ModuleList([cat(self.channels_blocks[0],self.channels_blocks[0], self.channels_blocks[0], False)])
        self.decode_blocks = nn.ModuleList([SeDoubleConv(self.channels_blocks[0], self.channels_blocks[1])])
        for i in range(1, self.num_base_layers):
            self.concat_blocks.append(cat(self.channels_blocks[i],self.channels_blocks[i], self.channels_blocks[i], do_upsample[i]))
            if i < self.num_base_layers-1:
                self.decode_blocks.append(SeDoubleConv(self.channels_blocks[i], self.channels_blocks[i+1]))
            else:
                self.decode_blocks.append(SeDoubleConv(self.channels_blocks[i], self.channels_blocks[i]))

        self.ASPP = ASPPModule(self.channels_blocks[-1], self.channels_blocks[-1])
        self.segmentation_head = nn.Sequential(
                        nn.Conv2d(self.channels_blocks[-1], 16,kernel_size=3, stride=1, padding=1),
                        nn.BatchNorm2d(16, momentum=bn_mom),
                        nn.ReLU(inplace=True),
                        torch.nn.Conv2d(16,num_class,kernel_size=1,stride=1,padding=0)
                        )
    
    def forward(self, x):
        x0_h, x0_w = x.size(2), x.size(3)
        layers = self.backbone(x)
        y = self.center(layers[-1])
        for i in range(self.num_base_layers) :
            y = self.concat_blocks[i](y, layers[self.num_base_layers-i-1])
            y = self.decode_blocks[i](y)
        y = self.ASPP(y)
        y = F.interpolate(y, size=(x0_h, x0_w), mode='bilinear', align_corners=True)
        y = self.segmentation_head(y)
        
        return y
