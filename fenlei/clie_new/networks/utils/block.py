import torch.nn as nn
import torch

bn_mom = 0.0003

class base_conv(torch.nn.Module):
    def __init__(
        self, in_chn, out_chn, kernel_size=3, stride=1, dilation=1, padding=1
    ):  # params:in_chn(input channel of double conv),out_chn(output channel of double conv)
        super(base_conv, self).__init__()  ##parent's init func

        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_chn,
                out_chn,
                kernel_size=3,
                stride=1,
                dilation=dilation,
                padding=dilation,
            ),
            nn.BatchNorm2d(out_chn, momentum=bn_mom),
            torch.nn.ReLU()
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class double_conv(torch.nn.Module):
    def __init__(
        self, in_chn, out_chn, stride=1, dilation=1
    ):  # params:in_chn(input channel of double conv),out_chn(output channel of double conv)
        super(double_conv, self).__init__()  ##parent's init func

        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_chn,
                out_chn,
                kernel_size=3,
                stride=1,
                dilation=dilation,
                padding=dilation,
            ),
            nn.BatchNorm2d(out_chn, momentum=bn_mom),
            torch.nn.ReLU(),
            torch.nn.Conv2d(out_chn, out_chn, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_chn, momentum=bn_mom),
            torch.nn.ReLU(),
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class cat(torch.nn.Module):
    def __init__(self, in_chn_high, in_chn_low, out_chn, upsample=False):
        super(cat, self).__init__()  ##parent's init func
        self.do_upsample = upsample
        self.upsample = torch.nn.Upsample(
            scale_factor=2, mode="nearest"
        )
        self.conv2d = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_chn_high + in_chn_low, out_chn, kernel_size=1, stride=1, padding=0
            ),
            nn.BatchNorm2d(out_chn, momentum=bn_mom),
            torch.nn.ReLU(inplace=True),
        )

    def forward(self, x, y):
        if self.do_upsample:
            x = self.upsample(x)
        x = torch.cat(
            (x, y), 1
        )  # x,y shape(batch_sizxe,channel,w,h), concat at the dim of channel
        return self.conv2d(x)


class cat_conv(cat):
    def __init__(self, in_chn_high, in_chn_low, out_chn, upsample=False):
        super(cat_conv, self).__init__(in_chn_high, in_chn_low, out_chn, upsample)  ##parent's init func
        # self.conv2d_1 = torch.nn.Sequential(
        #     torch.nn.Conv2d(
        #         in_chn_high + in_chn_low, out_chn, kernel_size=1, stride=1, padding=0
        #     ),
        #     nn.BatchNorm2d(out_chn, momentum=bn_mom),
        #     torch.nn.ReLU(inplace=True),
        # )
        self.conv2d_2 = torch.nn.Sequential(
            torch.nn.Conv2d(out_chn, out_chn, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_chn, momentum=bn_mom),
            torch.nn.ReLU(inplace=True),
        )

    def forward(self, x, y):
        if self.do_upsample:
            x = self.upsample(x)
        out = torch.cat(
            (x, y), 1
        )  # x,y shape(batch_sizxe,channel,w,h), concat at the dim of channel
        out = self.conv2d(out)
        return self.conv2d_2(out) + out


class bridge(torch.nn.Module):  # X16 -> X32
    def __init__(self, in_chn, output_stride=1):
        super(bridge, self).__init__()  ##parent's init func
        if output_stride not in [1, 2]:
            raise ValueError("UNet.py: invalid output_stride")
        self.downsample = (
            torch.nn.MaxPool2d(stride=2, kernel_size=2) if output_stride == 2 else None
        )
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_chn, in_chn * 2, kernel_size=3, stride=output_stride, padding=1
            ),
            nn.BatchNorm2d(in_chn * 2, momentum=bn_mom),
            torch.nn.ReLU(),
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(in_chn * 2, in_chn, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(in_chn, momentum=bn_mom),
            torch.nn.ReLU(),
        )

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample:
            x = self.downsample(x)
        return out + x
        
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
