import torch.nn as nn
import torch
import torch.nn.functional as F
from networks.utils.block import DoubleConv

class SEModule(nn.Module):
    def __init__(self, channels, reduction_channels):
        super(SEModule, self).__init__()
        # self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(
            channels, reduction_channels, kernel_size=1, padding=0, bias=True
        )
        self.ReLU = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(
            reduction_channels, channels, kernel_size=1, padding=0, bias=True
        )

    def forward(self, x):
        # x_se = self.avg_pool(x)
        x_se = (
            x.view(x.size(0), x.size(1), -1).mean(-1).view(x.size(0), x.size(1), 1, 1)
        )
        x_se = self.fc1(x_se)
        x_se = self.ReLU(x_se)
        x_se = self.fc2(x_se)
        return x * x_se.sigmoid()

class SeDoubleConv(DoubleConv):
    def __init__(self, in_ch, out_ch, residual=True, use_se=True):
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
            x = x + x0
        return x