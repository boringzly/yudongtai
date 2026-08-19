import torch
import torch.nn as nn
from networks.backbone.efficient_net import EfficientNet
import os

class efficientnet(nn.Module):
    def __init__(
        self,
        model_name="efficientnet-b0",
        band_num=3,
        layerlist=[],
        pretrained=False,
        drop_connect_rate=0,
        pretrained_path='',
        **kwargs
    ):
        super(efficientnet, self).__init__()
        self.band_num = band_num
        self.model_name = model_name
        self.pretrained = pretrained
        if 'memory_efficient' in kwargs:
            memory_efficient = kwargs['memory_efficient']
        else:
            memory_efficient = True
        self.efficientbase = EfficientNet.from_name(
            model_name,
            band_num,
            layerlist,
            override_params={"drop_connect_rate": drop_connect_rate,
                            "memory_efficient": memory_efficient},
        )
        if pretrained:
            self.load_pretrained(pretrained_path)

    def forward(self, inputs):
        layers = self.efficientbase(inputs)
        # no head
        return layers

    def load_pretrained(self, pretrained_path):
        if self.model_name == "efficientnet-b0":
            old_dict = torch.load(os.path.join(pretrained_path, "efficientnet-b0-355c32eb.pth"), map_location="cpu")
        elif self.model_name == "efficientnet-b1":
            old_dict = torch.load(os.path.join(pretrained_path, "efficientnet-b1-f1951068.pth"), map_location="cpu")
        elif self.model_name == "efficientnet-b2":
            old_dict = torch.load(os.path.join(pretrained_path, "efficientnet-b2-8bb594d6.pth"), map_location="cpu")
        elif self.model_name == "efficientnet-b3":
            old_dict = torch.load(os.path.join(pretrained_path, "efficientnet-b3-5fb5a3c3.pth"), map_location="cpu")
        elif self.model_name == "efficientnet-b4":
            old_dict = torch.load(os.path.join(pretrained_path, "efficientnet-b4-6ed6700e.pth"), map_location="cpu")
        elif self.model_name == "efficientnet-b5":
            old_dict = torch.load(os.path.join(pretrained_path, "efficientnet-b5-b6417697.pth"), map_location="cpu")
        elif self.model_name == "efficientnet-b6":
            old_dict = torch.load(os.path.join(pretrained_path, "efficientnet-b6-c76e70fd.pth"), map_location="cpu")
        elif self.model_name == "efficientnet-b7":
            old_dict = torch.load(os.path.join(pretrained_path, "efficientnet-b7-dcc49843.pth"), map_location="cpu")

        conv1_weight = old_dict["_conv_stem.weight"]
        for i in range(3, self.band_num):
            conv1_weight = torch.cat(
                (conv1_weight, conv1_weight[:, (i % 3) : (i % 3 + 1), :, :]), 1
            )
        model_dict = self.efficientbase.state_dict()
        old_dict = {k: v for k, v in old_dict.items() if (k in model_dict)}
        old_dict["_conv_stem.weight"] = conv1_weight
        model_dict.update(old_dict)
        self.efficientbase.load_state_dict(model_dict)


"""
params:
    r: repeat bolck times
    s: stride
    X: downsample times
    C: out channels

details:
    b0: len = 16, {r1 s1}, {r2 s2}, {r2 s2}, {r3 s2}, {r3 s1}, {r4 s2}, {r1 s1}
            C32  X2 C16   X4 C24   X8 40   X16 80   X16 112  X32 192  X32 320
            use: idx = [2, 4, 10, 15], chanels = [24, 40, 112, 320]

    b1: len = 23, {r2 s1}, {r3 s2}, {r3 s2}, {r4 s2}, {r4 s1}, {r5 s2}, {r2 s1}
            C32  X2 C16   X4 C24   X8 40   X16 80   X16 112  X32 192  X32 320
            use: idx = [4, 7, 15, 22], chanels = [24, 40, 112, 320]

    b2: len = 23, {r2 s1}, {r3 s2}, {r3 s2}, {r4 s2}, {r4 s1}, {r5 s2}, {r2 s1}
            C32  X2 C16   X4 C24   X8 48   X16 88   X16 120  X32 208  X32 352
            use: idx = [4, 7, 15, 22], chanels = [24, 48, 120, 352]
                    
    b3: len = 25, {r2 s1}, {r3 s2}, {r3 s2}, {r5 s2}, {r5 s1}, {r6 s2}, {r2 s1}
            C40  X2 C24   X4 C32   X8 48   X16 96   X16 136  X32 232  X32 384
            use: idx = [4, 7, 17, 25], chanels = [32, 48, 136, 384]

    b4: len = 31, {r2 s1}, {r4 s2}, {r4 s2}, {r5 s2}, {r6 s1}, {r8 s2}, {r2 s1}
            C40  X2 C24   X4 C32   X8 56   X16 112   X16 160  X32 272  X32 448
            use: idx = [5, 9, 20, 30], chanels = [32, 56, 160, 448]

    b5: len = 39, {r3 s1}, {r5 s2}, {r5 s2}, {r7 s2}, {r7 s1}, {r9 s2}, {3 s1}
            C40  X2 C24   X4 C40   X8 64   X16 128   X16 176  X32 304  X32 512
            use: idx = [7, 12, 26, 38], chanels = [40, 64, 176, 512]

    b6: len = 44, {r3 s1}, {r6 s2}, {r6 s2}, {r8 s2}, {r8 s1}, {r11 s2}, {3 s1}
            C40  X2 32   X4 C40   X8 72   X16 144   X16 200  X32 344  X32 576
            use: idx = [8, 14, 30, 43], chanels = [40, 72, 200, 576]
            
    b7: len = 54, {r4 s1}, {r7 s2}, {r7 s2}, {r10 s2}, {r10 s1}, {r13 s2}, {4 s1}
            C40  X2 32   X4 C48   X8 80   X16 160   X16 224  X32 384  X32 640
            use: idx = [10, 17, 37, 53], chanels = [48, 80, 224, 640]
"""
