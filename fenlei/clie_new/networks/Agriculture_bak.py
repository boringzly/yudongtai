import torch
from torch import nn
from torch.nn import functional as F
from einops.layers.torch import Rearrange
from networks import tab_network

class DeepMlp(nn.Module):
    def __init__(self, in_dim,out_dim,mid_dim,block,drop_rate):
        super().__init__()
        _block_list = []
        for i in range(block):
            _block_list.append(nn.Sequential(
                nn.LayerNorm(in_dim),
                FeedForward(in_dim, mid_dim, drop_rate),
            ))
        self.block_list = nn.ModuleList(_block_list)
        self.block = block
    def forward(self, x):
        for i in range(self.block):
            residual = x
            if i != 0:
                x = x + self.block_list[i](x) + residual
            else:
                x = x + self.block_list[i](x)

        return x 
        
class FCCore(nn.Module):

    def __init__(self, in_dim=32, out_dim=16, mid_dim=64, repeat=3, drop_rate=0.1):
        super(FCCore, self).__init__()
        self.fc1 = nn.Linear(in_dim, mid_dim)
        self.bn1 = nn.BatchNorm1d(mid_dim)
        self.relu1 = nn.ReLU(inplace=True)
        self.dropout1 = nn.Dropout(drop_rate)
        _fc_list = []
        _bn_list = []
        _relu_list = []
        _dropout_list = []
        for i in range(repeat):
            _fc_list.append(nn.Linear(mid_dim, mid_dim))
            _bn_list.append(nn.BatchNorm1d(mid_dim))
            _relu_list.append(nn.ReLU(inplace=True))
            _dropout_list.append(nn.Dropout(drop_rate))
        self.fc_list = nn.ModuleList(_fc_list)
        self.bn_list = nn.ModuleList(_bn_list)
        self.relu_list = nn.ModuleList(_relu_list)
        self.dropout_list = nn.ModuleList(_dropout_list)
        self.fc2 = nn.Linear(mid_dim, out_dim)
        self.bn2 = nn.BatchNorm1d(out_dim)
        self.relu2 = nn.ReLU(inplace=True)
        self.dropout2 = nn.Dropout(drop_rate)
        self.repeat = repeat
        self.drop_rate = drop_rate

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        if self.drop_rate > 0:
            x = self.dropout1(x)
        for i in range(self.repeat):
            x = self.fc_list[i](x)
            x = self.bn_list[i](x)
            x = self.relu_list[i](x)
            if self.drop_rate > 0:
                x = self.dropout_list[i](x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        if self.drop_rate > 0:
            x = self.dropout2(x)

        return x

class FeedForward(nn.Module): #基本mlp
    def __init__(self, dim, hidden_dim, dropout = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class MixerBlock(nn.Module):

    def __init__(self, dim, num_patch, token_dim, channel_dim, dropout = 0.):
        super().__init__()

        self.token_mix = nn.Sequential(
            nn.LayerNorm(dim),
            Rearrange('b n d -> b d n'),
            FeedForward(num_patch, token_dim, dropout),
            Rearrange('b d n -> b n d')
        )

        self.channel_mix = nn.Sequential(
            nn.LayerNorm(dim),
            FeedForward(dim, channel_dim, dropout),
        )

    def forward(self, x):

        x = x + self.token_mix(x)

        x = x + self.channel_mix(x)

        return x

class sk_cnn_src(nn.Module):

    def __init__(self, in_dim=13, num_classes=4, drop_rate=0.1):
        super(sk_cnn_src, self).__init__()
        out_channels = 36
        self.out_channels=out_channels
        self.conv_3 = nn.Sequential(nn.Conv2d(in_dim,out_channels,3,padding=1,bias=False),
                                           nn.BatchNorm2d(out_channels),
                                           nn.ReLU(inplace=True))
        self.conv_5 = nn.Sequential(nn.Conv2d(in_dim,out_channels,5,padding=1,bias=False),
                                           nn.BatchNorm2d(out_channels),
                                           nn.ReLU(inplace=True))
        self.conv_7 = nn.Sequential(nn.Conv2d(in_dim,out_channels,7,padding=1,bias=False),
                                           nn.BatchNorm2d(out_channels),
                                           nn.ReLU(inplace=True))
        self.global_pool=nn.AdaptiveAvgPool2d(1)
        self.pool = nn.AvgPool2d(5)
        self.fc1=nn.Sequential(nn.Conv2d(out_channels,20,1,bias=False),
                               nn.BatchNorm2d(20),
                               nn.ReLU(inplace=True))   
        self.fc2=nn.Conv2d(20,out_channels*3,1,bias=False)  
        self.softmax=nn.Softmax(dim=1) 

        self.to_patch_embedding = nn.Sequential(
                                        nn.Conv2d(out_channels, 64, 3, padding=1),
                                        nn.BatchNorm2d(64),
                                        nn.ReLU(inplace=True),
                                        nn.Dropout(drop_rate),
                                        Rearrange('b c h w -> b (h w) c'))

        self.mixer_blocks = nn.ModuleList([])
        for _ in range(3):
            self.mixer_blocks.append(MixerBlock(64, 9, 128, 512, dropout=0.2))
        
        self.layer_norm = nn.LayerNorm(64)
        self.core = FCCore(64, 64, 128, 3, drop_rate)
        self.fc3 = nn.Sequential(nn.Linear(64, 16), 
                                nn.ReLU(inplace=True),
                                nn.Dropout(drop_rate))
        self.fc4 = nn.Linear(16, num_classes)

        self.core_mlp = FCCore(in_dim,64,32,4,drop_rate)

    
    def forward(self, x_7):
        
        x_7 = x_7.view(x_7.shape[0], x_7.shape[1], x_7.shape[2], x_7.shape[3])

        x_5 = x_7[:,:,1:6,1:6]
        x_3 = x_7[:,:,2:5,2:5]
        src = x_7[:,:,3,3]

        src = src.view(src.shape[0], src.shape[1])

        src = self.core_mlp(src)

        src = src.view(src.shape[0], src.shape[1])

        batch_size = x_7.shape[0]

        U_7 = self.conv_7(x_7)
        U_5 = self.conv_5(x_5)
        U_3 = self.conv_3(x_3)

        U = U_5+U_3+U_7

        s = self.global_pool(U)
        z = self.fc1(s)
        a_b = self.fc2(z)
        a_b=a_b.reshape(batch_size,3,self.out_channels,-1) 
        a_b=self.softmax(a_b) 
        a_b=list(a_b.chunk(3,dim=1))
        a_b=list(map(lambda x:x.reshape(batch_size,self.out_channels,1,1),a_b)) 
        V1 = U_5 * a_b[1]
        V2 = U_3 * a_b[2]
        V3 = U_7 * a_b[0]

        V_out = V1 + V2 + V3

        x = self.to_patch_embedding(V_out)
        for mixer_block in self.mixer_blocks:
            x = mixer_block(x)
        
        x = self.layer_norm(x)

        x = x.mean(dim=1)

        x = x * src

        x = x.view(x.shape[0],x.shape[1])

        x = self.core(x)
        x = self.fc3(x)
        x = self.fc4(x)

        x = x.view(x.shape[0], x.shape[1], 1, 1)

        return x


class skcnn_TabNet(nn.Module):

    def __init__(self, in_dim=13, num_classes=4, drop_rate=0.2, use_residual=True):
        super(skcnn_TabNet, self).__init__()
        out_channels = 64
        self.out_channels=out_channels
        self.conv_3 = nn.Sequential(nn.Conv2d(in_dim,out_channels,3,padding=1,bias=False),
                                           nn.BatchNorm2d(out_channels),
                                           nn.ReLU(inplace=True))
        self.conv_5 = nn.Sequential(nn.Conv2d(in_dim,out_channels,5,padding=1,bias=False),
                                           nn.BatchNorm2d(out_channels),
                                           nn.ReLU(inplace=True))
        self.conv_7 = nn.Sequential(nn.Conv2d(in_dim,out_channels,7,padding=1,bias=False),
                                           nn.BatchNorm2d(out_channels),
                                           nn.ReLU(inplace=True))
        self.global_pool=nn.AdaptiveAvgPool2d(1)
        self.fc1=nn.Sequential(nn.Conv2d(out_channels,20,1,bias=False),
                               nn.BatchNorm2d(20),
                               nn.ReLU(inplace=True))   
        self.fc2=nn.Conv2d(20,out_channels*3,1,bias=False)  
        self.softmax=nn.Softmax(dim=1) 

        self.to_patch_embedding = nn.Sequential(
                                        nn.Conv2d(64, 128, 3, padding=1),
                                        nn.BatchNorm2d(128),
                                        nn.ReLU(inplace=True),
                                        nn.Dropout(drop_rate),
                                        Rearrange('b c h w -> b (h w) c'))

        self.mixer_blocks = nn.ModuleList([])
        for _ in range(3):
            self.mixer_blocks.append(MixerBlock(128, 9, 128, 512, dropout=0.2))
        
        self.layer_norm = nn.LayerNorm(128)

        self.core = FCCore(128, 64, 64, 1, drop_rate)
        self.fc3 = nn.Sequential(nn.Linear(64, 16), 
                                nn.ReLU(inplace=True),
                                nn.Dropout(drop_rate))
        self.fc4 = nn.Linear(16, num_classes)

        self.core1 = FCCore(128, 64, 64, 1, drop_rate)
        self.fc31 = nn.Sequential(nn.Linear(64, 16), 
                                nn.ReLU(inplace=True),
                                nn.Dropout(drop_rate))
        self.fc41 = nn.Linear(16, num_classes)

        self.tab_net = tab_network.TabNet(in_dim, 64)
        self.head_mlp = FCCore(in_dim,128,64,2,drop_rate)
        self.core_mlp = DeepMlp(128,128,128,6,drop_rate)

        self.use_residual = use_residual
        
        # self.class_tabnet = tab_network.TabNet(64,4,n_d=4,n_a=4)

    
    def forward(self, x_7):
        
        x_7 = x_7.view(x_7.shape[0], x_7.shape[1], x_7.shape[2], x_7.shape[3])
        x_5 = x_7[:,:,1:6,1:6]
        x_3 = x_7[:,:,2:5,2:5]
        src = x_7[:,:,3,3]

        src = src.view(src.shape[0], src.shape[1])
        src1 = src

        src = self.tab_net(src)[0]

        src = src.view(src.shape[0], src.shape[1],1,1)
        src1 = self.head_mlp(src1)
        src1 = self.core_mlp(src1)
        src1 = self.core1(src1)

        src1 = src1.view(src1.shape[0], src1.shape[1],1,1)

        batch_size = x_7.shape[0]

        U_7 = self.conv_7(x_7)
        U_5 = self.conv_5(x_5)
        U_3 = self.conv_3(x_3)

        U = U_5+U_3+U_7

        s = self.global_pool(U)
        z = self.fc1(s)
        a_b = self.fc2(z)
        a_b=a_b.reshape(batch_size,3,self.out_channels,-1) 
        a_b=self.softmax(a_b) 
        a_b=list(a_b.chunk(3,dim=1))
        a_b=list(map(lambda x:x.reshape(batch_size,self.out_channels,1,1),a_b)) 
        V1 = U_5 * a_b[1]
        V2 = U_3 * a_b[2]
        V3 = U_7 * a_b[0]

        V_out = V1 + V2 + V3
        V_out = V_out * src * src1

        x = self.to_patch_embedding(V_out)
        for mixer_block in self.mixer_blocks:
            x = mixer_block(x)
        
        x = self.layer_norm(x)

        x = x.mean(dim=1)
        
        src1 = src1.view(src.shape[0], src.shape[1])
        y = self.fc31(src1)
        y = self.fc41(y)

        x = x.view(x.shape[0],x.shape[1])

        x = self.core(x)
        x = self.fc3(x)
        x = self.fc4(x)

        # x = self.class_tabnet(x)[0]

        x = x.view(x.shape[0], x.shape[1], 1, 1)
        y = y.view(y.shape[0], y.shape[1], 1, 1)

        return x, y

