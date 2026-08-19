'''initialize'''
import torch
from torch.nn import BatchNorm1d, BatchNorm2d, BatchNorm3d


class BatchNorm2d_CP(torch.nn.Module):
    def __init__(self, dim, data_format='channels_first'):
        super(BatchNorm2d_CP, self).__init__()
        self.norm = torch.nn.BatchNorm2d(dim)
        self.data_format = data_format
    
    def forward(self, x):
        if self.data_format == 'channels_last':
            x = x.permute(0, 3, 1, 2)
        x = self.norm(x)
        if self.data_format == 'channels_last':
            x = x.permute(0, 2, 3, 1)
        return x

class BatchNorm1d_CP(torch.nn.Module):
    def __init__(self, dim, data_format='channels_first'):
        super(BatchNorm1d_CP, self).__init__()
        self.norm = torch.nn.BatchNorm1d(dim)
        self.data_format = data_format
    
    def forward(self, x):
        if self.data_format == 'channels_last':
            if len(x.shape) == 3:
                x = x.permute(0, 2, 1)
        x = self.norm(x)
        if self.data_format == 'channels_last':
            if len(x.shape) == 3:
                x = x.permute(0, 2, 1)
        return x