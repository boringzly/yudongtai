'''initialize'''
import torch
import torch.nn as nn
import torch.nn.functional as F

class MaskNorm_(nn.Module):
    r""" Use mask to normalize the feature.
    """
    def __init__(self, normalized_shape, eps=1e-6, **kwargs):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x, mask):
        """
        x:    b, c, h, w
        mask: b, 1, h, w
        """
        b, c, h, w = x.shape
        mask = mask.expand(-1, self.normalized_shape[0], -1, -1)
        num_valid = mask.reshape(b, -1).sum(1)
        x_masked = x * mask
        mean = x_masked.reshape(b, -1).sum(1) / num_valid
        mean = mean.reshape(b, 1, 1, 1).expand(-1, -1, h, w)
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - mean) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x
    
class MaskNorm(nn.Module):
    r""" Use mask to normalize the feature.
    """
    def __init__(self, normalized_shape, eps=1e-6, **kwargs):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x, mask):
        """
        x:    b, c, h, w
        mask: b, 1, h, w
        """
        b, c, h, w = x.shape
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        u[mask == 1] = 0
        s[mask == 1] = 1
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x

if __name__ == "__main__":
    import numpy as np
    x = torch.tensor(np.random.rand(2,5,8,8)).cuda()
    print(x)
    mask = torch.zeros(2,1,8,8).cuda()
    mask[:,:,3:6,3:6] = 1
    mn = MaskNorm(5).cuda()
    x_norm = mn(x, mask)
    print(x_norm)