import torch
from torch import nn

class GaussianKLDiv(nn.Module):

    def __init__(self):
        super(GaussianKLDiv, self).__init__()

    def forward(self, mean, std):
        assert mean.size() == std.size()
        return 0.5 * ( -torch.log(std * std) + mean * mean + std * std - 1).mean()