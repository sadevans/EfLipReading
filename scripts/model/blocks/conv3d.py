import torch
import torch.nn as nn
import yaml
import numpy as np


class Swish(nn.Module):
    """Construct an Swish object."""

    def forward(self, x):
        """Return Swich activation function."""
        return x * torch.sigmoid(x)  
    

class Conv3D(nn.Module):
    """Convolution 3D block"""
    def __init__(self, in_channels=1, out_channels=24, kernel=(3, 5, 5), loss_type='relu', if_maxpool=False):
        super(Conv3D, self).__init__()
        self.if_maxpool = if_maxpool
        if loss_type == "relu":
            self.act = nn.ReLU()
        elif loss_type == 'swish':
            self.act = Swish()

        self.conv3d = nn.Conv3d(in_channels, out_channels, kernel, (1, 2, 2), (1, 2, 2))
        self.bn = nn.BatchNorm3d(out_channels, momentum=0.99)

        if self.if_maxpool:
            self.maxpool = nn.MaxPool3d((1, 3, 3), (1, 2, 2), (0, 1, 1))


    def forward(self, x):
        x = self.conv3d(x)
        x = self.bn(x)
        x = self.act(x)

        if self.if_maxpool:
            x = self.maxpool(x)
        return x


def init_3dconv(model):
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm3d)):
            m.momentum = 0.99


def get_conv_3d(config, model_size="S"):
    info_el = config['frontend-3d'][0]
    out_channels = config['efficient-net-blocks'][model_size][0][3]
    model = Conv3D(in_channels=info_el[0], out_channels=out_channels, kernel=tuple(info_el[2]), loss_type=info_el[3], if_maxpool=info_el[4])
    init_3dconv(model)
    return model