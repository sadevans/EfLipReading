import torch.nn as nn
from .conv3d import get_conv_3d
from .efficientnet import get_efficientnet_v2


def threeD_to_2D_tensor(x):
    n_batch, n_channels, s_time, sx, sy = x.shape
    x = x.transpose(1, 2)
    return x.reshape(n_batch * s_time, n_channels, sx, sy)


class Conv3DEfficientNetV2(nn.Module):
    def __init__(self, config, efficient_net_size="B"):
        super(Conv3DEfficientNetV2, self).__init__()
        self.conv3d = get_conv_3d(config, model_size=efficient_net_size)
        self.efnet = get_efficientnet_v2(config, model_size=efficient_net_size)

    def forward(self, x):
        if (x.shape[1] != 1 or x.shape[1] != 3) and (x.shape[2] == 1 or x.shape[2] == 3):
            x = x.permute(0, 2, 1, 3, 4)
        B, C, T, H, W = x.shape
        x = self.conv3d(x)                                         # After efnet x shoud be size: Frames x Channels x H x W
        Tnew = x.shape[2]
        x = threeD_to_2D_tensor(x)
        x = self.efnet(x)                                          # After efnet x shoud be size: Frames x 384
        x = x.view(B, Tnew, x.size(1))

        return x
