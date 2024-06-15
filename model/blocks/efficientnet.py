import torch.nn as nn
from collections import OrderedDict
from ..efficientnet_layers.mbconv import MBConv, MBConvConfig
from .conv3d import get_conv_3d
import copy
import yaml

    
class EfficientNetV2(nn.Module):
    """Pytorch Implementation of EfficientNetV2

    paper: https://arxiv.org/abs/2104.00298

    - reference 1 (pytorch): https://github.com/d-li14/efficientnetv2.pytorch/blob/main/effnetv2.py
    - reference 1 (official): https://github.com/google/automl/blob/master/efficientnetv2/effnetv2_configs.py

    :arg
        - layer_infos: list of MBConvConfig
        - out_channels: bottleneck channel
        - dropout: dropout probability before classifier layer
        - stochastic depth: stochastic depth probability
        - block: basic block
        - act_layer: basic activation
        - norm_layer: basic normalization
    """
    def __init__(self, layer_infos, out_channels=384, dropout=0.3, stochastic_depth=0.8,
                 block=MBConv, act_layer=nn.SiLU, norm_layer=nn.BatchNorm2d):
        super(EfficientNetV2, self).__init__()
        self.layer_infos = layer_infos
        self.norm_layer = norm_layer
        self.act = act_layer

        self.in_channel = layer_infos[0].in_ch
        self.final_stage_channel = layer_infos[-1].out_ch

        self.cur_block = 0
        self.num_block = sum(stage.num_layers for stage in layer_infos)
        self.stochastic_depth = stochastic_depth

        self.blocks = nn.Sequential(*self.make_stages(layer_infos, block))
        self.stage_7 = nn.Sequential(OrderedDict([
            ('pointwise', nn.Conv2d(self.final_stage_channel, 768, kernel_size=1, stride=1, padding=0, groups=1, bias=False)),
            ('avgpool', nn.AdaptiveAvgPool2d((1, 1))),
            ('GLU', nn.GLU(dim=1))
        ]))


    def make_stages(self, layer_infos, block):
        return [layer for layer_info in layer_infos for layer in self.make_layers(copy.copy(layer_info), block)]


    def make_layers(self, layer_info, block):
        layers = []
        for i in range(layer_info.num_layers):
            layers.append(block(layer_info, sd_prob=self.get_sd_prob()))
            layer_info.in_ch = layer_info.out_ch
            layer_info.stride = 1
        return layers


    def get_sd_prob(self):
        sd_prob = self.stochastic_depth * (self.cur_block / self.num_block)
        self.cur_block += 1
        return sd_prob


    def forward(self, x, debug=False):
        """forward.

        :param x: torch.Tensor, input tensor with input size (B, C, T, H, W).
        """
        if debug: print("INPUT SHAPE IN EFFICIENT NET V2: ", x.shape)
        for i, block in enumerate(self.blocks):
            if debug: print(f"NOW IN BLOCK {i}: ", block)
            x = block(x)
            if debug: print(f"SHAPE AFTER BLOCK {i}: ", x.shape)

        x = self.stage_7(x)

        return x

def efficientnet_v2_init(model):
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm2d)):
            m.momentum = 0.99


def get_efficientnet_v2(config, model_size="B", pretrained=False, dropout=0.3, stochastic_depth=0.8, **kwargs):
    residual_config = [MBConvConfig(*layer_config) for layer_config in get_efficientnet_v2_structure(config, model_size)]
    model = EfficientNetV2(residual_config, dropout=dropout, stochastic_depth=stochastic_depth, block=MBConv, act_layer=nn.SiLU)
    efficientnet_v2_init(model)
    return model


def get_efficientnet_v2_structure(config, model_size='B'):
    # with open(config, 'r') as file:
    #     info = yaml.safe_load(file)
    efficientnet_config = config['efficient-net-blocks'][model_size]
    return config['efficient-net-blocks'][model_size]
