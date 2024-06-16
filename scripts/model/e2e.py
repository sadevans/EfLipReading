import torch
import torch.nn as nn
from .blocks.frontend import Conv3DEfficientNetV2
from .blocks.transformer import TransformerEncoder
from .blocks.temporal import TCN, tcn_init
import numpy as np


class E2E(nn.Module):
    def __init__(self, config,  dropout=0.3, in_channels=1, num_classes=34, efficient_net_size="S") :
        super(E2E, self).__init__()

        self.dropout = dropout
        self.num_classes = num_classes
        self.frontend_3d = Conv3DEfficientNetV2(config, efficient_net_size=efficient_net_size)

        self.transformer_encoder = TransformerEncoder(dropout=self.dropout)
        self.tcn_block = TCN(dropout=self.dropout)

        self.temporal_avg = nn.AdaptiveAvgPool1d(1)

        self.fc_layer = nn.Linear(463, num_classes)
        self.logsoftmax = nn.LogSoftmax(dim=-1)


    def forward(self, x):
        """
        :arg
            x: Input. 
               Shape: (batch_size, channels, sequence_length, height, width)
        """
        x = self.frontend_3d(x)                                 # After frontend3d x shoud be size: Batch x Frames x Channels

        x = self.transformer_encoder(x)                         # After transformer x shoud be size: Frames x Channels

        x = self.tcn_block(x)                                   # After TCN x should be size: Frames x NewChannels
        x = x.transpose(1, 0)
        x = self.temporal_avg(x)
        x = x.transpose(1, 0)
        x = x.squeeze()
        x = self.fc_layer(x)
        return self.logsoftmax(x)

