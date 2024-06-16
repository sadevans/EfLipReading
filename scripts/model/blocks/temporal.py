import torch
import torch.nn as nn


class TemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, dilation, dropout=0.2):
        super(TemporalBlock, self).__init__()

        self.conv_block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation, padding=padding),
            nn.BatchNorm1d(out_channels, momentum=0.99),
            nn.PReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(out_channels, out_channels, kernel_size, dilation=dilation, padding=padding),
            nn.BatchNorm1d(out_channels, momentum=0.99),
            nn.PReLU(),
            nn.Dropout(dropout),
            nn.PReLU()
        )
        self.residual = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        out = self.conv_block(x)
        return self.residual(x) + out  # Residual connection
    

class TCN(nn.Module):
    def __init__(self, input_dim=384, hidden_dim=463, num_layers=4, dropout=0.2):
        super(TCN, self).__init__()

        self.tcn_layers = nn.ModuleList([
            TemporalBlock(input_dim if l == 0 else hidden_dim, hidden_dim, kernel_size=3, padding=2**l, dilation=2**l, dropout=dropout)
            for l in range(num_layers)
        ])
        self.avg_pool = nn.AdaptiveAvgPool1d(1)


    def forward(self, x):
        """
        :arg
            x: Input from the Transformer encoder. 
               Shape: (batch_size, sequence_length, embedding_dim)
        """
        x = x.transpose(1, 2)   # Transpose dimensions for convolution layers
        for layer in self.tcn_layers:
            x = layer(x)
        return x
    

def tcn_init(model):
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            m.momentum = 0.99