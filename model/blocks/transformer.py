import torch.nn as nn
import torch


class TransformerEncoder(nn.Module):
    def __init__(self, num_channels=384, num_heads=8, dim_feedforward=128, dropout=0.1):
        super(TransformerEncoder, self).__init__()
        self.self_attn = nn.MultiheadAttention(num_channels, num_heads, dropout=dropout)

        self.linear1 = nn.Linear(num_channels, dim_feedforward)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(dim_feedforward, num_channels)

        self.norm1 = nn.LayerNorm(num_channels)
        self.norm2 = nn.LayerNorm(num_channels)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
        
    def forward(self, x):
        """
        :arg
            x: Input from the 3d frontend with efficient net. 
               Shape: (batch_size, sequence_length, channels)
        """
        
        attn_output, _ = self.self_attn(x, x, x, attn_mask=None, key_padding_mask=None)
        x = x + self.dropout1(attn_output)                  # Add & Norm after residual connection here
        x = self.norm1(x)

        x2 = self.linear2(self.relu(self.linear1(x)))
        x = x + self.dropout2(x2)                           # Add & Norm after residual connection here
        x = self.norm2(x)
        # max_len = x.size(1)
        # lengths_after_extractor = (x != 0).sum(dim=2).squeeze(1)
        # max_len = x.size(1)
        # key_padding_mask = torch.arange(max_len)[None, :] >= lengths_after_extractor[:, None]
        # key_padding_mask = key_padding_mask.to(x.device)

        # attn_output, _ = self.self_attn(x, x, x, key_padding_mask=key_padding_mask)
        # x = x + self.dropout1(attn_output)
        # x = self.norm1(x)

        # x2 = self.linear2(self.relu(self.linear1(x)))
        # x = x + self.dropout2(x2)
        # x = self.norm2(x)

        return x