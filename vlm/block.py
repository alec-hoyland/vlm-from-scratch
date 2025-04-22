import torch.nn as nn
from .attention import MultiHeadAttention


class Block(nn.Module):
    def __init__(self, n_embed, num_heads, dropout=0.1, is_decoder=False):
        super().__init__()

        self.layer_norm1 = nn.LayerNorm(n_embed)
        self.attention = MultiHeadAttention(n_embed, num_heads, dropout, is_decoder)
        self.layer_norm2 = nn.LayerNorm(n_embed)
        self.feedforward = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed), nn.GELU(), nn.Linear(4 * n_embed, n_embed)
        )

    def forward(self, x):
        x0 = x

        x = self.layer_norm1(x)
        attention_output = self.attention(x)
        x = x0 + attention_output
        x = self.layer_norm2(x)
        feedforward_output = self.feedforward(x)

        return x + feedforward_output
