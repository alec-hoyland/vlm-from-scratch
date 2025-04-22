import torch
import torch.nn as nn
from torch.nn import functional as F


class Head(nn.Module):
    def __init__(self, n_embed, head_size, dropout=0.1, is_decoder=False):
        super().__init__()
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.is_decoder = is_decoder

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)

        w = q @ k.transpose(-2, 1) * (C**-0.5)

        if self.is_decoder:
            # lower triangular mask
            tril = torch.tril(torch.ones(T, T, dtype=torch.bool, device=x.device))
            w = w.masked_fill(tril == 0, float("-inf"))

        w = F.softmax(w, dim=-1)
        w = self.dropout(w)

        return w @ v


class MultiHeadAttention(nn.Module):
    def __init__(self, num_embed, num_heads, dropout=0.1, is_decoder=False):
        super().__init__()
        assert num_embed % num_heads == 0, (
            "Embedding dimension must be divisible by num_heads"
        )

        self.heads = nn.ModuleList(
            [
                Head(num_embed, num_embed // num_heads, dropout, is_decoder)
                for _ in range(num_heads)
            ]
        )
        self.proj = nn.Linear(num_embed, num_embed)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out
