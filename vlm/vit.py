import torch
import torch.nn as nn
from .patch import PatchEmbeddings
from .block import Block


class ViT(nn.Module):
    def __init__(
        self,
        image_size,
        patch_size,
        num_hidden,
        num_heads,
        num_blocks,
        emb_dropout,
        block_dropout,
    ):
        super().__init__()

        self.patch_embedding = PatchEmbeddings(image_size, patch_size, num_hidden)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, num_hidden))
        num_patches = (image_size // patch_size) ** 2
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, num_hidden))
        self.dropout = nn.Dropout(emb_dropout)
        self.blocks = nn.ModuleList(
            [
                Block(
                    n_embed=num_hidden,
                    num_heads=num_heads,
                    dropout=block_dropout,
                    is_decoder=False,
                )
                for _ in range(num_blocks)
            ]
        )
        self.layer_norm = nn.LayerNorm(num_hidden)

    def forward(self, x):
        x = self.patch_embedding(x)
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embedding
        x = self.dropout(x)
        for block in self.blocks:
            x = block(x)
        return self.layer_norm(x[:, 0])
