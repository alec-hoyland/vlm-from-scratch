import torch.nn as nn


class MultiModalProjector(nn.Module):
    def __init__(self, n_embed, image_embed_dim, dropout=0.1):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(image_embed_dim, 4 * image_embed_dim),
            nn.GELU(),
            nn.Linear(4 * image_embed_dim, image_embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)
