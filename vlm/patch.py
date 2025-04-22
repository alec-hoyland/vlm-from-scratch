import torch.nn as nn


class PatchEmbeddings(nn.Module):
    def __init__(self, image_size=96, patch_size=16, hidden_dim=512):
        super().__init__()

        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        self.conv = nn.Conv2d(
            in_channels=3,
            out_channels=hidden_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.flatten(2)
        return x.transpose(1, 2)
