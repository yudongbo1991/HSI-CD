"""Fixed VRWKV trunk used by the final detector."""
import torch.nn as nn

from .vrwkv import HSI_RWKV


class HiRWKV(nn.Module):
    def __init__(self, in_channels=128, hidden_dim=128, depth=3):
        super().__init__()
        self.patch_embedding = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
            nn.GroupNorm(1, hidden_dim),
            nn.SiLU(),
        )
        layers = []
        for index in range(depth):
            if index:
                layers.extend((nn.MaxPool2d(3, stride=1, padding=1), nn.SiLU()))
            layers.append(HSI_RWKV(hidden_dim, (1,)))
        self.trunk = nn.Sequential(*layers)

    def forward(self, features):
        return self.trunk(self.patch_embedding(features))
