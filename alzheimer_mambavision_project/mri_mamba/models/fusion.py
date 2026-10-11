import torch
import torch.nn as nn


class GatedFeatureFusion(nn.Module):
    """Learn a per-channel balance between local and global features."""

    def __init__(self, feature_dim=256):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.GELU(),
            nn.Linear(128, feature_dim),
            nn.Sigmoid(),
        )

    def forward(self, global_features, local_features):
        gate = self.gate(torch.abs(global_features - local_features))
        return gate * global_features + (1.0 - gate) * local_features
