import torch.nn as nn


class ECA1D(nn.Module):
    """Lightweight channel attention for a feature vector."""

    def __init__(self, kernel_size=5):
        super().__init__()
        self.conv = nn.Conv1d(
            1,
            1,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2,
            bias=False,
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, features):
        channel_weights = self.sigmoid(self.conv(features.unsqueeze(1))).squeeze(1)
        return features * channel_weights
