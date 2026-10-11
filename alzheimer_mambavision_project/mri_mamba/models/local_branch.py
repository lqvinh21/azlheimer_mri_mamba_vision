import torch.nn as nn


class LocalDetailBranch(nn.Module):
    """CNN path for local MRI texture and edge features."""

    def __init__(self, output_dim=128):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, output_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(output_dim),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )

    def forward(self, grayscale_image):
        return self.features(grayscale_image)
