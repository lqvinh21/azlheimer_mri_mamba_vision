import argparse

import torch
import torch.nn as nn
from mambavision import create_model

from .attention import ECA1D
from .fusion import GatedFeatureFusion
from .local_branch import LocalDetailBranch


class MRI_MambaFusionClassifier(nn.Module):
    """MambaVision global encoder + local CNN + gated fusion + ECA + classifier."""

    def __init__(self, num_classes=4, pretrained=True, dropout=0.35, model_path="mamba_vision_tiny_1k.pth.tar"):
        super().__init__()
        if pretrained:
            # Official NVIDIA checkpoint metadata includes argparse.Namespace.
            torch.serialization.add_safe_globals([argparse.Namespace])
        self.backbone = create_model(
            "mamba_vision_T",
            pretrained=pretrained,
            model_path=str(model_path),
        )
        global_dim = self.backbone.head.in_features
        self.backbone.head = nn.Identity()

        self.local_branch = LocalDetailBranch(output_dim=128)
        self.global_projection = nn.Sequential(
            nn.Linear(global_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
        )
        self.local_projection = nn.Sequential(
            nn.Linear(128, 256),
            nn.LayerNorm(256),
            nn.GELU(),
        )
        self.feature_fusion = GatedFeatureFusion(feature_dim=256)
        self.channel_attention = ECA1D(kernel_size=5)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(dropout / 2),
            nn.Linear(128, num_classes),
        )

    def forward(self, image):
        grayscale_image = image[:, :1]
        global_features = self.global_projection(self.backbone(image))
        local_features = self.local_projection(self.local_branch(grayscale_image))
        fused_features = self.feature_fusion(global_features, local_features)
        attended_features = self.channel_attention(fused_features)
        return self.classifier(attended_features)
