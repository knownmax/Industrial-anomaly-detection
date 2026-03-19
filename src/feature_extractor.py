"""
Feature extractor using WideResNet-101-2 backbone (timm).
Extracts multi-scale patch descriptors from layer2 (stride 8) and layer3 (stride 16).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class PatchFeatureExtractor(nn.Module):
    """
    Extracts L2-normalized patch descriptors from WideResNet-101-2.

    For a 224×224 input:
      - layer2 output: [B, 512,  28, 28]
      - layer3 output: [B, 1024, 14, 14]

    Both are spatially aligned to 28×28 via adaptive avg pooling on layer3,
    then concatenated → [B, 1536, 28, 28].
    Each of the 28×28 = 784 spatial locations yields a 1536-dim descriptor.

    No training — runs in eval mode with torch.no_grad().
    """

    def __init__(self, backbone: str = "wide_resnet101_2", pretrained: bool = True):
        super().__init__()
        # features_only=True exposes intermediate feature maps
        self.backbone = timm.create_model(
            backbone,
            pretrained=pretrained,
            features_only=True,
            out_indices=(1, 2),   # layer2=index1 (512ch), layer3=index2 (1024ch)
        )
        self.backbone.eval()
        for param in self.backbone.parameters():
            param.requires_grad_(False)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, 3, 224, 224] ImageNet-normalised images.
        Returns:
            patch_features: [B, H*W, 1536] L2-normalised patch descriptors
                            where H=W=28 for 224px input.
        """
        features = self.backbone(x)   # list: [layer2, layer3]
        f2 = features[0]              # [B, 512,  28, 28]
        f3 = features[1]              # [B, 1024, 14, 14]

        # Upsample layer3 to match layer2 spatial size
        f3_up = F.adaptive_avg_pool2d(f3, output_size=f2.shape[-2:])  # [B, 1024, 28, 28]

        # Concatenate along channel dim
        combined = torch.cat([f2, f3_up], dim=1)  # [B, 1536, 28, 28]

        B, C, H, W = combined.shape
        # Reshape to [B, H*W, C]
        patch_features = combined.permute(0, 2, 3, 1).reshape(B, H * W, C)

        # L2 normalize each descriptor
        patch_features = F.normalize(patch_features, p=2, dim=-1)

        return patch_features

    @torch.no_grad()
    def extract_patch_features(self, images: torch.Tensor) -> torch.Tensor:
        """
        Convenience method: flatten all patches across batch.
        Args:
            images: [B, 3, 224, 224]
        Returns:
            [B * H * W, 1536]  — ready for coreset subsampling or memory bank
        """
        patch_features = self.forward(images)  # [B, 784, 1536]
        B, N, C = patch_features.shape
        return patch_features.reshape(B * N, C)
