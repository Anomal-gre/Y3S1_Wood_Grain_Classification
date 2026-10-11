"""
Model architectures for Wood Species Classification.
Supports ResNet18, ResNet50, and ConvNeXt-Tiny with modular custom heads
and backbone freeze/unfreeze mechanisms for two-stage transfer learning.
"""

from typing import Tuple, List, Dict
import torch
import torch.nn as nn
from torchvision import models

def build_backbone(name: str = "resnet18", pretrained: bool = True) -> Tuple[nn.Module, int]:
    """Build backbone feature extractor and return feature dimension."""
    name = name.lower()
    if name == "resnet18":
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        m = models.resnet18(weights=weights)
        in_features = m.fc.in_features
        m.fc = nn.Identity()
    elif name == "resnet50":
        weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        m = models.resnet50(weights=weights)
        in_features = m.fc.in_features
        m.fc = nn.Identity()
    elif name == "convnext_tiny":
        weights = models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
        m = models.convnext_tiny(weights=weights)
        in_features = m.classifier[2].in_features
        m.classifier[2] = nn.Identity()
    else:
        raise ValueError(f"Unsupported backbone: {name}. Supported: ['resnet18', 'resnet50', 'convnext_tiny']")
    return m, in_features

class WoodNet(nn.Module):
    """
    Deep Convolutional Network for Wood Grain Texture Classification.
    Features a pretrained backbone with a custom MLP classification head.
    """
    def __init__(
        self,
        backbone: str = "resnet18",
        num_classes: int = 12,
        pretrained: bool = True,
        dropout_rate: float = 0.4
    ):
        super().__init__()
        self.backbone_name = backbone
        self.num_classes = num_classes
        self.backbone, in_features = build_backbone(backbone, pretrained=pretrained)
        
        self.head = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(x)
        return self.head(feat)

    def freeze_backbone(self) -> None:
        """Freeze all parameters in the backbone for Stage 1 warm-up."""
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self) -> None:
        """Unfreeze all parameters in the backbone for Stage 2 fine-tuning."""
        for param in self.backbone.parameters():
            param.requires_grad = True

    def get_parameter_groups(
        self,
        lr_backbone: float = 1e-4,
        lr_head: float = 1e-3,
        weight_decay: float = 1e-2
    ) -> List[Dict]:
        """Configure differential learning rates for Stage 2 fine-tuning."""
        return [
            {"params": self.backbone.parameters(), "lr": lr_backbone, "weight_decay": weight_decay},
            {"params": self.head.parameters(), "lr": lr_head, "weight_decay": weight_decay}
        ]
