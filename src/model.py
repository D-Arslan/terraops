"""Model definition: ResNet-18 with transfer learning for EuroSAT."""

import torch.nn as nn
from torchvision import models


def build_model(num_classes: int = 10, pretrained: bool = True,
                dropout: float = 0.3) -> nn.Module:
    """Build ResNet-18 with pretrained ImageNet weights.

    Why ResNet-18 over alternatives:
    - vs. ResNet-50: ResNet-18 is sufficient for 64x64 images (fewer params,
      faster training, less overfitting risk on 27k samples)
    - vs. VGG-16: ResNet's skip connections converge faster and generalize
      better on small datasets

    Transfer learning: freeze early layers, replace final FC for 10 classes.
    """
    weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.resnet18(weights=weights)

    # Freeze early convolutional layers (layers 1-3)
    for name, param in model.named_parameters():
        if "layer4" not in name and "fc" not in name:
            param.requires_grad = False

    # Replace classifier head for 10 EuroSAT classes
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(dropout),
        nn.Linear(in_features, num_classes)
    )

    # Count trainable parameters
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Model: ResNet-18 | Trainable: {trainable:,} / {total:,} params")

    return model
