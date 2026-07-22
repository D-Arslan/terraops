"""Model definitions for EuroSAT: ResNet-18 or MobileNetV3-Small, with a
configurable freeze/unfreeze policy. Both knobs live in params.yaml (model.arch,
model.unfreeze) so DVC re-trains when they change and MLflow logs them as params.
"""

import torch.nn as nn
from torchvision import models


def build_model(num_classes: int = 10, pretrained: bool = True,
                dropout: float = 0.3, arch: str = "resnet18",
                unfreeze: str = "last_block") -> nn.Module:
    """Build a pretrained backbone with a fresh classification head.

    arch: 'resnet18' | 'mobilenet_v3_small'
      - resnet18: the Sprint-1 baseline (11.2M params)
      - mobilenet_v3_small: ~2.5M params — the efficiency challenger, to make
        the accuracy-vs-inference-cost trade-off visible in the promotion gate

    unfreeze: which parameters train (everything else keeps ImageNet weights)
      - 'none':       head only (cheapest, pure linear probing)
      - 'last_block': last conv block + head (Sprint-1 default behavior)
      - 'all':        full fine-tuning (most capacity, most overfitting risk)
    """
    if arch == "resnet18":
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.resnet18(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, num_classes)
        )
        head_prefixes = ("fc",)
        last_block_prefixes = ("layer4", "fc")

    elif arch == "mobilenet_v3_small":
        weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.mobilenet_v3_small(weights=weights)
        # Head = classifier[2] (Dropout) + classifier[3] (final Linear):
        # override both so model.dropout applies here too, like on ResNet.
        model.classifier[2] = nn.Dropout(dropout)
        in_features = model.classifier[3].in_features
        model.classifier[3] = nn.Linear(in_features, num_classes)
        head_prefixes = ("classifier",)
        # features.11-12 = last inverted-residual block + final conv
        last_block_prefixes = ("features.11", "features.12", "classifier")

    else:
        raise ValueError(f"Unknown arch '{arch}' (resnet18 | mobilenet_v3_small)")

    if unfreeze == "all":
        trainable_prefixes = None                 # train everything
    elif unfreeze == "last_block":
        trainable_prefixes = last_block_prefixes
    elif unfreeze == "none":
        trainable_prefixes = head_prefixes
    else:
        raise ValueError(f"Unknown unfreeze '{unfreeze}' (none | last_block | all)")

    if trainable_prefixes is not None:
        for name, param in model.named_parameters():
            param.requires_grad = name.startswith(trainable_prefixes)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Model: {arch} | unfreeze={unfreeze} | "
          f"Trainable: {trainable:,} / {total:,} params")

    return model
