"""Frozen feature extractor plus a separately initialized binary linear head."""
import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18


class FrozenResNet18(nn.Module):
    def __init__(self, weights="IMAGENET1K_V1", *, initialize=True):
        super().__init__()
        if weights not in {"IMAGENET1K_V1", "RANDOM_SMOKE_ONLY"}:
            raise ValueError("unsupported initialization")
        pretrained = ResNet18_Weights.IMAGENET1K_V1 if initialize and weights == "IMAGENET1K_V1" else None
        try:
            self.backbone = resnet18(weights=pretrained)
        except Exception as exc:
            raise RuntimeError(f"PRETRAINED_WEIGHTS_UNAVAILABLE: {type(exc).__name__}: {exc}. "
                               "No random fallback; use explicit RANDOM_SMOKE_ONLY for a smoke run.") from exc
        feature_dim = self.backbone.fc.in_features
        if feature_dim != 512:
            raise RuntimeError("unexpected ResNet18 feature dimension")
        self.backbone.fc = nn.Identity()
        self.backbone.requires_grad_(False)
        self.head = nn.Linear(feature_dim, 2)
        self.backbone.eval()

    def train(self, mode=True):
        super().train(mode)
        # requires_grad=False alone does NOT stop BatchNorm running-stat updates.
        self.backbone.eval()
        return self

    def extract_features(self, x):
        if x.ndim != 4 or x.shape[1] != 3 or min(x.shape[-2:]) < 32:
            raise ValueError("expected NCHW RGB tensor with spatial dimensions >=32")
        if not x.is_floating_point() or not torch.isfinite(x).all():
            raise ValueError("expected finite floating-point inputs")
        with torch.no_grad():
            return self.backbone(x)

    def forward(self, x):
        return self.head(self.extract_features(x))
