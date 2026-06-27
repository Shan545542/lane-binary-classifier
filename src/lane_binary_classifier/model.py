from __future__ import annotations

import torch
from torch import nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class LaneBinaryNet(nn.Module):
    """Small CNN for lane in/out binary classification from an RGB ROI."""

    def __init__(self, dropout: float = 0.25) -> None:
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(3, 24),
            ConvBlock(24, 48),
            ConvBlock(48, 96),
            ConvBlock(96, 128),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(p=dropout),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.classifier(self.features(x))
        return logits.squeeze(1)


def build_model(name: str = "lane_cnn", dropout: float = 0.25) -> nn.Module:
    if name != "lane_cnn":
        raise ValueError(f"Unsupported model name: {name}")
    return LaneBinaryNet(dropout=dropout)
