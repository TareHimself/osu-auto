import torch
from torch import nn


class ConvBlock(nn.Sequential):
    def __init__(self, cin: int, cout: int, stride: int):
        super().__init__(
            nn.Conv2d(cin, cout, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
        )


class AimNet(nn.Module):
    """Small conv net: K stacked grayscale frames + last key state -> (x, y) in [0, 1] capture-box coordinates."""

    def __init__(self, stack: int = 4, width: int = 32):
        super().__init__()
        self.stack = stack
        self.features = nn.Sequential(
            ConvBlock(stack, width, 2),
            ConvBlock(width, width, 1),
            ConvBlock(width, width * 2, 2),
            ConvBlock(width * 2, width * 2, 1),
            ConvBlock(width * 2, width * 4, 2),
            ConvBlock(width * 4, width * 4, 1),
            ConvBlock(width * 4, width * 8, 2),
            nn.AdaptiveAvgPool2d((3, 4)),
            nn.Flatten(),
        )
        feat = width * 8 * 12
        self.head = nn.Sequential(
            nn.Linear(feat + 2, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, 2),
        )

    def forward(self, frames: torch.Tensor, prev_keys: torch.Tensor) -> torch.Tensor:
        f = self.features(frames)
        return self.head(torch.cat([f, prev_keys], dim=1))
