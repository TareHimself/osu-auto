import torch
from torch import nn


class ConvBlock(nn.Sequential):
    def __init__(self, cin: int, cout: int, stride: int):
        super().__init__(
            nn.Conv2d(cin, cout, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
        )


class FrameNet(nn.Module):
    """Small conv net: K stacked grayscale frames + the agent's last key state -> `outputs` values.

    Aim: outputs are (x, y) in [0, 1] capture-box coordinates. Click: outputs are logits (press, hold).
    """

    def __init__(self, stack: int = 4, width: int = 32, outputs: int = 2, heads: list[int] | None = None, bounded_aim: bool = False):
        """`heads`: output sizes of separate MLP heads on the shared trunk (e.g. [2, 2] for aim and click); None = one head.
        `bounded_aim`: sigmoid on the first two outputs, for playfield-normalised aim targets."""
        super().__init__()
        self.stack = stack
        self.bounded_aim = bounded_aim
        self.head_sizes = list(heads) if heads else None
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

        def mlp(n: int) -> nn.Sequential:
            return nn.Sequential(nn.Linear(feat + 2, 512), nn.ReLU(inplace=True), nn.Dropout(0.2), nn.Linear(512, n))

        if self.head_sizes:
            self.heads = nn.ModuleList(mlp(n) for n in self.head_sizes)
        else:
            self.head = mlp(outputs)

    def forward(self, frames: torch.Tensor, prev_keys: torch.Tensor) -> torch.Tensor:
        f = torch.cat([self.features(frames), prev_keys], dim=1)
        out = torch.cat([h(f) for h in self.heads], dim=1) if self.head_sizes else self.head(f)
        if self.bounded_aim:
            out = torch.cat([torch.sigmoid(out[:, :2]), out[:, 2:]], dim=1)
        return out


AimNet = FrameNet
