from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

AIM_FROM_BOX = ((0.625, 0.1875), (0.625, 0.17708333))


class ConvBlock(nn.Sequential):
    def __init__(self, cin: int, cout: int, stride: int):
        super().__init__(
            nn.Conv2d(cin, cout, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
        )


class SoftArgmaxAim(nn.Module):
    """Scores every stride 8 cell and aims at the attention-weighted mean of the cell centres. Context comes from the upsampled
    stride 16 map; the aux inputs (latency, keys) modulate the score features."""

    def __init__(self, c8: int, c16: int, aux: int, hidden: int = 64):
        super().__init__()
        self.mix = nn.Sequential(ConvBlock(c8 + c16, hidden, 1), ConvBlock(hidden, hidden, 1))
        self.film = nn.Linear(aux, hidden * 2)
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)
        self.score = nn.Conv2d(hidden, 1, 1)
        self.temp = nn.Parameter(torch.ones(()))

    def forward(self, s8: torch.Tensor, s16: torch.Tensor, aux: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.mix(torch.cat([s8, F.interpolate(s16, size=s8.shape[-2:], mode="bilinear", align_corners=False)], dim=1))
        film = self.film(aux)
        scale, shift = film[:, : film.shape[1] // 2], film[:, film.shape[1] // 2 :]
        h = torch.relu(h * (1 + scale[:, :, None, None]) + shift[:, :, None, None])
        gh, gw = s8.shape[-2:]
        logits = (self.score(h).flatten(1) * self.temp).float()
        att = logits.softmax(dim=1)
        v, u = torch.meshgrid((torch.arange(gh, device=s8.device) + 0.5) / gh, (torch.arange(gw, device=s8.device) + 0.5) / gw, indexing="ij")
        (ax, bx), (ay, by) = AIM_FROM_BOX
        x = bx + ax * (att * u.flatten()).sum(1)
        y = by + ay * (att * v.flatten()).sum(1)
        return torch.stack([x, y], dim=1), logits.view(-1, gh, gw)


class FrameNet(nn.Module):
    """Stacked grayscale frames plus 2 aux inputs (latency, unused) -> (x, y, press logit, down logit).
    x, y are in the aim encoding of targets.py. `aim_head` is "softargmax" (attention over stride 8 cells) or "mlp"."""

    def __init__(self, stack: int = 4, width: int = 32, aim_head: str = "softargmax"):
        super().__init__()
        self.stack = stack
        self.aim_head = aim_head
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

        def mlp() -> nn.Sequential:
            return nn.Sequential(nn.Linear(feat + 2, 512), nn.ReLU(inplace=True), nn.Dropout(0.2), nn.Linear(512, 2))

        if aim_head == "softargmax":
            self.aim = SoftArgmaxAim(width * 4, width * 8, 2)
            self.heads = nn.ModuleList([mlp()])
        else:
            self.heads = nn.ModuleList([mlp(), mlp()])

    def forward(self, frames: torch.Tensor, aux: torch.Tensor, return_logits: bool = False):
        """`return_logits` also returns the softargmax attention logits (B, 15, 20) for the heatmap loss."""
        if self.aim_head == "softargmax":
            s8 = self.features[:6](frames)
            s16 = self.features[6](s8)
            f = torch.cat([self.features[7:](s16), aux], dim=1)
            aim, logits = self.aim(s8, s16, aux)
            out = torch.cat([aim.to(f.dtype), self.heads[0](f)], dim=1)
            return (out, logits) if return_logits else out
        f = torch.cat([self.features(frames), aux], dim=1)
        return torch.cat([torch.sigmoid(self.heads[0](f)), self.heads[1](f)], dim=1)


def load_checkpoint(path: str | Path, device: str | torch.device = "cpu") -> tuple[FrameNet, dict]:
    ck = torch.load(path, map_location=device)
    model = FrameNet(stack=ck["stack"], width=ck["width"], aim_head=ck.get("aim_head", "mlp"))
    model.load_state_dict(ck["model"])
    return model.to(device).eval(), ck
