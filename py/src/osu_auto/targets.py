"""Aim target encoding: playfield position (0..1 over 512x384) mapped into the middle of the sigmoid range."""
from __future__ import annotations

import numpy as np

from osu_auto.geometry import PLAYFIELD_H, PLAYFIELD_W, PlayArea

AIM_LO, AIM_HI = 0.25, 0.75


def encode_playfield(pf: np.ndarray) -> np.ndarray:
    return (AIM_LO + (AIM_HI - AIM_LO) * pf).astype(np.float32)


def decode_to_playfield(nx: float, ny: float, ckpt: dict, area: PlayArea) -> tuple[float, float]:
    """Model aim output -> playfield coordinates, for both the playfield encoding and the older capture-box encoding."""
    if ckpt.get("target_space") == "playfield":
        px = (nx - AIM_LO) / (AIM_HI - AIM_LO)
        py = (ny - AIM_LO) / (AIM_HI - AIM_LO)
        return px * PLAYFIELD_W, py * PLAYFIELD_H
    cap = area.capture
    return area.window_to_playfield(nx * cap.w + cap.x, ny * cap.h + cap.y)


def error_scale(ckpt_or_space) -> np.ndarray:
    """Multiply (output - target) by this to get playfield pixels for the playfield encoding."""
    space = ckpt_or_space if isinstance(ckpt_or_space, str) else ckpt_or_space.get("target_space")
    if space == "playfield":
        return np.array([PLAYFIELD_W, PLAYFIELD_H]) / (AIM_HI - AIM_LO)
    return np.array([640.0, 480.0])
