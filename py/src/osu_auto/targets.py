"""Aim encoding: playfield position (0..1 over 512x384) mapped into the middle of the sigmoid range."""
import numpy as np

from osu_auto.geometry import PLAYFIELD_H, PLAYFIELD_W

AIM_LO, AIM_HI = 0.25, 0.75
PX_PER_UNIT = np.array([PLAYFIELD_W, PLAYFIELD_H]) / (AIM_HI - AIM_LO)


def encode_playfield(pf: np.ndarray) -> np.ndarray:
    return (AIM_LO + (AIM_HI - AIM_LO) * pf).astype(np.float32)


def decode_to_playfield(nx: float, ny: float) -> tuple[float, float]:
    return (nx - AIM_LO) / (AIM_HI - AIM_LO) * PLAYFIELD_W, (ny - AIM_LO) / (AIM_HI - AIM_LO) * PLAYFIELD_H
