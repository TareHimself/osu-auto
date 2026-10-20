from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from osu_auto.cursor import CursorSprite
from osu_auto.geometry import PlayArea
from osu_auto.labels import load_export


@dataclass
class FrameExport:
    path: Path
    frames: np.ndarray
    cursor: CursorSprite
    factor: int
    capture_wh: tuple[float, float]
    cursor_xy: np.ndarray
    t_ms: np.ndarray
    target_xy: np.ndarray
    k1: np.ndarray
    k2: np.ndarray
    press_k1: np.ndarray
    press_k2: np.ndarray


def load_frame_export(path: str | Path) -> FrameExport:
    ex = load_export(path)
    fm = ex.meta["frames"]
    frames = np.load(ex.path / fm["path"], mmap_mode="r")
    n = len(frames)
    window = fm["window_client"]
    area = PlayArea(window[0], window[1])
    cursor = CursorSprite.load(ex.path / fm["cursor"])
    cap = area.capture
    cursor_xy = np.stack([np.array(area.playfield_to_capture(float(x), float(y)), dtype=np.float32) for x, y in zip(ex.labels.x[:n], ex.labels.y[:n])])
    target_xy = (cursor_xy / np.array([cap.w, cap.h], dtype=np.float32)).astype(np.float32)
    return FrameExport(ex.path, frames, cursor, int(fm["factor"]), (cap.w, cap.h), cursor_xy, ex.labels.t_ms[:n], target_xy,
                       ex.labels.k1[:n], ex.labels.k2[:n], ex.labels.press_k1[:n], ex.labels.press_k2[:n])


class AimDataset(Dataset):
    """Stack of the last `stack` frames (oldest first) -> expert cursor position `horizon` steps later, normalised to the capture box.

    Frames are stored without the cursor; a calibrated sprite is composited per example:
      real - at the expert position of each frame
      none - no cursor
      fake - at one random position for the whole stack
    The target is the expert's future position, and with `none`/`fake` the cursor carries no information about it,
    which kills the copy-the-cursor shortcut.
    """

    def __init__(self, exports: list[FrameExport], stack: int = 4, stride: int = 1, horizon: int = 1,
                 variant_probs: tuple[float, float, float] = (1 / 3, 1 / 3, 1 / 3), real_lag: int = 0, seed: int = 0):
        """real_lag: the real-cursor variant is drawn where the expert was up to this many steps earlier (random per sample),
        so a horizon-0 label never coincides with the visible cursor and the picture matches an agent that trails the expert."""
        self.exports = exports
        self.stack = stack
        self.horizon = horizon
        self.real_lag = real_lag
        self.variant_probs = np.array(variant_probs) / sum(variant_probs)
        self.rng = np.random.default_rng(seed)
        self.index: list[tuple[int, int]] = []
        for e, ex in enumerate(exports):
            for i in range(stack - 1, len(ex.frames) - horizon, stride):
                self.index.append((e, i))

    def __len__(self) -> int:
        return len(self.index)

    def _frames(self, ex: FrameExport, i: int) -> np.ndarray:
        lo, hi = i - self.stack + 1, i + 1
        none = np.array(ex.frames[lo:hi])
        variant = self.rng.choice(3, p=self.variant_probs)
        if variant == 1:
            return none

        if variant == 0:
            lag = int(self.rng.integers(0, self.real_lag + 1)) if self.real_lag else 0
            lag = min(lag, lo)
            positions = ex.cursor_xy[lo - lag:hi - lag]
        else:
            fake = self.rng.uniform((0, 0), ex.capture_wh)
            positions = np.repeat(fake[None], hi - lo, axis=0)

        return np.stack([ex.cursor.composite(f, float(x), float(y), ex.factor) for f, (x, y) in zip(none, positions)])

    def __getitem__(self, idx: int):
        e, i = self.index[idx]
        ex = self.exports[e]
        x = torch.from_numpy(self._frames(ex, i)).float().div_(255.0)
        prev_keys = torch.tensor([ex.k1[i], ex.k2[i]], dtype=torch.float32)
        target = torch.from_numpy(ex.target_xy[i + self.horizon])
        return x, prev_keys, target
