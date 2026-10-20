from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


def gray(bgra: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(np.ascontiguousarray(bgra), cv2.COLOR_BGRA2GRAY)


def _shift(layer: np.ndarray, fx: float, fy: float) -> np.ndarray:
    """Bilinear shift of a layer by (fx, fy) in [0, 1); the result is one pixel larger on each axis, anchored at the same integer origin."""
    h, w = layer.shape
    q = np.zeros((h + 2, w + 2), np.float32)
    q[1:1 + h, 1:1 + w] = layer
    return ((1 - fy) * (1 - fx) * q[1:, 1:] + (1 - fy) * fx * q[1:, :-1] + fy * (1 - fx) * q[:-1, 1:] + fy * fx * q[:-1, :-1])


@dataclass
class CursorSprite:
    """The gameplay cursor as an alpha-premultiplied grayscale layer at full capture resolution.

    `offset` is the sprite's top-left corner relative to the cursor hotspot, in capture pixels.
    """

    premult: np.ndarray
    alpha: np.ndarray
    offset: tuple[float, float]

    @classmethod
    def from_calibration(cls, white: np.ndarray, white_only: np.ndarray, black: np.ndarray, black_only: np.ndarray,
                         hotspot: tuple[float, float]) -> CursorSprite:
        """Four captures of the same state: cursor over white, white alone, cursor over black, black alone.

        Over black the capture is src*a, over white it is src*a + 255*(1-a), so the pair gives both layers exactly.
        """
        w, w0, b, b0 = (gray(a).astype(np.float32) for a in (white, white_only, black, black_only))
        denom = w0 - b0
        valid = denom > 128
        premult = np.clip(b - b0, 0, 255)
        alpha = np.zeros_like(premult)
        alpha[valid] = np.clip(1 - (w[valid] - b[valid]) / denom[valid], 0, 1)
        mask = valid & ((alpha > 1 / 255) | (premult > 0.5))
        ys, xs = np.nonzero(mask)
        if len(ys) == 0:
            raise RuntimeError("no cursor pixels found in calibration captures")
        y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        m = mask[y0:y1, x0:x1]
        return cls(premult[y0:y1, x0:x1] * m, alpha[y0:y1, x0:x1] * m, (float(x0 - hotspot[0]), float(y0 - hotspot[1])))

    def save(self, path: Path) -> None:
        np.savez(path, premult=self.premult.astype(np.float32), alpha=self.alpha.astype(np.float32), offset=np.array(self.offset, dtype=np.float32))

    @classmethod
    def load(cls, path: Path) -> CursorSprite:
        with np.load(path) as z:
            return cls(z["premult"], z["alpha"], (float(z["offset"][0]), float(z["offset"][1])))

    def layers_at(self, cx: float, cy: float, factor: int) -> tuple[np.ndarray, np.ndarray, int, int]:
        """Premultiplied and alpha layers downsampled by `factor`, aligned to the low-res grid for a hotspot at (cx, cy).

        The sub-pixel part of the position is applied as a bilinear shift, matching how the renderer samples the sprite.
        """
        x = cx + self.offset[0]
        y = cy + self.offset[1]
        ix, iy = int(np.floor(x)), int(np.floor(y))
        premult, alpha = (_shift(layer, x - ix, y - iy) for layer in (self.premult, self.alpha))
        px, py = ix % factor, iy % factor
        sh, sw = premult.shape
        ph = -(-(py + sh) // factor) * factor
        pw = -(-(px + sw) // factor) * factor
        p = np.zeros((ph, pw), np.float32)
        a = np.zeros((ph, pw), np.float32)
        p[py:py + sh, px:px + sw] = premult
        a[py:py + sh, px:px + sw] = alpha
        p = p.reshape(ph // factor, factor, pw // factor, factor).mean((1, 3))
        a = a.reshape(ph // factor, factor, pw // factor, factor).mean((1, 3))
        return p, a, ix // factor, iy // factor

    def composite(self, frame: np.ndarray, cx: float, cy: float, factor: int) -> np.ndarray:
        """Draw the cursor onto a grayscale frame stored at 1/factor of the capture resolution. (cx, cy) is the hotspot in capture pixels."""
        p, a, lx, ly = self.layers_at(cx, cy, factor)
        h, w = frame.shape
        y0, x0 = max(ly, 0), max(lx, 0)
        y1, x1 = min(ly + p.shape[0], h), min(lx + p.shape[1], w)
        out = frame.copy()
        if y1 <= y0 or x1 <= x0:
            return out
        ps = p[y0 - ly:y1 - ly, x0 - lx:x1 - lx]
        as_ = a[y0 - ly:y1 - ly, x0 - lx:x1 - lx]
        region = out[y0:y1, x0:x1].astype(np.float32)
        out[y0:y1, x0:x1] = np.clip(region * (1 - as_) + ps + 0.5, 0, 255).astype(np.uint8)
        return out
