from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from osu_auto.cursor import CursorSprite
from osu_auto.framestore import read_all
from osu_auto.geometry import PlayArea
from osu_auto.labels import KIND_SPINNER, load_export
from osu_auto.targets import encode_playfield


@dataclass
class FrameExport:
    path: Path
    frames_path: Path
    count: int
    cursor: CursorSprite
    factor: int
    cursor_xy: np.ndarray
    t_ms: np.ndarray
    target_pf: np.ndarray
    k1: np.ndarray
    k2: np.ndarray
    press_k1: np.ndarray
    press_k2: np.ndarray
    down: np.ndarray
    press: np.ndarray
    spinner: np.ndarray
    frames_meta: dict | None = None
    strain: np.ndarray | None = None
    active: np.ndarray | None = None
    _frames: np.ndarray | None = None

    @property
    def frame_hw(self) -> tuple[int, int]:
        return int(self.frames_meta["height"]), int(self.frames_meta["width"])

    @property
    def compressed(self) -> bool:
        return self.frames_meta.get("codec") == "zstd"

    @property
    def frames(self) -> np.ndarray:
        """Opened lazily and per process: a memmap for .npy exports, decoded into memory for compressed ones."""
        if self._frames is None:
            self._frames = read_all(self.frames_path, self.frames_meta) if self.compressed else np.load(self.frames_path, mmap_mode="r")
        return self._frames

    def load_frames(self, into: np.ndarray) -> None:
        """Fill `into` (count, h, w) with every frame, decoding in parallel for compressed exports."""
        if self.compressed:
            read_all(self.frames_path, self.frames_meta, into=into)
        else:
            into[:] = self.frames[:self.count]

    def __getstate__(self):
        return {**self.__dict__, "_frames": None}


def preempt_ms(ar: float) -> float:
    return 1200 + 600 * (5 - ar) / 5 if ar < 5 else 1200 - 750 * (ar - 5) / 5


def strain_per_frame(ex, t_ms: np.ndarray) -> np.ndarray | None:
    """Aim plus speed difficulty of the object each frame is approaching (the hardest one in its approach window), 0 elsewhere."""
    st = ex.meta.get("strains")
    if not st:
        return None
    per = st["per_object"]
    off = int(st["object_offset"])
    d = np.asarray(per["aim"], dtype=np.float32) + np.asarray(per["speed"], dtype=np.float32)
    starts = ex.objects.start_ms[off:off + len(d)] / float(st.get("clock_rate", 1.0))
    pre = preempt_ms(float(ex.meta["approach_rate"]))
    out = np.zeros(len(t_ms), dtype=np.float32)
    for s, v in zip(starts, d):
        lo, hi = np.searchsorted(t_ms, s - pre), np.searchsorted(t_ms, s, side="right")
        out[lo:hi] = np.maximum(out[lo:hi], v)
    return out


def active_mask(ex, t_ms: np.ndarray, down: np.ndarray, spinner: np.ndarray, press: np.ndarray, after_steps: int = 30) -> np.ndarray:
    """Frames with something to do: an object in its approach window, a key held (slider), a spinner, or just after a press.
    Everything else is lead-in or break and carries no signal."""
    pre = preempt_ms(float(ex.meta["approach_rate"]))
    active = np.zeros(len(t_ms), dtype=bool)
    for s, e in zip(ex.objects.start_ms, ex.objects.end_ms):
        lo, hi = np.searchsorted(t_ms, s - pre), np.searchsorted(t_ms, max(e, s), side="right")
        active[lo:hi] = True
    for i in np.flatnonzero(press > 0.5):
        active[i:i + after_steps] = True
    return (active | (down > 0.5) | spinner) & ~wrong_way_spins(ex, t_ms, spinner)


SPIN_DIRECTION = -1.0


def wrong_way_spins(ex, t_ms: np.ndarray, spinner: np.ndarray) -> np.ndarray:
    """Frames inside spinners where the expert spins against SPIN_DIRECTION. Autoplay picks a direction per spinner; the model
    is taught one direction only, so these frames are left out rather than relabelled (the rendered disc turns the other way)."""
    L = ex.labels
    x = (L.x if L.target_x is None else L.target_x)[:len(t_ms)] - 256.0
    y = (L.y if L.target_y is None else L.target_y)[:len(t_ms)] - 192.0
    out = np.zeros(len(t_ms), dtype=bool)
    for s, e, k in zip(ex.objects.start_ms, ex.objects.end_ms, ex.objects.kind):
        if k != KIND_SPINNER:
            continue
        idx = np.flatnonzero((t_ms > s) & (t_ms < e))
        if len(idx) < 3:
            continue
        d = np.angle(np.exp(1j * np.diff(np.arctan2(y[idx], x[idx]))))
        if np.sign(np.median(d)) != SPIN_DIRECTION:
            out[idx] = True
    return out


def load_frame_export(path: str | Path) -> FrameExport:
    ex = load_export(path)
    fm = ex.meta["frames"]
    n = int(fm["count"])
    window = fm["window_client"]
    area = PlayArea(window[0], window[1])
    cursor = CursorSprite.load(ex.path / fm["cursor"])
    L = ex.labels
    cursor_xy = np.stack([np.array(area.playfield_to_capture(float(x), float(y)), dtype=np.float32) for x, y in zip(L.x[:n], L.y[:n])])
    tx = L.x if L.target_x is None else L.target_x
    ty = L.y if L.target_y is None else L.target_y
    target_pf = encode_playfield(np.stack([tx[:n] / 512.0, ty[:n] / 384.0], axis=1))
    down = (L.k1[:n] | L.k2[:n]).astype(np.float32)
    press = (L.press_k1[:n] | L.press_k2[:n]).astype(np.float32)
    spinner = np.zeros(n, dtype=bool)
    for s, e, k in zip(ex.objects.start_ms, ex.objects.end_ms, ex.objects.kind):
        if k == KIND_SPINNER:
            spinner[(L.t_ms[:n] >= s) & (L.t_ms[:n] <= e)] = True
    return FrameExport(ex.path, ex.path / fm["path"], n, cursor, int(fm["factor"]), cursor_xy, L.t_ms[:n], target_pf,
                       L.k1[:n], L.k2[:n], L.press_k1[:n], L.press_k2[:n], down, press, spinner, fm, strain_per_frame(ex, L.t_ms[:n]),
                       active_mask(ex, L.t_ms[:n], down, spinner, press))
