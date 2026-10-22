from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, Subset, get_worker_info

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
    capture_wh: tuple[float, float]
    cursor_xy: np.ndarray
    t_ms: np.ndarray
    target_xy: np.ndarray
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
    cap = area.capture
    L = ex.labels
    cursor_xy = np.stack([np.array(area.playfield_to_capture(float(x), float(y)), dtype=np.float32) for x, y in zip(L.x[:n], L.y[:n])])
    target_xy = (cursor_xy / np.array([cap.w, cap.h], dtype=np.float32)).astype(np.float32)
    tx = L.x if L.target_x is None else L.target_x
    ty = L.y if L.target_y is None else L.target_y
    target_pf = encode_playfield(np.stack([tx[:n] / 512.0, ty[:n] / 384.0], axis=1))
    down = (L.k1[:n] | L.k2[:n]).astype(np.float32)
    press = (L.press_k1[:n] | L.press_k2[:n]).astype(np.float32)
    spinner = np.zeros(n, dtype=bool)
    for s, e, k in zip(ex.objects.start_ms, ex.objects.end_ms, ex.objects.kind):
        if k == KIND_SPINNER:
            spinner[(L.t_ms[:n] >= s) & (L.t_ms[:n] <= e)] = True
    return FrameExport(ex.path, ex.path / fm["path"], n, cursor, int(fm["factor"]), (cap.w, cap.h), cursor_xy, L.t_ms[:n], target_xy, target_pf,
                       L.k1[:n], L.k2[:n], L.press_k1[:n], L.press_k2[:n], down, press, spinner, fm, strain_per_frame(ex, L.t_ms[:n]),
                       active_mask(ex, L.t_ms[:n], down, spinner, press))


class FrameStackDataset(Dataset):
    """Stack of the last `stack` frames (oldest first) with a cursor composited per example.

    Frames are stored without the cursor; a calibrated sprite is drawn in one of three ways:
      real - where the expert was, up to `real_lag` steps earlier (random per sample)
      none - no cursor
      fake - at one random position for the whole stack
    With `none`/`fake` the cursor carries no information about the label, which kills the copy-the-cursor shortcut.
    """

    def __init__(self, exports: list[FrameExport], stack: int = 4, stride: int = 1, horizon: int = 0,
                 variant_probs: tuple[float, float, float] = (1, 1, 0), real_lag: int = 5, seed: int = 0, target_space: str = "playfield"):
        self.exports = exports
        self.stack = stack
        self.horizon = horizon
        self.real_lag = real_lag
        self.target_space = target_space
        self.variant_probs = np.array(variant_probs) / sum(variant_probs)
        self.rng = np.random.default_rng(seed)
        self.index: list[tuple[int, int]] = []
        for e, ex in enumerate(exports):
            for i in range(stack, ex.count - horizon, stride):
                self.index.append((e, i))

    def __len__(self) -> int:
        return len(self.index)

    def _frames(self, ex: FrameExport, i: int) -> np.ndarray:
        """During a spinner the cursor is always real with at most one step of lag: the rotation phase is only readable from it."""
        lo, hi = i - self.stack + 1, i + 1
        none = np.array(ex.frames[lo:hi])
        in_spinner = bool(ex.spinner[i])
        variant = 0 if in_spinner else self.rng.choice(3, p=self.variant_probs)
        if variant == 1:
            return none

        if variant == 0:
            max_lag = 1 if in_spinner else self.real_lag
            lag = int(self.rng.integers(0, max_lag + 1)) if max_lag else 0
            lag = min(lag, lo)
            positions = ex.cursor_xy[lo - lag:hi - lag]
        else:
            fake = self.rng.uniform((0, 0), ex.capture_wh)
            positions = np.repeat(fake[None], hi - lo, axis=0)

        return np.stack([ex.cursor.composite(f, float(x), float(y), ex.factor) for f, (x, y) in zip(none, positions)])

    def _target(self, ex: FrameExport, j: int) -> np.ndarray:
        """Aim label: the expert position normalised over the playfield (0..1 across 512x384) or over the capture box."""
        return ex.target_pf[j] if self.target_space == "playfield" else ex.target_xy[j]

    def _prev_keys(self, ex: FrameExport, i: int) -> torch.Tensor:
        """The agent's own key state coming into step i: a key is down, and a press happened on the previous step."""
        return torch.tensor([ex.down[i - 1], ex.press[i - 1]], dtype=torch.float32)


class AimDataset(FrameStackDataset):
    """-> expert cursor position `horizon` steps ahead, normalised to the capture box."""

    def __getitem__(self, idx: int):
        e, i = self.index[idx]
        ex = self.exports[e]
        x = torch.from_numpy(self._frames(ex, i)).float().div_(255.0)
        target = torch.from_numpy(self._target(ex, i + self.horizon))
        return x, self._prev_keys(ex, i), target


class ClickDataset(FrameStackDataset):
    """-> (press, hold) `horizon` steps after the last frame, with a mask saying the hold decision applies.

    The frame at time t already shows the hit animation of a press made at t, so a horizon of 0 leaks the label;
    horizon 1 matches the closed loop, where the decision taken from the frame at t is applied at t + step.

    Keys are normalised to one stream: `press` is any key going down this step (a new press while the other key is
    held counts, the executor alternates keys), `hold` is a key staying down without a new press.
    With `use_keys=False` the model gets no key state and the second head becomes "a key should be down now".
    """

    SINCE_PRESS_CLIP = 50

    def __init__(self, *args, use_keys: bool = True, since_press: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_keys = use_keys
        self.since_press = since_press

    def _since_press(self, ex: FrameExport, i: int) -> torch.Tensor:
        """Steps since the last press at or before frame i, clipped and scaled; the executor supplies the same at play time."""
        if not hasattr(ex, "_last_press"):
            idx = np.flatnonzero(ex.press > 0.5)
            last = np.full(ex.count, -10 ** 6, dtype=np.int64)
            if len(idx):
                pos = np.searchsorted(idx, np.arange(ex.count), side="right") - 1
                last[pos >= 0] = idx[pos[pos >= 0]]
            ex._last_press = last
        since = min(i - int(ex._last_press[i]), self.SINCE_PRESS_CLIP)
        return torch.tensor([since / self.SINCE_PRESS_CLIP, 0.0], dtype=torch.float32)

    def __getitem__(self, idx: int):
        e, i = self.index[idx]
        ex = self.exports[e]
        x = torch.from_numpy(self._frames(ex, i)).float().div_(255.0)
        j = i + self.horizon
        press = float(ex.press[j])
        if self.use_keys:
            target = torch.tensor([press, float(ex.down[j]) * (1.0 - press)], dtype=torch.float32)
            return x, self._prev_keys(ex, j), target, torch.tensor(float(ex.down[j - 1]))
        target = torch.tensor([press, float(ex.down[j])], dtype=torch.float32)
        return x, self._since_press(ex, i) if self.since_press else torch.zeros(2), target, torch.tensor(1.0)


def worker_init(worker_id: int) -> None:
    """Give each data loader worker its own random stream for cursor variants."""
    info = get_worker_info()
    ds = info.dataset.dataset if isinstance(info.dataset, Subset) else info.dataset
    ds.rng = np.random.default_rng(info.seed % (2 ** 32))


class JointDataset(FrameStackDataset):
    """-> aim target at the frame (normalised capture box), click target (press, down) one step after the frame."""

    def __init__(self, *args, click_horizon: int = 1, **kwargs):
        super().__init__(*args, **kwargs)
        self.click_horizon = click_horizon
        self.index = [(e, i) for e, i in self.index if i + click_horizon < self.exports[e].count]

    def __getitem__(self, idx: int):
        e, i = self.index[idx]
        ex = self.exports[e]
        x = torch.from_numpy(self._frames(ex, i)).float().div_(255.0)
        j = i + self.click_horizon
        aim = torch.from_numpy(self._target(ex, i + self.horizon))
        click = torch.tensor([float(ex.press[j]), float(ex.down[j])], dtype=torch.float32)
        return x, torch.zeros(2), aim, click
