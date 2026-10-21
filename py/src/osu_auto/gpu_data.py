"""Training data served from GPU memory in chunks: frames stay uint8 on the device, stacks are gathered and the
cursor is composited there, so no loader workers and no per-sample CPU work."""
from __future__ import annotations

import threading
from dataclasses import dataclass

import numpy as np
import torch
from torch.nn import functional as F

from osu_auto.dataset import FrameExport


@dataclass
class Chunk:
    exports: list[int]
    frames: torch.Tensor
    export_id: torch.Tensor
    local: torch.Tensor
    cursor_xy: torch.Tensor
    aim: torch.Tensor
    press: torch.Tensor
    down: torch.Tensor
    spinner: torch.Tensor
    rows: torch.Tensor
    weight: torch.Tensor | None = None


class GpuCursor:
    """All exports' cursor sprites padded to one size, composited in a batch with the same maths as CursorSprite."""

    def __init__(self, exports: list[FrameExport], device: torch.device):
        h = max(e.cursor.premult.shape[0] for e in exports)
        w = max(e.cursor.premult.shape[1] for e in exports)
        premult = np.zeros((len(exports), h, w), np.float32)
        alpha = np.zeros((len(exports), h, w), np.float32)
        for k, e in enumerate(exports):
            sh, sw = e.cursor.premult.shape
            premult[k, :sh, :sw] = e.cursor.premult
            alpha[k, :sh, :sw] = e.cursor.alpha
        self.premult = torch.from_numpy(premult).to(device)
        self.alpha = torch.from_numpy(alpha).to(device)
        self.offset = torch.tensor([e.cursor.offset for e in exports], dtype=torch.float32, device=device)
        self.factor = torch.tensor([e.factor for e in exports], dtype=torch.int64, device=device)
        self.device = device

    @staticmethod
    def _shift(layer: torch.Tensor, fx: torch.Tensor, fy: torch.Tensor) -> torch.Tensor:
        fx, fy = fx[:, None, None], fy[:, None, None]
        return ((1 - fy) * (1 - fx) * F.pad(layer, (0, 1, 0, 1)) + (1 - fy) * fx * F.pad(layer, (1, 0, 0, 1))
                + fy * (1 - fx) * F.pad(layer, (0, 1, 1, 0)) + fy * fx * F.pad(layer, (1, 0, 1, 0)))

    def composite(self, frames: torch.Tensor, export_id: torch.Tensor, xy: torch.Tensor) -> torch.Tensor:
        """frames (N, H, W) float in 0..255, xy (N, 2) hotspot in capture pixels -> frames with the cursor drawn, rounded to whole values."""
        n, h, w = frames.shape
        factor = self.factor[export_id]
        if not bool((factor == factor[0]).all()):
            raise ValueError("all exports in a chunk must share the downsample factor")
        f = int(factor[0])
        pos = xy + self.offset[export_id]
        ipos = torch.floor(pos)
        frac = pos - ipos
        ipos = ipos.long()
        premult = self._shift(self.premult[export_id], frac[:, 0], frac[:, 1])
        alpha = self._shift(self.alpha[export_id], frac[:, 0], frac[:, 1])
        sh, sw = premult.shape[1:]
        ph = -(-(f - 1 + sh) // f) * f
        pw = -(-(f - 1 + sw) // f) * f
        py, px = ipos[:, 1] % f, ipos[:, 0] % f
        rows = (torch.arange(ph, device=self.device)[None] - py[:, None]) % ph
        cols = (torch.arange(pw, device=self.device)[None] - px[:, None]) % pw
        canvas_p = F.pad(premult, (0, pw - sw, 0, ph - sh))
        canvas_a = F.pad(alpha, (0, pw - sw, 0, ph - sh))
        canvas_p = canvas_p.gather(1, rows[:, :, None].expand(-1, -1, pw)).gather(2, cols[:, None, :].expand(-1, ph, -1))
        canvas_a = canvas_a.gather(1, rows[:, :, None].expand(-1, -1, pw)).gather(2, cols[:, None, :].expand(-1, ph, -1))
        p = F.avg_pool2d(canvas_p[:, None], f)[:, 0]
        a = F.avg_pool2d(canvas_a[:, None], f)[:, 0]
        lh, lw = p.shape[1:]
        ly = torch.div(ipos[:, 1], f, rounding_mode="floor")
        lx = torch.div(ipos[:, 0], f, rounding_mode="floor")
        ys = ly[:, None] + torch.arange(lh, device=self.device)[None]
        xs = lx[:, None] + torch.arange(lw, device=self.device)[None]
        valid = ((ys >= 0) & (ys < h))[:, :, None] & ((xs >= 0) & (xs < w))[:, None, :]
        flat = (torch.arange(n, device=self.device)[:, None, None] * h + ys.clamp(0, h - 1)[:, :, None]) * w + xs.clamp(0, w - 1)[:, None, :]
        out = frames.reshape(-1)
        region = out[flat]
        new = torch.clamp(region * (1 - a) + p + 0.5, 0, 255).floor()
        out[flat[valid]] = new[valid]
        return out.view(n, h, w)


class GpuChunkLoader:
    """Iterates (frames, keys, aim, click) batches like JointDataset, one GPU-resident chunk of exports at a time.

    Rows are frame indices into the chunk; the stack ends at the row, the aim label is at the row and the click
    label `click_horizon` rows later. Each chunk's rows are shuffled, chunk order is shuffled per epoch, and the
    next chunk is read into pinned memory while the current one trains.
    """

    def __init__(self, exports: list[FrameExport], device: torch.device, stack: int = 4, click_horizon: int = 1, chunk_bytes: int = 3 << 30,
                 real_prob: float = 0.5, real_lag: int = 5, batch: int = 128, shuffle: bool = True, drop_last: bool = True, seed: int = 0,
                 stride: int = 1, pool: int = 1, heat: float = 0.0, heat_floor: float = 0.25):
        """`heat` > 0 samples rows in proportion to (strain of the approached object, normalised to mean 1) + press windows,
        mixed with uniform: weight = heat_floor + heat * score. 0 keeps uniform sampling."""
        self.exports = exports
        self.device = device
        self.stack = stack
        self.click_horizon = click_horizon
        self.real_prob = real_prob
        self.real_lag = real_lag
        self.batch = batch
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.stride = stride
        self.pool = pool
        self.heat = heat
        self.heat_floor = heat_floor
        if heat > 0:
            if any(e.strain is None for e in exports):
                raise ValueError("heat sampling needs strain data in every export's meta (run backfill_labels)")
            self.strain_scale = float(np.concatenate([e.strain for e in exports]).mean()) or 1.0
        self.gen = torch.Generator(device=device)
        self.gen.manual_seed(seed)
        self.rng = np.random.default_rng(seed)
        self.cursor = GpuCursor(exports, device)
        self.chunk_bytes = chunk_bytes
        self.groups = self._group(list(range(len(exports))))

    def _group(self, order: list[int]) -> list[list[int]]:
        groups: list[list[int]] = []
        size = 0
        for k in order:
            nbytes = self.exports[k].count * int(np.prod(self.exports[k].frame_hw))
            if groups and size + nbytes > self.chunk_bytes:
                groups.append([])
                size = 0
            if not groups:
                groups.append([])
            groups[-1].append(k)
            size += nbytes
        return groups

    def _rows(self, e: FrameExport) -> np.ndarray:
        return np.arange(self.stack - 1, e.count - self.click_horizon, self.stride)

    def _weights(self, e: FrameExport, rows: np.ndarray) -> np.ndarray:
        score = e.strain[rows] / self.strain_scale
        press = np.flatnonzero(e.press > 0.5)
        near = np.zeros(e.count, dtype=np.float32)
        for i in press:
            near[max(i - 5, 0):i + 6] = 1.0
        return (self.heat_floor + self.heat * (score + near[rows])).astype(np.float32)

    def __len__(self) -> int:
        n = 0
        for g in self.groups:
            rows = sum(len(self._rows(self.exports[k])) for k in g)
            n += rows // self.batch if self.drop_last else -(-rows // self.batch)
        return n

    def _read(self, group: list[int]) -> Chunk:
        exps = [self.exports[k] for k in group]
        total = sum(e.count for e in exps)
        h, w = exps[0].frame_hw
        frames = torch.empty((total, h, w), dtype=torch.uint8).pin_memory()
        ids, local, cxy, aim, press, down, spin, rows, weight = [], [], [], [], [], [], [], [], []
        base = 0
        for k, e in zip(group, exps):
            e.load_frames(frames[base:base + e.count].numpy())
            if self.heat > 0:
                weight.append(self._weights(e, self._rows(e)))
            ids.append(np.full(e.count, k, np.int64))
            local.append(np.arange(e.count))
            cxy.append(e.cursor_xy)
            aim.append(e.target_pf)
            press.append(e.press)
            down.append(e.down)
            spin.append(e.spinner)
            rows.append(self._rows(e) + base)
            base += e.count
        t = lambda arrs, dtype: torch.from_numpy(np.concatenate(arrs)).to(dtype)
        return Chunk(group, frames, t(ids, torch.int64), t(local, torch.int64), t(cxy, torch.float32), t(aim, torch.float32),
                     t(press, torch.float32), t(down, torch.float32), t(spin, torch.bool), t(rows, torch.int64),
                     t(weight, torch.float32) if weight else None)

    def _to_device(self, c: Chunk) -> Chunk:
        d = self.device
        return Chunk(c.exports, c.frames.to(d), c.export_id.to(d), c.local.to(d), c.cursor_xy.to(d), c.aim.to(d),
                     c.press.to(d), c.down.to(d), c.spinner.to(d), c.rows.to(d), None if c.weight is None else c.weight.to(d))

    def _batch(self, c: Chunk, rows: torch.Tensor):
        b = len(rows)
        s = self.stack
        offsets = torch.arange(-s + 1, 1, device=self.device)
        idx = rows[:, None] + offsets[None]
        frames = c.frames[idx].float()
        export_id = c.export_id[rows]
        lo = c.local[rows] - s + 1
        in_spinner = c.spinner[rows]
        real = (torch.rand(b, generator=self.gen, device=self.device) < self.real_prob) | in_spinner
        max_lag = torch.where(in_spinner, torch.ones_like(lo), torch.full_like(lo, self.real_lag))
        lag = (torch.rand(b, generator=self.gen, device=self.device) * (max_lag + 1).float()).long().clamp(max=self.real_lag)
        lag = torch.minimum(lag, lo)
        if bool(real.any()):
            sel = real.nonzero()[:, 0]
            pos_idx = (idx[sel] - lag[sel][:, None]).reshape(-1)
            xy = c.cursor_xy[pos_idx]
            flat = frames[sel].reshape(-1, *frames.shape[2:])
            drawn = self.cursor.composite(flat, export_id[sel].repeat_interleave(s), xy)
            frames[sel] = drawn.view(len(sel), s, *frames.shape[2:])
        if self.pool > 1:
            frames = F.avg_pool2d(frames, self.pool).add_(0.5).floor_()
        frames.div_(255.0)
        j = rows + self.click_horizon
        click = torch.stack([c.press[j], c.down[j]], dim=1)
        return frames, torch.zeros(b, 2, device=self.device), c.aim[rows], click

    def __iter__(self):
        """Shuffling regroups exports into fresh chunks each epoch, so a chunk mixes maps and skins instead of following directory order."""
        if self.shuffle:
            exports = list(range(len(self.exports)))
            self.rng.shuffle(exports)
            self.groups = self._group(exports)
        order = list(range(len(self.groups)))
        if self.shuffle:
            self.rng.shuffle(order)
        pending: dict[int, Chunk] = {}
        lock = threading.Lock()

        def read(i: int) -> None:
            c = self._read(self.groups[order[i]])
            with lock:
                pending[i] = c

        thread = threading.Thread(target=read, args=(0,))
        thread.start()
        for n in range(len(order)):
            thread.join()
            with lock:
                chunk = self._to_device(pending.pop(n))
            if n + 1 < len(order):
                thread = threading.Thread(target=read, args=(n + 1,))
                thread.start()
            rows = chunk.rows
            if self.shuffle and chunk.weight is not None:
                rows = rows[torch.multinomial(chunk.weight, len(rows), replacement=True, generator=self.gen)]
            elif self.shuffle:
                rows = rows[torch.randperm(len(rows), generator=self.gen, device=self.device)]
            stop = len(rows) - len(rows) % self.batch if self.drop_last else len(rows)
            for start in range(0, stop, self.batch):
                yield self._batch(chunk, rows[start:start + self.batch])
            del chunk
            torch.cuda.empty_cache()


def split_exports(exports: list[FrameExport], holdout: float, seed: int = 0) -> tuple[list[FrameExport], list[FrameExport]]:
    """Hold out whole maps (every skin and mod variant of a held-out map goes to validation)."""
    keys = sorted({e.path.name.split("-")[0] for e in exports})
    rng = np.random.default_rng(seed)
    rng.shuffle(keys)
    held = set(keys[:max(1, int(round(len(keys) * holdout)))])
    train = [e for e in exports if e.path.name.split("-")[0] not in held]
    val = [e for e in exports if e.path.name.split("-")[0] in held]
    return train, val
