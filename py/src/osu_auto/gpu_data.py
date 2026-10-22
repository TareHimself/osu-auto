"""Training data served from GPU memory in chunks: frames stay uint8 on the device, stacks are gathered and the
cursor is composited there, so no loader workers and no per-sample CPU work."""
from __future__ import annotations

import math
import threading
from dataclasses import dataclass

import numpy as np
import torch
from torch.nn import functional as F

from osu_auto.dataset import SPIN_DIRECTION, FrameExport
from osu_auto.targets import AIM_HI, AIM_LO


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
    click_w: torch.Tensor | None = None


class GpuCursor:
    """All exports' cursor sprites padded to one size, composited in a batch with the same maths as CursorSprite."""

    def __init__(self, exports: list[FrameExport], device: torch.device, max_scale: float = 1.0):
        h = max(e.cursor.premult.shape[0] for e in exports)
        w = max(e.cursor.premult.shape[1] for e in exports)
        pad = int(math.ceil(max(h, w) * (max_scale - 1))) + 1 if max_scale > 1 else 0
        premult = np.zeros((len(exports), h + 2 * pad, w + 2 * pad), np.float32)
        alpha = np.zeros_like(premult)
        for k, e in enumerate(exports):
            sh, sw = e.cursor.premult.shape
            premult[k, pad:pad + sh, pad:pad + sw] = e.cursor.premult
            alpha[k, pad:pad + sh, pad:pad + sw] = e.cursor.alpha
        self.premult = torch.from_numpy(premult).to(device)
        self.alpha = torch.from_numpy(alpha).to(device)
        self.offset = torch.tensor([e.cursor.offset for e in exports], dtype=torch.float32, device=device) - pad
        self.factor = torch.tensor([e.factor for e in exports], dtype=torch.int64, device=device)
        self.device = device

    @staticmethod
    def _shift(layer: torch.Tensor, fx: torch.Tensor, fy: torch.Tensor) -> torch.Tensor:
        fx, fy = fx[:, None, None], fy[:, None, None]
        return ((1 - fy) * (1 - fx) * F.pad(layer, (0, 1, 0, 1)) + (1 - fy) * fx * F.pad(layer, (1, 0, 0, 1))
                + fy * (1 - fx) * F.pad(layer, (0, 1, 1, 0)) + fy * fx * F.pad(layer, (1, 0, 1, 0)))

    def _scaled(self, export_id: torch.Tensor, scale: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Sprites resized by `scale` about their hotspot, on the same canvas."""
        premult, alpha = self.premult[export_id], self.alpha[export_id]
        sh, sw = premult.shape[1:]
        hot = -self.offset[export_id]
        inv = 1 / scale
        cx = 2 * hot[:, 0] / sw - 1
        cy = 2 * hot[:, 1] / sh - 1
        zero = torch.zeros_like(inv)
        theta = torch.stack([torch.stack([inv, zero, cx * (1 - inv)], 1), torch.stack([zero, inv, cy * (1 - inv)], 1)], 1)
        grid = F.affine_grid(theta, (len(scale), 1, sh, sw), align_corners=False)
        both = F.grid_sample(torch.stack([premult, alpha], 1), grid, mode="bilinear", padding_mode="zeros", align_corners=False)
        return both[:, 0], both[:, 1]

    def composite(self, frames: torch.Tensor, export_id: torch.Tensor, xy: torch.Tensor, scale: torch.Tensor | None = None) -> torch.Tensor:
        """frames (N, H, W) float in 0..255, xy (N, 2) hotspot in capture pixels, scale (N,) cursor size factor or None
        -> frames with the cursor drawn, rounded to whole values."""
        n, h, w = frames.shape
        factor = self.factor[export_id]
        if not bool((factor == factor[0]).all()):
            raise ValueError("all exports in a chunk must share the downsample factor")
        f = int(factor[0])
        pos = xy + self.offset[export_id]
        ipos = torch.floor(pos)
        frac = pos - ipos
        ipos = ipos.long()
        if scale is None:
            premult, alpha = self.premult[export_id], self.alpha[export_id]
        else:
            premult, alpha = self._scaled(export_id, scale)
        premult = self._shift(premult, frac[:, 0], frac[:, 1])
        alpha = self._shift(alpha, frac[:, 0], frac[:, 1])
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
                 stride: int = 1, pool: int = 1, heat: float = 0.0, heat_floor: float = 0.25, click_reweight: bool = False, latency_max_ms: float = 0.0, step_ms: float = 10.0,
                 active_only: bool = True, offsets: list[int] | None = None, spin_recover: float = 0.0, spin_weight: float = 2.0,
                 cursor_scale: tuple[float, float] = (1.0, 1.0), cursor_scale_p: float = 0.0):
        """`heat` > 0 samples rows in proportion to (strain of the approached object, normalised to mean 1) + press windows,
        mixed with uniform: weight = heat_floor + heat * score. 0 keeps uniform sampling."""
        self.exports = exports
        self.device = device
        self.offsets = sorted(offsets or list(range(stack)))
        self.span = self.offsets[-1] + 1
        self.stack = len(self.offsets)
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
        self.click_reweight = click_reweight and heat > 0
        self.active_only = active_only
        self.spin_recover = spin_recover
        self.spin_weight = spin_weight
        self._spin_setup()
        self.latency_max_ms = latency_max_ms
        self.step_ms = step_ms
        self.tail = max(click_horizon, int(math.ceil(latency_max_ms / step_ms)))
        if heat > 0:
            if any(e.strain is None for e in exports):
                raise ValueError("heat sampling needs strain data in every export's meta (run backfill_labels)")
            self.strain_scale = float(np.concatenate([e.strain for e in exports]).mean()) or 1.0
        self.gen = torch.Generator(device=device)
        self.gen.manual_seed(seed)
        self.rng = np.random.default_rng(seed)
        self.cursor_scale = cursor_scale if cursor_scale_p > 0 and cursor_scale != (1.0, 1.0) else None
        self.cursor_scale_p = cursor_scale_p
        self.cursor = GpuCursor(exports, device, max_scale=self.cursor_scale[1] if self.cursor_scale else 1.0)
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
        rows = np.arange(self.span - 1, e.count - self.tail, self.stride)
        return rows[e.active[rows]] if self.active_only else rows

    def _spin_setup(self) -> None:
        """Playfield to capture mapping per export, and autoplay's spin direction, radius and speed, for synthetic recovery examples."""
        from osu_auto.geometry import PlayArea

        scale, off = [], []
        for e in self.exports:
            area = PlayArea(*e.frames_meta["window_client"])
            scale.append(area.scale)
            off.append(area.playfield_to_capture(0.0, 0.0))
        self.pf_scale = torch.tensor(scale, dtype=torch.float32, device=self.device)
        self.pf_off = torch.tensor(off, dtype=torch.float32, device=self.device)
        dth, rad = [], []
        for e in self.exports:
            if not e.spinner.any():
                continue
            pf = (e.target_pf - AIM_LO) / (AIM_HI - AIM_LO) * np.array([512.0, 384.0]) - np.array([256.0, 192.0])
            idx = np.flatnonzero(e.spinner[1:] & e.spinner[:-1]) + 1
            ang = np.arctan2(pf[:, 1], pf[:, 0])
            keep = e.active[idx]
            dth.append(np.angle(np.exp(1j * (ang[idx] - ang[idx - 1])))[keep])
            rad.append(np.hypot(*pf[idx].T))
        if dth:
            d = np.concatenate(dth)
            self.spin_dir = SPIN_DIRECTION
            self.spin_step = float(np.median(np.abs(d))) if len(d) else 0.49
            self.spin_radius = float(np.median(np.concatenate(rad)))
        else:
            self.spin_dir, self.spin_step, self.spin_radius = SPIN_DIRECTION, 0.49, 50.0

    def _weights(self, e: FrameExport, rows: np.ndarray) -> np.ndarray:
        score = e.strain[rows] / self.strain_scale
        press = np.flatnonzero(e.press > 0.5)
        held = np.roll(e.down, 1) > 0.5
        near = np.zeros(e.count, dtype=np.float32)
        for i in press:
            near[max(i - 5, 0):i + 6] = np.maximum(near[max(i - 5, 0):i + 6], 3.0 if held[i] else 1.0)
        return (self.heat_floor + self.heat * (score + near[rows] + self.spin_weight * e.spinner[rows])).astype(np.float32)

    def _click_weights(self, e: FrameExport, rows: np.ndarray, weight: np.ndarray, edge: int = 10) -> np.ndarray:
        """Undo the spinner oversampling for the click loss: spinner rows count as often as they would without spin_weight,
        except near a spinner's start and end where presses and releases happen."""
        cw = np.ones(e.count, dtype=np.float32)
        spin = e.spinner[rows] > 0
        cw[rows[spin]] = (weight[spin] - self.heat * self.spin_weight) / weight[spin]
        s = e.spinner.astype(np.int8)
        for i in np.flatnonzero(np.diff(s) != 0):
            cw[max(i - edge, 0):i + edge + 1] = 1.0
        return cw

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
        ids, local, cxy, aim, press, down, spin, rows, weight, click_w = [], [], [], [], [], [], [], [], [], []
        base = 0
        for k, e in zip(group, exps):
            e.load_frames(frames[base:base + e.count].numpy())
            if self.heat > 0:
                weight.append(self._weights(e, self._rows(e)))
                if self.click_reweight:
                    click_w.append(self._click_weights(e, self._rows(e), weight[-1]))
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
                     t(weight, torch.float32) if weight else None, t(click_w, torch.float32) if click_w else None)

    def _to_device(self, c: Chunk) -> Chunk:
        d = self.device
        return Chunk(c.exports, c.frames.to(d), c.export_id.to(d), c.local.to(d), c.cursor_xy.to(d), c.aim.to(d),
                     c.press.to(d), c.down.to(d), c.spinner.to(d), c.rows.to(d), None if c.weight is None else c.weight.to(d),
                     None if c.click_w is None else c.click_w.to(d))

    def _batch(self, c: Chunk, rows: torch.Tensor):
        b = len(rows)
        s = self.stack
        offsets = -torch.tensor(self.offsets[::-1], device=self.device)
        idx = rows[:, None] + offsets[None]
        frames = c.frames[idx].float()
        export_id = c.export_id[rows]
        lo = c.local[rows] - self.span + 1
        in_spinner = c.spinner[rows]
        synth = in_spinner & (torch.rand(b, generator=self.gen, device=self.device) < self.spin_recover)
        real = ((torch.rand(b, generator=self.gen, device=self.device) < self.real_prob) | in_spinner) & ~synth
        max_lag = torch.where(in_spinner, torch.ones_like(lo), torch.full_like(lo, self.real_lag))
        lag = (torch.rand(b, generator=self.gen, device=self.device) * (max_lag + 1).float()).long().clamp(max=self.real_lag)
        lag = torch.minimum(lag, lo)
        cscale = None
        if self.cursor_scale:
            lo_s, hi_s = math.log(self.cursor_scale[0]), math.log(self.cursor_scale[1])
            drawn_scale = torch.exp(lo_s + (hi_s - lo_s) * torch.rand(b, generator=self.gen, device=self.device))
            cscale = torch.where(torch.rand(b, generator=self.gen, device=self.device) < self.cursor_scale_p, drawn_scale, torch.ones_like(drawn_scale))
        if bool(real.any()):
            sel = real.nonzero()[:, 0]
            pos_idx = (idx[sel] - lag[sel][:, None]).reshape(-1)
            xy = c.cursor_xy[pos_idx]
            flat = frames[sel].reshape(-1, *frames.shape[2:])
            drawn = self.cursor.composite(flat, export_id[sel].repeat_interleave(s), xy, None if cscale is None else cscale[sel].repeat_interleave(s))
            frames[sel] = drawn.view(len(sel), s, *frames.shape[2:])
        spin_theta = None
        if bool(synth.any()):
            sel = synth.nonzero()[:, 0]
            n = len(sel)
            g = lambda: torch.rand(n, generator=self.gen, device=self.device)
            theta0 = g() * 2 * math.pi
            radius = 10 + g() * 55
            speed = g() * self.spin_step
            back = torch.tensor(self.offsets[::-1], dtype=torch.float32, device=self.device)
            theta = theta0[:, None] - self.spin_dir * speed[:, None] * back[None]
            pf = torch.stack([256 + radius[:, None] * torch.cos(theta), 192 + radius[:, None] * torch.sin(theta)], dim=-1)
            eid = export_id[sel]
            xy = pf * self.pf_scale[eid][:, None, None] + self.pf_off[eid][:, None, :]
            flat = frames[sel].reshape(-1, *frames.shape[2:])
            drawn = self.cursor.composite(flat, eid.repeat_interleave(s), xy.reshape(-1, 2), None if cscale is None else cscale[sel].repeat_interleave(s))
            frames[sel] = drawn.view(n, s, *frames.shape[2:])
            spin_theta = (sel, theta0)
        if self.pool > 1:
            frames = F.avg_pool2d(frames, self.pool).add_(0.5).floor_()
        frames.div_(255.0)
        aux = torch.zeros(b, 2, device=self.device)
        if self.latency_max_ms > 0:
            lat = torch.rand(b, generator=self.gen, device=self.device) * self.latency_max_ms
            steps = lat / self.step_ms
            lo = steps.floor().long()
            frac = (steps - lo.float())[:, None]
            aim = c.aim[rows + lo] * (1 - frac) + c.aim[rows + lo + 1] * frac
            j = rows + torch.clamp(steps.round().long(), min=self.click_horizon)
            aux[:, 0] = lat / self.latency_max_ms
        else:
            aim = c.aim[rows]
            j = rows + self.click_horizon
        click = torch.stack([c.press[j], c.down[j]], dim=1)
        if c.click_w is not None:
            click = torch.cat([click, c.click_w[rows][:, None]], dim=1)
        if bool(in_spinner.any()):
            aim = self._spin_labels(c, rows, export_id, in_spinner, synth, spin_theta, lag, aux, aim)
        return frames, aux, aim, click

    def _spin_labels(self, c: Chunk, rows, export_id, in_spinner, synth, spin_theta, lag, aux, aim):
        """Spinner aim label: the angle of the cursor drawn in the newest frame, advanced by (1 + lag + latency) full-speed steps in the
        fixed direction, on autoplay's circle. Lead grows with frame staleness, so a stale frame still spins at full speed."""
        theta = torch.zeros(len(rows), device=self.device)
        sel = (in_spinner & ~synth).nonzero()[:, 0]
        if len(sel):
            eid = export_id[sel]
            xy = c.cursor_xy[rows[sel] - lag[sel]]
            pf = (xy - self.pf_off[eid]) / self.pf_scale[eid][:, None]
            theta[sel] = torch.atan2(pf[:, 1] - 192, pf[:, 0] - 256)
        ahead = 1.0 + lag.float()
        if spin_theta is not None:
            ssel, theta0 = spin_theta
            theta[ssel] = theta0
            ahead[ssel] = 1.0
        if self.latency_max_ms > 0:
            ahead = ahead + aux[:, 0] * self.latency_max_ms / self.step_ms
        th = theta + self.spin_dir * self.spin_step * ahead
        pf = torch.stack([256 + self.spin_radius * torch.cos(th), 192 + self.spin_radius * torch.sin(th)], dim=-1)
        target = AIM_LO + (AIM_HI - AIM_LO) * pf / torch.tensor([512.0, 384.0], device=self.device)
        aim = aim.clone()
        aim[in_spinner] = target[in_spinner]
        return aim

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
