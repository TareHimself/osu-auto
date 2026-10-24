"""Synthetic beatmaps: write the label distribution we want (positions, jump vectors, tempo, slider bodies) and let autoplay label it."""
from __future__ import annotations

import argparse
import io
import json
import math
import struct
import wave
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from osu_auto.paths import BEATMAPS

OUT = BEATMAPS / "synthetic"
W, H = 512, 384
LEAD_IN_MS = 2000
NEW_COMBO = 4


@dataclass
class HitObject:
    x: float
    y: float
    t: float
    kind: str = "circle"
    body: list[tuple[float, float]] = field(default_factory=list)
    curve: str = "L"
    repeats: int = 1
    length: float = 0.0
    sv_mult: float = 1.0
    end_t: float = 0.0
    end_xy: tuple[float, float] | None = None

    @property
    def end(self) -> tuple[float, float]:
        return self.end_xy or (self.x, self.y)


class Histo:
    """Draws the bin with the fewest draws so far, with a little noise so ties break randomly: every bin ends up equally used."""

    def __init__(self, edges: np.ndarray, rng: np.random.Generator):
        self.edges = edges
        self.counts = np.zeros(len(edges) - 1)
        self.rng = rng

    def draw(self) -> float:
        k = int(np.argmin(self.counts + self.rng.random(len(self.counts)) * 0.5))
        self.counts[k] += 1
        return float(self.rng.uniform(self.edges[k], self.edges[k + 1]))


class Histo2:
    def __init__(self, xedges: np.ndarray, yedges: np.ndarray, rng: np.random.Generator):
        self.xe, self.ye = xedges, yedges
        self.counts = np.zeros((len(xedges) - 1, len(yedges) - 1))
        self.rng = rng

    def draw(self, mask: np.ndarray | None = None) -> tuple[float, float]:
        c = self.counts + self.rng.random(self.counts.shape) * 0.5
        if mask is not None:
            c = np.where(mask, c, np.inf)
        i, j = np.unravel_index(int(np.argmin(c)), c.shape)
        self.counts[i, j] += 1
        return float(self.rng.uniform(self.xe[i], self.xe[i + 1])), float(self.rng.uniform(self.ye[j], self.ye[j + 1]))


def preempt_ms(ar: float) -> float:
    return 1200 + 600 * (5 - ar) / 5 if ar < 5 else 1200 - 750 * (ar - 5) / 5


def radius(cs: float) -> float:
    return 54.4 - 4.48 * cs


class Generator:
    MAX_SPEED = 4.0

    def __init__(self, seed: int, cs: float, ar: float, sv: float = 1.4, bpm: float = 180.0):
        self.rng = np.random.default_rng(seed)
        self.cs, self.ar, self.sv, self.bpm = cs, ar, sv, bpm
        self.beat_ms = 60000.0 / bpm
        self.objects: list[HitObject] = []
        self.t = float(LEAD_IN_MS)
        self.pos = Histo2(np.linspace(0, W, 17), np.linspace(0, H, 13), self.rng)
        self.jump = Histo2(np.linspace(0, 600, 13), np.linspace(-math.pi, math.pi, 17), self.rng)
        self.gap = Histo(np.geomspace(60, 900, 13), self.rng)
        self.slen = Histo(np.linspace(40, 600, 12), self.rng)
        self.svel = Histo(np.linspace(0.5, 3.0, 8), self.rng)
        self.spin = Histo(np.linspace(500, 4000, 8), self.rng)

    def last_end(self) -> tuple[float, float]:
        return self.objects[-1].end if self.objects else (W / 2, H / 2)

    def next_pos(self, gap: float) -> tuple[float, float]:
        """The reachable (distance, angle) bin whose landing cell plus own count is lowest, so jumps and positions flatten together."""
        x0, y0 = self.last_end()
        best, best_score = None, math.inf
        jc = self.jump.counts + self.rng.random(self.jump.counts.shape) * 0.5
        for i in range(jc.shape[0]):
            for j in range(jc.shape[1]):
                d = self.rng.uniform(self.jump.xe[i], self.jump.xe[i + 1])
                a = self.rng.uniform(self.jump.ye[j], self.jump.ye[j + 1])
                if d / gap > self.MAX_SPEED:
                    continue
                x, y = x0 + d * math.cos(a), y0 + d * math.sin(a)
                if not (0 <= x <= W and 0 <= y <= H):
                    continue
                cx, cy = min(int(x / W * 16), 15), min(int(y / H * 12), 11)
                score = jc[i, j] + 2 * self.pos.counts[cx, cy]
                if score < best_score:
                    best, best_score = (i, j, x, y, cx, cy), score
        if best is None:
            return self.pos.draw()
        i, j, x, y, cx, cy = best
        self.jump.counts[i, j] += 1
        self.pos.counts[cx, cy] += 1
        return x, y

    def circle(self, gap: float | None = None) -> HitObject:
        gap = self.gap.draw() if gap is None else gap
        self.t += gap
        x, y = self.next_pos(gap)
        o = HitObject(x, y, self.t)
        o.end_t = self.t
        self.objects.append(o)
        return o

    def slider(self, gap: float | None = None) -> HitObject:
        gap = self.gap.draw() if gap is None else gap
        self.t += gap
        x, y = self.next_pos(gap)
        length = self.slen.draw()
        vel = self.svel.draw()
        curve = self.rng.choice(["L", "B", "P"], p=[0.3, 0.45, 0.25])
        body = self.body(x, y, length, curve)
        repeats = int(self.rng.choice([1, 1, 1, 2, 3, 4]))
        o = HitObject(x, y, self.t, "slider", body, curve, repeats, length, vel)
        base_vel = 100 * self.sv / self.beat_ms
        o.end_t = self.t + length / (base_vel * vel) * repeats
        o.end_xy = body[-1] if repeats % 2 == 1 else (x, y)
        self.t = o.end_t
        self.objects.append(o)
        return o

    def body(self, x: float, y: float, length: float, curve: str) -> list[tuple[float, float]]:
        """Control points after the head whose rendered curve has about `length` px and stays inside the field."""
        for _ in range(64):
            a = self.rng.uniform(-math.pi, math.pi)
            if curve == "L":
                pts = [(x + length * math.cos(a), y + length * math.sin(a))]
            elif curve == "P":
                bend = self.rng.uniform(-1.2, 1.2)
                mid = (x + length / 2 * math.cos(a) + length * 0.3 * bend * math.cos(a + math.pi / 2),
                       y + length / 2 * math.sin(a) + length * 0.3 * bend * math.sin(a + math.pi / 2))
                end = (x + length * math.cos(a), y + length * math.sin(a))
                pts = [mid, end]
            else:
                n = int(self.rng.integers(2, 6))
                pts, px, py, ang = [], x, y, a
                for _ in range(n):
                    ang += self.rng.uniform(-1.5, 1.5)
                    step = length / n * self.rng.uniform(0.7, 1.4)
                    px, py = px + step * math.cos(ang), py + step * math.sin(ang)
                    pts.append((px, py))
            if all(0 <= px <= W and 0 <= py <= H for px, py in pts):
                return pts
        return [(min(max(x + length, 0), W), y)]

    def chain(self) -> None:
        """Sliders back to back with 30 to 150 ms between one's end and the next head: autoplay swaps keys without releasing."""
        n = int(self.rng.integers(2, 7))
        long = self.slen
        self.slen = Histo(np.linspace(40, 220, 7), self.rng)
        for _ in range(n):
            self.slider(float(self.rng.uniform(30, 150)))
        self.slen = long

    def pattern(self) -> None:
        """A shape jump: regular polygon, star, zigzag or cross at constant spacing and tempo, random size, rotation and position."""
        kind = str(self.rng.choice(["polygon", "star", "zigzag", "cross"], p=[0.4, 0.2, 0.3, 0.1]))
        spacing = float(self.rng.uniform(80, 320))
        tempo = float(self.rng.uniform(max(90, spacing / self.MAX_SPEED), 360))
        rot = self.rng.uniform(0, 2 * math.pi)
        if kind == "polygon":
            n = int(self.rng.integers(3, 7))
            r = spacing / (2 * math.sin(math.pi / n))
            pts = [(r * math.cos(rot + 2 * math.pi * k / n), r * math.sin(rot + 2 * math.pi * k / n)) for k in range(n)]
            pts = pts * int(self.rng.integers(1, 3)) + pts[:1]
        elif kind == "star":
            n = 5
            r = spacing / (2 * math.sin(2 * math.pi / n))
            pts = [(r * math.cos(rot + 4 * math.pi * k / n), r * math.sin(rot + 4 * math.pi * k / n)) for k in range(n)] + [(r * math.cos(rot), r * math.sin(rot))]
        elif kind == "zigzag":
            n = int(self.rng.integers(4, 10))
            pts = [((k % 2) * spacing * math.cos(rot), (k % 2) * spacing * math.sin(rot)) for k in range(n)]
            drift = self.rng.uniform(0, 60)
            pts = [(x + k * drift * math.cos(rot + math.pi / 2), y + k * drift * math.sin(rot + math.pi / 2)) for k, (x, y) in enumerate(pts)]
        else:
            r = spacing / 2
            pts = [(r, 0), (-r, 0), (0, r), (0, -r), (r, 0)]
            pts = [(x * math.cos(rot) - y * math.sin(rot), x * math.sin(rot) + y * math.cos(rot)) for x, y in pts]
        xs, ys = zip(*pts)
        w, h = max(xs) - min(xs), max(ys) - min(ys)
        if w > W or h > H:
            return
        ox = self.rng.uniform(-min(xs), W - max(xs))
        oy = self.rng.uniform(-min(ys), H - max(ys))
        self.t += self.gap.draw()
        for k, (x, y) in enumerate(pts):
            o = HitObject(ox + x, oy + y, self.t)
            o.end_t = self.t
            self.objects.append(o)
            if k + 1 < len(pts):
                self.t += tempo

    def stream(self) -> None:
        n = int(self.rng.integers(3, 17))
        spacing = float(self.rng.uniform(10, 80))
        tempo = float(self.rng.uniform(50, 120))
        x, y = self.next_pos(self.gap.draw())
        self.t += self.gap.draw()
        ang = self.rng.uniform(-math.pi, math.pi)
        bend = self.rng.uniform(-0.3, 0.3)
        for k in range(n):
            o = HitObject(min(max(x, 0), W), min(max(y, 0), H), self.t)
            o.end_t = self.t
            self.objects.append(o)
            ang += bend
            x, y = x + spacing * math.cos(ang), y + spacing * math.sin(ang)
            self.t += tempo

    def stack(self) -> None:
        n = int(self.rng.integers(2, 6))
        tempo = float(self.rng.uniform(60, 150))
        gap = self.gap.draw()
        self.t += gap
        x, y = self.next_pos(gap)
        for _ in range(n):
            o = HitObject(x, y, self.t)
            o.end_t = self.t
            self.objects.append(o)
            self.t += tempo

    def spinner(self) -> None:
        self.t += self.gap.draw()
        length = self.spin.draw()
        o = HitObject(W / 2, H / 2, self.t, "spinner")
        o.end_t = self.t + length
        self.t = o.end_t
        self.objects.append(o)

    def fill(self, family: str, seconds: float) -> None:
        end = LEAD_IN_MS + seconds * 1000
        weights = {
            "jumps": {"circle": 1.0},
            "sliders": {"slider": 1.0},
            "chains": {"chain": 0.7, "slider": 0.2, "circle": 0.1},
            "patterns": {"pattern": 0.8, "circle": 0.2},
            "spinners": {"spinner": 0.6, "circle": 0.2, "slider": 0.2},
            "streams": {"stream": 0.6, "stack": 0.4},
            "dense": {"circle": 0.5, "stack": 0.25, "stream": 0.25},
            "mixed": {"circle": 0.35, "slider": 0.22, "pattern": 0.1, "chain": 0.1, "stream": 0.08, "stack": 0.08, "spinner": 0.07},
        }[family]
        names, probs = list(weights), np.array(list(weights.values()))
        if family == "spinners":
            self.spin = Histo(np.linspace(1000, 8000, 8), self.rng)
        if family == "dense":
            self.gap = Histo(np.linspace(60, 250, 8), self.rng)
        while self.t < end:
            kind = str(self.rng.choice(names, p=probs / probs.sum()))
            getattr(self, kind)()

    def osu_text(self, name: str, od: float = 8.0) -> str:
        lines = [
            "osu file format v14", "", "[General]", "AudioFilename: audio.wav", "AudioLeadIn: 0", "Mode: 0", "StackLeniency: 0.7", "",
            "[Metadata]", f"Title:synthetic {name}", f"TitleUnicode:synthetic {name}", "Artist:osu-auto", "ArtistUnicode:osu-auto",
            "Creator:osu-auto", f"Version:{name}", "Source:", "Tags:synthetic", "BeatmapID:0", "BeatmapSetID:-1", "",
            "[Difficulty]", "HPDrainRate:5", f"CircleSize:{self.cs}", f"OverallDifficulty:{od}", f"ApproachRate:{self.ar}",
            f"SliderMultiplier:{self.sv}", "SliderTickRate:1", "",
            "[Events]", "//Background and Video events", "//Break Periods", "//Storyboard Layer 0 (Background)", "",
            "[TimingPoints]", f"0,{self.beat_ms:.6f},4,1,0,100,1,0",
        ]
        for o in self.objects:
            if o.kind == "slider" and abs(o.sv_mult - 1.0) > 1e-6:
                lines.append(f"{int(o.t)},{-100 / o.sv_mult:.4f},4,1,0,100,0,0")
        lines += ["", "[Colours]", "Combo1 : 255,96,96", "Combo2 : 96,255,96", "Combo3 : 96,160,255", "Combo4 : 255,255,96", "", "[HitObjects]"]
        for i, o in enumerate(self.objects):
            combo = NEW_COMBO if i % int(self.rng.integers(4, 9)) == 0 else 0
            if o.kind == "circle":
                lines.append(f"{int(round(o.x))},{int(round(o.y))},{int(o.t)},{1 | combo},0,0:0:0:0:")
            elif o.kind == "slider":
                pts = "|".join(f"{int(round(px))}:{int(round(py))}" for px, py in o.body)
                lines.append(f"{int(round(o.x))},{int(round(o.y))},{int(o.t)},{2 | combo},0,{o.curve}|{pts},{o.repeats},{o.length:.2f}")
            else:
                lines.append(f"256,192,{int(o.t)},{8 | combo},0,{int(o.end_t)},0:0:0:0:")
        return "\n".join(lines) + "\n"


def silent_wav(ms: float) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(8000)
        w.writeframes(struct.pack("<h", 0) * int(8000 * ms / 1000))
    return buf.getvalue()


def write_osz(name: str, osu: str, length_ms: float) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{name}.osz"
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr(f"osu-auto - synthetic {name} (osu-auto) [{name}].osu", osu)
        z.writestr("audio.wav", silent_wav(length_ms + 3000))
    return path


def generate(family: str, cs: float, ar: float, seconds: float, seed: int) -> tuple[Path, str, dict]:
    g = Generator(seed, cs, ar)
    g.fill(family, seconds)
    name = f"{family}-cs{cs:g}-ar{ar:g}-s{seed}"
    path = write_osz(name, g.osu_text(name), g.t)
    kinds = {}
    for o in g.objects:
        kinds[o.kind] = kinds.get(o.kind, 0) + 1
    return path, name, {"objects": len(g.objects), "kinds": kinds, "seconds": round((g.t - LEAD_IN_MS) / 1000)}


def main() -> None:
    p = argparse.ArgumentParser(description="Generate synthetic osu! beatmaps with flat position, jump and tempo distributions")
    p.add_argument("--families", default="jumps,sliders,streams,dense,mixed")
    p.add_argument("--cs", default="3,4,5")
    p.add_argument("--ar", default="8,9,10")
    p.add_argument("--seconds", type=float, default=90)
    p.add_argument("--seeds", type=int, default=1)
    p.add_argument("--jobs", default=None, help="write a job list (json) for export_jobs")
    args = p.parse_args()
    jobs = []
    for family in args.families.split(","):
        for cs in (float(v) for v in args.cs.split(",")):
            for ar in (float(v) for v in args.ar.split(",")):
                for seed in range(args.seeds):
                    path, name, info = generate(family, cs, ar, args.seconds, seed)
                    print(f"{path.name}: {info}")
                    jobs.append({"map": f"synthetic/{path.name}", "diff": name, "cs": cs, "ar": ar, "title": f"synthetic {name}", "family": family})
    if args.jobs:
        Path(args.jobs).write_text(json.dumps(jobs, indent=2))
        print(f"{len(jobs)} jobs -> {args.jobs}")


if __name__ == "__main__":
    main()
