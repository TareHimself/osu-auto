import argparse
import io
import json
import re
import urllib.request
import zipfile
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
BEATMAPS = ROOT / "data/beatmaps"
MIRROR = "https://osu.direct/api"
USER_AGENT = "osu-auto/0.1 (github.com/TareHimself)"


def fetch(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read()


def search(pages: int, amount: int = 100, sort: str = "play_count:desc") -> list[dict]:
    """Ranked osu! standard beatmap sets from the mirror's search, most played first."""
    sets = []
    for page in range(pages):
        batch = json.loads(fetch(f"{MIRROR}/v2/search?mode=0&status=1&amount={amount}&offset={page * amount}&sort={sort}"))
        if not batch:
            break
        sets.extend(batch)
    return sets


def candidates(sets: list[dict], stars: tuple[float, float], cs: tuple[float, float], ar: tuple[float, float],
               length: tuple[int, int]) -> list[dict]:
    out = []
    for s in sets:
        for b in s.get("beatmaps", []):
            if b.get("mode_int", 0) != 0:
                continue
            if not (stars[0] <= b["difficulty_rating"] < stars[1] and cs[0] <= b["cs"] <= cs[1] and ar[0] <= b["ar"] <= ar[1]
                    and length[0] <= b["total_length"] <= length[1]):
                continue
            out.append({"set_id": s["id"], "title": s["title"], "artist": s["artist"], "play_count": s.get("play_count", 0),
                        "diff": b["version"], "stars": round(b["difficulty_rating"], 2), "cs": b["cs"], "ar": b["ar"], "od": b.get("accuracy"),
                        "bpm": b.get("bpm"), "length": b["total_length"]})
    return out


def pick(cands: list[dict], count: int, per_set: int = 1, per_cell: int = 1) -> list[dict]:
    """Up to `per_cell` most-played diffs per (stars, CS, AR) cell from different songs, then thin evenly to `count`."""
    cells: dict[tuple[int, int, int], list[dict]] = defaultdict(list)
    for c in cands:
        cells[(int(c["stars"]), round(c["cs"]), round(c["ar"]))].append(c)
    chosen = []
    used: dict[int, int] = defaultdict(int)
    for key in sorted(cells):
        taken = 0
        for c in sorted(cells[key], key=lambda c: -c["play_count"]):
            if taken >= per_cell:
                break
            if used[c["set_id"]] < per_set:
                chosen.append(c)
                used[c["set_id"]] += 1
                taken += 1
    if len(chosen) > count:
        step = len(chosen) / count
        chosen = [chosen[int(i * step)] for i in range(count)]
    return chosen


def jump_scores(osz: Path) -> dict[str, float]:
    """Per difficulty: 90th percentile distance (osu! pixels) between consecutive objects that are at most 250 ms apart."""
    return {k: v[0] for k, v in diff_scores(osz).items()}


def spinner_scores(osz: Path) -> dict[str, float]:
    """Per difficulty: fraction of the map's length spent inside spinners."""
    return {k: v[1] for k, v in diff_scores(osz).items()}


def stream_scores(osz: Path) -> dict[str, float]:
    """Per difficulty: fraction of consecutive objects under 100 ms apart at short spacing (streams)."""
    return {k: v[2] for k, v in diff_scores(osz).items()}


def diff_scores(osz: Path) -> dict[str, tuple[float, float, float]]:
    """Per difficulty: (jump p90 spacing in px, spinner time fraction, stream fraction)."""
    import numpy as np

    out = {}
    with zipfile.ZipFile(osz) as z:
        for name in z.namelist():
            if not name.lower().endswith(".osu"):
                continue
            text = z.read(name).decode("utf-8", errors="replace")
            mode = re.search(r"^Mode:\s*(\d+)", text, re.M)
            if mode and mode.group(1) != "0":
                continue
            version = re.search(r"^Version:\s*(.*)$", text, re.M)
            section = text.split("[HitObjects]", 1)
            if not version or len(section) < 2:
                continue
            pts, spin_ms = [], 0.0
            for line in section[1].splitlines():
                parts = line.split(",")
                if len(parts) < 5:
                    continue
                try:
                    x, y, t, kind = float(parts[0]), float(parts[1]), float(parts[2]), int(parts[3])
                except ValueError:
                    continue
                if kind & 8:
                    try:
                        spin_ms += max(float(parts[5]) - t, 0.0)
                    except (ValueError, IndexError):
                        pass
                    continue
                pts.append((x, y, t))
            if len(pts) < 20:
                continue
            a = np.array(pts)
            d = np.hypot(np.diff(a[:, 0]), np.diff(a[:, 1]))
            dt = np.diff(a[:, 2])
            fast = d[(dt > 0) & (dt <= 250)]
            length = max(a[-1, 2] - a[0, 2], 1.0)
            stream = float(np.mean((dt > 0) & (dt <= 100) & (d <= 150)))
            out[version.group(1).strip()] = (float(np.percentile(fast, 90)) if len(fast) >= 10 else 0.0, float(spin_ms / length), stream)
    return out


def download(set_id: int) -> Path:
    path = BEATMAPS / f"{set_id}.osz"
    if not path.exists():
        BEATMAPS.mkdir(parents=True, exist_ok=True)
        path.write_bytes(fetch(f"{MIRROR}/d/{set_id}?noVideo=1"))
    return path


def main() -> None:
    p = argparse.ArgumentParser(description="Pick ranked maps that cover a grid of star rating, circle size and approach rate, download them, and write a job list")
    p.add_argument("--count", type=int, default=12)
    p.add_argument("--pages", type=int, default=10, help="search pages of 100 most-played sets to consider")
    p.add_argument("--stars", default="3,6.5")
    p.add_argument("--cs", default="3,5.5")
    p.add_argument("--ar", default="8,10.3")
    p.add_argument("--length", default="60,240", help="seconds")
    p.add_argument("--per-set", type=int, default=1)
    p.add_argument("--per-cell", type=int, default=1, help="diffs per (stars, CS, AR) cell")
    p.add_argument("--mods", default="", help="extra job variants per pick, semicolon-separated specs, e.g. 'HD;HR'")
    p.add_argument("--jumps", type=int, default=0, help="download this many candidate sets, score diffs by spacing, keep the top --count")
    p.add_argument("--spinners", type=int, default=0, help="download this many candidate sets, score diffs by spinner time, keep the top --count")
    p.add_argument("--streams", type=int, default=0, help="download this many candidate sets, score diffs by stream density, keep the top --count")
    p.add_argument("--download", action="store_true")
    p.add_argument("--jobs", default=None, help="write the job list here (json)")
    args = p.parse_args()

    rng = lambda s: tuple(float(v) for v in s.split(","))
    sets = search(args.pages)
    cands = candidates(sets, rng(args.stars), rng(args.cs), rng(args.ar), tuple(int(v) for v in args.length.split(",")))
    if args.jumps or args.spinners or args.streams:
        by_set: dict[int, list[dict]] = defaultdict(list)
        for c in cands:
            by_set[c["set_id"]].append(c)
        scored = []
        for set_id in sorted(by_set, key=lambda i: -by_set[i][0]["play_count"])[:(args.jumps or args.spinners or args.streams)]:
            scores = diff_scores(download(set_id))
            for c in by_set[set_id]:
                c["jump"], c["spin"], c["stream"] = scores.get(c["diff"], (0.0, 0.0, 0.0))
                scored.append(c)
        key = "jump" if args.jumps else "spin" if args.spinners else "stream"
        scored.sort(key=lambda c: -c[key])
        chosen, used = [], set()
        for c in scored:
            if c["set_id"] not in used and len(chosen) < args.count:
                chosen.append(c)
                used.add(c["set_id"])
    else:
        chosen = pick(cands, args.count, args.per_set, args.per_cell)
    print(f"{len(sets)} sets searched, {len(cands)} candidate diffs, {len(chosen)} chosen")
    print(f"{'set':>8} {'stars':>5} {'cs':>4} {'ar':>4} {'bpm':>5} {'len':>4}  title [diff]")
    for c in chosen:
        print(f"{c['set_id']:>8} {c['stars']:>5.2f} {c['cs']:>4.1f} {c['ar']:>4.1f} {c['bpm']:>5.0f} {c['length']:>4d}  {c['artist']} - {c['title']} [{c['diff']}]" + (f"  jump p90 {c['jump']:.0f} px  spinner {c['spin'] * 100:.0f}%  stream {c['stream'] * 100:.0f}%" if "jump" in c else ""))

    if args.download:
        for c in chosen:
            print(f"downloading {c['set_id']} ...", end=" ", flush=True)
            print(download(c["set_id"]).name)

    if args.jobs:
        mods = [m for m in args.mods.split(";") if m.strip()]
        jobs = []
        for c in chosen:
            base = {"map": f"{c['set_id']}.osz", "diff": c["diff"], "stars": c["stars"], "cs": c["cs"], "ar": c["ar"], "title": c["title"]}
            jobs.append(base)
            for m in mods:
                jobs.append({**base, "mods": [m]})
        Path(args.jobs).write_text(json.dumps(jobs, indent=2))
        print(f"{len(jobs)} jobs -> {args.jobs}")


if __name__ == "__main__":
    main()
