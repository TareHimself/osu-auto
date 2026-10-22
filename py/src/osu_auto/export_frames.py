import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
from osu_auto.cursor import CursorSprite, gray
from osu_auto.framestore import FILE, FrameWriter
from osu_auto.geometry import PlayArea
from osu_auto.host import HostClient
from osu_auto.labels import Export, load_export

CHUNK = 500


def to_gray(bgra: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    return cv2.resize(gray(bgra), size, interpolation=cv2.INTER_AREA)


def calibrate_cursor(host: HostClient, map_path: str, diff: str | None, mods: list[str] | None, area: PlayArea) -> CursorSprite:
    """Capture the cursor over a white and a black backdrop and solve for its premultiplied layer and alpha.

    Uses a synthetic-input load so the cursor sits at the playfield centre on an empty field before the first object;
    the autoplay cursor starts off-screen there.
    """
    kwargs = {"map": map_path, "mods": list(mods or []), "agent": True, "input": "user", "clicks": "auto"}
    if diff:
        kwargs["diff"] = diff
    info = host.call("load", **kwargs)
    host.cursor(True)
    host.seek(float(info["t0"]))
    st = host.status()
    cap = area.capture
    hotspot = (st["cursor"][0] - cap.x, st["cursor"][1] - cap.y)
    host.backdrop(True)
    white, _ = host.grab()
    host.cursor(False)
    white_only, _ = host.grab()
    host.backdrop(False)
    black_only, _ = host.grab()
    host.cursor(True)
    black, _ = host.grab()
    return CursorSprite.from_calibration(white, white_only, black, black_only, hotspot)


def self_check(host: HostClient, export: Export, sprite: CursorSprite, area: PlayArea, size: tuple[int, int], factor: int) -> dict:
    """Determinism of the export path (stepping forward to a time twice), parity of the host's gray path with the cv2 path,
    and fidelity of the composited cursor. A long seek can leave Hidden fades in a different state than stepping does,
    so that difference is only reported."""
    labels = export.labels
    picks = np.linspace(len(labels) // 10, len(labels) - 1, 6).astype(int)
    j = int(picks[0])
    t = float(labels.t_ms[j])
    walk = labels.t_ms[j - 40:j + 1]

    host.cursor(False)
    host.seek(float(walk[0]))
    a = host.export_frames(walk, *size)[0][-1:]
    host.seek(t + 2000)
    host.seek(float(walk[0]))
    b = host.export_frames(walk, *size)[0][-1:]
    same = float(np.abs(a.astype(np.int16) - b.astype(np.int16)).mean())

    host.seek(t + 2000)
    jump = host.export_frames([t], *size)[0]
    jump_diff = float(np.abs(jump.astype(np.int16) - a.astype(np.int16)).mean())

    _, box, _ = host.seek_grab(t)
    parity = float(np.abs(to_gray(box, size).astype(np.int16) - jump[0].astype(np.int16)).mean())
    if same >= 1 or parity >= 1:
        for name, img in (("a", a[0]), ("b", b[0]), ("jump", jump[0]), ("cv2", to_gray(box, size))):
            cv2.imwrite(str(export.path / f"selfcheck_{name}.png"), img)

    fidelity = []
    for j in picks:
        tj = float(labels.t_ms[j])
        host.cursor(True)
        _, real_box, _ = host.seek_grab(tj)
        real = to_gray(real_box, size)
        host.cursor(False)
        none, _ = host.export_frames([tj], *size)
        cx, cy = area.playfield_to_capture(float(labels.x[j]), float(labels.y[j]))
        comp = sprite.composite(none[0], cx, cy, factor)
        diff = np.abs(comp.astype(np.int16) - real.astype(np.int16))
        fidelity.append((float(diff.mean()), int(diff.max()), float(np.abs(none[0].astype(np.int16) - real.astype(np.int16)).mean())))

    return {"same_time_diff": same, "jump_vs_walk_diff": jump_diff, "gray_parity": parity,
            "composite_mean_diff": float(np.mean([f[0] for f in fidelity])), "composite_max_diff": int(max(f[1] for f in fidelity)),
            "cursor_effect": float(np.mean([f[2] for f in fidelity]))}


def export_frames(host: HostClient, map_path: str, export_dir: str | Path, diff: str | None = None, osr: str | None = None,
                  mods: list[str] | None = None, size: tuple[int, int] = (160, 120), limit: int | None = None, log=print,
                  sprite: CursorSprite | None = None, strict: bool = True) -> dict:
    export = load_export(export_dir)
    labels = export.labels
    w, h = size
    n = len(labels) if limit is None else min(limit, len(labels))

    client = host.status()["client_size"]
    area = PlayArea(client[0], client[1])
    factor = round(area.capture.w / w)
    if sprite is None:
        sprite = calibrate_cursor(host, map_path, diff, None if osr else mods, area)

    info = host.load(map_path, diff=diff, osr=osr, mods=mods)
    log(f"loaded {info['title']} [{info['difficulty']}] mods={mods or []} client={client} -> {w}x{h} (factor {factor}), {n} steps, cursor sprite {sprite.premult.shape[1]}x{sprite.premult.shape[0]}")

    checks = self_check(host, export, sprite, area, size, factor)
    log(f"self-check: {json.dumps(checks)}")
    if checks["same_time_diff"] >= 1:
        if strict:
            raise RuntimeError("readback is not deterministic across seeks")
        log("warning: readback differs across seeks (spinner state depends on history); export walks forward only, so kept")
    if checks["gray_parity"] >= 1:
        raise RuntimeError("host grayscale path disagrees with cv2")
    if checks["composite_mean_diff"] > max(0.5, checks["cursor_effect"] * 0.25):
        raise RuntimeError("composited cursor does not reproduce the rendered cursor")

    host.cursor(False)
    for stale in ("frames_nocursor.npy", "frames.npy"):
        (export.path / stale).unlink(missing_ok=True)
    writer = FrameWriter(export.path / FILE, h, w)
    clocks = np.zeros(n, dtype=np.float64)

    t0 = time.perf_counter()
    for start in range(0, n, CHUNK):
        stop = min(start + CHUNK, n)
        block, clocks[start:stop] = host.export_frames(labels.t_ms[start:stop], w, h)
        writer.append(block)
        rate = stop / (time.perf_counter() - t0)
        log(f"{stop}/{n} t={labels.t_ms[stop - 1]:.0f} {rate:.1f} steps/s eta {(n - stop) / max(rate, 1e-6):.0f} s")

    store = writer.close()
    np.save(export.path / "frame_clock_ms.npy", clocks)
    sprite.save(export.path / "cursor.npz")
    host.cursor(True)

    bad = int(np.sum(np.abs(clocks - labels.t_ms[:n]) > 0.5))
    meta = dict(export.meta)
    meta["frames"] = {**store, "count": n, "width": w, "height": h, "gray": True, "box": "4:3 play area", "cursor_hidden": True,
                      "cursor": "cursor.npz", "factor": factor, "clock_mismatches": bad, "window_client": list(client), "capture": "readback",
                      "self_check": checks, "steps_per_s": n / (time.perf_counter() - t0)}
    (export.path / "meta.json").write_text(json.dumps(meta, indent=2))
    log(f"done: {n} frames -> {export.path / FILE} ({writer.offset / 2**20:.1f} MB), clock mismatches={bad}, {meta['frames']['steps_per_s']:.1f} steps/s")
    return meta["frames"]


def main() -> None:
    p = argparse.ArgumentParser(description="Export one grayscale frame per label step with the host rendering, cropping and downsampling in-process")
    p.add_argument("map_path")
    p.add_argument("--export", required=True, help="label export dir produced by the host export-labels command")
    p.add_argument("--diff")
    p.add_argument("--osr")
    p.add_argument("--mods", default="", help="semicolon-separated mod specs, e.g. 'HD;DA:circle_size=4'")
    p.add_argument("--size", default="160x120", help="stored frame size WxH, 4:3")
    p.add_argument("--limit", type=int, default=None, help="only export the first N steps")
    p.add_argument("--endpoint", default="tcp://127.0.0.1:5555")
    args = p.parse_args()

    w, h = (int(v) for v in args.size.split("x"))
    mods = [m for m in args.mods.split(";") if m.strip()]

    with HostClient(args.endpoint) as host:
        export_frames(host, args.map_path, args.export, diff=args.diff, osr=args.osr, mods=mods, size=(w, h), limit=args.limit)
        host.quit()


if __name__ == "__main__":
    main()
