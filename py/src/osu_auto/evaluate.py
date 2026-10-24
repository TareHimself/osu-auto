import argparse
import json
import time
from pathlib import Path
from collections import deque

import numpy as np
import torch

from osu_auto.env import OsuEnv
from osu_auto.host import HostClient
from osu_auto.keys import KeyExecutor, timing_report
from osu_auto.models import load_checkpoint
from osu_auto.paths import MODELS
from osu_auto.targets import decode_to_playfield
from osu_auto.tracking import log_eval


def object_starts(map_path: str, diff: str | None) -> np.ndarray:
    """Hit object start times of the chosen difficulty (lowest-named when none), read straight from the .osz.
    Spinners also contribute a time every second through to their end, so gap skipping never jumps over one."""
    import re
    import zipfile

    with zipfile.ZipFile(map_path) as z:
        names = [n for n in z.namelist() if n.endswith(".osu")]
        chosen = names[0]
        for n in names:
            text = z.read(n).decode("utf-8", "replace")
            m = re.search(r"^Version:\s*(.*)$", text, re.M)
            if diff and m and m.group(1).strip().lower() == diff.lower():
                chosen = n
                break
        text = z.read(chosen).decode("utf-8", "replace")
    starts = []
    for line in text.split("[HitObjects]", 1)[1].splitlines():
        parts = line.split(",")
        if len(parts) >= 4:
            try:
                starts.append(float(parts[2]))
                if int(parts[3]) & 8 and len(parts) >= 6:
                    starts.extend(np.arange(float(parts[2]), float(parts[5]), 1000.0)[1:].tolist() + [float(parts[5])])
            except ValueError:
                pass
    return np.array(sorted(starts))


def main() -> None:
    p = argparse.ArgumentParser(description="Closed-loop evaluation in the host: the model moves the cursor and presses, the game judges")
    p.add_argument("map_path")
    p.add_argument("--diff")
    p.add_argument("--model", default=str(MODELS / "current/joint_best.pt"))
    p.add_argument("--export", default=None, help="label export of this map, for press timing diagnostics")
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--stop-misses", type=int, default=0)
    p.add_argument("--start-ms", type=float, default=None)
    p.add_argument("--press-threshold", type=float, default=0.5)
    p.add_argument("--hold-threshold", type=float, default=0.5)
    p.add_argument("--min-press-gap", type=int, default=2)
    p.add_argument("--latency", type=float, default=None, help="latency in ms fed to a latency-conditioned model (default: delay * step)")
    p.add_argument("--delay", type=int, default=0, help="apply each action this many steps late, to simulate pipeline latency")
    p.add_argument("--wandb", default=None, help="also log the result to the model's wandb run")
    p.add_argument("--release", type=float, default=0.25, help="once a key is down, release only when p_down falls below this")
    p.add_argument("--no-edge", action="store_true", help="do not take a rising edge of the hold probability as a press")
    p.add_argument("--trace", default=None, help="save per-step t, x, y, p_press, p_down, k1, k2 to this .npz")
    p.add_argument("--skip-gaps", type=float, default=4000, help="jump over stretches with no object start for this many ms (breaks, intro); 0 = play everything")
    p.add_argument("--endpoint", default="tcp://127.0.0.1:5555")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ck = load_checkpoint(args.model, device)
    h, w = ck["frame_hw"]
    offsets = ck.get("offsets") or list(range(ck["stack"]))
    span = max(offsets) + 1

    def gather(hist):
        return np.stack([hist[-1 - k] for k in reversed(offsets)])

    with HostClient(args.endpoint, timeout_ms=600_000) as host:
        env = OsuEnv(host, None, obs_size=(w, h))
        res = env.reset(args.map_path, diff=args.diff, clicks="agent", mods=[], start_ms=args.start_ms)
        stack = deque([res.obs] * span, maxlen=span)
        starts = object_starts(args.map_path, args.diff) if args.skip_gaps else np.array([])
        skipped_ms = 0.0
        keys = KeyExecutor(args.min_press_gap, args.press_threshold, 1.0, args.release)
        zeros = torch.zeros(1, 2, device=device)
        latency_max = float(ck.get("latency_max_ms", 0.0))
        latency = args.delay * env.step_ms if args.latency is None else args.latency
        if latency_max:
            zeros[0, 0] = min(latency, latency_max) / latency_max
            print(f"latency input {latency:.0f} ms (model max {latency_max:.0f})")
        pending = deque()
        hits: dict[str, int] = {}
        press_times: list[float] = []
        trace: list[tuple] = []
        steps = 0
        consecutive_misses = 0
        stopped = ""
        t0 = time.perf_counter()

        while not env.done and (args.max_steps is None or steps < args.max_steps):
            frames = torch.from_numpy(gather(stack)).float().div_(255.0).unsqueeze(0).to(device)
            with torch.no_grad():
                o = model(frames, zeros)[0]
            nx, ny = o[:2].tolist()
            p_press, p_hold = torch.sigmoid(o[2:]).tolist()
            x, y = decode_to_playfield(nx, ny)
            k1, k2 = keys.apply(p_press, p_hold > args.hold_threshold, None if args.no_edge else p_hold)
            if keys.pressed:
                press_times.append(env.t + env.step_ms)

            pending.append((x, y, k1, k2))
            if len(pending) <= args.delay:
                x, y, k1, k2 = pending[0][0], pending[0][1], False, False
            else:
                x, y, k1, k2 = pending.popleft()
            res = env.step(x, y, k1, k2)
            trace.append((env.t, x, y, p_press, p_hold, k1, k2))
            stack.append(res.obs)
            if len(starts):
                nxt = starts[np.searchsorted(starts, env.t)] if np.searchsorted(starts, env.t) < len(starts) else None
                if nxt is not None and nxt - env.t > args.skip_gaps:
                    target = nxt - 1500
                    skipped_ms += target - env.t
                    env.t = target - env.step_ms
                    pending.clear()
                    res = env.step(x, y, False, False)
                    stack = deque([res.obs] * span, maxlen=span)
            for k, v in res.hits.items():
                hits[k] = hits.get(k, 0) + v
            misses = res.hits.get("Miss", 0)
            scored = res.hits.get("Great", 0) or res.hits.get("Ok", 0) or res.hits.get("Meh", 0)
            consecutive_misses = consecutive_misses + misses if misses else (0 if scored else consecutive_misses)
            steps += 1
            if steps % 500 == 0:
                print(f"t={res.t_ms:.0f} combo={res.combo} acc={res.accuracy:.4f} presses={len(press_times)} hits={hits} {steps / (time.perf_counter() - t0):.1f} steps/s")
            if args.stop_misses and consecutive_misses >= args.stop_misses:
                stopped = f"{consecutive_misses} consecutive misses"
                break

        judged = sum(v for k, v in hits.items() if k in ("Great", "Ok", "Meh", "Miss"))
        circle_acc = (hits.get("Great", 0) + hits.get("Ok", 0) + hits.get("Meh", 0)) / max(judged, 1)
        print(f"final: t={res.t_ms:.0f} combo={res.combo} max_combo? circle_hit_rate={circle_acc:.3f} ({judged} circles) game_accuracy={res.accuracy:.4f} "
              f"score={res.score} presses={len(press_times)} hits={hits} steps={steps} skipped={skipped_ms / 1000:.0f}s" + (f" stopped early: {stopped}" if stopped else ""))
        timing = timing_report(press_times, args.export, env.step_ms) if args.export else None
        if timing:
            print("timing:", json.dumps(timing))
        log_eval(Path(args.model).parent, {"map": f"{Path(args.map_path).stem}[{args.diff or 'lowest'}]", "model": args.model, "delay": args.delay, "latency": latency if latency_max else 0.0,
                             "circle_hit_rate": circle_acc, "game_accuracy": res.accuracy, "score": res.score, "max_combo": res.combo, "circles": judged, "presses": len(press_times),
                             "hits": hits, "timing": timing, "when": time.strftime("%Y-%m-%d %H:%M")}, args.wandb)
        if args.trace:
            np.savez(args.trace, trace=np.array(trace, dtype=np.float64))
        host.quit()


if __name__ == "__main__":
    main()
