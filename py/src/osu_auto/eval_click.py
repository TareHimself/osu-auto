import argparse
import json
import time
from collections import deque

import numpy as np
import torch

from osu_auto.env import OsuEnv
from osu_auto.host import HostClient
from osu_auto.labels import load_export
from osu_auto.models import FrameNet


class KeyExecutor:
    """Turns per-step press/hold decisions into k1/k2 states, alternating keys on every new press."""

    def __init__(self, min_press_gap: int = 2, press_threshold: float = 0.5, rearm: float = 0.3):
        self.down = False
        self.key = 1
        self.pressed = False
        self.min_press_gap = min_press_gap
        self.since_press = 10 ** 6
        self.press_threshold = press_threshold
        self.rearm = rearm
        self.armed = True

    def apply(self, p_press: float, hold: bool) -> tuple[bool, bool]:
        """One press per excursion of the press probability: it must fall below `rearm` before it can fire again."""
        if p_press < self.rearm:
            self.armed = True
        press = self.armed and p_press > self.press_threshold and self.since_press >= self.min_press_gap
        self.since_press += 1
        if press:
            self.key = 1 - self.key
            self.down = True
            self.since_press = 0
            self.armed = False
        elif not (self.down and hold):
            self.down = False
        self.pressed = press
        return self.down and self.key == 0, self.down and self.key == 1

    @property
    def state(self) -> list[float]:
        return [float(self.down), float(self.pressed)]


def timing_report(press_times: list[float], export_dir: str, step_ms: float) -> dict:
    """Agent presses against the expert's: offset of each agent press to the nearest expert press, and expert presses with no agent press nearby."""
    L = load_export(export_dir).labels
    expert = L.t_ms[(L.press_k1 | L.press_k2).astype(bool)]
    agent = np.array(press_times)
    if len(agent) == 0 or len(expert) == 0:
        return {}
    offsets = np.array([agent[np.argmin(np.abs(agent - e))] - e for e in expert])
    matched = np.abs(offsets) <= 2.5 * step_ms
    stray = np.array([np.min(np.abs(expert - a)) > 2.5 * step_ms for a in agent])
    return {"expert_presses": int(len(expert)), "agent_presses": int(len(agent)), "expert_matched": float(matched.mean()),
            "agent_stray": float(stray.mean()), "offset_ms_median": float(np.median(offsets[matched])) if matched.any() else None,
            "offset_ms_p90_abs": float(np.percentile(np.abs(offsets[matched]), 90)) if matched.any() else None}


def main() -> None:
    p = argparse.ArgumentParser(description="Closed-loop click evaluation: the model presses and holds, Autopilot moves the cursor")
    p.add_argument("map_path")
    p.add_argument("--diff")
    p.add_argument("--model", default="F:/osu-auto/data/models/click/click_best.pt")
    p.add_argument("--export", default=None, help="label export of this map, for press timing diagnostics")
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--stop-misses", type=int, default=15, help="stop after this many consecutive misses (0 = never)")
    p.add_argument("--min-acc", type=float, default=0.5, help="stop when the circle hit rate falls below this after --min-acc-objects judged")
    p.add_argument("--min-acc-objects", type=int, default=40)
    p.add_argument("--start-ms", type=float, default=None)
    p.add_argument("--press-threshold", type=float, default=0.5)
    p.add_argument("--hold-threshold", type=float, default=0.5)
    p.add_argument("--min-press-gap", type=int, default=2, help="steps between presses, so one decision spread over two steps is one press")
    p.add_argument("--rearm", type=float, default=0.3, help="press probability must drop below this before another press can fire (1.0 disables)")
    p.add_argument("--trace", default=None, help="save per-step (t, p_press, p_hold, pressed, down) to this .npz")
    p.add_argument("--dump-frames", default=None, help="save every observation to this .npy (steps x h x w)")
    p.add_argument("--endpoint", default="tcp://127.0.0.1:5555")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.model, map_location=device)
    model = FrameNet(stack=ckpt["stack"], width=ckpt["width"], outputs=2, heads=ckpt.get("heads")).to(device).eval()
    model.load_state_dict(ckpt["model"])
    h, w = ckpt["frame_hw"]
    use_keys = ckpt.get("use_keys", True)
    since_press = ckpt.get("since_press", False)

    with HostClient(args.endpoint, timeout_ms=600_000) as host:
        env = OsuEnv(host, None, obs_size=(w, h))
        res = env.reset(args.map_path, diff=args.diff, clicks="agent", mods=["AP"], start_ms=args.start_ms)
        stack = deque([res.obs] * ckpt["stack"], maxlen=ckpt["stack"])
        keys = KeyExecutor(args.min_press_gap, args.press_threshold, args.rearm)
        hits: dict[str, int] = {}
        press_times: list[float] = []
        trace: list[tuple[float, float, float, int, int]] = []
        dumped: list[np.ndarray] = []
        press_while_down = 0
        steps = 0
        consecutive_misses = 0
        stopped = ""
        t0 = time.perf_counter()

        while not env.done and (args.max_steps is None or steps < args.max_steps):
            frames = torch.from_numpy(np.stack(stack)).float().div_(255.0).unsqueeze(0).to(device)
            if use_keys:
                state = keys.state
            elif since_press:
                state = [min(keys.since_press, 50) / 50, 0.0]
            else:
                state = [0.0, 0.0]
            prev = torch.tensor([state], device=device)
            with torch.no_grad():
                p_press, p_hold = torch.sigmoid(model(frames, prev))[0].tolist()
            was_down = keys.down
            k1, k2 = keys.apply(p_press, p_hold > args.hold_threshold)
            if keys.pressed:
                press_times.append(env.t + env.step_ms)
                press_while_down += int(was_down)
            trace.append((env.t + env.step_ms, p_press, p_hold, int(keys.pressed), int(keys.down)))
            if args.dump_frames:
                dumped.append(stack[-1])

            res = env.step(256.0, 192.0, k1, k2)
            stack.append(res.obs)
            for k, v in res.hits.items():
                hits[k] = hits.get(k, 0) + v
            misses = res.hits.get("Miss", 0)
            scored = res.hits.get("Great", 0) or res.hits.get("Ok", 0) or res.hits.get("Meh", 0)
            consecutive_misses = consecutive_misses + misses if misses else (0 if scored else consecutive_misses)
            judged = sum(v for k, v in hits.items() if k in ("Great", "Ok", "Meh", "Miss"))
            steps += 1
            if steps % 500 == 0:
                print(f"t={res.t_ms:.0f} combo={res.combo} acc={res.accuracy:.4f} presses={len(press_times)} hits={hits} {steps / (time.perf_counter() - t0):.1f} steps/s")

            if args.stop_misses and consecutive_misses >= args.stop_misses:
                stopped = f"{consecutive_misses} consecutive misses"
                break
            circle_acc = (hits.get("Great", 0) + hits.get("Ok", 0) + hits.get("Meh", 0)) / max(judged, 1)
            if judged >= args.min_acc_objects and circle_acc < args.min_acc:
                stopped = f"circle hit rate {circle_acc:.3f} < {args.min_acc} after {judged} objects"
                break

        judged = sum(v for k, v in hits.items() if k in ("Great", "Ok", "Meh", "Miss"))
        circle_acc = (hits.get("Great", 0) + hits.get("Ok", 0) + hits.get("Meh", 0)) / max(judged, 1)
        print(f"final: t={res.t_ms:.0f} combo={res.combo} circle_hit_rate={circle_acc:.3f} ({judged} circles) game_accuracy={res.accuracy:.4f} "
              f"score={res.score} presses={len(press_times)} press_while_down={press_while_down} hits={hits} steps={steps}"
              + (f" stopped early: {stopped}" if stopped else ""))
        if args.export:
            print("timing:", json.dumps(timing_report(press_times, args.export, env.step_ms)))
        if args.dump_frames:
            np.save(args.dump_frames, np.stack(dumped))
        if args.trace:
            tr = np.array(trace, dtype=np.float64)
            np.savez(args.trace, t=tr[:, 0], p_press=tr[:, 1], p_hold=tr[:, 2], pressed=tr[:, 3].astype(np.uint8), down=tr[:, 4].astype(np.uint8))
        host.quit()


if __name__ == "__main__":
    main()
