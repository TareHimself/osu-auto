import argparse
import json
import time
from collections import deque

import numpy as np
import torch

from osu_auto.env import OsuEnv
from osu_auto.eval_click import KeyExecutor, timing_report
from osu_auto.geometry import PlayArea
from osu_auto.host import HostClient
from osu_auto.models import FrameNet
from osu_auto.targets import decode_to_playfield


def load(path: str, device: torch.device) -> tuple[FrameNet, dict]:
    ckpt = torch.load(path, map_location=device)
    model = FrameNet(stack=ckpt["stack"], width=ckpt["width"], outputs=ckpt.get("outputs", 2), heads=ckpt.get("heads"), bounded_aim=ckpt.get("bounded_aim", False)).to(device).eval()
    model.load_state_dict(ckpt["model"])
    return model, ckpt


def main() -> None:
    p = argparse.ArgumentParser(description="Closed-loop evaluation with the aim model moving the cursor and the click model pressing, no mods")
    p.add_argument("map_path")
    p.add_argument("--diff")
    p.add_argument("--aim", default="F:/osu-auto/data/models/aim_grid/aim_best.pt")
    p.add_argument("--click", default="F:/osu-auto/data/models/click_grid_h1/click_best.pt")
    p.add_argument("--joint", default=None, help="a unified model (outputs x, y, press, down) used instead of --aim and --click")
    p.add_argument("--export", default=None, help="label export of this map, for press timing diagnostics")
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--stop-misses", type=int, default=0)
    p.add_argument("--start-ms", type=float, default=None)
    p.add_argument("--press-threshold", type=float, default=0.5)
    p.add_argument("--hold-threshold", type=float, default=0.5)
    p.add_argument("--min-press-gap", type=int, default=2)
    p.add_argument("--endpoint", default="tcp://127.0.0.1:5555")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.joint:
        joint, aim_ck = load(args.joint, device)
        aim = click = None
    else:
        aim, aim_ck = load(args.aim, device)
        click, click_ck = load(args.click, device)
        if aim_ck["frame_hw"] != click_ck["frame_hw"] or aim_ck["stack"] != click_ck["stack"]:
            raise SystemExit("aim and click models must share frame size and stack")
    h, w = aim_ck["frame_hw"]
    stack_n = aim_ck["stack"]

    with HostClient(args.endpoint, timeout_ms=600_000) as host:
        env = OsuEnv(host, None, obs_size=(w, h))
        client = host.status()["client_size"]
        area = PlayArea(client[0], client[1])
        res = env.reset(args.map_path, diff=args.diff, clicks="agent", mods=[], start_ms=args.start_ms)
        stack = deque([res.obs] * stack_n, maxlen=stack_n)
        keys = KeyExecutor(args.min_press_gap, args.press_threshold, 1.0)
        zeros = torch.zeros(1, 2, device=device)
        hits: dict[str, int] = {}
        press_times: list[float] = []
        steps = 0
        consecutive_misses = 0
        stopped = ""
        t0 = time.perf_counter()

        while not env.done and (args.max_steps is None or steps < args.max_steps):
            frames = torch.from_numpy(np.stack(stack)).float().div_(255.0).unsqueeze(0).to(device)
            with torch.no_grad():
                if args.joint:
                    o = joint(frames, zeros)[0]
                    nx, ny = o[:2].tolist()
                    p_press, p_hold = torch.sigmoid(o[2:]).tolist()
                else:
                    nx, ny = aim(frames, zeros)[0].tolist()
                    p_press, p_hold = torch.sigmoid(click(frames, zeros))[0].tolist()
            x, y = decode_to_playfield(nx, ny, aim_ck, area)
            k1, k2 = keys.apply(p_press, p_hold > args.hold_threshold)
            if keys.pressed:
                press_times.append(env.t + env.step_ms)

            res = env.step(x, y, k1, k2)
            stack.append(res.obs)
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
              f"score={res.score} presses={len(press_times)} hits={hits} steps={steps}" + (f" stopped early: {stopped}" if stopped else ""))
        if args.export:
            print("timing:", json.dumps(timing_report(press_times, args.export, env.step_ms)))
        host.quit()


if __name__ == "__main__":
    main()
