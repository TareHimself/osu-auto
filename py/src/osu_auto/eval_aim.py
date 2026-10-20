import argparse
import time
from collections import deque

import numpy as np
import torch

from osu_auto.capture import WindowCapture
from osu_auto.env import OsuEnv
from osu_auto.geometry import PlayArea
from osu_auto.host import HostClient
from osu_auto.models import AimNet


def main() -> None:
    p = argparse.ArgumentParser(description="Closed-loop aim evaluation: the model moves the cursor, autoplay supplies the clicks")
    p.add_argument("map_path")
    p.add_argument("--diff")
    p.add_argument("--model", default="F:/osu-auto/data/models/aim/aim_best.pt")
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--stop-misses", type=int, default=15, help="stop after this many consecutive misses (0 = never)")
    p.add_argument("--min-acc", type=float, default=0.5, help="stop when accuracy falls below this after --min-acc-objects judged")
    p.add_argument("--min-acc-objects", type=int, default=40)
    p.add_argument("--start-ms", type=float, default=None, help="start gameplay at this time (e.g. the held-out tail of a training map)")
    p.add_argument("--title", default="osu-auto")
    p.add_argument("--endpoint", default="tcp://127.0.0.1:5555")
    p.add_argument("--capture", choices=["readback", "wgc"], default="readback",
                   help="readback: the host returns its own backbuffer with each step; wgc: capture the window from the screen")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.model, map_location=device)
    model = AimNet(stack=ckpt["stack"], width=ckpt["width"]).to(device).eval()
    model.load_state_dict(ckpt["model"])
    h, w = ckpt["frame_hw"]

    with HostClient(args.endpoint, timeout_ms=600_000) as host:
        cap = WindowCapture(args.title) if args.capture == "wgc" else None
        env = OsuEnv(host, cap, obs_size=(w, h))
        client = host.status()["client_size"]
        area = PlayArea(client[0], client[1])

        res = env.reset(args.map_path, diff=args.diff, clicks="agent", mods=["RX"], start_ms=args.start_ms)
        stack = deque([res.obs] * ckpt["stack"], maxlen=ckpt["stack"])
        hits: dict[str, int] = {}
        steps = 0
        consecutive_misses = 0
        stopped = ""
        t0 = time.perf_counter()
        x, y = 256.0, 192.0

        while not env.done and (args.max_steps is None or steps < args.max_steps):
            frames = torch.from_numpy(np.stack(stack)).float().div_(255.0).unsqueeze(0).to(device)
            keys = torch.zeros(1, 2, device=device)
            with torch.no_grad():
                nx, ny = model(frames, keys)[0].tolist()
            cx, cy = nx * area.capture.w, ny * area.capture.h
            x, y = area.window_to_playfield(cx + area.capture.x, cy + area.capture.y)

            res = env.step(x, y, False, False)
            stack.append(res.obs)
            for k, v in res.hits.items():
                hits[k] = hits.get(k, 0) + v
            misses = res.hits.get("Miss", 0)
            consecutive_misses = consecutive_misses + misses if misses else (0 if res.hits.get("Great", 0) or res.hits.get("Ok", 0) or res.hits.get("Meh", 0) else consecutive_misses)
            judged = sum(v for k, v in hits.items() if k in ("Great", "Ok", "Meh", "Miss"))
            steps += 1
            if steps % 500 == 0:
                print(f"t={res.t_ms:.0f} combo={res.combo} acc={res.accuracy:.4f} score={res.score} hits={hits} {steps / (time.perf_counter() - t0):.1f} steps/s")

            if args.stop_misses and consecutive_misses >= args.stop_misses:
                stopped = f"{consecutive_misses} consecutive misses"
                break
            circle_acc = (hits.get("Great", 0) + hits.get("Ok", 0) + hits.get("Meh", 0)) / max(judged, 1)
            if judged >= args.min_acc_objects and circle_acc < args.min_acc:
                stopped = f"circle hit rate {circle_acc:.3f} < {args.min_acc} after {judged} objects"
                break

        judged = sum(v for k, v in hits.items() if k in ("Great", "Ok", "Meh", "Miss"))
        circle_acc = (hits.get("Great", 0) + hits.get("Ok", 0) + hits.get("Meh", 0)) / max(judged, 1)
        print(f"final: t={res.t_ms:.0f} combo={res.combo} circle_hit_rate={circle_acc:.3f} ({judged} circles) game_accuracy={res.accuracy:.4f} score={res.score} hits={hits} steps={steps}"
              + (f" stopped early: {stopped}" if stopped else ""))
        if cap is not None:
            cap.close()
        host.quit()


if __name__ == "__main__":
    main()
