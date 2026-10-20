import argparse
import time

from osu_auto.capture import WindowCapture
from osu_auto.env import OsuEnv
from osu_auto.host import HostClient
from osu_auto.labels import load_export


def main() -> None:
    p = argparse.ArgumentParser(description="Drive the lockstep env with autoplay label positions; expect autoplay-quality judgements")
    p.add_argument("map_path")
    p.add_argument("--export", required=True)
    p.add_argument("--diff")
    p.add_argument("--steps", type=int, default=1500)
    p.add_argument("--clicks", default="auto", choices=["auto", "agent"])
    p.add_argument("--title", default="osu-auto")
    args = p.parse_args()

    labels = load_export(args.export).labels

    with HostClient() as host:
        cap = WindowCapture(args.title)
        env = OsuEnv(host, cap)
        res = env.reset(args.map_path, diff=args.diff, clicks=args.clicks, mods=[])
        print(f"reset: t={res.t_ms:.0f} clock={res.clock_ms:.0f} obs={res.obs.shape}")

        hits: dict[str, int] = {}
        t0 = time.perf_counter()
        n = min(args.steps, len(labels) - 1)
        for i in range(1, n + 1):
            x, y = float(labels.x[i]), float(labels.y[i])
            res = env.step(x, y, bool(labels.k1[i]), bool(labels.k2[i]))
            for k, v in res.hits.items():
                hits[k] = hits.get(k, 0) + v
            if abs(res.clock_ms - res.t_ms) > 0.5:
                print(f"clock mismatch at step {i}: t={res.t_ms} clock={res.clock_ms}")
            if i % 300 == 0 or i == n:
                print(f"step {i} t={res.t_ms:.0f} combo={res.combo} acc={res.accuracy:.4f} hits={hits} {i / (time.perf_counter() - t0):.1f} steps/s")

        cap.close()
        host.quit()


if __name__ == "__main__":
    main()
