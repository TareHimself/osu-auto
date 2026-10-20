import argparse
import time

import numpy as np

from osu_auto.geometry import PlayArea
from osu_auto.host import HostClient
from osu_auto.labels import load_export


def main() -> None:
    p = argparse.ArgumentParser(description="Exercise the host control channel: load, seek through objects, compare play rect")
    p.add_argument("map_path")
    p.add_argument("--export", help="matching export dir, used to seek to every object start time")
    p.add_argument("--diff")
    p.add_argument("--endpoint", default="tcp://127.0.0.1:5555")
    p.add_argument("--quit", action="store_true", help="tell the host to exit at the end")
    args = p.parse_args()

    with HostClient(args.endpoint) as host:
        t = time.perf_counter()
        info = host.load(args.map_path, diff=args.diff)
        print(f"loaded {info['title']} [{info['difficulty']}] objects={info['objects']} t0={info['t0']:.0f} t1={info['t1']:.0f} in {time.perf_counter() - t:.1f}s")

        st = host.status()
        w, h = st["draw_size"]
        area = PlayArea(w, h)
        pf = area.playfield
        measured = st["play_rect"]
        formula = (pf.x, pf.y, pf.right, pf.bottom)
        delta = max(abs(a - b) for a, b in zip(measured, formula))
        print(f"draw_size={w:.0f}x{h:.0f} play_rect measured={[round(v, 1) for v in measured]} formula={[round(v, 1) for v in formula]} max_delta={delta:.2f}px")

        if args.export:
            objects = load_export(args.export).objects
            times = objects.start_ms
        else:
            times = np.linspace(info["t0"], info["t1"], 20)

        t = time.perf_counter()
        last_combo = 0
        for i, target in enumerate(times):
            clock = host.seek(float(target) + 1)
            st = host.status()
            if i % max(1, len(times) // 10) == 0 or i == len(times) - 1:
                print(f"seek {target:8.0f} -> clock {clock:8.0f} combo={st['combo']:4d} max_combo={st['max_combo']:4d} acc={st['accuracy']:.4f} completed={st['completed']}")
            last_combo = st["max_combo"]
        elapsed = time.perf_counter() - t
        print(f"{len(times)} seeks in {elapsed:.2f}s ({len(times) / elapsed:.0f} seeks/s), final max_combo={last_combo}")

        if args.quit:
            host.quit()


if __name__ == "__main__":
    main()
