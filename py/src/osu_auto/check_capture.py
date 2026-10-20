import argparse
import time
from pathlib import Path

import cv2
import numpy as np

from osu_auto.capture import WindowCapture
from osu_auto.host import HostClient
from osu_auto.labels import load_export


def main() -> None:
    p = argparse.ArgumentParser(description="Capture the windowed host after seeks and overlay computed geometry plus label cursor")
    p.add_argument("map_path")
    p.add_argument("--export", required=True)
    p.add_argument("--diff")
    p.add_argument("--out", default="F:/osu-auto/data/capture_check")
    p.add_argument("--times", type=float, nargs="*", default=None, help="ms times to capture; default: a few object starts")
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    export = load_export(args.export)

    with HostClient() as host:
        info = host.load(args.map_path, diff=args.diff)
        st = host.status()
        print(f"loaded {info['title']} [{info['difficulty']}] client_size={st['client_size']} title={st['window_title']!r}")

        cap = WindowCapture(st["window_title"])
        print(f"client={cap.client.rounded()} bounds={cap.bounds.rounded()} offset={cap.client_offset} capture_box={cap.area.capture.rounded()} playfield={cap.area.playfield.rounded()} scale={cap.area.scale:.3f}")

        first = cap.wait_frame(0, timeout=5)
        print(f"captured frame shape={first.bgra.shape} (expected bounds {cap.bounds.rounded()[2:]})")

        pf = cap.area.playfield
        measured = st["play_rect"]
        print(f"play_rect host={[round(v, 1) for v in measured]} formula={[round(v, 1) for v in (pf.x, pf.y, pf.right, pf.bottom)]}")

        times = args.times or [float(export.objects.start_ms[i]) for i in (10, 60, 120, 200)]
        seek_times = []
        for t in times:
            t0 = time.perf_counter()
            before = cap.count
            host.seek(t)
            frame = cap.wait_frame(before + 1)
            frame = cap.wait_frame(frame.index)
            seek_times.append(time.perf_counter() - t0)

            img = cap.client_view(frame).copy()
            x, y, w, h = cap.area.capture.rounded()
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 255, 255), 1)
            px, py, pw, ph = cap.area.playfield.rounded()
            cv2.rectangle(img, (px, py), (px + pw, py + ph), (0, 255, 0, 255), 1)

            i = export.labels.index_at(t)
            lx, ly = cap.area.playfield_to_window(float(export.labels.x[i]), float(export.labels.y[i]))
            cv2.circle(img, (int(lx), int(ly)), 6, (0, 0, 255, 255), 1)
            cv2.drawMarker(img, (int(lx), int(ly)), (0, 0, 255, 255), cv2.MARKER_CROSS, 14, 1)

            path = out / f"t{int(t)}.png"
            cv2.imwrite(str(path), img)
            print(f"t={t:.0f} label=({export.labels.x[i]:.0f},{export.labels.y[i]:.0f}) -> {path}")

        print(f"seek+capture: mean {np.mean(seek_times) * 1000:.1f} ms, max {np.max(seek_times) * 1000:.1f} ms")

        t0 = time.perf_counter()
        n0 = cap.count
        time.sleep(1.0)
        print(f"idle capture rate: {(cap.count - n0) / (time.perf_counter() - t0):.0f} fps")

        cap.close()
        host.quit()


if __name__ == "__main__":
    main()
