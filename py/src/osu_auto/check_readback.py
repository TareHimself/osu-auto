import argparse

import numpy as np

from osu_auto.capture import WindowCapture
from osu_auto.host import HostClient
from osu_auto.labels import load_export


def main() -> None:
    p = argparse.ArgumentParser(description="Compare the host's in-process readback with a WGC window capture of the same state")
    p.add_argument("map_path")
    p.add_argument("--export", required=True)
    p.add_argument("--diff")
    p.add_argument("--title", default="osu-auto")
    args = p.parse_args()

    labels = load_export(args.export).labels

    with HostClient() as host:
        host.load(args.map_path, diff=args.diff)
        cap = WindowCapture(args.title)
        cap.wait_frame(0, timeout=5)

        for frac in (0.2, 0.5, 0.8):
            t = float(labels.t_ms[int(len(labels) * frac)])
            before = cap.count
            clock, box, _ = host.seek_grab(t)
            frame = cap.wait_frame(before + 1)
            frame = cap.wait_frame(frame.index)
            wgc = cap.play_box(frame)
            if wgc.shape != box.shape:
                print(f"t={t:.0f} shape mismatch readback={box.shape} wgc={wgc.shape}")
                continue
            d = np.abs(wgc[..., :3].astype(np.int16) - box[..., :3].astype(np.int16))
            print(f"t={t:.0f} clock={clock:.0f} readback vs WGC: mean abs diff={d.mean():.4f} max={d.max()} pixels differing={(d.max(axis=2) > 0).mean():.4%}")

        cap.close()
        host.quit()


if __name__ == "__main__":
    main()
