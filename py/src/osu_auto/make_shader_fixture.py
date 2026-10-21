"""Save one full-resolution window frame and what play.py's ObsCapture turns it into, to test the native shader on the same pixels."""
import argparse
import ctypes
import json
from pathlib import Path

import cv2
import numpy as np

from .capture import WindowCapture
from .play import list_windows


def main() -> None:
    ctypes.windll.shcore.SetProcessDpiAwareness(2)
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--filter", default="osu!")
    ap.add_argument("--size", type=int, nargs=2, default=[160, 120], metavar=("W", "H"))
    args = ap.parse_args()

    items = list_windows(args.filter)
    if not items:
        raise SystemExit(f"no window matching '{args.filter}'")
    cap = WindowCapture(hwnd=items[0][0])
    frame = cap.wait_frame(0)
    cap.refresh_geometry()
    cap.close()

    ox, oy = cap.client_offset
    x, y, w, h = cap.area.capture.rounded()
    box = frame.bgra[oy + y:oy + y + h, ox + x:ox + x + w]
    expected = cv2.resize(cv2.cvtColor(np.ascontiguousarray(box), cv2.COLOR_BGRA2GRAY), tuple(args.size), interpolation=cv2.INTER_AREA)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "frame.npy", np.ascontiguousarray(frame.bgra))
    np.save(out / "expected.npy", expected)
    np.save(out / "crop.npy", np.array([ox + x, oy + y, w, h], dtype=np.int32))
    (out / "meta.json").write_text(json.dumps({"frame": list(frame.bgra.shape), "client": [cap.client.w, cap.client.h], "crop": [ox + x, oy + y, w, h]}))
    print(f"frame {frame.bgra.shape}, crop {ox + x},{oy + y} {w}x{h}")


if __name__ == "__main__":
    main()
