"""Write the same folder as `play.py --dump-frames`, from a training frame export instead of a live window.

Real frames go through the torch model and play.py's post-processing, so the native runner can be checked against them.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .dataset import load_frame_export
from .keys import KeyExecutor
from .models import load_checkpoint
from .paths import MODELS
from .targets import decode_to_playfield


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("export")
    ap.add_argument("out")
    ap.add_argument("--model", default=str(MODELS / "current/joint_best.pt"))
    ap.add_argument("--start", type=int, default=2000)
    ap.add_argument("--ticks", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    model, ck = load_checkpoint(args.model)
    offsets = ck.get("offsets") or list(range(ck["stack"]))
    steps = list(reversed(offsets))
    latency_max = float(ck.get("latency_max_ms", 0.0))

    frames = np.asarray(load_frame_export(args.export).frames)
    pool = frames.shape[1] // ck["frame_hw"][0]
    if pool > 1:
        n, h, w = frames.shape
        sums = frames.reshape(n, h // pool, pool, w // pool, pool).sum(axis=(2, 4), dtype=np.int64)
        frames = ((sums + pool * pool // 2) // (pool * pool)).astype(np.uint8)
    rng = np.random.default_rng(args.seed)
    keys = KeyExecutor(2, 0.5, 1.0)
    stacks, auxes, outs, posts = [], [], [], []
    for i in range(args.start, args.start + args.ticks):
        stack = np.stack([frames[i - k] for k in steps]).astype(np.uint8)
        aux0 = float(rng.uniform(0.0, 1.0)) if latency_max else 0.0
        x = torch.from_numpy(stack).float().div_(255.0).unsqueeze(0)
        with torch.no_grad():
            o = model(x, torch.tensor([[aux0, 0.0]], dtype=torch.float32))[0]
        nx, ny = o[:2].tolist()
        p_press, p_down = torch.sigmoid(o[2:]).tolist()
        px, py = decode_to_playfield(nx, ny)
        k1, k2 = keys.apply(p_press, p_down > 0.5, p_down)
        stacks.append(stack)
        auxes.append(aux0)
        outs.append((nx, ny, p_press, p_down))
        posts.append((px, py, float(k1), float(k2)))

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "frames.npy", np.stack(stacks))
    np.save(out / "aux.npy", np.array(auxes, dtype=np.float32))
    np.save(out / "out.npy", np.array(outs, dtype=np.float64))
    np.save(out / "post.npy", np.array(posts, dtype=np.float64))
    (out / "meta.json").write_text(json.dumps({"ticks": len(stacks), "model": args.model}))
    presses = int(np.sum(np.diff(np.array(posts)[:, 2:].sum(axis=1), prepend=0) > 0))
    print(f"wrote {len(stacks)} ticks to {out}, key-down transitions ~{presses}")


if __name__ == "__main__":
    main()
