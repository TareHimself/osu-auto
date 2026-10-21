import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from osu_auto.dataset import AimDataset, load_frame_export, worker_init
from osu_auto.models import AimNet
from osu_auto.targets import error_scale


def split_indices(n: int, holdout: float) -> tuple[list[int], list[int]]:
    """Hold out the last `holdout` fraction of time, not random frames, so validation is unseen gameplay."""
    cut = int(n * (1 - holdout))
    return list(range(cut)), list(range(cut, n))


def main() -> None:
    p = argparse.ArgumentParser(description="Behaviour-clone aim from exported frames")
    p.add_argument("exports", nargs="+")
    p.add_argument("--out", default="F:/osu-auto/data/models/aim")
    p.add_argument("--stack", type=int, default=4)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--holdout", type=float, default=0.15)
    p.add_argument("--width", type=int, default=32)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--horizon", type=int, default=0, help="predict the expert position this many steps ahead (0: where the expert is now)")
    p.add_argument("--real-lag", type=int, default=5, help="draw the real-cursor variant up to this many steps behind the expert")
    p.add_argument("--variants", default="1,1,0", help="relative weights of the real, none and fake cursor variants")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    exports = [load_frame_export(e) for e in args.exports]
    ds = AimDataset(exports, stack=args.stack, horizon=args.horizon, real_lag=args.real_lag,
                    variant_probs=tuple(float(v) for v in args.variants.split(",")), target_space="playfield")
    train_idx, val_idx = split_indices(len(ds), args.holdout)
    train = DataLoader(Subset(ds, train_idx), batch_size=args.batch, shuffle=True, num_workers=args.workers, drop_last=True, worker_init_fn=worker_init, persistent_workers=args.workers > 0)
    val = DataLoader(Subset(ds, val_idx), batch_size=args.batch, shuffle=False, num_workers=args.workers, worker_init_fn=worker_init, persistent_workers=args.workers > 0)
    print(f"device={device} samples train={len(train_idx)} val={len(val_idx)} stack={args.stack} frame={exports[0].frame_hw}")

    model = AimNet(stack=args.stack, width=args.width, bounded_aim=True).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.epochs * len(train))
    loss_fn = nn.SmoothL1Loss(beta=0.02)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    best = float("inf")
    history = []

    for epoch in range(args.epochs):
        model.train()
        t0 = time.perf_counter()
        total, count = 0.0, 0
        for frames, keys, target in train:
            frames, keys, target = frames.to(device), keys.to(device), target.to(device)
            pred = model(frames, keys)
            loss = loss_fn(pred, target)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            total += loss.item() * len(frames)
            count += len(frames)

        model.eval()
        errs = []
        with torch.no_grad():
            for frames, keys, target in val:
                pred = model(frames.to(device), keys.to(device)).cpu()
                errs.append((pred - target).numpy())
        err = np.concatenate(errs)
        px = err * error_scale("playfield")
        dist = np.sqrt((px ** 2).sum(1))
        stats = {"epoch": epoch + 1, "train_loss": total / count, "val_mean_px": float(dist.mean()), "val_median_px": float(np.median(dist)),
                 "val_within_30px": float((dist < 30).mean()), "seconds": time.perf_counter() - t0}
        history.append(stats)
        print(json.dumps(stats))

        if stats["val_mean_px"] < best:
            best = stats["val_mean_px"]
            torch.save({"model": model.state_dict(), "stack": args.stack, "width": args.width, "horizon": args.horizon, "frame_hw": list(exports[0].frame_hw),
                        "target_space": "playfield", "bounded_aim": True}, out / "aim_best.pt")

    (out / "history.json").write_text(json.dumps(history, indent=2))
    print(f"best val mean error {best:.1f} playfield units -> {out / 'aim_best.pt'}")


if __name__ == "__main__":
    main()
