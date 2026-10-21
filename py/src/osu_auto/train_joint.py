import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset

from osu_auto.dataset import JointDataset, load_frame_export, worker_init
from osu_auto.gpu_data import GpuChunkLoader, split_exports
from osu_auto.models import FrameNet
from osu_auto.targets import error_scale
from osu_auto.train_aim import split_indices
from osu_auto.train_click import tolerant_match


def main() -> None:
    p = argparse.ArgumentParser(description="Behaviour-clone aim and click together: one net, outputs (x, y, press, down)")
    p.add_argument("exports", nargs="+")
    p.add_argument("--out", default="F:/osu-auto/data/models/joint")
    p.add_argument("--stack", type=int, default=4)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--holdout", type=float, default=0.15)
    p.add_argument("--width", type=int, default=32)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--real-lag", type=int, default=5)
    p.add_argument("--press-weight", type=float, default=8.0)
    p.add_argument("--click-weight", type=float, default=1.0, help="weight of the click losses against the aim loss")
    p.add_argument("--shared-head", action="store_true", help="one MLP for all outputs instead of separate aim and click heads")
    p.add_argument("--chunk-gb", type=float, default=3.0, help="serve batches from GPU-resident chunks of this size; 0 uses the CPU data loader")
    p.add_argument("--val-stride", type=int, default=2)
    p.add_argument("--heat", type=float, default=0.0, help="sample training windows by strain and press proximity (weight = 0.25 + heat * score); 0 = uniform")
    p.add_argument("--pool", type=int, default=1, help="area-downsample exported frames by this factor on the GPU (2: a 320x240 export trains at 160x120)")
    p.add_argument("--no-amp", action="store_true", help="disable bfloat16 autocast")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    exports = [load_frame_export(e) for e in args.exports]
    frame_hw = [v // args.pool for v in exports[0].frame_hw]
    if args.chunk_gb > 0:
        train_ex, val_ex = split_exports(exports, args.holdout)
        chunk = int(args.chunk_gb * (1 << 30))
        train = GpuChunkLoader(train_ex, device, stack=args.stack, real_lag=args.real_lag, batch=args.batch, chunk_bytes=chunk, pool=args.pool, heat=args.heat)
        val = GpuChunkLoader(val_ex, device, stack=args.stack, real_lag=args.real_lag, batch=args.batch, chunk_bytes=chunk, shuffle=False, drop_last=False, stride=args.val_stride, pool=args.pool)
        print(f"device={device} exports train={len(train_ex)} val={len(val_ex)} chunks={len(train.groups)}+{len(val.groups)} batches={len(train)}+{len(val)} stack={args.stack} frame={frame_hw}")
    else:
        ds = JointDataset(exports, stack=args.stack, horizon=0, real_lag=args.real_lag, target_space="playfield")
        train_idx, val_idx = split_indices(len(ds), args.holdout)
        train = DataLoader(Subset(ds, train_idx), batch_size=args.batch, shuffle=True, num_workers=args.workers, drop_last=True,
                           worker_init_fn=worker_init, persistent_workers=args.workers > 0)
        val = DataLoader(Subset(ds, val_idx), batch_size=args.batch, shuffle=False, num_workers=args.workers,
                         worker_init_fn=worker_init, persistent_workers=args.workers > 0)
        print(f"device={device} samples train={len(train_idx)} val={len(val_idx)} stack={args.stack} frame={exports[0].frame_hw}")

    heads = None if args.shared_head else [2, 2]
    model = FrameNet(stack=args.stack, width=args.width, outputs=4, heads=heads, bounded_aim=True).to(device, memory_format=torch.channels_last)
    amp = not args.no_amp
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.epochs * len(train))
    aim_loss = nn.SmoothL1Loss(beta=0.02)
    press_weight = torch.tensor([args.press_weight], device=device)

    def loss_fn(out, aim, click):
        la = aim_loss(out[:, :2], aim)
        lp = F.binary_cross_entropy_with_logits(out[:, 2], click[:, 0], pos_weight=press_weight)
        ld = F.binary_cross_entropy_with_logits(out[:, 3], click[:, 1])
        return la + args.click_weight * (lp + ld)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    best = -1e9
    history = []

    for epoch in range(args.epochs):
        model.train()
        t0 = time.perf_counter()
        total, count = 0.0, 0
        for frames, keys, aim, click in train:
            frames = frames.to(device, memory_format=torch.channels_last)
            keys, aim, click = keys.to(device), aim.to(device), click.to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                out = model(frames, keys)
            loss = loss_fn(out.float(), aim, click)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            total += loss.item() * len(frames)
            count += len(frames)

        model.eval()
        outs, aims, clicks = [], [], []
        with torch.no_grad():
            for frames, keys, aim, click in val:
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                    outs.append(model(frames.to(device, memory_format=torch.channels_last), keys.to(device)).float().cpu().numpy())
                aims.append(aim.cpu().numpy())
                clicks.append(click.cpu().numpy())
        out = np.concatenate(outs)
        aim = np.concatenate(aims)
        click = np.concatenate(clicks)
        dist = np.sqrt((((out[:, :2] - aim) * error_scale("playfield")) ** 2).sum(1))
        p_press = 1 / (1 + np.exp(-out[:, 2]))
        p_down = 1 / (1 + np.exp(-out[:, 3]))
        recall1, precision1 = tolerant_match(np.flatnonzero(click[:, 0] > 0.5), np.flatnonzero(p_press > 0.5), 1)
        down_acc = float(((p_down > 0.5) == (click[:, 1] > 0.5)).mean())
        score = (recall1 + precision1) / 2 - dist.mean() / 100
        stats = {"epoch": epoch + 1, "train_loss": total / count, "val_mean_px": float(dist.mean()), "val_median_px": float(np.median(dist)),
                 "press_recall_tol1": recall1, "press_precision_tol1": precision1, "down_acc": down_acc, "seconds": time.perf_counter() - t0}
        history.append(stats)
        print(json.dumps(stats))
        if score > best:
            best = score
            torch.save({"model": model.state_dict(), "stack": args.stack, "width": args.width, "outputs": 4, "joint": True, "heads": heads, "target_space": "playfield", "bounded_aim": True,
                        "frame_hw": frame_hw, "horizon": 0, "click_horizon": 1}, out_dir / "joint_best.pt")

    (out_dir / "history.json").write_text(json.dumps(history, indent=2))
    print(f"best -> {out_dir / 'joint_best.pt'}")


if __name__ == "__main__":
    main()
