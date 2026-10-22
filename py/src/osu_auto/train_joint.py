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
from osu_auto.export_onnx import export as export_onnx
from osu_auto.models import AIM_FROM_BOX, FrameNet
from osu_auto.targets import error_scale
from osu_auto.train_aim import split_indices
from osu_auto.train_click import tolerant_match
from osu_auto.tracking import Tracker


def main() -> None:
    p = argparse.ArgumentParser(description="Behaviour-clone aim and click together: one net, outputs (x, y, press, down)", fromfile_prefix_chars="@")
    p.add_argument("exports", nargs="+")
    p.add_argument("--out", default="F:/osu-auto/data/models/joint")
    p.add_argument("--stack", type=int, default=4)
    p.add_argument("--offsets", default=None, help="frame offsets in steps back from now, e.g. 0,1,2,4,8 (overrides --stack)")
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
    p.add_argument("--latency-max", type=float, default=0.0, help="condition on a per-sample latency in 0..this many ms: labels shift ahead by it and it is fed as input")
    p.add_argument("--wandb", default=None, help="wandb project to log to (needs `wandb login` once)")
    p.add_argument("--run-name", default=None)
    p.add_argument("--spin-recover", type=float, default=0.5, help="share of spinner samples drawn with a synthetic off-phase or stalled cursor, labelled one step ahead around the centre")
    p.add_argument("--spin-weight", type=float, default=2.0, help="extra heat weight for spinner rows")
    p.add_argument("--no-onnx", action="store_true", help="skip exporting joint_best.onnx for the runner at the end")
    p.add_argument("--cursor-scale", default="1,1", help="min,max cursor size factor drawn log-uniformly in training (lazer's cursor size setting)")
    p.add_argument("--cursor-scale-p", type=float, default=0.0, help="share of samples drawn with a random cursor size")
    p.add_argument("--click-reweight", action="store_true", help="weight spinner rows in the click loss by their unboosted sampling rate")
    p.add_argument("--aim-head", default="mlp", choices=["mlp", "softargmax"])
    p.add_argument("--att-loss", type=float, default=0.1, help="weight of the heatmap cross entropy on the softargmax attention")
    p.add_argument("--pool", type=int, default=1, help="area-downsample exported frames by this factor on the GPU (2: a 320x240 export trains at 160x120)")
    p.add_argument("--no-amp", action="store_true", help="disable bfloat16 autocast")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    exports = [load_frame_export(e.strip()) for e in args.exports if e.strip()]
    offsets = [int(v) for v in args.offsets.split(",")] if args.offsets else list(range(args.stack))
    args.stack = len(offsets)
    frame_hw = [v // args.pool for v in exports[0].frame_hw]
    if args.chunk_gb > 0:
        train_ex, val_ex = split_exports(exports, args.holdout)
        chunk = int(args.chunk_gb * (1 << 30))
        train = GpuChunkLoader(train_ex, device, stack=args.stack, real_lag=args.real_lag, batch=args.batch, chunk_bytes=chunk, pool=args.pool, heat=args.heat, click_reweight=args.click_reweight, cursor_scale=tuple(float(v) for v in args.cursor_scale.split(",")), cursor_scale_p=args.cursor_scale_p, latency_max_ms=args.latency_max, offsets=offsets, spin_recover=args.spin_recover, spin_weight=args.spin_weight)
        val = GpuChunkLoader(val_ex, device, stack=args.stack, real_lag=args.real_lag, batch=args.batch, chunk_bytes=chunk, shuffle=False, drop_last=False, stride=args.val_stride, pool=args.pool, offsets=offsets)
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
    model = FrameNet(stack=args.stack, width=args.width, outputs=4, heads=heads, bounded_aim=True, aim_head=args.aim_head).to(device, memory_format=torch.channels_last)
    amp = not args.no_amp
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    total_steps = args.epochs * (len(train) + 50)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=total_steps)
    aim_loss = nn.SmoothL1Loss(beta=0.02)
    press_weight = torch.tensor([args.press_weight], device=device)

    def att_loss(logits, aim):
        gh, gw = logits.shape[1:]
        (ax, bx), (ay, by) = AIM_FROM_BOX
        u, v = (aim[:, 0] - bx) / ax, (aim[:, 1] - by) / ay
        cy = (torch.arange(gh, device=aim.device) + 0.5) / gh
        cx = (torch.arange(gw, device=aim.device) + 0.5) / gw
        g = torch.exp(-0.5 * (((cy[None, :, None] - v[:, None, None]) * gh) ** 2 + ((cx[None, None, :] - u[:, None, None]) * gw) ** 2)).flatten(1)
        g = g / g.sum(1, keepdim=True).clamp_min(1e-8)
        return -(g * logits.flatten(1).log_softmax(1)).sum(1).mean()

    def loss_fn(out, aim, click, logits=None):
        parts = {"aim": aim_loss(out[:, :2], aim)}
        if logits is not None and args.att_loss > 0:
            parts["att"] = args.att_loss * att_loss(logits, aim)
        w = click[:, 2] if click.shape[1] > 2 else torch.ones_like(click[:, 0])
        parts["press"] = (w * F.binary_cross_entropy_with_logits(out[:, 2], click[:, 0], pos_weight=press_weight, reduction="none")).mean()
        parts["down"] = (w * F.binary_cross_entropy_with_logits(out[:, 3], click[:, 1], reduction="none")).mean()
        loss = parts["aim"] + parts.get("att", 0) + args.click_weight * (parts["press"] + parts["down"])
        return loss, parts

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    best = -1e9
    history = []
    config = {k: v for k, v in vars(args).items() if k != "exports"}
    config["n_exports"] = len(exports)
    (out_dir / "config.json").write_text(json.dumps({**config, "exports": [str(e.path) for e in exports]}, indent=2))
    tracker = Tracker(args.wandb, args.run_name or out_dir.name, config, out_dir)
    if tracker.id:
        (out_dir / "wandb_run.json").write_text(json.dumps({"id": tracker.id, "project": args.wandb}))

    for epoch in range(args.epochs):
        model.train()
        t0 = time.perf_counter()
        total, count, part_sums = 0.0, 0, {}
        for frames, keys, aim, click in train:
            frames = frames.to(device, memory_format=torch.channels_last)
            keys, aim, click = keys.to(device), aim.to(device), click.to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                out, logits = model(frames, keys, return_logits=True) if args.aim_head == "softargmax" else (model(frames, keys), None)
            loss, parts = loss_fn(out.float(), aim, click, logits)
            for k, v in parts.items():
                part_sums[k] = part_sums.get(k, 0.0) + v.item() * len(frames)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            if sched.last_epoch + 1 < total_steps:
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
                 "press_recall_tol1": recall1, "press_precision_tol1": precision1, "down_acc": down_acc, "seconds": time.perf_counter() - t0,
                 **{f"loss_{k}": v / count for k, v in part_sums.items()}}
        history.append(stats)
        print(json.dumps(stats))
        tracker.log({**stats, "score": score}, step=epoch + 1)
        if score > best:
            best = score
            torch.save({"model": model.state_dict(), "stack": args.stack, "width": args.width, "outputs": 4, "joint": True, "heads": heads, "target_space": "playfield", "bounded_aim": True, "aim_head": args.aim_head,
                        "frame_hw": frame_hw, "horizon": 0, "click_horizon": 1, "latency_max_ms": args.latency_max, "offsets": offsets}, out_dir / "joint_best.pt")

    (out_dir / "history.json").write_text(json.dumps(history, indent=2))
    tracker.artifact(out_dir / "joint_best.pt", out_dir.name, metadata={"best_score": best})
    tracker.finish({"best_score": best})
    print(f"best -> {out_dir / 'joint_best.pt'}")
    if not args.no_onnx:
        try:
            export_onnx(out_dir / "joint_best.pt", frames_from=exports[-1].path)
        except Exception as e:
            print(f"onnx export failed: {e!r}; run python -m osu_auto.export_onnx {out_dir / 'joint_best.pt'}")


if __name__ == "__main__":
    main()
