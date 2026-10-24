"""Behaviour-clone aim and clicks together: one net, outputs (x, y, press, down). Defaults are the current best recipe."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from osu_auto.config import parse_args as parse_with_config
from osu_auto.dataset import load_frame_export
from osu_auto.export_onnx import export as export_onnx
from osu_auto.gpu_data import GpuChunkLoader, split_exports
from osu_auto.models import AIM_FROM_BOX, FrameNet
from osu_auto.paths import MODELS
from osu_auto.targets import PX_PER_UNIT
from osu_auto.tracking import Tracker


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("exports", nargs="*", help="frame export folders, or .txt files listing one per line (export_jobs --list)")
    p.add_argument("--out", default=str(MODELS / "joint"))
    p.add_argument("--stack", type=int, default=4)
    p.add_argument("--offsets", default=None, help="frame offsets in steps back from now, e.g. 0,1,2,4,8 (overrides --stack)")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--holdout", type=float, default=0.15, help="share of exports held out for validation")
    p.add_argument("--width", type=int, default=32)
    p.add_argument("--aim-head", default="softargmax", choices=["softargmax", "mlp"])
    p.add_argument("--att-loss", type=float, default=0.1, help="weight of the heatmap cross entropy on the softargmax attention")
    p.add_argument("--real-lag", type=int, default=5, help="max steps the composited cursor trails the label")
    p.add_argument("--press-weight", type=float, default=8.0, help="positive class weight of the press loss")
    p.add_argument("--click-weight", type=float, default=1.0, help="weight of the click losses against the aim loss")
    p.add_argument("--click-reweight", action=argparse.BooleanOptionalAction, default=True,
                   help="weight spinner rows in the click loss by their unboosted sampling rate")
    p.add_argument("--chunk-gb", type=float, default=3.0, help="size of the GPU-resident frame chunks batches are drawn from")
    p.add_argument("--val-stride", type=int, default=4)
    p.add_argument("--pool", type=int, default=2, help="area-downsample exported frames by this factor (2: a 320x240 export trains at 160x120)")
    p.add_argument("--heat", type=float, default=1.0, help="sample training windows by strain and press proximity (weight = 0.25 + heat * score); 0 = uniform")
    p.add_argument("--latency-max", type=float, default=40.0, help="condition on a per-sample latency in 0..this many ms: labels shift ahead by it and it is fed as input")
    p.add_argument("--spin-recover", type=float, default=0.25, help="share of spinner samples drawn with a synthetic off-phase or stalled cursor")
    p.add_argument("--spin-weight", type=float, default=1.0, help="extra heat weight for spinner rows")
    p.add_argument("--cursor-scale", default="0.5,1.6", help="min,max cursor size factor drawn log-uniformly (lazer's cursor size setting)")
    p.add_argument("--cursor-scale-p", type=float, default=0.5, help="share of samples drawn with a random cursor size")
    p.add_argument("--no-amp", action="store_true", help="disable bfloat16 autocast")
    p.add_argument("--no-onnx", action="store_true", help="skip exporting joint_best.onnx for the runner at the end")
    p.add_argument("--wandb", default=None, help="wandb project to log to (needs `wandb login` once)")
    p.add_argument("--run-name", default=None)
    args = parse_with_config(p, "train", path_keys=("exports", "out"))
    args.exports = [line.strip() for e in args.exports for line in (Path(e).read_text().splitlines() if e.endswith(".txt") else [e]) if line.strip()]
    if not args.exports:
        p.error("no exports given")
    return args


def att_loss(logits: torch.Tensor, aim: torch.Tensor) -> torch.Tensor:
    """Cross entropy of the attention map against a Gaussian (1 cell sigma) around the target."""
    gh, gw = logits.shape[1:]
    (ax, bx), (ay, by) = AIM_FROM_BOX
    u, v = (aim[:, 0] - bx) / ax, (aim[:, 1] - by) / ay
    cy = (torch.arange(gh, device=aim.device) + 0.5) / gh
    cx = (torch.arange(gw, device=aim.device) + 0.5) / gw
    g = torch.exp(-0.5 * (((cy[None, :, None] - v[:, None, None]) * gh) ** 2 + ((cx[None, None, :] - u[:, None, None]) * gw) ** 2)).flatten(1)
    g = g / g.sum(1, keepdim=True).clamp_min(1e-8)
    return -(g * logits.flatten(1).log_softmax(1)).sum(1).mean()


def tolerant_match(true_idx: np.ndarray, pred_idx: np.ndarray, tol: int) -> tuple[float, float]:
    """Recall and precision of press events allowing a timing slack of `tol` steps."""
    if len(true_idx) == 0 or len(pred_idx) == 0:
        return 0.0, 0.0
    recall = np.mean([np.min(np.abs(pred_idx - t)) <= tol for t in true_idx])
    precision = np.mean([np.min(np.abs(true_idx - p)) <= tol for p in pred_idx])
    return float(recall), float(precision)


def validate(model: FrameNet, val: GpuChunkLoader, device: torch.device, amp: bool) -> dict:
    model.eval()
    outs, aims, clicks = [], [], []
    with torch.no_grad():
        for frames, keys, aim, click in val:
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                outs.append(model(frames.to(device, memory_format=torch.channels_last), keys.to(device)).float().cpu().numpy())
            aims.append(aim.cpu().numpy())
            clicks.append(click.cpu().numpy())
    out, aim, click = np.concatenate(outs), np.concatenate(aims), np.concatenate(clicks)
    dist = np.sqrt((((out[:, :2] - aim) * PX_PER_UNIT) ** 2).sum(1))
    p_press = 1 / (1 + np.exp(-out[:, 2]))
    p_down = 1 / (1 + np.exp(-out[:, 3]))
    recall, precision = tolerant_match(np.flatnonzero(click[:, 0] > 0.5), np.flatnonzero(p_press > 0.5), 1)
    return {"val_mean_px": float(dist.mean()), "val_median_px": float(np.median(dist)), "press_recall_tol1": recall,
            "press_precision_tol1": precision, "down_acc": float(((p_down > 0.5) == (click[:, 1] > 0.5)).mean())}


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    exports = [load_frame_export(e) for e in args.exports]
    offsets = [int(v) for v in args.offsets.split(",")] if args.offsets else list(range(args.stack))
    args.stack = len(offsets)
    frame_hw = [v // args.pool for v in exports[0].frame_hw]
    train_ex, val_ex = split_exports(exports, args.holdout)
    chunk = int(args.chunk_gb * (1 << 30))
    train = GpuChunkLoader(train_ex, device, stack=args.stack, real_lag=args.real_lag, batch=args.batch, chunk_bytes=chunk, pool=args.pool, heat=args.heat,
                           click_reweight=args.click_reweight, cursor_scale=tuple(float(v) for v in args.cursor_scale.split(",")), cursor_scale_p=args.cursor_scale_p,
                           latency_max_ms=args.latency_max, offsets=offsets, spin_recover=args.spin_recover, spin_weight=args.spin_weight)
    val = GpuChunkLoader(val_ex, device, stack=args.stack, real_lag=args.real_lag, batch=args.batch, chunk_bytes=chunk, shuffle=False, drop_last=False,
                         stride=args.val_stride, pool=args.pool, offsets=offsets)
    print(f"device={device} exports train={len(train_ex)} val={len(val_ex)} chunks={len(train.groups)}+{len(val.groups)} "
          f"batches={len(train)}+{len(val)} stack={args.stack} frame={frame_hw}")

    model = FrameNet(stack=args.stack, width=args.width, aim_head=args.aim_head).to(device, memory_format=torch.channels_last)
    amp = not args.no_amp
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    total_steps = args.epochs * (len(train) + 50)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=total_steps)
    aim_loss = nn.SmoothL1Loss(beta=0.02)
    press_weight = torch.tensor([args.press_weight], device=device)

    def loss_fn(out, aim, click, logits):
        parts = {"aim": aim_loss(out[:, :2], aim)}
        if logits is not None and args.att_loss > 0:
            parts["att"] = args.att_loss * att_loss(logits, aim)
        w = click[:, 2] if click.shape[1] > 2 else torch.ones_like(click[:, 0])
        parts["press"] = (w * F.binary_cross_entropy_with_logits(out[:, 2], click[:, 0], pos_weight=press_weight, reduction="none")).mean()
        parts["down"] = (w * F.binary_cross_entropy_with_logits(out[:, 3], click[:, 1], reduction="none")).mean()
        return parts["aim"] + parts.get("att", 0) + args.click_weight * (parts["press"] + parts["down"]), parts

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    config = {**{k: v for k, v in vars(args).items() if k != "exports"}, "n_exports": len(exports)}
    (out_dir / "config.json").write_text(json.dumps({**config, "exports": [str(e.path) for e in exports]}, indent=2))
    tracker = Tracker(args.wandb, args.run_name or out_dir.name, config, out_dir)
    if tracker.id:
        (out_dir / "wandb_run.json").write_text(json.dumps({"id": tracker.id, "project": args.wandb}))

    best = -1e9
    history = []
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

        val_stats = validate(model, val, device, amp)
        score = (val_stats["press_recall_tol1"] + val_stats["press_precision_tol1"]) / 2 - val_stats["val_mean_px"] / 100
        stats = {"epoch": epoch + 1, "train_loss": total / count, **val_stats, "seconds": time.perf_counter() - t0,
                 **{f"loss_{k}": s / count for k, s in part_sums.items()}}
        history.append(stats)
        print(json.dumps(stats))
        tracker.log({**stats, "score": score}, step=epoch + 1)
        if score > best:
            best = score
            torch.save({"model": model.state_dict(), "stack": args.stack, "width": args.width, "aim_head": args.aim_head, "frame_hw": frame_hw,
                        "latency_max_ms": args.latency_max, "offsets": offsets}, out_dir / "joint_best.pt")

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
