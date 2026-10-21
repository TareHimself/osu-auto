import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset

from osu_auto.dataset import ClickDataset, load_frame_export, worker_init
from osu_auto.models import FrameNet
from osu_auto.train_aim import split_indices


def tolerant_match(true_idx: np.ndarray, pred_idx: np.ndarray, tol: int) -> tuple[float, float]:
    """Recall and precision of press events allowing a timing slack of `tol` steps."""
    if len(true_idx) == 0 or len(pred_idx) == 0:
        return 0.0, 0.0
    recall = np.mean([np.min(np.abs(pred_idx - t)) <= tol for t in true_idx])
    precision = np.mean([np.min(np.abs(true_idx - p)) <= tol for p in pred_idx])
    return float(recall), float(precision)


def main() -> None:
    p = argparse.ArgumentParser(description="Behaviour-clone clicking from exported frames: press and hold decisions per step")
    p.add_argument("exports", nargs="+")
    p.add_argument("--out", default="F:/osu-auto/data/models/click")
    p.add_argument("--stack", type=int, default=4)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--holdout", type=float, default=0.15)
    p.add_argument("--width", type=int, default=32)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--real-lag", type=int, default=2)
    p.add_argument("--press-weight", type=float, default=8.0, help="positive class weight for the press head")
    p.add_argument("--horizon", type=int, default=1, help="label this many steps after the last frame (1 = the closed-loop delay, 0 leaks the hit animation)")
    p.add_argument("--since-press", action="store_true", help="feed steps since the last press (with --no-keys)")
    p.add_argument("--no-keys", action="store_true", help="no key-state input; second head predicts 'a key is down' instead of 'keep holding'")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    exports = [load_frame_export(e) for e in args.exports]
    ds = ClickDataset(exports, stack=args.stack, horizon=args.horizon, real_lag=args.real_lag, use_keys=not args.no_keys, since_press=args.since_press)
    train_idx, val_idx = split_indices(len(ds), args.holdout)
    train = DataLoader(Subset(ds, train_idx), batch_size=args.batch, shuffle=True, num_workers=args.workers, drop_last=True, worker_init_fn=worker_init, persistent_workers=args.workers > 0)
    val = DataLoader(Subset(ds, val_idx), batch_size=args.batch, shuffle=False, num_workers=args.workers, worker_init_fn=worker_init, persistent_workers=args.workers > 0)
    print(f"device={device} samples train={len(train_idx)} val={len(val_idx)} stack={args.stack} frame={exports[0].frame_hw}")

    model = FrameNet(stack=args.stack, width=args.width, outputs=2).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.epochs * len(train))
    press_weight = torch.tensor([args.press_weight], device=device)

    def loss_fn(logits, target, hold_mask):
        press = F.binary_cross_entropy_with_logits(logits[:, 0], target[:, 0], pos_weight=press_weight)
        hold = F.binary_cross_entropy_with_logits(logits[:, 1], target[:, 1], reduction="none")
        hold = (hold * hold_mask).sum() / hold_mask.sum().clamp(min=1)
        return press + hold

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    best = -1.0
    history = []

    for epoch in range(args.epochs):
        model.train()
        t0 = time.perf_counter()
        total, count = 0.0, 0
        for frames, keys, target, hold_mask in train:
            frames, keys, target, hold_mask = frames.to(device), keys.to(device), target.to(device), hold_mask.to(device)
            loss = loss_fn(model(frames, keys), target, hold_mask)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            total += loss.item() * len(frames)
            count += len(frames)

        model.eval()
        preds, targets, masks = [], [], []
        with torch.no_grad():
            for frames, keys, target, hold_mask in val:
                preds.append(torch.sigmoid(model(frames.to(device), keys.to(device))).cpu().numpy())
                targets.append(target.numpy())
                masks.append(hold_mask.numpy())
        pred = np.concatenate(preds)
        tgt = np.concatenate(targets)
        mask = np.concatenate(masks).astype(bool)
        true_press = np.flatnonzero(tgt[:, 0] > 0.5)
        pred_press = np.flatnonzero(pred[:, 0] > 0.5)
        recall0, precision0 = tolerant_match(true_press, pred_press, 0)
        recall1, precision1 = tolerant_match(true_press, pred_press, 1)
        hold_acc = float(((pred[mask, 1] > 0.5) == (tgt[mask, 1] > 0.5)).mean()) if mask.any() else 0.0
        score = (recall1 + precision1) / 2
        stats = {"epoch": epoch + 1, "train_loss": total / count, "press_recall_exact": recall0, "press_precision_exact": precision0,
                 "press_recall_tol1": recall1, "press_precision_tol1": precision1, "hold_acc": hold_acc,
                 "true_presses": int(len(true_press)), "pred_presses": int(len(pred_press)), "seconds": time.perf_counter() - t0}
        history.append(stats)
        print(json.dumps(stats))

        if score > best:
            best = score
            torch.save({"model": model.state_dict(), "stack": args.stack, "width": args.width, "outputs": 2, "frame_hw": list(exports[0].frame_hw),
                        "use_keys": not args.no_keys, "since_press": args.since_press, "horizon": args.horizon}, out / "click_best.pt")

    (out / "history.json").write_text(json.dumps(history, indent=2))
    print(f"best press score (mean of recall and precision at +-1 step) {best:.3f} -> {out / 'click_best.pt'}")


if __name__ == "__main__":
    main()
