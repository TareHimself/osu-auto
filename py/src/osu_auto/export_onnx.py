import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from .dataset import load_frame_export
from .models import FrameNet
from .targets import AIM_HI, AIM_LO, PLAYFIELD_H, PLAYFIELD_W


def load_model(ck: dict) -> FrameNet:
    model = FrameNet(stack=ck["stack"], width=ck["width"], outputs=ck.get("outputs", 2), heads=ck.get("heads"), bounded_aim=ck.get("bounded_aim", False), aim_head=ck.get("aim_head", "mlp"))
    model.load_state_dict(ck["model"])
    return model.eval()


def _sample_stacks(frames_from: str | Path | None, offsets: list[int], hw: tuple[int, int], samples: int, rng: np.random.Generator):
    """Real frame stacks from an export when given, else uniform noise."""
    h, w = hw
    if frames_from is None:
        for _ in range(samples):
            yield rng.integers(0, 256, (len(offsets), h, w), dtype=np.uint8)
        return
    frames = np.asarray(load_frame_export(frames_from).frames)
    pool = frames.shape[1] // h
    for i in rng.integers(max(offsets), len(frames), samples):
        stack = np.stack([frames[i - k] for k in reversed(offsets)])
        if pool > 1:
            stack = stack.reshape(len(offsets), h, pool, w, pool).mean(axis=(2, 4)).round().astype(np.uint8)
        yield stack


def export(checkpoint: str | Path, out: str | Path | None = None, samples: int = 100, frames_from: str | Path | None = None,
           tol_px: float = 0.5, tol_prob: float = 0.01) -> Path:
    ck = torch.load(checkpoint, map_location="cpu")
    model = load_model(ck)
    h, w = ck["frame_hw"]
    stack = ck["stack"]
    out = Path(out) if out else Path(checkpoint).with_suffix(".onnx")

    frames = torch.zeros(1, stack, h, w)
    aux = torch.zeros(1, 2)
    torch.onnx.export(model, (frames, aux), str(out), opset_version=17, input_names=["frames", "aux"], output_names=["out"], dynamo=True)

    meta = {"stack": stack, "offsets": ck.get("offsets"), "latency_max_ms": float(ck.get("latency_max_ms", 0.0)), "frame_hw": [h, w], "outputs": int(ck.get("outputs", 2))}
    out.with_suffix(".json").write_text(json.dumps(meta, indent=2))

    rng = np.random.default_rng(0)
    sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    px_per_unit = np.array([PLAYFIELD_W, PLAYFIELD_H]) / (AIM_HI - AIM_LO)
    worst_px = worst_prob = 0.0
    for s in _sample_stacks(frames_from, ck.get("offsets") or list(range(stack)), (h, w), samples, rng):
        f = torch.from_numpy(s[None]).float() / 255.0
        a = torch.tensor([[rng.random(), 0.0]], dtype=torch.float32)
        with torch.no_grad():
            ref = model(f, a)[0].double()
        got = torch.from_numpy(sess.run(None, {"frames": f.numpy(), "aux": a.numpy()})[0][0]).double()
        worst_px = max(worst_px, float(np.hypot(*((ref[:2] - got[:2]).numpy() * px_per_unit))))
        worst_prob = max(worst_prob, float((torch.sigmoid(ref[2:]) - torch.sigmoid(got[2:])).abs().max()))
    source = "noise" if frames_from is None else "real frames"
    print(f"wrote {out}, worst over {samples} {source}: aim {worst_px:.2e} px, click prob {worst_prob:.2e}")
    if worst_px > tol_px or worst_prob > tol_prob:
        raise RuntimeError(f"onnx mismatch: aim {worst_px:.2e} px (max {tol_px}), click prob {worst_prob:.2e} (max {tol_prob})")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Export a joint checkpoint to ONNX for the native runner.")
    ap.add_argument("checkpoint")
    ap.add_argument("--out", default=None, help="output .onnx path (default: next to the checkpoint)")
    ap.add_argument("--samples", type=int, default=100)
    ap.add_argument("--frames-from", default=None, help="frame export to draw real test stacks from (default: noise)")
    ap.add_argument("--tol-px", type=float, default=0.5, help="max aim difference in playfield pixels")
    ap.add_argument("--tol-prob", type=float, default=0.01, help="max click probability difference")
    args = ap.parse_args()
    try:
        export(args.checkpoint, args.out, args.samples, args.frames_from, args.tol_px, args.tol_prob)
    except RuntimeError as e:
        raise SystemExit(str(e))


if __name__ == "__main__":
    main()
