import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from .models import FrameNet


def load_model(ck: dict) -> FrameNet:
    model = FrameNet(stack=ck["stack"], width=ck["width"], outputs=ck.get("outputs", 2), heads=ck.get("heads"), bounded_aim=ck.get("bounded_aim", False))
    model.load_state_dict(ck["model"])
    return model.eval()


def main() -> None:
    ap = argparse.ArgumentParser(description="Export a joint checkpoint to ONNX for the native runner.")
    ap.add_argument("checkpoint")
    ap.add_argument("--out", default=None, help="output .onnx path (default: next to the checkpoint)")
    ap.add_argument("--samples", type=int, default=100)
    ap.add_argument("--tol", type=float, default=1e-3)
    args = ap.parse_args()

    ck = torch.load(args.checkpoint, map_location="cpu")
    model = load_model(ck)
    h, w = ck["frame_hw"]
    stack = ck["stack"]
    out = Path(args.out) if args.out else Path(args.checkpoint).with_suffix(".onnx")

    frames = torch.zeros(1, stack, h, w)
    aux = torch.zeros(1, 2)
    torch.onnx.export(model, (frames, aux), str(out), opset_version=17, input_names=["frames", "aux"], output_names=["out"], dynamo=True)

    meta = {"stack": stack, "offsets": ck.get("offsets"), "latency_max_ms": float(ck.get("latency_max_ms", 0.0)), "frame_hw": [h, w], "outputs": int(ck.get("outputs", 2))}
    out.with_suffix(".json").write_text(json.dumps(meta, indent=2))

    rng = np.random.default_rng(0)
    sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    worst = 0.0
    for _ in range(args.samples):
        f = torch.from_numpy(rng.integers(0, 256, (1, stack, h, w), dtype=np.uint8)).float() / 255.0
        a = torch.tensor([[rng.random(), 0.0]], dtype=torch.float32)
        with torch.no_grad():
            ref = model(f, a).numpy()
        got = sess.run(None, {"frames": f.numpy(), "aux": a.numpy()})[0]
        worst = max(worst, float(np.abs(ref - got).max()))
    print(f"wrote {out}, max abs diff over {args.samples} samples: {worst:.2e}")
    if worst > args.tol:
        raise SystemExit(f"diff {worst:.2e} exceeds {args.tol:.0e}")


if __name__ == "__main__":
    main()
