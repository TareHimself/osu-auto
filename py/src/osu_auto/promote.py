import argparse
import json
import shutil
from pathlib import Path

FILES = ("joint_best.onnx", "joint_best.onnx.data", "joint_best.json")


def main() -> None:
    ap = argparse.ArgumentParser(description="Make a run's ONNX export the runner's default model (data/models/current).")
    ap.add_argument("run", help="run folder under data/models, or a path to it")
    ap.add_argument("--models", default="../data/models")
    args = ap.parse_args()

    models = Path(args.models)
    run = Path(args.run) if Path(args.run).is_dir() else models / args.run
    missing = [f for f in FILES if not (run / f).exists()]
    if missing:
        raise SystemExit(f"{run} is missing {', '.join(missing)}; run python -m osu_auto.export_onnx {run / 'joint_best.pt'}")
    current = models / "current"
    tmp = models / "current.tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir()
    for f in FILES:
        shutil.copy2(run / f, tmp / f)
    (tmp / "source.json").write_text(json.dumps({"run": run.resolve().name}, indent=2))
    shutil.rmtree(current, ignore_errors=True)
    tmp.rename(current)
    print(f"current -> {run.resolve().name}")


if __name__ == "__main__":
    main()
