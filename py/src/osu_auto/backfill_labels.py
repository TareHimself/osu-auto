"""Re-run the label export for existing exports so their meta gains the strain data; frames are kept."""
import argparse
import json
import time
from pathlib import Path

from osu_auto.export_jobs import Host, map_path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("jobs")
    p.add_argument("roots", nargs="+", help="export roots, one per skin")
    p.add_argument("--skin", default="F:/osu-auto/data/skins/angelmegumin-ai-friendly.osk")
    args = p.parse_args()
    jobs = json.loads(Path(args.jobs).read_text())
    log_dir = Path(args.roots[0]) / "_logs"
    log_dir.mkdir(exist_ok=True)
    host = Host(9, args.skin, argparse.Namespace(window="640x480"), log_dir)
    t0 = time.perf_counter()
    done = 0
    try:
        for root in args.roots:
            for job in jobs:
                d = Path(host.client.export_labels(map_path(job), root, diff=job.get("diff"), osr=job.get("osr"), mods=job.get("mods")))
                meta = json.loads((d / "meta.json").read_text())
                done += 1
                if done % 20 == 0:
                    print(f"{done} done, {time.perf_counter() - t0:.0f}s, last {d.name} frames={'frames' in meta} strains={'strains' in meta}", flush=True)
    finally:
        host.close()
    print(f"{done} exports relabelled in {time.perf_counter() - t0:.0f}s")


if __name__ == "__main__":
    main()
