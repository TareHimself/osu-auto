import argparse
import json
import queue
import re
import socket
import subprocess
import threading
import time
from pathlib import Path

from osu_auto.cursor import CursorSprite
from osu_auto.export_frames import calibrate_cursor, export_frames
from osu_auto.geometry import PlayArea
from osu_auto.host import HostClient

ROOT = Path(__file__).resolve().parents[3]
HOST_EXE = ROOT / "host/OsuAuto.Host/bin/Release/net10.0/OsuAuto.Host.exe"
HOST_CWD = ROOT / "host"
BEATMAPS = ROOT / "data/beatmaps"


def map_path(job: dict) -> str:
    """Job maps are file names under data/beatmaps, or absolute paths."""
    p = Path(job["map"])
    return str(p if p.is_absolute() else BEATMAPS / p)


def wait_port(port: int, timeout: float = 90.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        with socket.socket() as s:
            s.settimeout(0.5)
            if s.connect_ex(("127.0.0.1", port)) == 0:
                return
        time.sleep(0.5)
    raise TimeoutError(f"host on port {port} did not come up")


class Host:
    def __init__(self, i: int, skin: str, args, log_dir: Path):
        self.port = 5555 + i
        self.skin = skin
        self.log = open(log_dir / f"host-{i}-{Path(skin).stem}.log", "w")
        self.proc = subprocess.Popen([str(HOST_EXE), "serve", "--window", args.window, "--skin", skin, "--instance", str(i)],
                                     cwd=HOST_CWD, stdout=self.log, stderr=subprocess.STDOUT)
        wait_port(self.port)
        self.client = HostClient(f"tcp://127.0.0.1:{self.port}", timeout_ms=600_000)
        self.sprite: CursorSprite | None = None

    def cursor_sprite(self, job: dict) -> CursorSprite:
        """The cursor depends on the skin and window, not the map, so calibrate once per host."""
        if self.sprite is None:
            client = self.client.status()["client_size"]
            self.sprite = calibrate_cursor(self.client, map_path(job), job.get("diff"), None if job.get("osr") else job.get("mods"), PlayArea(client[0], client[1]))
        return self.sprite

    def close(self) -> None:
        try:
            self.client.quit()
        except Exception:
            pass
        self.client.close()
        try:
            self.proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.proc.kill()
        self.log.close()


def has_spinner(export_dir: Path) -> bool:
    """Spinner visuals depend on how the spinner was spun before, so a seek-based determinism check cannot pass on them."""
    import numpy as np

    with np.load(export_dir / "objects.npz") as z:
        return bool((z["kind"] == 2).any()) if "kind" in z.files else False


class Progress:
    """Shared export status, written to _logs/status.json once a second for `export_status` to display."""

    def __init__(self, path: Path, total: int, wandb_project: str | None = None):
        self.path = path
        self.total = total
        self.t0 = time.time()
        self.lock = threading.Lock()
        self.hosts: dict[int, dict] = {}
        self.done = 0
        self.failed = 0
        self.frames = 0
        self.failures: list[dict] = []
        self.run = None
        self._last = 0.0
        if wandb_project:
            import wandb

            self.run = wandb.init(project=wandb_project, name=f"export-{time.strftime('%m%d-%H%M')}", job_type="export", group="exports", tags=["export"], config={"jobs": total})

    def update(self, i: int, **fields) -> None:
        with self.lock:
            self.hosts.setdefault(i, {}).update(fields)
        self.flush()

    def finished(self, frames: int | None, error: str | None, name: str = "") -> None:
        with self.lock:
            if error:
                self.failed += 1
                self.failures.append({"map": name, "error": error.splitlines()[0][:200]})
            else:
                self.done += 1
                self.frames += frames or 0
        self.flush(force=True)

    def flush(self, force: bool = False) -> None:
        now = time.time()
        if not force and now - self._last < 1.0:
            return
        self._last = now
        with self.lock:
            elapsed = now - self.t0
            finished = self.done + self.failed
            per_job = elapsed / finished if finished else None
            snap = {"updated": now, "started": self.t0, "elapsed_s": elapsed, "total": self.total, "done": self.done, "failed": self.failed,
                    "remaining": self.total - finished, "frames": self.frames, "frames_per_s": self.frames / max(elapsed, 1e-6),
                    "eta_s": per_job * (self.total - finished) if per_job else None, "hosts": self.hosts, "failures": self.failures}
        try:
            self.path.write_text(json.dumps(snap, indent=1))
        except OSError:
            pass
        if self.run:
            self.run.log({"done": snap["done"], "failed": snap["failed"], "frames_per_s": snap["frames_per_s"]})

    def close(self) -> None:
        self.flush(force=True)
        if self.run:
            self.run.finish()


def worker(i: int, jobs: "queue.Queue[tuple[dict, str, Path]]", args, results: list, log_dir: Path, progress: Progress) -> None:
    tag = f"[w{i}]"

    def log(msg: str) -> None:
        print(f"{tag} {time.strftime('%H:%M:%S')} {msg}", flush=True)
        m = re.match(r"(\d+)/(\d+) t=\S+ ([\d.]+) steps/s", msg)
        if m:
            progress.update(i, stage="frames", step=int(m.group(1)), steps=int(m.group(2)), steps_per_s=float(m.group(3)))
        elif msg.startswith("loaded "):
            progress.update(i, stage="loading", step=0)
        elif msg.startswith("self-check"):
            progress.update(i, stage="self-check")

    host: Host | None = None
    try:
        while True:
            try:
                job, skin, out = jobs.get_nowait()
            except queue.Empty:
                break
            if host is None or host.skin != skin:
                if host is not None:
                    host.close()
                log(f"host with skin {Path(skin).stem}")
                host = Host(i, skin, args, log_dir)
            started = time.perf_counter()
            progress.update(i, map=f"{Path(job['map']).stem} [{job.get('diff') or 'lowest'}]", skin=Path(skin).stem, stage="labels", step=0, steps=0)
            try:
                export_dir = Path(host.client.export_labels(map_path(job), str(out), diff=job.get("diff"), osr=job.get("osr"), mods=job.get("mods"), spin_recovery=job.get("spin_recovery")))
                osr = job.get("osr") or json.loads((export_dir / "meta.json").read_text()).get("osr_path")
                log(f"labels -> {export_dir}")
                done = json.loads((export_dir / "meta.json").read_text()).get("frames")
                if done and (export_dir / done["path"]).exists():
                    log(f"already exported, skipping {export_dir.name}")
                    progress.finished(0, None)
                    continue
                fm = export_frames(host.client, map_path(job), export_dir, diff=job.get("diff"), osr=osr, mods=None if osr else job.get("mods"), size=args.size_wh, limit=args.limit, log=log,
                                   sprite=host.cursor_sprite(job), strict=not has_spinner(export_dir))
                results.append({"job": job, "skin": skin, "export": str(export_dir), "frames": fm["count"], "steps_per_s": fm["steps_per_s"], "seconds": time.perf_counter() - started})
                progress.finished(fm["count"], None)
            except Exception as e:
                log(f"FAILED {job}: {e}")
                results.append({"job": job, "skin": skin, "error": str(e)})
                progress.finished(None, str(e), f"{Path(job['map']).stem} [{job.get('diff') or 'lowest'}]")
    finally:
        progress.update(i, stage="idle", map="")
        if host is not None:
            host.close()


def main() -> None:
    p = argparse.ArgumentParser(description="Export labels and frames for a list of (map, diff, mods) jobs across parallel host processes")
    p.add_argument("jobs", help="json file: list of {map (file name under data/beatmaps), diff?, osr?, mods?: [spec, ...]}")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--out", default=str(ROOT / "data/exports"))
    p.add_argument("--skin", action="append", help="skin .osk; repeat for several skins, exports then go to --out/<skin name>/")
    p.add_argument("--window", default="640x480")
    p.add_argument("--size", default="160x120")
    p.add_argument("--limit", type=int, default=None, help="only export the first N steps of each job (benchmarks)")
    p.add_argument("--wandb", default=None, help="log progress of this batch to a wandb project")
    args = p.parse_args()
    args.size_wh = tuple(int(v) for v in args.size.split("x"))

    jobs_list = json.loads(Path(args.jobs).read_text())
    skins = args.skin or [str(ROOT / "data/skins/angelmegumin-ai-friendly.osk")]
    jobs: "queue.Queue[tuple[dict, str, Path]]" = queue.Queue()
    for skin in skins:
        out = Path(args.out) / Path(skin).stem if len(skins) > 1 else Path(args.out)
        for j in jobs_list:
            jobs.put((j, skin, out))

    log_dir = Path(args.out) / "_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict] = []
    t0 = time.perf_counter()
    progress = Progress(log_dir / "status.json", jobs.qsize(), args.wandb)

    threads = [threading.Thread(target=worker, args=(i, jobs, args, results, log_dir, progress), daemon=True) for i in range(min(args.workers, jobs.qsize()))]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    progress.close()

    total = time.perf_counter() - t0
    frames = sum(r.get("frames", 0) for r in results)
    print(f"\n{len(results)} jobs, {frames} frames in {total:.0f} s ({frames / max(total, 1e-6):.0f} frames/s aggregate)")
    for r in results:
        j = r["job"]
        name = f"{Path(j['map']).stem} [{j.get('diff') or 'lowest'}] {'+'.join(j.get('mods') or [])} {Path(r['skin']).stem}"
        print(f"  {name}: " + (f"{r['frames']} frames, {r['steps_per_s']:.0f} steps/s, {r['seconds']:.0f} s" if "frames" in r else f"FAILED {r['error']}"))
    (log_dir / "last_run.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
