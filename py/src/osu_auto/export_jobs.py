import argparse
import json
import queue
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


def worker(i: int, jobs: "queue.Queue[tuple[dict, str, Path]]", args, results: list, log_dir: Path) -> None:
    tag = f"[w{i}]"

    def log(msg: str) -> None:
        print(f"{tag} {time.strftime('%H:%M:%S')} {msg}", flush=True)

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
            try:
                export_dir = Path(host.client.export_labels(map_path(job), str(out), diff=job.get("diff"), osr=job.get("osr"), mods=job.get("mods")))
                log(f"labels -> {export_dir}")
                done = json.loads((export_dir / "meta.json").read_text()).get("frames")
                if done and (export_dir / done["path"]).exists():
                    log(f"already exported, skipping {export_dir.name}")
                    continue
                fm = export_frames(host.client, map_path(job), export_dir, diff=job.get("diff"), osr=job.get("osr"), mods=job.get("mods"), size=args.size_wh, limit=args.limit, log=log,
                                   sprite=host.cursor_sprite(job))
                results.append({"job": job, "skin": skin, "export": str(export_dir), "frames": fm["count"], "steps_per_s": fm["steps_per_s"], "seconds": time.perf_counter() - started})
            except Exception as e:
                log(f"FAILED {job}: {e}")
                results.append({"job": job, "skin": skin, "error": str(e)})
    finally:
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

    threads = [threading.Thread(target=worker, args=(i, jobs, args, results, log_dir), daemon=True) for i in range(min(args.workers, jobs.qsize()))]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

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
