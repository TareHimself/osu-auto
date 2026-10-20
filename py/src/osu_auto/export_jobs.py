import argparse
import json
import queue
import re
import socket
import subprocess
import threading
import time
from pathlib import Path

from osu_auto.export_frames import export_frames
from osu_auto.host import HostClient

HOST_EXE = Path("F:/osu-auto/host/OsuAuto.Host/bin/Release/net10.0/OsuAuto.Host.exe")
HOST_CWD = Path("F:/osu-auto/host")
LABELS_INSTANCE_BASE = 50


def wait_port(port: int, timeout: float = 90.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        with socket.socket() as s:
            s.settimeout(0.5)
            if s.connect_ex(("127.0.0.1", port)) == 0:
                return
        time.sleep(0.5)
    raise TimeoutError(f"host on port {port} did not come up")


def export_labels(job: dict, out: Path, instance: int, log) -> Path:
    cmd = [str(HOST_EXE), "export-labels", job["map"], "--out", str(out), "--instance", str(instance)]
    if job.get("diff"):
        cmd += ["--diff", job["diff"]]
    if job.get("osr"):
        cmd += ["--osr", job["osr"]]
    if job.get("mods"):
        cmd += ["--mods", ";".join(job["mods"])]
    r = subprocess.run(cmd, cwd=HOST_CWD, capture_output=True, text=True)
    m = re.search(r"-> (.+)$", r.stdout, re.M)
    if r.returncode != 0 or not m:
        raise RuntimeError(f"export-labels failed for {job}:\n{r.stdout}\n{r.stderr}")
    log(f"labels -> {m.group(1).strip()}")
    return Path(m.group(1).strip())


def worker(i: int, jobs: "queue.Queue[dict]", args, results: list, log_dir: Path) -> None:
    tag = f"[w{i}]"

    def log(msg: str) -> None:
        print(f"{tag} {msg}", flush=True)

    port = 5555 + i
    host_log = open(log_dir / f"host-{i}.log", "w")
    proc = subprocess.Popen([str(HOST_EXE), "serve", "--window", args.window, "--skin", args.skin, "--instance", str(i)],
                            cwd=HOST_CWD, stdout=host_log, stderr=subprocess.STDOUT)
    try:
        wait_port(port)
        with HostClient(f"tcp://127.0.0.1:{port}", timeout_ms=600_000) as host:
            while True:
                try:
                    job = jobs.get_nowait()
                except queue.Empty:
                    break
                started = time.perf_counter()
                try:
                    export_dir = export_labels(job, Path(args.out), LABELS_INSTANCE_BASE + i, log)
                    fm = export_frames(host, job["map"], export_dir, diff=job.get("diff"), osr=job.get("osr"), mods=job.get("mods"), size=args.size_wh, log=log)
                    results.append({"job": job, "export": str(export_dir), "frames": fm["count"], "steps_per_s": fm["steps_per_s"], "seconds": time.perf_counter() - started})
                except Exception as e:
                    log(f"FAILED {job}: {e}")
                    results.append({"job": job, "error": str(e)})
            host.quit()
    finally:
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
        host_log.close()


def main() -> None:
    p = argparse.ArgumentParser(description="Export labels and frames for a list of (map, diff, mods) jobs across parallel host processes")
    p.add_argument("jobs", help="json file: list of {map, diff?, osr?, mods?: [spec, ...]}")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--out", default="F:/osu-auto/data/exports")
    p.add_argument("--skin", default="F:/osu-auto/data/skins/angelmegumin-ai-friendly.osk")
    p.add_argument("--window", default="640x480")
    p.add_argument("--size", default="160x120")
    args = p.parse_args()
    args.size_wh = tuple(int(v) for v in args.size.split("x"))

    jobs_list = json.loads(Path(args.jobs).read_text())
    jobs: "queue.Queue[dict]" = queue.Queue()
    for j in jobs_list:
        jobs.put(j)

    log_dir = Path(args.out) / "_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict] = []
    t0 = time.perf_counter()

    threads = [threading.Thread(target=worker, args=(i, jobs, args, results, log_dir), daemon=True) for i in range(min(args.workers, len(jobs_list)))]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    total = time.perf_counter() - t0
    frames = sum(r.get("frames", 0) for r in results)
    print(f"\n{len(results)} jobs, {frames} frames in {total:.0f} s ({frames / max(total, 1e-6):.0f} frames/s aggregate)")
    for r in results:
        j = r["job"]
        name = f"{Path(j['map']).stem} [{j.get('diff') or 'lowest'}] {'+'.join(j.get('mods') or [])}"
        print(f"  {name}: " + (f"{r['frames']} frames, {r['steps_per_s']:.0f} steps/s, {r['seconds']:.0f} s" if "frames" in r else f"FAILED {r['error']}"))
    (log_dir / "last_run.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
