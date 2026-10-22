"""Export progress window: `python -m osu_auto.export_status [out_dir]` (default: the newest export root under data/).
`--once` prints a text snapshot instead."""
import argparse
import json
import time
import tkinter as tk
from pathlib import Path
from tkinter import ttk

ROOT = Path(__file__).resolve().parents[3]


def newest_status() -> Path | None:
    found = sorted(ROOT.glob("data/**/_logs/status.json"), key=lambda p: p.stat().st_mtime)
    return found[-1] if found else None


def minutes(s: float | None) -> str:
    return "?" if s is None else f"{s / 60:.1f} min"


def render_text(snap: dict) -> str:
    lines = [f"{snap['done']}/{snap['total']} done, {snap['failed']} failed, {snap['remaining']} left | {snap['frames_per_s']:.0f} frames/s | "
             f"elapsed {minutes(snap['elapsed_s'])} | eta {minutes(snap.get('eta_s'))}"]
    for i in sorted(snap["hosts"], key=int):
        h = snap["hosts"][i]
        frac = h.get("step", 0) / h["steps"] if h.get("steps") else 0
        lines.append(f"host {i:>2} {h.get('stage', ''):>10} {frac:4.0%} {h.get('map', '')}")
    lines += [f"FAILED {f['map']}: {f['error']}" for f in snap.get("failures", [])]
    return "\n".join(lines)


class Window:
    def __init__(self, path: Path, close_after: float | None):
        self.path = path
        self.root = tk.Tk()
        self.root.title("osu-auto export")
        self.root.geometry("760x520")
        self.root.configure(padx=14, pady=12)
        style = ttk.Style(self.root)
        style.configure("Big.TLabel", font=("Segoe UI", 13, "bold"))

        self.source = ttk.Label(self.root, text=str(path), foreground="#666")
        self.source.pack(anchor="w")
        self.headline = ttk.Label(self.root, style="Big.TLabel", text="waiting for status")
        self.headline.pack(anchor="w", pady=(6, 2))
        self.total_bar = ttk.Progressbar(self.root, maximum=1.0, length=720)
        self.total_bar.pack(fill="x")
        self.detail = ttk.Label(self.root, text="")
        self.detail.pack(anchor="w", pady=(4, 10))

        self.hosts_frame = ttk.Frame(self.root)
        self.hosts_frame.pack(fill="x")
        self.host_rows: dict[str, tuple] = {}

        ttk.Label(self.root, text="Failures").pack(anchor="w", pady=(12, 2))
        self.failures = tk.Text(self.root, height=6, wrap="word", relief="flat", background="#f6f6f6")
        self.failures.pack(fill="both", expand=True)

        self.root.after(0, self.refresh)
        if close_after:
            self.root.after(int(close_after * 1000), self.root.destroy)

    def host_row(self, key: str):
        if key not in self.host_rows:
            row = ttk.Frame(self.hosts_frame)
            row.pack(fill="x", pady=2)
            name = ttk.Label(row, text=f"host {key}", width=8)
            name.pack(side="left")
            stage = ttk.Label(row, width=10)
            stage.pack(side="left")
            bar = ttk.Progressbar(row, maximum=1.0, length=180)
            bar.pack(side="left", padx=6)
            rate = ttk.Label(row, width=9)
            rate.pack(side="left")
            label = ttk.Label(row, width=48)
            label.pack(side="left", fill="x")
            self.host_rows[key] = (stage, bar, rate, label)
        return self.host_rows[key]

    def refresh(self) -> None:
        try:
            snap = json.loads(self.path.read_text())
        except (OSError, ValueError):
            self.root.after(1000, self.refresh)
            return
        finished = snap["done"] + snap["failed"]
        age = time.time() - snap["updated"]
        state = "finished" if snap["remaining"] == 0 else ("stalled or stopped" if age > 15 else "running")
        self.headline.configure(text=f"{finished}/{snap['total']} jobs  ·  {state}")
        self.total_bar["value"] = finished / max(snap["total"], 1)
        self.detail.configure(text=f"{snap['done']} done, {snap['failed']} failed, {snap['remaining']} left  ·  {snap['frames_per_s']:.0f} frames/s  ·  "
                                   f"elapsed {minutes(snap['elapsed_s'])}  ·  eta {minutes(snap.get('eta_s'))}  ·  updated {age:.0f} s ago")
        for key in sorted(snap["hosts"], key=int):
            h = snap["hosts"][key]
            stage, bar, rate, label = self.host_row(key)
            stage.configure(text=h.get("stage", ""))
            bar["value"] = h.get("step", 0) / h["steps"] if h.get("steps") else 0
            rate.configure(text=f"{h.get('steps_per_s', 0):.0f}/s" if h.get("stage") == "frames" else "")
            label.configure(text=h.get("map", "")[:60])
        self.failures.delete("1.0", "end")
        self.failures.insert("end", "\n".join(f"{f['map']}: {f['error']}" for f in snap.get("failures", [])) or "none")
        self.root.after(1000, self.refresh)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("out_dir", nargs="?", help="export output root (the one passed to export_jobs --out); default: newest")
    p.add_argument("--once", action="store_true", help="print a text snapshot and exit")
    p.add_argument("--close-after", type=float, default=None, help=argparse.SUPPRESS)
    args = p.parse_args()
    path = Path(args.out_dir) / "_logs" / "status.json" if args.out_dir else newest_status()
    if path is None:
        raise SystemExit("no export status found under data/")
    if args.once:
        print(render_text(json.loads(path.read_text())))
        return
    Window(path, args.close_after).root.mainloop()


if __name__ == "__main__":
    main()
