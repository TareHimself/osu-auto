"""Play osu! from the screen: capture a game window, run the aim and click models, inject mouse and keys.

Start it, pick the osu! window, press Start (or F8 in the game). F8 toggles, F9 stops. The game must have the AI-friendly
skin, 100% background dim, beatmap skins and colours off, hit lighting off, no cursor trail, HUD hidden, and raw input off.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import ctypes
import threading
import time
import tkinter as tk
from collections import deque
from ctypes import wintypes
from tkinter import ttk

import cv2
import numpy as np
import torch
import win32gui

from osu_auto.capture import WindowCapture
from osu_auto.eval_click import KeyExecutor
from windows_capture import Frame, InternalCaptureControl
from osu_auto.models import FrameNet
from osu_auto.targets import decode_to_playfield

ROOT = __import__("pathlib").Path(__file__).resolve().parents[3]
LOG_DIR = ROOT / "data/play_logs"
STEP_MS = 10.0
HOTKEY_TOGGLE, HOTKEY_STOP = 1, 2
VK_F8, VK_F9 = 0x77, 0x78


class Injector:
    """SendInput: absolute mouse moves on the virtual desktop and scan-code key events."""

    class MOUSEINPUT(ctypes.Structure):
        _fields_ = [("dx", wintypes.LONG), ("dy", wintypes.LONG), ("mouseData", wintypes.DWORD), ("dwFlags", wintypes.DWORD),
                    ("time", wintypes.DWORD), ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong))]

    class KEYBDINPUT(ctypes.Structure):
        _fields_ = [("wVk", wintypes.WORD), ("wScan", wintypes.WORD), ("dwFlags", wintypes.DWORD), ("time", wintypes.DWORD),
                    ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong))]

    def __init__(self):
        class U(ctypes.Union):
            _fields_ = [("mi", Injector.MOUSEINPUT), ("ki", Injector.KEYBDINPUT)]

        class INPUT(ctypes.Structure):
            _fields_ = [("type", wintypes.DWORD), ("u", U)]

        self.INPUT = INPUT
        self.user32 = ctypes.windll.user32
        self.user32.SendInput.argtypes = (wintypes.UINT, ctypes.POINTER(INPUT), ctypes.c_int)
        self.vx, self.vy = self.user32.GetSystemMetrics(76), self.user32.GetSystemMetrics(77)
        self.vw, self.vh = self.user32.GetSystemMetrics(78), self.user32.GetSystemMetrics(79)

    def _send(self, *inputs) -> None:
        arr = (self.INPUT * len(inputs))(*inputs)
        self.user32.SendInput(len(inputs), arr, ctypes.sizeof(self.INPUT))

    def move(self, x: float, y: float) -> None:
        """SetCursorPos rather than SendInput: synthetic mouse events are rejected on some systems while cursor moves are not."""
        self.user32.SetCursorPos(int(round(x)), int(round(y)))

    def key(self, scan: int, down: bool) -> None:
        inp = self.INPUT(type=1)
        inp.u.ki = self.KEYBDINPUT(0, scan, 0x0008 | (0 if down else 0x0002), 0, None)
        self._send(inp)

    def button(self, index: int, down: bool) -> None:
        """Mouse button 0 (left) or 1 (right) at the current cursor position."""
        flags = (0x0002 if down else 0x0004) if index == 0 else (0x0008 if down else 0x0010)
        inp = self.INPUT(type=0)
        inp.u.mi = self.MOUSEINPUT(0, 0, 0, flags, 0, None)
        self._send(inp)

    @staticmethod
    def scan_code(char: str) -> int:
        vk = ctypes.windll.user32.VkKeyScanW(ord(char)) & 0xFF
        return ctypes.windll.user32.MapVirtualKeyW(vk, 0)


class FrameRing:
    """Recent observations with capture timestamps, so a 10 ms spaced stack can be sampled at any moment."""

    def __init__(self, maxlen: int = 128):
        self.items: deque[tuple[float, np.ndarray]] = deque(maxlen=maxlen)
        self.lock = threading.Lock()

    def push(self, t: float, obs: np.ndarray) -> None:
        with self.lock:
            self.items.append((t, obs))

    def stack(self, now: float, n: int, spacing_s: float, offsets: list[int] | None = None) -> tuple[np.ndarray, float] | None:
        with self.lock:
            if not self.items:
                return None
            ts = np.array([t for t, _ in self.items])
            steps = list(reversed(offsets)) if offsets else list(range(n - 1, -1, -1))
            frames = [self.items[int(np.argmin(np.abs(ts - (now - k * spacing_s))))][1] for k in steps]
            return np.stack(frames), now - ts[-1]


class ObsCapture(WindowCapture):
    """Window capture that turns each frame into the model's observation inside the callback: crop the play box from the
    frame buffer view, gray, downsample, push into the ring. No full-frame copy and no extra Python thread."""

    def __init__(self, hwnd: int, ring: "FrameRing", size: tuple[int, int], min_interval_s: float = 0.008):
        self.ring = ring
        self.size = size
        self.min_interval_s = min_interval_s
        self.last_t = 0.0
        self.fps = 0.0
        self._n = 0
        self._t0 = time.perf_counter()
        super().__init__(hwnd=hwnd)

    def on_frame_arrived(self, frame: Frame, control: InternalCaptureControl) -> None:
        t = time.perf_counter()
        if t - self.last_t < self.min_interval_s:
            return
        self.last_t = t
        try:
            buf = np.asarray(frame.frame_buffer)
            ox, oy = self.client_offset
            x, y, w, h = self.area.capture.rounded()
            box = buf[oy + y:oy + y + h, ox + x:ox + x + w]
            gray = cv2.cvtColor(np.ascontiguousarray(box), cv2.COLOR_BGRA2GRAY)
            self.ring.push(t, cv2.resize(gray, self.size, interpolation=cv2.INTER_AREA))
        except Exception as e:
            print(f"[capture] frame conversion failed: {e}", flush=True)
            return
        self._n += 1
        if self._n % 60 == 0:
            now = time.perf_counter()
            self.fps = 60 / (now - self._t0)
            self._t0 = now
            self.refresh_geometry()


class OneEuro:
    """Adaptive low-pass on the aim output: heavy smoothing while slow, little while fast, so jitter goes without adding lag to jumps."""

    def __init__(self, dt: float, min_cutoff: float, beta: float, d_cutoff: float = 10.0):
        self.dt, self.min_cutoff, self.beta, self.d_cutoff = dt, min_cutoff, beta, d_cutoff
        self.x = None
        self.dx = np.zeros(2)

    def _alpha(self, cutoff: float) -> float:
        r = 2 * math.pi * cutoff * self.dt
        return r / (r + 1)

    def __call__(self, x: tuple[float, float]) -> tuple[float, float]:
        x = np.asarray(x, dtype=np.float64)
        if self.x is None or self.min_cutoff <= 0:
            self.x = x
            return float(x[0]), float(x[1])
        dx = (x - self.x) / self.dt
        self.dx = self.dx + self._alpha(self.d_cutoff) * (dx - self.dx)
        a = self._alpha(self.min_cutoff + self.beta * float(np.hypot(*self.dx)))
        self.x = self.x + a * (x - self.x)
        return float(self.x[0]), float(self.x[1])


class Player:
    def __init__(self, hwnd: int, aim_path: str, click_path: str, log=print, inject: bool = True, buttons: str = "keyboard", keys: str = "zx",
                 smooth_cutoff: float = 2.0, smooth_beta: float = 0.05, latency_extra: float = 8.0, smooth_dcut: float = 10.0, dump_frames: str | None = None, dump_ticks: int = 2000):
        self.smooth_args = (0.01, smooth_cutoff, smooth_beta, smooth_dcut)
        self.smooth = OneEuro(*self.smooth_args)
        self.dump_dir = Path(dump_frames) if dump_frames else None
        self.dump_ticks = dump_ticks
        self.dump: list[tuple] = []
        self.inject_ms = latency_extra
        self.hwnd = hwnd
        self.log = log
        self.inject_enabled = inject
        self.use_mouse_buttons = buttons == "mouse"
        self.scans = [Injector.scan_code(keys[0]), Injector.scan_code(keys[1])]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.aim_path = aim_path
        self.aim, aim_ck = self._load(aim_path)
        self.aim_ck = aim_ck
        self.joint = bool(aim_ck.get("joint"))
        self.click = None if self.joint else self._load(click_path)[0]
        self.h, self.w = aim_ck["frame_hw"]
        self.stack_n = aim_ck["stack"]
        self.ring = FrameRing()
        self.cap = ObsCapture(hwnd, self.ring, (self.w, self.h))
        self.inject = Injector()
        self.keys = KeyExecutor(2, 0.5, 1.0)
        self.active = False
        self.closing = False
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        self.logfile = open(LOG_DIR / f"play-{time.strftime('%Y%m%d-%H%M%S')}.jsonl", "a")
        self.stats = {"capture_fps": 0.0, "loop_ms": 0.0, "frame_age_ms": 0.0, "presses": 0, "ticks": 0, "target": (0, 0), "p_press": 0.0, "p_down": 0.0, "tick_gap_max_ms": 0.0, "frame_age_max_ms": 0.0, "infer_max_ms": 0.0}
        self.zeros = torch.zeros(1, 2, device=self.device)
        self.latency_max = float(aim_ck.get("latency_max_ms", 0.0))
        self.offsets = aim_ck.get("offsets")
        self.last_infer_ms = 1.0
        self.trace: list[tuple] = []
        self.trace_path = LOG_DIR / f"play-{time.strftime('%Y%m%d-%H%M%S')}-trace.npz"
        self.event("init", model=aim_path, click=None if self.joint else click_path, window=win32gui.GetWindowText(hwnd),
                   client=[self.cap.client.x, self.cap.client.y, self.cap.client.w, self.cap.client.h] if hasattr(self.cap, "client") else None,
                   smooth_cutoff=smooth_cutoff, smooth_beta=smooth_beta, smooth_dcut=smooth_dcut, latency_extra=latency_extra, latency_max=self.latency_max, offsets=self.offsets)
        ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x00000080)
        warm = torch.zeros(1, self.stack_n, self.h, self.w, device=self.device)
        with torch.no_grad():
            for _ in range(20):
                self.aim(warm, self.zeros)
                if self.click is not None:
                    self.click(warm, self.zeros)
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        threading.Thread(target=self._run, daemon=True).start()

    def _load(self, path: str):
        ck = torch.load(path, map_location=self.device)
        m = FrameNet(stack=ck["stack"], width=ck["width"], outputs=ck.get("outputs", 2), heads=ck.get("heads"), bounded_aim=ck.get("bounded_aim", False)).to(self.device).eval()
        m.load_state_dict(ck["model"])
        return m, ck

    def _run(self) -> None:
        ctypes.windll.kernel32.SetThreadPriority(ctypes.windll.kernel32.GetCurrentThread(), 2)
        period = STEP_MS / 1000
        next_tick = time.perf_counter()
        loop_ms = deque(maxlen=100)
        last_tick = time.perf_counter()
        gap_max = age_max = infer_max = 0.0
        window_start = last_tick
        while not self.closing:
            next_tick += period
            if not self.active:
                time.sleep(0.02)
                next_tick = time.perf_counter()
                continue
            t_start = time.perf_counter()
            gap_max = max(gap_max, (t_start - last_tick) * 1000)
            last_tick = t_start
            if t_start - window_start >= 1.0:
                self.stats["tick_gap_max_ms"], self.stats["frame_age_max_ms"], self.stats["infer_max_ms"] = gap_max, age_max, infer_max
                self.stats["capture_fps"] = self.cap.fps
                self.event("stats", active=self.active, **{k: (list(v) if isinstance(v, tuple) else v) for k, v in self.stats.items()})
                gap_max = age_max = infer_max = 0.0
                window_start = t_start
            sampled = self.ring.stack(t_start, self.stack_n, period, self.offsets)
            if sampled is not None:
                frames, age = sampled
                x = torch.from_numpy(frames).float().div_(255.0).unsqueeze(0).to(self.device)
                aux = self.zeros
                if self.latency_max:
                    latency_ms = age * 1000 + self.last_infer_ms + self.inject_ms
                    aux = torch.tensor([[min(latency_ms, self.latency_max) / self.latency_max, 0.0]], dtype=torch.float32, device=self.device)
                    self.stats["latency_ms"] = latency_ms
                t_inf = time.perf_counter()
                with torch.no_grad():
                    if self.joint:
                        o = self.aim(x, aux)[0]
                        nx, ny = o[:2].tolist()
                        p_press, p_down = torch.sigmoid(o[2:]).tolist()
                    else:
                        nx, ny = self.aim(x, self.zeros)[0].tolist()
                        p_press, p_down = torch.sigmoid(self.click(x, self.zeros))[0].tolist()
                self.last_infer_ms = (time.perf_counter() - t_inf) * 1000
                infer_max = max(infer_max, self.last_infer_ms)
                px, py = self.smooth(decode_to_playfield(nx, ny, self.aim_ck, self.cap.area))
                wx, wy = self.cap.area.playfield_to_window(px, py)
                sx, sy = self.cap.client.x + wx, self.cap.client.y + wy
                before = (self.keys.down, self.keys.key)
                k1, k2 = self.keys.apply(p_press, p_down > 0.5, p_down)
                if self.inject_enabled:
                    self.inject.move(sx, sy)
                    self._apply_keys(before, (self.keys.down, self.keys.key))
                self.stats["presses"] += int(self.keys.pressed)
                self.stats["frame_age_ms"] = age * 1000
                age_max = max(age_max, age * 1000)
                self.stats["target"] = (int(sx), int(sy))
                self.stats["p_press"], self.stats["p_down"] = p_press, p_down
                if self.active:
                    self.trace.append((t_start, px, py, p_press, p_down, float(k1), float(k2), age * 1000, self.last_infer_ms))
                    if self.dump_dir and len(self.dump) < self.dump_ticks:
                        self.dump.append((frames, float(aux[0, 0]), (nx, ny, p_press, p_down), (px, py, float(k1), float(k2))))
            loop_ms.append((time.perf_counter() - t_start) * 1000)
            self.stats["loop_ms"] = float(np.mean(loop_ms))
            self.stats["ticks"] += 1
            remaining = next_tick - time.perf_counter()
            if remaining > 0.0015:
                time.sleep(remaining - 0.001)
            while time.perf_counter() < next_tick:
                pass

    def _apply_keys(self, before: tuple[bool, int], after: tuple[bool, int]) -> None:
        if before == after:
            return
        was_down, was_key = before
        down, key = after
        if was_down and (not down or key != was_key):
            self._key(was_key, False)
        if down and (not was_down or key != was_key):
            self._key(key, True)

    def _key(self, index: int, down: bool) -> None:
        if self.use_mouse_buttons:
            self.inject.button(index, down)
        else:
            self.inject.key(self.scans[index], down)

    def event(self, kind: str, **fields) -> None:
        """One JSON line per event or per-second stats window, so a session can be read back later."""
        try:
            self.logfile.write(json.dumps({"t": time.time(), "event": kind, **fields}, default=str) + "\n")
            self.logfile.flush()
        except Exception:
            pass

    def start(self) -> None:
        self.smooth = OneEuro(*self.smooth_args)
        self.keys = KeyExecutor(2, 0.5, 1.0)
        self.active = True
        self.log("agent on")
        self.event("start", window=win32gui.GetWindowText(self.hwnd) if hasattr(self, "hwnd") else None)

    def stop(self) -> None:
        self.active = False
        for index in (0, 1):
            self._key(index, False)
        self.log("agent off")
        self.event("stop", presses=self.stats["presses"], ticks=self.stats["ticks"], trace=str(self.trace_path) if self.trace else None)
        if self.dump_dir and self.dump:
            self.dump_dir.mkdir(parents=True, exist_ok=True)
            np.save(self.dump_dir / "frames.npy", np.stack([d[0] for d in self.dump]).astype(np.uint8))
            np.save(self.dump_dir / "aux.npy", np.array([d[1] for d in self.dump], dtype=np.float32))
            np.save(self.dump_dir / "out.npy", np.array([d[2] for d in self.dump], dtype=np.float64))
            np.save(self.dump_dir / "post.npy", np.array([d[3] for d in self.dump], dtype=np.float64))
            (self.dump_dir / "meta.json").write_text(json.dumps({"ticks": len(self.dump), "smooth": list(self.smooth_args), "model": str(self.aim_path)}))
            self.dump = []
        if self.trace:
            np.savez_compressed(self.trace_path, trace=np.array(self.trace, dtype=np.float64), columns=np.array(["t", "x", "y", "p_press", "p_down", "k1", "k2", "frame_age_ms", "infer_ms"]))

    def close(self) -> None:
        self.stop()
        self.closing = True
        self.cap.close()


def list_windows(filter_text: str = "osu!") -> list[tuple[int, str]]:
    """Visible windows whose title starts with the filter (lazer titles are "osu!" or "osu! - artist - title [diff]")."""
    out = []

    def cb(hwnd, _):
        if win32gui.IsWindowVisible(hwnd):
            title = win32gui.GetWindowText(hwnd)
            if title and title.lower().startswith(filter_text.lower()):
                out.append((hwnd, title))

    win32gui.EnumWindows(cb, None)
    return out


class Hotkeys(threading.Thread):
    """Global F8 (toggle) and F9 (stop) through RegisterHotKey on a thread with its own message loop."""

    def __init__(self, on_toggle, on_stop):
        super().__init__(daemon=True)
        self.on_toggle, self.on_stop = on_toggle, on_stop

    def run(self) -> None:
        user32 = ctypes.windll.user32
        user32.RegisterHotKey(None, HOTKEY_TOGGLE, 0, VK_F8)
        user32.RegisterHotKey(None, HOTKEY_STOP, 0, VK_F9)
        msg = wintypes.MSG()
        while user32.GetMessageW(ctypes.byref(msg), None, 0, 0) != 0:
            if msg.message == 0x0312:
                (self.on_toggle if msg.wParam == HOTKEY_TOGGLE else self.on_stop)()


def gui(args) -> None:
    root = tk.Tk()
    root.title("osu-auto play")
    root.geometry("640x300")
    state = {"player": None}
    windows = tk.StringVar()
    status = tk.StringVar(value="pick the osu! window, then Start or F8 (F9 stops)")
    combo = ttk.Combobox(root, textvariable=windows, state="readonly", width=60)
    items: list[tuple[int, str]] = []

    def refresh():
        items[:] = list_windows(args.filter)
        combo["values"] = [f"{t}  [{h}]" for h, t in items]
        if items and not windows.get():
            combo.current(0)

    def log(msg):
        status.set(msg)

    def start():
        if state["player"] is None:
            if not items or combo.current() < 0:
                log("no window selected")
                return
            hwnd = items[combo.current()][0]
            state["player"] = Player(hwnd, args.model, args.click, log, buttons=args.buttons, keys=args.keys, smooth_cutoff=args.smooth_cutoff, smooth_beta=args.smooth_beta, latency_extra=args.latency_extra, smooth_dcut=args.smooth_dcut, dump_frames=args.dump_frames, dump_ticks=args.dump_ticks)
            state["title"] = items[combo.current()][1]
            root.after(500, lambda: state["player"].start())
        else:
            state["player"].start()
        btn.config(text="Stop (F8)")

    def stop():
        if state["player"]:
            state["player"].stop()
        btn.config(text="Start (F8)")

    def toggle():
        if state["player"] and state["player"].active:
            stop()
        else:
            start()

    def tick():
        p = state["player"]
        if p:
            s = p.stats
            lines = [
                f"agent      {'ON' if p.active else 'off'}   model {'unified' if p.joint else 'aim+click'}   inject {'on' if p.inject_enabled else 'off'}",
                f"window     {state.get('title', '')}",
                f"client     {int(p.cap.client.w)}x{int(p.cap.client.h)} at ({int(p.cap.client.x)}, {int(p.cap.client.y)})",
                f"capture    {s['capture_fps']:.0f} fps   frame age {s['frame_age_ms']:.1f} ms",
                f"loop       {s['loop_ms']:.2f} ms   presses {s['presses']}",
                f"stall max  tick {s['tick_gap_max_ms']:.0f} ms   frame {s['frame_age_max_ms']:.0f} ms   infer {s['infer_max_ms']:.0f} ms",
                f"model out  target ({s['target'][0]}, {s['target'][1]})   press {s['p_press']:.2f}   down {s['p_down']:.2f}",
            ]
            stats_var.set("\n".join(lines))
        root.after(200, tick)

    ttk.Label(root, text="Game window").pack(anchor="w", padx=10, pady=(10, 0))
    combo.pack(fill="x", padx=10)
    row = ttk.Frame(root)
    row.pack(fill="x", padx=10, pady=8)
    ttk.Button(row, text="Refresh", command=refresh).pack(side="left")
    btn = ttk.Button(row, text="Start (F8)", command=lambda: toggle())
    btn.pack(side="left", padx=8)
    stats_var = tk.StringVar(value="")
    ttk.Label(root, textvariable=status, wraplength=440).pack(anchor="w", padx=10)
    ttk.Label(root, textvariable=stats_var, font=("Consolas", 10), justify="left").pack(anchor="w", padx=10, pady=(6, 0))
    refresh()
    Hotkeys(lambda: root.after(0, toggle), lambda: root.after(0, stop)).start()
    root.after(200, tick)

    def on_close():
        if state["player"]:
            state["player"].close()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)
    root.mainloop()


def headless(args) -> None:
    """No GUI: attach to the first matching window, run for a fixed time, print loop statistics. For testing."""
    items = list_windows(args.filter)
    if not items:
        raise SystemExit(f"no window matching '{args.filter}'")
    p = Player(items[0][0], args.model, args.click, inject=not args.no_inject, buttons=args.buttons, keys=args.keys, smooth_cutoff=args.smooth_cutoff, smooth_beta=args.smooth_beta, latency_extra=args.latency_extra, smooth_dcut=args.smooth_dcut, dump_frames=args.dump_frames, dump_ticks=args.dump_ticks)
    time.sleep(1.0)
    p.start()
    t_end = time.perf_counter() + args.duration
    while time.perf_counter() < t_end:
        time.sleep(1.0)
        s = p.stats
        print(f"capture {s['capture_fps']:.0f} fps | frame age {s['frame_age_ms']:.1f} ms | loop {s['loop_ms']:.2f} ms | ticks {s['ticks']} | presses {s['presses']} | stall max tick {s['tick_gap_max_ms']:.0f} frame {s['frame_age_max_ms']:.0f} infer {s['infer_max_ms']:.0f} ms", flush=True)
    p.close()


def main() -> None:
    ctypes.windll.shcore.SetProcessDpiAwareness(2)
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", default=str(ROOT / "data/models/joint_v3_lat2/joint_best.pt"), help="unified checkpoint (x, y, press, down), or an aim-only checkpoint used together with --click")
    ap.add_argument("--click", default=str(ROOT / "data/models/click_grid_h1/click_best.pt"))
    ap.add_argument("--filter", default="osu!", help="window title prefix")
    ap.add_argument("--smooth-cutoff", type=float, default=2.0, help="1 euro filter minimum cutoff in Hz for the aim output; 0 disables smoothing")
    ap.add_argument("--latency-extra", type=float, default=8.0, help="ms added to the measured frame age and inference time before it is fed to the model: injection, capture and display delay. Raise it if hits land late")
    ap.add_argument("--smooth-beta", type=float, default=0.05, help="1 euro filter speed coefficient: higher follows fast moves more closely")
    ap.add_argument("--smooth-dcut", type=float, default=10.0, help="1 euro filter derivative cutoff in Hz: how fast the speed estimate reacts (1 lagged big jumps badly)")
    ap.add_argument("--buttons", choices=["mouse", "keyboard"], default="keyboard", help="press with mouse buttons (left/right) or keyboard keys")
    ap.add_argument("--keys", default="zx", help="the two keyboard keys when --buttons keyboard")
    ap.add_argument("--headless", action="store_true", help="no GUI: attach to the first matching window and run for --duration seconds")
    ap.add_argument("--duration", type=float, default=20.0)
    ap.add_argument("--no-inject", action="store_true", help="headless only: run capture and models but send no input")
    ap.add_argument("--dump-frames", default=None, help="folder for the per-tick model inputs, outputs and post-processing results (frames.npy, aux.npy, out.npy, post.npy), used to test the native runner")
    ap.add_argument("--dump-ticks", type=int, default=2000, help="most ticks --dump-frames keeps")
    args = ap.parse_args()
    headless(args) if args.headless else gui(args)


if __name__ == "__main__":
    main()
