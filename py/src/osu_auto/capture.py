from __future__ import annotations

import ctypes
import threading
import time
from ctypes import wintypes
from dataclasses import dataclass

import numpy as np
import win32gui
from windows_capture import Frame, InternalCaptureControl, WindowsCapture

from osu_auto.geometry import PlayArea, Rect

DWMWA_EXTENDED_FRAME_BOUNDS = 9


def find_window(title: str) -> int:
    hwnd = win32gui.FindWindow(None, title)
    if not hwnd:
        raise RuntimeError(f"window '{title}' not found")
    return hwnd


def client_rect_screen(hwnd: int) -> Rect:
    left, top, right, bottom = win32gui.GetClientRect(hwnd)
    sx, sy = win32gui.ClientToScreen(hwnd, (left, top))
    return Rect(sx, sy, right - left, bottom - top)


def frame_bounds_screen(hwnd: int) -> Rect:
    rect = wintypes.RECT()
    ctypes.windll.dwmapi.DwmGetWindowAttribute(wintypes.HWND(hwnd), ctypes.c_uint(DWMWA_EXTENDED_FRAME_BOUNDS), ctypes.byref(rect), ctypes.sizeof(rect))
    return Rect(rect.left, rect.top, rect.right - rect.left, rect.bottom - rect.top)


@dataclass
class Captured:
    index: int
    bgra: np.ndarray
    time: float


class WindowCapture:
    """Windows.Graphics.Capture of one window, cropped to its client area.

    Runs the capture on a background thread and keeps the latest frame plus a counter,
    so callers can wait for frames that arrived after a known point in time.
    """

    def __init__(self, title: str = "osu-auto", hwnd: int | None = None):
        self.hwnd = hwnd or find_window(title)
        self._lock = threading.Condition()
        self._latest: Captured | None = None
        self._count = 0
        self._closed = False
        self._capture = WindowsCapture(cursor_capture=False, draw_border=False, window_hwnd=self.hwnd, minimum_update_interval=0)
        self._capture.event(self.on_frame_arrived)
        self._capture.event(self.on_closed)
        self._control = self._capture.start_free_threaded()
        self.refresh_geometry()

    def refresh_geometry(self) -> None:
        self.client = client_rect_screen(self.hwnd)
        self.bounds = frame_bounds_screen(self.hwnd)
        self.client_offset = (int(self.client.x - self.bounds.x), int(self.client.y - self.bounds.y))
        self.area = PlayArea(self.client.w, self.client.h)

    def on_frame_arrived(self, frame: Frame, control: InternalCaptureControl) -> None:
        buf = np.array(frame.frame_buffer, copy=True)
        with self._lock:
            self._count += 1
            self._latest = Captured(self._count, buf, time.perf_counter())
            self._lock.notify_all()

    def on_closed(self) -> None:
        print(f"[capture] session closed (hwnd={self.hwnd}, frames={self._count}, window alive={win32gui.IsWindow(self.hwnd)})", flush=True)
        with self._lock:
            self._closed = True
            self._lock.notify_all()

    @property
    def count(self) -> int:
        with self._lock:
            return self._count

    def wait_frame(self, after_index: int, timeout: float = 2.0) -> Captured:
        deadline = time.perf_counter() + timeout
        with self._lock:
            while self._latest is None or self._latest.index <= after_index:
                if self._closed:
                    raise RuntimeError("capture closed")
                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    raise TimeoutError(f"no frame after index {after_index} within {timeout}s (count={self._count})")
                self._lock.wait(remaining)
            return self._latest

    def client_view(self, captured: Captured) -> np.ndarray:
        ox, oy = self.client_offset
        w, h = int(self.client.w), int(self.client.h)
        return captured.bgra[oy:oy + h, ox:ox + w]

    def play_box(self, captured: Captured) -> np.ndarray:
        x, y, w, h = self.area.capture.rounded()
        return self.client_view(captured)[y:y + h, x:x + w]

    def close(self) -> None:
        try:
            self._control.stop()
        except Exception:
            pass
