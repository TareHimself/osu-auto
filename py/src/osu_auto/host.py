from __future__ import annotations

from types import TracebackType

import msgpack
import zmq


class HostError(RuntimeError):
    pass


def _box(pixels: bytes, width: int, height: int) -> "np.ndarray":
    import numpy as np

    return np.frombuffer(pixels, dtype=np.uint8).reshape(height, width, 4)


class HostClient:
    """Client for the OsuAuto.Host `serve` control channel (ZeroMQ REQ/REP, msgpack)."""

    def __init__(self, endpoint: str = "tcp://127.0.0.1:5555", timeout_ms: int = 60_000):
        self._ctx = zmq.Context.instance()
        self._sock = self._ctx.socket(zmq.REQ)
        self._sock.setsockopt(zmq.RCVTIMEO, timeout_ms)
        self._sock.setsockopt(zmq.LINGER, 0)
        self._sock.connect(endpoint)

    def __enter__(self) -> HostClient:
        return self

    def __exit__(self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: TracebackType | None) -> None:
        self.close()

    def close(self) -> None:
        self._sock.close()

    def call(self, cmd: str, **kwargs) -> dict:
        self._sock.send(msgpack.packb({"cmd": cmd, **kwargs}))
        reply = msgpack.unpackb(self._sock.recv(), raw=False)
        if not reply.get("ok", False):
            raise HostError(reply.get("error", "unknown host error"))
        return reply

    def load(self, map_path: str, diff: str | None = None, osr: str | None = None, mods: list[str] | None = None) -> dict:
        kwargs = {"map": map_path, "mods": list(mods or [])}
        if diff:
            kwargs["diff"] = diff
        if osr:
            kwargs["osr"] = osr
        return self.call("load", **kwargs)

    def backdrop(self, visible: bool) -> None:
        self.call("backdrop", visible=bool(visible))

    def export_frames(self, times, width: int, height: int) -> tuple["np.ndarray", "np.ndarray"]:
        """Seek to each time with the host rendering and downsampling in-process: returns (gray frames n x h x w, clock_ms per frame)."""
        import numpy as np

        r = self.call("export_frames", t=[float(t) for t in times], width=int(width), height=int(height))
        return np.frombuffer(r["frames"], dtype=np.uint8).reshape(len(times), height, width), np.asarray(r["clocks"], dtype=np.float64)

    def seek(self, t_ms: float) -> float:
        return float(self.call("seek", t=float(t_ms))["clock_ms"])

    def seek_grab(self, t_ms: float, pair: bool = False) -> tuple[float, "np.ndarray", "np.ndarray | None"]:
        """Seek, then read the host's own backbuffer: returns (clock_ms, play box BGRA, no-cursor play box BGRA or None)."""
        r = self.call("seek", t=float(t_ms), grab=True, pair=bool(pair))
        return float(r["clock_ms"]), _box(r["pixels"], r["width"], r["height"]), (_box(r["pixels_nocursor"], r["width"], r["height"]) if pair else None)

    def grab(self, pair: bool = False) -> tuple["np.ndarray", "np.ndarray | None"]:
        r = self.call("grab", pair=bool(pair))
        return _box(r["pixels"], r["width"], r["height"]), (_box(r["pixels_nocursor"], r["width"], r["height"]) if pair else None)

    def status(self) -> dict:
        return self.call("status")

    def cursor(self, visible: bool, wait: bool = True) -> None:
        self.call("cursor", visible=bool(visible), wait=bool(wait))

    def quit(self) -> None:
        self.call("quit")
