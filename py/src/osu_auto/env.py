from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from osu_auto.capture import WindowCapture
from osu_auto.host import HostClient


@dataclass
class StepResult:
    obs: np.ndarray
    t_ms: float
    clock_ms: float
    hits: dict[str, int]
    combo: int
    accuracy: float
    score: int
    completed: bool


class OsuEnv:
    """Lockstep environment over the host `serve` channel with the agent driving the cursor and keys.

    Each step moves the cursor (playfield units), sets the key state, advances the gameplay clock by one step,
    waits for the render, and returns the grayscale play box. With `capture=None` the host reads its own backbuffer
    and returns the frame in the step reply (identical to the export path); with a WindowCapture the frame is taken
    from the screen like a deployed agent would.
    """

    def __init__(self, host: HostClient, capture: WindowCapture | None = None, obs_size: tuple[int, int] = (160, 120), step_ms: float = 10.0,
                 frames_after_reply: int = 2):
        self.host = host
        self.cap = capture
        self.obs_size = obs_size
        self.step_ms = step_ms
        self.frames_after_reply = frames_after_reply
        self.t = 0.0
        self.t0 = 0.0
        self.t1 = 0.0

    def reset(self, map_path: str, diff: str | None = None, clicks: str = "agent", mods: list[str] | None = None, input_mode: str = "user",
              start_ms: float | None = None) -> StepResult:
        """mods: lazer mod specs, e.g. ["RX"] for aim-only (game clicks), ["AP"] for click-only (game aims).
        input_mode: "user" drives a regular player through synthetic mouse input (mods behave as in real play);
                    "replay" appends frames to a streaming replay (Relax and Autopilot do not act on replays)."""
        kwargs = {"map": map_path, "agent": True, "clicks": clicks, "mods": list(mods or []), "input": input_mode}
        if diff:
            kwargs["diff"] = diff
        info = self.host.call("load", **kwargs)
        self.t0 = float(info["t0"])
        self.t1 = float(info["t1"])
        self.t = self.t0 if start_ms is None else max(self.t0, float(start_ms))
        if self.cap is not None:
            self.cap.refresh_geometry()
        return self.step(256.0, 192.0, False, False, first=True)

    def _observe_capture(self, before: int) -> np.ndarray:
        assert self.cap is not None
        frame = self.cap.wait_frame(before + self.frames_after_reply - 1)
        if self.frames_after_reply > 1:
            frame = self.cap.wait_frame(frame.index)
        gray = cv2.cvtColor(self.cap.play_box(frame), cv2.COLOR_BGRA2GRAY)
        return cv2.resize(gray, self.obs_size, interpolation=cv2.INTER_AREA)

    def step(self, x: float, y: float, k1: bool, k2: bool, first: bool = False) -> StepResult:
        if not first:
            self.t += self.step_ms
        w, h = self.obs_size
        if self.cap is None:
            r = self.host.call("step", t=float(self.t), x=float(x), y=float(y), k1=bool(k1), k2=bool(k2), obs=[w, h])
            obs = np.frombuffer(r["obs"], dtype=np.uint8).reshape(h, w)
        else:
            before = self.cap.count
            r = self.host.call("step", t=float(self.t), x=float(x), y=float(y), k1=bool(k1), k2=bool(k2))
            obs = self._observe_capture(before)
        return StepResult(obs, self.t, float(r["clock_ms"]), dict(r.get("hits", {})), int(r["combo"]), float(r["accuracy"]), int(r["score"]), bool(r["completed"]))

    @property
    def done(self) -> bool:
        return self.t >= self.t1
