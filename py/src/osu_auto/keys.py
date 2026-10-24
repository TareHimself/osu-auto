
import numpy as np

from osu_auto.labels import load_export


class KeyExecutor:
    """Turns per-step press/hold decisions into k1/k2 states, alternating keys on every new press."""

    def __init__(self, min_press_gap: int = 2, press_threshold: float = 0.5, rearm: float = 0.3, release: float = 0.25):
        self.down = False
        self.key = 1
        self.pressed = False
        self.min_press_gap = min_press_gap
        self.since_press = 10 ** 6
        self.press_threshold = press_threshold
        self.rearm = rearm
        self.armed = True
        self.prev_down = 0.0
        self.release = release

    def apply(self, p_press: float, hold: bool, p_down: float | None = None) -> tuple[bool, bool]:
        """One press per excursion of the press probability: it must fall below `rearm` before it can fire again.
        A rising edge of the hold probability is also a press (that is how the labels define one), caught when no key is held."""
        if p_press < self.rearm:
            self.armed = True
        if p_down is not None and self.down:
            hold = p_down > self.release
        edge = p_down is not None and self.prev_down < 0.5 <= p_down and not self.down
        self.prev_down = p_down if p_down is not None else 0.0
        press = (self.armed and p_press > self.press_threshold or edge) and self.since_press >= self.min_press_gap
        self.since_press += 1
        if press:
            self.key = 1 - self.key
            self.down = True
            self.since_press = 0
            self.armed = False
        elif not (self.down and hold):
            self.down = False
        self.pressed = press
        return self.down and self.key == 0, self.down and self.key == 1

    @property
    def state(self) -> list[float]:
        return [float(self.down), float(self.pressed)]


def timing_report(press_times: list[float], export_dir: str, step_ms: float) -> dict:
    """Agent presses against the expert's: offset of each agent press to the nearest expert press, and expert presses with no agent press nearby."""
    L = load_export(export_dir).labels
    expert = L.t_ms[(L.press_k1 | L.press_k2).astype(bool)]
    agent = np.array(press_times)
    if len(agent) == 0 or len(expert) == 0:
        return {}
    offsets = np.array([agent[np.argmin(np.abs(agent - e))] - e for e in expert])
    matched = np.abs(offsets) <= 2.5 * step_ms
    stray = np.array([np.min(np.abs(expert - a)) > 2.5 * step_ms for a in agent])
    return {"expert_presses": int(len(expert)), "agent_presses": int(len(agent)), "expert_matched": float(matched.mean()),
            "agent_stray": float(stray.mean()), "offset_ms_median": float(np.median(offsets[matched])) if matched.any() else None,
            "offset_ms_p90_abs": float(np.percentile(np.abs(offsets[matched]), 90)) if matched.any() else None}

