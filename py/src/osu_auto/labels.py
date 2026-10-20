import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

KIND_CIRCLE = 0
KIND_SLIDER = 1
KIND_SPINNER = 2


@dataclass
class Labels:
    t_ms: np.ndarray
    x: np.ndarray
    y: np.ndarray
    k1: np.ndarray
    k2: np.ndarray
    press_k1: np.ndarray
    press_k2: np.ndarray

    def __len__(self) -> int:
        return len(self.t_ms)

    @property
    def step_ms(self) -> float:
        return float(self.t_ms[1] - self.t_ms[0]) if len(self.t_ms) > 1 else 0.0

    def hold_mask(self, key: int) -> np.ndarray:
        """Frames where the key is already down coming into this step (hold decision applies)."""
        state = self.k1 if key == 1 else self.k2
        prev = np.concatenate(([0], state[:-1]))
        return prev.astype(bool)

    def press_mask(self, key: int) -> np.ndarray:
        """Frames where the key is up coming into this step (press decision applies)."""
        return ~self.hold_mask(key)

    def index_at(self, t_ms: float) -> int:
        return int(np.clip(np.searchsorted(self.t_ms, t_ms), 0, len(self.t_ms) - 1))


@dataclass
class HitObjects:
    start_ms: np.ndarray
    end_ms: np.ndarray
    x: np.ndarray
    y: np.ndarray
    end_x: np.ndarray
    end_y: np.ndarray
    kind: np.ndarray
    radius: np.ndarray
    combo: np.ndarray

    def __len__(self) -> int:
        return len(self.start_ms)


@dataclass
class Export:
    path: Path
    meta: dict
    labels: Labels
    objects: HitObjects


def load_export(path: str | Path) -> Export:
    path = Path(path)
    meta = json.loads((path / "meta.json").read_text())

    with np.load(path / "labels.npz") as z:
        labels = Labels(**{k: z[k] for k in z.files})

    with np.load(path / "objects.npz") as z:
        objects = HitObjects(**{k: z[k] for k in z.files})

    return Export(path, meta, labels, objects)
