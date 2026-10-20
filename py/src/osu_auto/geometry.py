from dataclasses import dataclass

PLAYFIELD_W = 512.0
PLAYFIELD_H = 384.0
PLAYFIELD_SIZE_ADJUST = 0.8
STORYBOARD_SHIFT_UNITS = 8.0


@dataclass(frozen=True)
class Rect:
    x: float
    y: float
    w: float
    h: float

    @property
    def right(self) -> float:
        return self.x + self.w

    @property
    def bottom(self) -> float:
        return self.y + self.h

    def rounded(self) -> tuple[int, int, int, int]:
        return round(self.x), round(self.y), round(self.w), round(self.h)


@dataclass(frozen=True)
class PlayArea:
    """Playfield geometry derived purely from the lazer window client size.

    Mirrors OsuPlayfieldAdjustmentContainer: 80% of the gameplay area, 4:3 fit,
    scaled from 512x384, shifted down 8 playfield units for storyboard alignment.
    """

    window_w: float
    window_h: float

    @property
    def capture(self) -> Rect:
        box_w = min(self.window_w, self.window_h * 4 / 3)
        box_h = box_w * 3 / 4
        return Rect((self.window_w - box_w) / 2, (self.window_h - box_h) / 2, box_w, box_h)

    @property
    def scale(self) -> float:
        return PLAYFIELD_SIZE_ADJUST * self.capture.w / PLAYFIELD_W

    @property
    def playfield(self) -> Rect:
        pf_w = PLAYFIELD_W * self.scale
        pf_h = PLAYFIELD_H * self.scale
        cx = self.window_w / 2
        cy = self.window_h / 2 + STORYBOARD_SHIFT_UNITS * self.scale
        return Rect(cx - pf_w / 2, cy - pf_h / 2, pf_w, pf_h)

    def playfield_to_window(self, x: float, y: float) -> tuple[float, float]:
        pf = self.playfield
        return pf.x + x * self.scale, pf.y + y * self.scale

    def window_to_playfield(self, px: float, py: float) -> tuple[float, float]:
        pf = self.playfield
        return (px - pf.x) / self.scale, (py - pf.y) / self.scale

    def playfield_to_capture(self, x: float, y: float) -> tuple[float, float]:
        wx, wy = self.playfield_to_window(x, y)
        cap = self.capture
        return wx - cap.x, wy - cap.y

    def playfield_to_capture_normalised(self, x: float, y: float) -> tuple[float, float]:
        cx, cy = self.playfield_to_capture(x, y)
        cap = self.capture
        return cx / cap.w, cy / cap.h
