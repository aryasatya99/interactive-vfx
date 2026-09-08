"""Small shared helpers: FPS counter, text drawing, smoothing math, and glow
compositing used by the visual modules (jellyfish, cube, particles)."""

from __future__ import annotations

import time
from collections import deque

import cv2
import numpy as np

FONT = cv2.FONT_HERSHEY_SIMPLEX

# --- Palette (BGR) - neon cyan/blue, used sparingly for minimal indicators --
COLOR_TEXT = (235, 235, 235)
COLOR_MUTED = (160, 158, 155)
COLOR_OK = (120, 255, 170)
COLOR_WARN = (60, 190, 255)


class FPSCounter:
    """Rolling-average FPS over the last `window` frames (avoids flicker)."""

    def __init__(self, window: int = 30) -> None:
        self._times: deque[float] = deque(maxlen=window)

    def tick(self) -> float:
        self._times.append(time.perf_counter())
        if len(self._times) < 2:
            return 0.0
        elapsed = self._times[-1] - self._times[0]
        if elapsed <= 0:
            return 0.0
        return (len(self._times) - 1) / elapsed


def put_text(img: np.ndarray, text: str, org: tuple[int, int], scale: float = 0.5,
             color: tuple[int, int, int] = COLOR_TEXT, thickness: int = 1) -> None:
    cv2.putText(img, text, org, FONT, scale, color, thickness, cv2.LINE_AA)


def text_width(text: str, scale: float = 0.5, thickness: int = 1) -> int:
    return cv2.getTextSize(text, FONT, scale, thickness)[0][0]


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def ema(previous: float, target: float, alpha: float) -> float:
    """Exponential moving average: alpha closer to 1 tracks the target
    faster, closer to 0 smooths harder. Used everywhere a tracked position
    or size must not jitter frame-to-frame."""
    return previous + (target - previous) * alpha


def ema_point(previous: tuple[float, float], target: tuple[float, float],
              alpha: float) -> tuple[float, float]:
    return (ema(previous[0], target[0], alpha), ema(previous[1], target[1], alpha))


# --- Cheap glow compositing --------------------------------------------------
# Instead of Gaussian-blurring the whole frame every frame (expensive), glow
# is faked by drawing shapes onto a throwaway black layer with extra
# thickness/radius, then adding that layer onto the real frame with
# `cv2.add`, which saturates at 255 instead of wrapping - overlaps simply
# brighten instead of corrupting colour. One `cv2.add` per frame is far
# cheaper than a real blur pass and reads as "glow" at video framerate.

def new_glow_layer(frame: np.ndarray) -> np.ndarray:
    return np.zeros_like(frame)


def blend_glow(frame: np.ndarray, glow: np.ndarray, intensity: float = 0.9) -> np.ndarray:
    scaled = cv2.convertScaleAbs(glow, alpha=intensity, beta=0)
    return cv2.add(frame, scaled)
