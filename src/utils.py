"""Small shared helpers: FPS counter, drawing, and non-overwriting capture save."""

from __future__ import annotations

import time
from collections import deque
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

# --- Palette (BGR) ----------------------------------------------------------
COLOR_BG = (24, 22, 20)
COLOR_PANEL = (38, 35, 32)
COLOR_TEXT = (235, 235, 235)
COLOR_MUTED = (150, 148, 145)
COLOR_OK = (120, 220, 120)
COLOR_WARN = (60, 190, 255)
COLOR_BAD = (90, 90, 240)
COLOR_ACCENT = (255, 190, 90)

FONT = cv2.FONT_HERSHEY_SIMPLEX


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


def fit_into(image: np.ndarray | None, width: int, height: int) -> np.ndarray:
    """Resize keeping aspect ratio and letterbox onto a panel-coloured canvas."""
    canvas = np.full((height, width, 3), COLOR_PANEL, dtype=np.uint8)
    if image is None or image.size == 0:
        return canvas
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    h, w = image.shape[:2]
    scale = min(width / w, height / h)
    new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    resized = cv2.resize(image, (new_w, new_h), interpolation=interp)
    x0, y0 = (width - new_w) // 2, (height - new_h) // 2
    canvas[y0:y0 + new_h, x0:x0 + new_w] = resized
    return canvas


def save_capture(frame: np.ndarray, out_dir: Path) -> Path:
    """Save `frame` as captures/capture_YYYYMMDD_HHMMSS.jpg.

    Never overwrites: if a file for this exact second already exists (e.g.
    SPACE pressed twice within the same second), a numeric suffix is added.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = out_dir / f"capture_{stamp}.jpg"
    n = 2
    while path.exists():
        path = out_dir / f"capture_{stamp}_{n}.jpg"
        n += 1
    cv2.imwrite(str(path), frame)
    return path
