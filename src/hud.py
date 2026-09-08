"""Futuristic HUD drawing primitives: brackets, rings, crosshair, panels, glow.

Performance note: instead of Gaussian-blurring the whole frame every frame
(expensive), "glow" is faked cheaply by drawing HUD shapes twice onto a
throwaway black layer — once thick/soft, once crisp — then adding that layer
onto the camera frame with `cv2.add` (which saturates at 255 instead of
wrapping, so overlaps just brighten instead of corrupting colour). One
`cv2.add` per frame is a lot cheaper than a real blur pass.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from .utils import put_text, text_width

# --- Palette (BGR) — a cyan/amber sci-fi HUD scheme -------------------------
CYAN = (255, 220, 40)
CYAN_DIM = (120, 90, 15)
GREEN = (120, 255, 120)
AMBER = (30, 170, 255)
RED = (60, 60, 255)
WHITE = (235, 235, 235)
PANEL_BG = (18, 16, 15)


def new_glow_layer(frame: np.ndarray) -> np.ndarray:
    return np.zeros_like(frame)


def blend_glow(frame: np.ndarray, glow: np.ndarray, intensity: float = 0.9) -> np.ndarray:
    """Additively blend the glow layer onto frame; cv2.add saturates cleanly."""
    scaled = cv2.convertScaleAbs(glow, alpha=intensity, beta=0)
    return cv2.add(frame, scaled)


def draw_panel(frame: np.ndarray, x: int, y: int, w: int, h: int,
               alpha: float = 0.55, border_color: tuple[int, int, int] = CYAN) -> None:
    """Semi-transparent dark panel with a thin glowing border, drawn in place."""
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + w, y + h), PANEL_BG, -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, dst=frame)
    cv2.rectangle(frame, (x, y), (x + w, y + h), border_color, 1, cv2.LINE_AA)


def draw_corner_brackets(frame: np.ndarray, x: int, y: int, w: int, h: int,
                          size: int = 26, color: tuple[int, int, int] = CYAN,
                          thickness: int = 2) -> None:
    """Four L-shaped brackets at the corners of a rect — the classic HUD frame."""
    corners = ((x, y, 1, 1), (x + w, y, -1, 1), (x, y + h, 1, -1), (x + w, y + h, -1, -1))
    for cx, cy, sx, sy in corners:
        cv2.line(frame, (cx, cy), (cx + sx * size, cy), color, thickness, cv2.LINE_AA)
        cv2.line(frame, (cx, cy), (cx, cy + sy * size), color, thickness, cv2.LINE_AA)


def draw_crosshair(frame: np.ndarray, center: tuple[int, int], size: int = 14,
                    color: tuple[int, int, int] = CYAN, thickness: int = 1) -> None:
    x, y = center
    gap = max(2, size // 3)
    cv2.line(frame, (x - size, y), (x - gap, y), color, thickness, cv2.LINE_AA)
    cv2.line(frame, (x + gap, y), (x + size, y), color, thickness, cv2.LINE_AA)
    cv2.line(frame, (x, y - size), (x, y - gap), color, thickness, cv2.LINE_AA)
    cv2.line(frame, (x, y + gap), (x, y + size), color, thickness, cv2.LINE_AA)
    cv2.circle(frame, (x, y), 2, color, -1, cv2.LINE_AA)


def draw_scanning_ring(frame: np.ndarray, center: tuple[int, int], radius: int,
                        sweep_angle: float, color: tuple[int, int, int],
                        active: bool = False, thickness: int = 2) -> None:
    """A faint full ring with tick marks, plus a brighter rotating arc.

    `sweep_angle` should advance every frame (e.g. by elapsed_ms * 0.2) to
    animate the "scanning" motion.
    """
    cv2.circle(frame, center, radius, color, 1, cv2.LINE_AA)
    for deg in range(0, 360, 30):
        rad = math.radians(deg)
        inner = radius - 6
        x1 = int(center[0] + inner * math.cos(rad))
        y1 = int(center[1] + inner * math.sin(rad))
        x2 = int(center[0] + radius * math.cos(rad))
        y2 = int(center[1] + radius * math.sin(rad))
        cv2.line(frame, (x1, y1), (x2, y2), color, 1, cv2.LINE_AA)
    sweep = 80 if active else 40
    thick = thickness + 1 if active else thickness
    cv2.ellipse(frame, center, (radius, radius), 0, sweep_angle, sweep_angle + sweep,
                color, thick, cv2.LINE_AA)


HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),          # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),          # index
    (5, 9), (9, 10), (10, 11), (11, 12),     # middle
    (9, 13), (13, 14), (14, 15), (15, 16),   # ring
    (13, 17), (17, 18), (18, 19), (19, 20),  # pinky
    (0, 17),                                 # palm base
]
TIP_INDICES = (4, 8)  # thumb tip, index tip — the two fingers this app cares about


def draw_hand_skeleton(frame: np.ndarray, glow: np.ndarray, landmarks_px: list[tuple[int, int]],
                        label: str, base_color: tuple[int, int, int]) -> None:
    """Draw one hand's skeleton with glowing thumb/index tips and a label."""
    for a, b in HAND_CONNECTIONS:
        cv2.line(glow, landmarks_px[a], landmarks_px[b], base_color, 5, cv2.LINE_AA)
        cv2.line(frame, landmarks_px[a], landmarks_px[b], base_color, 1, cv2.LINE_AA)

    for i, (x, y) in enumerate(landmarks_px):
        if i in TIP_INDICES:
            cv2.circle(glow, (x, y), 12, AMBER, -1, cv2.LINE_AA)
            cv2.circle(frame, (x, y), 5, AMBER, -1, cv2.LINE_AA)
            cv2.circle(frame, (x, y), 8, WHITE, 1, cv2.LINE_AA)
        else:
            cv2.circle(frame, (x, y), 3, base_color, -1, cv2.LINE_AA)

    wx, wy = landmarks_px[0]
    draw_crosshair(frame, (wx, wy), size=18, color=base_color)
    label_text = label.upper()
    put_text(frame, label_text, (wx - text_width(label_text, 0.5, 2) // 2, wy + 34),
             0.5, base_color, 2)


def status_row(frame: np.ndarray, x: int, y: int, name: str, ok: bool,
               true_text: str = "OK", false_text: str = "--") -> int:
    """Draw one `NAME  MARK` row; returns the x position right after it."""
    put_text(frame, name, (x, y), 0.5, WHITE, 1)
    x += text_width(name, 0.5, 1) + 10
    color = GREEN if ok else RED
    text = true_text if ok else false_text
    put_text(frame, text, (x, y), 0.5, color, 2)
    return x + text_width(text, 0.5, 2) + 24
