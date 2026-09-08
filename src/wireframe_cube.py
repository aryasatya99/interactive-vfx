"""A slowly rotating 3D wireframe cube, projected onto the 2D frame by hand.

OpenCV has no 3D renderer, so this does the whole pipeline itself: rotate 8
unit-cube vertices with standard rotation matrices, apply a simple
perspective projection, then draw the 12 edges as glowing lines. It's a
handful of NumPy matrix multiplies per frame - cheap enough for 30-60 FPS
alongside MediaPipe.
"""

from __future__ import annotations

import cv2
import numpy as np

# Unit cube vertices, centered at the origin.
_VERTICES = np.array([
    [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
    [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1],
], dtype=np.float64)

_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 0),  # back face
    (4, 5), (5, 6), (6, 7), (7, 4),  # front face
    (0, 4), (1, 5), (2, 6), (3, 7),  # connectors
]

_CAMERA_DISTANCE = 4.0  # bigger = flatter/less perspective distortion


def _rotation_matrix(angle_x: float, angle_y: float) -> np.ndarray:
    cx, sx = np.cos(angle_x), np.sin(angle_x)
    cy, sy = np.cos(angle_y), np.sin(angle_y)
    rot_x = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    rot_y = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    return rot_y @ rot_x


class WireframeCube:
    """Owns rotation state; `update()` advances it, `render()` draws it."""

    def __init__(self) -> None:
        self.angle_x = 0.3
        self.angle_y = 0.0
        self.base_spin = 0.4  # radians/sec, always-on idle rotation

    def update(self, dt: float, animation_intensity: float = 0.3) -> None:
        speed = self.base_spin + animation_intensity * 1.2
        self.angle_x += dt * speed * 0.4
        self.angle_y += dt * speed

    def render(self, frame: np.ndarray, glow: np.ndarray, center: tuple[float, float],
               radius_px: float, color: tuple[int, int, int],
               glow_intensity: float = 1.0) -> None:
        rot = _rotation_matrix(self.angle_x, self.angle_y)
        rotated = _VERTICES @ rot.T

        cx, cy = center
        points: list[tuple[int, int]] = []
        for x, y, z in rotated:
            perspective = _CAMERA_DISTANCE / (_CAMERA_DISTANCE + z)
            sx = cx + x * radius_px * perspective
            sy = cy + y * radius_px * perspective
            points.append((int(sx), int(sy)))

        glow_thickness = max(2, int(2 + glow_intensity * 3))
        for a, b in _EDGES:
            cv2.line(glow, points[a], points[b], color, glow_thickness, cv2.LINE_AA)
            cv2.line(frame, points[a], points[b], color, 1, cv2.LINE_AA)
        for p in points:
            cv2.circle(frame, p, 2, color, -1, cv2.LINE_AA)
