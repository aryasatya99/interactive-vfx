"""Procedural, real-time glowing jellyfish - the centerpiece visual.

Entirely generated each frame from a few sine waves; no static image or
external asset is involved. The bell "breathes" (pulses), and each tentacle
sways independently using its own phase offset so they don't move in
lockstep, which is most of what reads as "organic" here.
"""

from __future__ import annotations

import math
import random

import cv2
import numpy as np

DEFAULT_TENTACLE_COUNT = 8


def _tentacle_points(origin: tuple[float, float], lean: float, length: float,
                      sway_amplitude: float, phase: float, t: float,
                      segments: int = 12) -> list[tuple[int, int]]:
    """A wavy chain of points hanging from `origin`, swaying over time `t`.

    The sway grows toward the tip (scaled by `frac`) so the base stays
    anchored to the bell while the tip whips more freely, like a real
    tentacle.
    """
    points = []
    for i in range(segments + 1):
        frac = i / segments
        y = origin[1] + length * frac
        sway = sway_amplitude * frac * math.sin(t * 2.0 + phase + frac * 4.0)
        x = origin[0] + lean * frac + sway
        points.append((int(x), int(y)))
    return points


class Jellyfish:
    """Owns animation phase state; `update()` advances it, `render()` draws
    the current frame and returns each tentacle's tip position (useful as
    particle-emission points)."""

    def __init__(self, tentacle_count: int = DEFAULT_TENTACLE_COUNT) -> None:
        self.time = 0.0
        self.tentacle_count = tentacle_count
        self._phase_offsets = [random.uniform(0, math.tau) for _ in range(tentacle_count)]

    def update(self, dt: float, animation_intensity: float) -> None:
        # Idle motion never fully stops (0.5 floor) so the jellyfish always
        # feels alive; more animation_intensity (from GESTURE_CONFIG's
        # "animation" channel) speeds up the pulse/tentacle sway.
        self.time += dt * (0.5 + animation_intensity * 1.8)

    def render(
        self,
        frame: np.ndarray,
        glow: np.ndarray,
        position: tuple[float, float],
        scale: float,
        depth: float,
        color: tuple[int, int, int],
        base_radius: float = 70.0,
    ) -> list[tuple[int, int]]:
        pulse = 1.0 + 0.12 * math.sin(self.time * 2.2)
        bell_r = max(12.0, base_radius * scale * pulse)
        bell_center = (int(position[0]), int(position[1] - bell_r * 0.2))
        axes = (int(bell_r), int(bell_r * 0.75))

        # Translucent dome fill: blended at low alpha so the camera feed
        # still shows through, which is what sells the "translucent
        # jellyfish" look rather than a solid cyan blob.
        overlay = frame.copy()
        cv2.ellipse(overlay, bell_center, axes, 0, 180, 360, color, -1, cv2.LINE_AA)
        alpha = 0.15 + 0.15 * depth
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, dst=frame)

        # Glowing outline + a couple of inner rings for texture.
        cv2.ellipse(glow, bell_center, axes, 0, 180, 360, color, 6, cv2.LINE_AA)
        cv2.ellipse(frame, bell_center, axes, 0, 180, 360, color, 2, cv2.LINE_AA)
        for k in (0.55, 0.78):
            rr = (int(bell_r * k), int(bell_r * k * 0.75))
            cv2.ellipse(frame, bell_center, rr, 0, 180, 360, color, 1, cv2.LINE_AA)

        # Tentacles hang from evenly spaced points along the bell's bottom edge.
        base_y = bell_center[1] + axes[1]
        tips: list[tuple[int, int]] = []
        n = max(1, self.tentacle_count)
        for i in range(n):
            frac = (i / (n - 1) - 0.5) if n > 1 else 0.0
            origin = (bell_center[0] + frac * bell_r * 1.5, base_y)
            length = bell_r * 2.0
            sway_amp = bell_r * 0.22 * (0.6 + depth)
            points = _tentacle_points(origin, frac * bell_r * 0.4, length, sway_amp,
                                     self._phase_offsets[i], self.time)
            for a, b in zip(points, points[1:]):
                cv2.line(glow, a, b, color, 3, cv2.LINE_AA)
                cv2.line(frame, a, b, color, 1, cv2.LINE_AA)
            tips.append(points[-1])
        return tips
