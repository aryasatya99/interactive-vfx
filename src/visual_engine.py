"""Owns and composites the three visual layers - wireframe cube, jellyfish,
and particles - into the final frame. This is the only module that knows
how those three pieces fit together; main.py just calls update()/render().
"""

from __future__ import annotations

import random

import cv2
import numpy as np

from .interaction import InteractionState
from .jellyfish import Jellyfish
from .particles import ParticleSystem
from .utils import blend_glow, new_glow_layer
from .wireframe_cube import WireframeCube

CYAN = (255, 210, 40)     # BGR - jellyfish/cube neon colour
CYAN_DIM = (140, 110, 20)  # standby colour: same hue, much dimmer

CUBE_BASE_RADIUS = 150.0
JELLYFISH_BASE_RADIUS = 70.0


class VisualEngine:
    def __init__(self) -> None:
        self.jellyfish = Jellyfish()
        self.cube = WireframeCube()
        self.particles = ParticleSystem()

    def update(self, state: InteractionState, dt: float, active: bool) -> None:
        anim = state.animation_intensity if active else state.animation_intensity * 0.25
        self.jellyfish.update(dt, anim)
        self.cube.update(dt, anim)
        self.particles.update(dt)

        if not active:
            return  # STANDBY: tracking/animation still tick, but no new particles

        # Emission rate scales with the "particles" gesture channel and with
        # relative depth (closer hand = more particles), capped by
        # ParticleSystem's own MAX_PARTICLES regardless of these rates.
        rate = 0.4 + state.particle_intensity * 1.6
        cx, cy = state.position
        if random.random() < rate:
            self.particles.emit(cx, cy, CYAN, count=2)
        if state.two_hands_active and state.two_hand_points and random.random() < rate:
            (lx, ly), (rx, ry) = state.two_hand_points
            t = random.random()
            self.particles.emit(lx + (rx - lx) * t, ly + (ry - ly) * t, CYAN, count=1)

    def render(self, frame: np.ndarray, state: InteractionState, active: bool) -> np.ndarray:
        glow = new_glow_layer(frame)
        color = CYAN if active else CYAN_DIM
        depth_boost = state.depth if active else state.depth * 0.3

        cube_radius = CUBE_BASE_RADIUS * state.scale
        self.cube.render(frame, glow, state.position, cube_radius, color,
                         glow_intensity=0.4 + depth_boost)

        jelly_scale = state.scale if active else state.scale * 0.7
        tips = self.jellyfish.render(frame, glow, state.position, jelly_scale,
                                     depth_boost, color, base_radius=JELLYFISH_BASE_RADIUS)

        if active and state.particle_intensity > 0.05 and tips and random.random() < 0.5:
            tip = random.choice(tips)
            self.particles.emit(tip[0], tip[1], color, count=1)

        if state.two_hands_active and state.two_hand_points:
            p1, p2 = state.two_hand_points
            cv2.line(glow, p1, p2, color, 4, cv2.LINE_AA)
            cv2.line(frame, p1, p2, color, 1, cv2.LINE_AA)

        blended = blend_glow(frame, glow, intensity=0.9 if active else 0.35)
        self.particles.render(blended)
        return blended
