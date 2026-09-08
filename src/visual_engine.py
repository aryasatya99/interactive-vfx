"""Owns and composites the visual layers - background treatment, wireframe
cube, jellyfish, and particles - into the final frame, following the
render order: camera -> background treatment -> cube -> jellyfish ->
particles. main.py draws hand indicators and status text on top of what
this returns; everything below is the actual "visual art" layer.
"""

from __future__ import annotations

import random

import cv2
import numpy as np

from .interaction import InteractionState
from .jellyfish import Jellyfish
from .particles import ParticleSystem
from .utils import blend_glow, clamp, new_glow_layer
from .visual_mode import VISUAL_MODES, apply_background_treatment
from .wireframe_cube import WireframeCube

CYAN = (255, 210, 40)     # BGR - jellyfish/cube neon colour
CYAN_DIM = (140, 110, 20)  # standby colour: same hue, much dimmer

CUBE_BASE_RADIUS = 150.0
JELLYFISH_BASE_RADIUS = 70.0

_DEFAULT_MODE_PARAMS = VISUAL_MODES["normal"]


class VisualEngine:
    def __init__(self) -> None:
        self.jellyfish = Jellyfish()
        self.cube = WireframeCube()
        self.particles = ParticleSystem()

    def update(self, state: InteractionState, dt: float, active: bool,
               mode_params: dict[str, float] | None = None, presence: float = 1.0) -> None:
        mode_params = mode_params or _DEFAULT_MODE_PARAMS
        presence = clamp(presence, 0.0, 1.0) if active else 1.0

        anim = state.animation_intensity if active else state.animation_intensity * 0.25
        self.jellyfish.update(dt, anim)
        self.cube.update(dt, anim)
        self.particles.update(dt)

        if not active or presence <= 0.05:
            return  # STANDBY, or no hand recently seen: tracking still ticks, no new particles

        # Emission rate scales with the "particles" gesture channel, the
        # active mode's particle_intensity multiplier, relative depth
        # (closer hand = more particles), and hand-presence fade - all
        # capped by ParticleSystem's own MAX_PARTICLES regardless of rate.
        rate = (0.4 + state.particle_intensity * 1.6) * mode_params["particle_intensity"] * presence
        cx, cy = state.position
        if random.random() < rate:
            self.particles.emit(cx, cy, CYAN, count=2)
        if state.two_hands_active and state.two_hand_points and random.random() < rate:
            (lx, ly), (rx, ry) = state.two_hand_points
            t = random.random()
            self.particles.emit(lx + (rx - lx) * t, ly + (ry - ly) * t, CYAN, count=1)

    def render(self, frame: np.ndarray, state: InteractionState, active: bool,
               mode_params: dict[str, float] | None = None, presence: float = 1.0) -> np.ndarray:
        mode_params = mode_params or _DEFAULT_MODE_PARAMS
        presence = clamp(presence, 0.0, 1.0) if active else 1.0

        # 1-3: camera -> blur -> background treatment (brightness/contrast/
        # translucent blend). A no-op in NORMAL mode.
        frame = apply_background_treatment(frame, mode_params)

        glow = new_glow_layer(frame)
        color = CYAN if active else CYAN_DIM
        # presence_scale never quite hits 0 so the VFX eases to a faint point
        # rather than popping away the instant the last hand leaves frame.
        presence_scale = max(0.03, presence)
        depth_boost = (state.depth if active else state.depth * 0.3) * presence_scale

        vfx_scale = state.scale * mode_params["vfx_scale"] * presence_scale
        glow_mult = mode_params["glow_intensity"]

        # 4: wireframe cube
        cube_radius = CUBE_BASE_RADIUS * vfx_scale
        self.cube.render(frame, glow, state.position, cube_radius, color,
                         glow_intensity=(0.4 + depth_boost) * glow_mult)

        # 5: jellyfish
        jelly_scale = vfx_scale if active else vfx_scale * 0.7
        tips = self.jellyfish.render(frame, glow, state.position, jelly_scale,
                                     depth_boost, color, base_radius=JELLYFISH_BASE_RADIUS)

        if active and presence > 0.05 and state.particle_intensity > 0.05 and tips \
                and random.random() < 0.5:
            tip = random.choice(tips)
            self.particles.emit(tip[0], tip[1], color, count=1)

        if state.two_hands_active and state.two_hand_points and presence_scale > 0.1:
            p1, p2 = state.two_hand_points
            cv2.line(glow, p1, p2, color, 4, cv2.LINE_AA)
            cv2.line(frame, p1, p2, color, 1, cv2.LINE_AA)

        blend_intensity = (0.9 if active else 0.35) * glow_mult * max(0.15, presence_scale)
        blended = blend_glow(frame, glow, intensity=blend_intensity)

        # 6: particles
        self.particles.render(blended)
        return blended
