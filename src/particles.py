"""Lightweight particle effect that follows tracked hand positions.

Kept deliberately simple: a capped-size list of tiny circles with a
position, velocity, and lifetime. No physics engine, no external assets —
just enough motion to make the HUD feel alive without costing FPS.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

import cv2
import numpy as np

MAX_PARTICLES = 120  # hard cap so cost per frame stays bounded regardless of input


@dataclass
class Particle:
    x: float
    y: float
    vx: float
    vy: float
    life: float          # seconds remaining
    max_life: float
    radius: float
    color: tuple[int, int, int]

    def alive(self) -> bool:
        return self.life > 0

    def step(self, dt: float) -> None:
        self.x += self.vx * dt
        self.y += self.vy * dt
        self.vy += 12.0 * dt  # slight upward drift feel via damping, not real gravity
        self.vx *= 0.98
        self.vy *= 0.98
        self.life -= dt

    def draw(self, frame: np.ndarray) -> None:
        t = max(0.0, self.life / self.max_life)  # 1 -> just born, 0 -> about to die
        alpha = t
        radius = max(1, int(self.radius * (0.4 + 0.6 * t)))
        color = tuple(int(c * alpha) for c in self.color)
        cv2.circle(frame, (int(self.x), int(self.y)), radius, color, -1, cv2.LINE_AA)


@dataclass
class ParticleSystem:
    """Owns the particle list and the emission rate policy."""

    particles: list[Particle] = field(default_factory=list)
    max_particles: int = MAX_PARTICLES

    def emit(self, x: float, y: float, color: tuple[int, int, int], count: int = 1) -> None:
        """Spawn up to `count` particles at (x, y), respecting the hard cap."""
        room = self.max_particles - len(self.particles)
        for _ in range(max(0, min(count, room))):
            speed = random.uniform(20, 70)
            self.particles.append(Particle(
                x=x, y=y,
                vx=speed * random.uniform(-1, 1),
                vy=speed * random.uniform(-1, 1) - 10,
                life=random.uniform(0.4, 0.9),
                max_life=0.9,
                radius=random.uniform(1.5, 3.5),
                color=color,
            ))

    def update(self, dt: float) -> None:
        for p in self.particles:
            p.step(dt)
        self.particles = [p for p in self.particles if p.alive()]

    def render(self, frame: np.ndarray) -> None:
        for p in self.particles:
            p.draw(frame)

    def __len__(self) -> int:
        return len(self.particles)
