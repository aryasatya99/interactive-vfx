"""Tests for hand geometry: finger extension, finger counting, handedness,
palm/bbox helpers, and the particle system.

No camera and no MediaPipe model file is needed: hand landmarks are
synthetic points, so these run anywhere `pip install -r requirements.txt`
has succeeded.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.hand_detector import (  # noqa: E402
    bounding_box,
    build_reading,
    count_open_fingers,
    index_open,
    palm_center,
    thumb_open,
)
from src.particles import MAX_PARTICLES, ParticleSystem  # noqa: E402

Point = tuple[float, float]


def make_landmarks(open_fingers: set[str] | None = None) -> list[Point]:
    """A synthetic 21-point hand where each finger in `open_fingers` is
    extended (tip far from wrist) and every other finger is curled (tip
    near wrist). Positions are otherwise arbitrary but consistent."""
    open_fingers = open_fingers or set()
    wrist: Point = (0.5, 0.9)
    points: list[Point] = [wrist] + [(0.5, 0.5)] * 20

    def set_finger(open_: bool, mcp: int, mid: int, tip: int, lean: float) -> None:
        points[mcp] = (0.5 + lean * 0.05, 0.75)
        points[mid] = (0.5 + lean * 0.08, 0.6)
        points[tip] = (0.5 + lean * 0.12, 0.25) if open_ else (0.5 + lean * 0.06, 0.82)

    set_finger("thumb" in open_fingers, 2, 3, 4, lean=1.5)
    set_finger("index" in open_fingers, 5, 6, 8, lean=0.7)
    set_finger("middle" in open_fingers, 9, 10, 12, lean=0.0)
    set_finger("ring" in open_fingers, 13, 14, 16, lean=-0.7)
    set_finger("pinky" in open_fingers, 17, 18, 20, lean=-1.4)
    return points


def make_reading(label: str, open_fingers: set[str]):
    return build_reading(label, 0.95, make_landmarks(open_fingers), 1280, 720)


# --- Finger geometry ---------------------------------------------------------

def test_thumb_open_and_closed() -> None:
    assert thumb_open(make_landmarks({"thumb"})) is True
    assert thumb_open(make_landmarks(set())) is False


def test_index_open_and_closed() -> None:
    assert index_open(make_landmarks({"index"})) is True
    assert index_open(make_landmarks(set())) is False


def test_finger_count_zero_for_closed_fist() -> None:
    assert count_open_fingers(make_landmarks(set())) == 0


def test_finger_count_five_for_open_hand() -> None:
    all_fingers = {"thumb", "index", "middle", "ring", "pinky"}
    assert count_open_fingers(make_landmarks(all_fingers)) == 5


def test_finger_count_partial() -> None:
    assert count_open_fingers(make_landmarks({"index", "middle"})) == 2
    assert count_open_fingers(make_landmarks({"index"})) == 1
    assert count_open_fingers(make_landmarks({"index", "middle", "ring"})) == 3
    assert count_open_fingers(make_landmarks({"index", "middle", "ring", "pinky"})) == 4


# --- Handedness --------------------------------------------------------------

def test_left_right_label_comes_from_input_not_position() -> None:
    """Handedness must be whatever MediaPipe reports, never derived from
    where the hand sits on screen - identical landmarks, different labels."""
    landmarks = make_landmarks({"index"})
    left = build_reading("Left", 0.9, landmarks, 1280, 720)
    right = build_reading("Right", 0.9, landmarks, 1280, 720)
    assert left.is_left and not left.is_right
    assert right.is_right and not right.is_left


# --- Palm / bbox helpers ------------------------------------------------------

def test_palm_center_is_within_hand_bounds() -> None:
    landmarks = make_landmarks({"index", "middle"})
    x0, y0, x1, y1 = bounding_box(landmarks)
    px, py = palm_center(landmarks)
    assert x0 <= px <= x1
    assert y0 <= py <= y1


def test_reading_exposes_all_five_fingertips() -> None:
    reading = make_reading("Left", {"thumb", "index"})
    assert reading.thumb_tip and reading.index_tip
    assert reading.middle_tip and reading.ring_tip and reading.pinky_tip
    assert reading.relative_size > 0


# --- Particle system -----------------------------------------------------

def test_particle_emit_and_lifetime() -> None:
    system = ParticleSystem()
    system.emit(100, 100, (0, 255, 0), count=5)
    assert len(system) == 5
    for _ in range(50):  # advance well past any particle's max lifetime
        system.update(dt=0.1)
    assert len(system) == 0  # all particles must expire on their own


def test_particle_count_is_capped() -> None:
    system = ParticleSystem()
    for _ in range(200):
        system.emit(0, 0, (255, 255, 255), count=10)
    assert len(system) <= MAX_PARTICLES
