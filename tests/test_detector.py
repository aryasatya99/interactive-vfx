"""Tests for finger-state geometry, handedness handling, and gesture logic.

No camera and no MediaPipe model file is needed: hand landmarks are
synthetic points, so these run anywhere `pip install -r requirements.txt`
has succeeded.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.gesture_detector import GestureDebouncer, evaluate  # noqa: E402
from src.hand_detector import (  # noqa: E402
    HandReading,
    build_reading,
    index_open,
    thumb_open,
)

Point = tuple[float, float]


def make_landmarks(thumb_is_open: bool, index_is_open: bool) -> list[Point]:
    """Build a synthetic 21-point hand with a controllable thumb/index state.

    Only WRIST(0), THUMB_IP(3)/THUMB_TIP(4), and INDEX_PIP(6)/INDEX_TIP(8)
    matter to the geometry under test; the rest are filled with plausible
    placeholder positions.
    """
    wrist: Point = (0.5, 0.9)
    thumb_tip = (0.75, 0.55) if thumb_is_open else (0.52, 0.85)
    thumb_ip = (0.58, 0.75)
    index_tip = (0.5, 0.25) if index_is_open else (0.5, 0.75)
    index_pip = (0.5, 0.6)

    points: list[Point] = [wrist] + [(0.5, 0.5)] * 20
    points[1] = (0.53, 0.85)   # THUMB_CMC
    points[2] = (0.55, 0.8)    # THUMB_MCP
    points[3] = thumb_ip       # THUMB_IP
    points[4] = thumb_tip      # THUMB_TIP
    points[5] = (0.5, 0.7)     # INDEX_MCP
    points[6] = index_pip      # INDEX_PIP
    points[7] = (0.5, 0.45)    # INDEX_DIP
    points[8] = index_tip      # INDEX_TIP
    points[9] = (0.5, 0.7)     # MIDDLE_MCP
    points[17] = (0.45, 0.72)  # PINKY_MCP
    return points


def make_reading(label: str, thumb_is_open: bool, index_is_open: bool) -> HandReading:
    landmarks = make_landmarks(thumb_is_open, index_is_open)
    return build_reading(label, confidence=0.95, landmarks=landmarks,
                        frame_w=1280, frame_h=720)


# --- Finger geometry ---------------------------------------------------------

def test_thumb_open_detected() -> None:
    assert thumb_open(make_landmarks(thumb_is_open=True, index_is_open=True)) is True


def test_thumb_closed_detected() -> None:
    assert thumb_open(make_landmarks(thumb_is_open=False, index_is_open=True)) is False


def test_index_open_detected() -> None:
    assert index_open(make_landmarks(thumb_is_open=True, index_is_open=True)) is True


def test_index_closed_detected() -> None:
    assert index_open(make_landmarks(thumb_is_open=True, index_is_open=False)) is False


# --- Handedness --------------------------------------------------------------

def test_left_right_classification_comes_from_label_not_position() -> None:
    """Handedness must be whatever MediaPipe reports, never derived from
    where the hand sits on screen. Two hands at the very same landmark
    positions but different labels must classify accordingly."""
    landmarks = make_landmarks(thumb_is_open=True, index_is_open=True)
    left = build_reading("Left", 0.9, landmarks, 1280, 720)
    right = build_reading("Right", 0.9, landmarks, 1280, 720)
    assert left.is_left and not left.is_right
    assert right.is_right and not right.is_left


# --- Gesture validation -------------------------------------------------------

def test_valid_two_hand_gesture() -> None:
    readings = [
        make_reading("Left", thumb_is_open=True, index_is_open=True),
        make_reading("Right", thumb_is_open=True, index_is_open=True),
    ]
    assert evaluate(readings).valid is True


def test_single_hand_is_invalid() -> None:
    readings = [make_reading("Left", thumb_is_open=True, index_is_open=True)]
    snap = evaluate(readings)
    assert snap.valid is False
    assert snap.left.ok is True
    assert snap.right.detected is False


def test_missing_left_hand_is_invalid() -> None:
    readings = [make_reading("Right", thumb_is_open=True, index_is_open=True)]
    assert evaluate(readings).valid is False


def test_missing_right_hand_is_invalid() -> None:
    readings = [make_reading("Left", thumb_is_open=True, index_is_open=True)]
    assert evaluate(readings).valid is False


def test_left_thumb_closed_is_invalid() -> None:
    readings = [
        make_reading("Left", thumb_is_open=False, index_is_open=True),
        make_reading("Right", thumb_is_open=True, index_is_open=True),
    ]
    assert evaluate(readings).valid is False


def test_left_index_closed_is_invalid() -> None:
    readings = [
        make_reading("Left", thumb_is_open=True, index_is_open=False),
        make_reading("Right", thumb_is_open=True, index_is_open=True),
    ]
    assert evaluate(readings).valid is False


def test_right_thumb_closed_is_invalid() -> None:
    readings = [
        make_reading("Left", thumb_is_open=True, index_is_open=True),
        make_reading("Right", thumb_is_open=False, index_is_open=True),
    ]
    assert evaluate(readings).valid is False


def test_right_index_closed_is_invalid() -> None:
    readings = [
        make_reading("Left", thumb_is_open=True, index_is_open=True),
        make_reading("Right", thumb_is_open=True, index_is_open=False),
    ]
    assert evaluate(readings).valid is False


def test_gesture_is_not_just_hand_count() -> None:
    """Two hands with a failing finger state must NOT count as valid -
    guards against a naive `len(hands) == 2` implementation."""
    readings = [
        make_reading("Left", thumb_is_open=False, index_is_open=False),
        make_reading("Right", thumb_is_open=False, index_is_open=False),
    ]
    assert len(readings) == 2
    assert evaluate(readings).valid is False


# --- Debounce -----------------------------------------------------------------

def test_debounce_holds_before_flipping_on() -> None:
    d = GestureDebouncer(hold_ms=400)
    assert d.update(True, now_ms=0) is False       # just started, not yet held
    assert d.update(True, now_ms=200) is False      # still under 400ms
    assert d.update(True, now_ms=450) is True        # held long enough


def test_debounce_ignores_brief_flicker() -> None:
    d = GestureDebouncer(hold_ms=400)
    assert d.update(True, now_ms=0) is False
    assert d.update(False, now_ms=100) is False      # flicker resets the pending timer
    assert d.update(True, now_ms=150) is False        # pending restarted here
    assert d.update(True, now_ms=400) is False        # only 250ms since restart
    assert d.update(True, now_ms=560) is True
