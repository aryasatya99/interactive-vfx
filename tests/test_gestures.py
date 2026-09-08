"""Tests for gesture classification, temporal smoothing, GESTURE_CONFIG
mapping, and the interaction controller. All synthetic - no camera or model
file required.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.finger_tracker import FingerTracker  # noqa: E402
from src.gesture_detector import (  # noqa: E402
    CLOSED_HAND,
    FOUR_FINGERS,
    ONE_FINGER,
    OPEN_HAND,
    THREE_FINGERS,
    TWO_FINGERS,
    StableValue,
    evaluate,
    gesture_from_count,
    is_pinching,
)
from src.hand_detector import build_reading  # noqa: E402
from src.interaction import (  # noqa: E402
    GESTURE_CONFIG,
    ActivationController,
    InteractionController,
    action_for,
)

Point = tuple[float, float]


def make_landmarks(open_fingers: set[str], pinch: bool = False) -> list[Point]:
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

    if pinch:
        points[4] = points[8] = (0.5, 0.4)  # thumb tip == index tip
    return points


def make_reading(label: str, open_fingers: set[str], pinch: bool = False):
    return build_reading(label, 0.95, make_landmarks(open_fingers, pinch), 1280, 720)


# --- Gesture classification --------------------------------------------------

def test_gesture_from_count_covers_zero_through_five() -> None:
    assert gesture_from_count(0) == CLOSED_HAND
    assert gesture_from_count(1) == ONE_FINGER
    assert gesture_from_count(2) == TWO_FINGERS
    assert gesture_from_count(3) == THREE_FINGERS
    assert gesture_from_count(4) == FOUR_FINGERS
    assert gesture_from_count(5) == OPEN_HAND


def test_pinch_detected_when_thumb_and_index_touch() -> None:
    reading = make_reading("Right", {"middle", "ring", "pinky"}, pinch=True)
    assert is_pinching(reading) is True


def test_pinch_not_detected_on_open_hand() -> None:
    reading = make_reading("Right", {"thumb", "index", "middle", "ring", "pinky"})
    assert is_pinching(reading) is False


def test_evaluate_reads_handedness_not_order() -> None:
    readings = [
        make_reading("Right", {"index"}),
        make_reading("Left", set()),
    ]
    snap = evaluate(readings)
    assert snap.left.finger_count == 0
    assert snap.right.finger_count == 1
    assert snap.both_hands is True


def test_evaluate_missing_hand_is_not_detected() -> None:
    snap = evaluate([make_reading("Left", {"index", "middle"})])
    assert snap.left.detected is True
    assert snap.right.detected is False
    assert snap.right.finger_count == 0
    assert snap.both_hands is False


# --- Temporal stability (StableValue / FingerTracker) ------------------------

def test_stable_value_holds_before_flipping() -> None:
    sv = StableValue[int](hold_ms=400, initial=0)
    assert sv.update(3, now_ms=0) == 0        # not held long enough yet
    assert sv.update(3, now_ms=200) == 0
    assert sv.update(3, now_ms=450) == 3       # now held past 400ms


def test_stable_value_ignores_brief_flicker() -> None:
    sv = StableValue[int](hold_ms=400, initial=4)
    assert sv.update(5, now_ms=0) == 4
    assert sv.update(4, now_ms=100) == 4       # flicker back resets the pending timer
    assert sv.update(5, now_ms=150) == 4       # pending restarted here
    assert sv.update(5, now_ms=400) == 4        # only 250ms since restart
    assert sv.update(5, now_ms=560) == 5


def test_finger_tracker_smooths_noisy_counts() -> None:
    tracker = FingerTracker(hold_ms=300)
    now = 0.0
    # Rapid 4/5 flicker for under the hold time must not change the stable count.
    for count, dt in [(4, 0), (5, 50), (4, 100), (5, 150), (4, 200)]:
        readings = [make_reading("Left", _fingers_for_count(count))]
        snap = tracker.update(readings, now_ms=now + dt)
    assert snap.left.finger_count == 0  # started at 0, hasn't been held long enough

    # Now hold 5 fingers continuously past the hold window.
    for t in (400, 500, 650, 800):
        readings = [make_reading("Left", _fingers_for_count(5))]
        snap = tracker.update(readings, now_ms=t)
    assert snap.left.finger_count == 5
    assert snap.left.gesture == OPEN_HAND


def _fingers_for_count(n: int) -> set[str]:
    order = ["thumb", "index", "middle", "ring", "pinky"]
    return set(order[:n])


# --- GESTURE_CONFIG mapping ----------------------------------------------

def test_gesture_config_has_all_documented_channels() -> None:
    assert GESTURE_CONFIG["one_finger"] == "position"
    assert GESTURE_CONFIG["two_fingers"] == "scale"
    assert GESTURE_CONFIG["three_fingers"] == "particles"
    assert GESTURE_CONFIG["four_fingers"] == "animation"
    assert GESTURE_CONFIG["five_fingers"] == "activate"
    assert GESTURE_CONFIG["fist"] == "pause"


def test_action_for_maps_stable_hand_to_channel() -> None:
    tracker = FingerTracker(hold_ms=0)  # zero hold = stabilizes immediately for this test
    readings = [make_reading("Left", _fingers_for_count(1))]
    snap = tracker.update(readings, now_ms=0)
    snap = tracker.update(readings, now_ms=1)  # second call lets StableValue settle at 0ms hold
    assert action_for(snap.left) == "position"


def test_action_for_none_when_hand_not_detected() -> None:
    tracker = FingerTracker(hold_ms=0)
    snap = tracker.update([], now_ms=0)
    assert action_for(snap.left) is None


# --- Interaction / activation ------------------------------------------------

def test_interaction_controller_tracks_position_channel() -> None:
    tracker = FingerTracker(hold_ms=0)
    controller = InteractionController(1280, 720)
    readings = [make_reading("Left", _fingers_for_count(1))]
    for t in range(0, 20):
        snap = tracker.update(readings, now_ms=t)
        state = controller.update(snap)
    # Position should have moved substantially toward the hand's index tip
    # (top area of the synthetic hand, y around 0.25 * 720 = 180).
    assert state.position[1] < 700


def test_activation_controller_requires_sustained_open_hand() -> None:
    tracker = FingerTracker(hold_ms=0)  # zero hold: two calls are enough to latch
    activation = ActivationController(hold_ms=300)
    readings = [make_reading("Left", _fingers_for_count(5))]
    tracker.update(readings, now_ms=0)
    snap = tracker.update(readings, now_ms=0)  # finger_count now stably 5
    assert activation.update(snap, now_ms=0) is None       # not held long enough
    assert activation.update(snap, now_ms=100) is None
    assert activation.update(snap, now_ms=350) is True       # now sustained past hold_ms


def test_activation_controller_pause_on_closed_fist() -> None:
    tracker = FingerTracker(hold_ms=0)
    activation = ActivationController(hold_ms=300)
    readings = [make_reading("Left", _fingers_for_count(0))]
    tracker.update(readings, now_ms=0)
    snap = tracker.update(readings, now_ms=0)  # finger_count now stably 0 (CLOSED_HAND)
    activation.update(snap, now_ms=0)
    assert activation.update(snap, now_ms=350) is False
