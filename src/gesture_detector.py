"""Pure gesture classification: finger count -> gesture name, pinch test,
and a generic temporal-stability primitive used to debounce noisy readings.

Nothing in this module touches MediaPipe, the camera, or rendering — it only
turns the raw per-frame facts from `hand_detector.py` into named gesture
states. The stateful "hold this value for N frames" smoothing that actually
uses `StableValue` over time lives in `finger_tracker.py`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, TypeVar

from .hand_detector import HandReading

# --- Gesture names -----------------------------------------------------------
CLOSED_HAND = "CLOSED_HAND"
ONE_FINGER = "ONE_FINGER"
TWO_FINGERS = "TWO_FINGERS"
THREE_FINGERS = "THREE_FINGERS"
FOUR_FINGERS = "FOUR_FINGERS"
OPEN_HAND = "OPEN_HAND"
PINCH = "PINCH"  # a modifier, not mutually exclusive with the count above

_COUNT_TO_GESTURE = {
    0: CLOSED_HAND,
    1: ONE_FINGER,
    2: TWO_FINGERS,
    3: THREE_FINGERS,
    4: FOUR_FINGERS,
    5: OPEN_HAND,
}

# Pinch threshold as a fraction of the hand's own bounding-box diagonal, not
# a fixed pixel distance — this keeps pinch detection working whether the
# hand is close to the camera (large) or far away (small).
PINCH_RATIO_THRESHOLD = 0.12


def gesture_from_count(finger_count: int) -> str:
    """Map a 0-5 finger count to a gesture name."""
    return _COUNT_TO_GESTURE.get(max(0, min(5, finger_count)), CLOSED_HAND)


def is_pinching(reading: HandReading) -> bool:
    """Thumb tip touching index tip, scaled by the hand's own size on screen."""
    if reading.relative_size <= 0:
        return False
    # Normalize the thumb-index pixel distance by the hand's own bounding-box
    # diagonal (also in pixels) rather than a fixed pixel threshold, so pinch
    # detection keeps working whether the hand is near or far from the camera.
    x0, y0, x1, y1 = reading.bbox_px
    bbox_diag_px = ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5
    if bbox_diag_px <= 0:
        return False
    return (reading.thumb_index_distance_px() / bbox_diag_px) < PINCH_RATIO_THRESHOLD


@dataclass(frozen=True)
class HandGesture:
    """One hand's classified, single-frame state — still unsmoothed."""

    detected: bool = False
    finger_count: int = 0
    gesture: str = CLOSED_HAND
    pinch: bool = False
    reading: HandReading | None = None


def classify(reading: HandReading | None) -> HandGesture:
    if reading is None:
        return HandGesture()
    return HandGesture(
        detected=True,
        finger_count=reading.finger_count,
        gesture=gesture_from_count(reading.finger_count),
        pinch=is_pinching(reading),
        reading=reading,
    )


@dataclass(frozen=True)
class GestureSnapshot:
    """Per-frame left/right breakdown, keyed by MediaPipe handedness."""

    left: HandGesture
    right: HandGesture

    @property
    def both_hands(self) -> bool:
        return self.left.detected and self.right.detected

    @property
    def hand_count(self) -> int:
        return int(self.left.detected) + int(self.right.detected)


def evaluate(readings: list[HandReading]) -> GestureSnapshot:
    """Reduce this frame's hand readings into a left/right GestureSnapshot.

    If MediaPipe reports two hands with the same label (rare, low-confidence
    frames), the higher-confidence reading for that label wins.
    """
    best: dict[str, HandReading] = {}
    for r in readings:
        if r.label not in best or r.confidence > best[r.label].confidence:
            best[r.label] = r
    return GestureSnapshot(left=classify(best.get("Left")), right=classify(best.get("Right")))


# --- Generic temporal stability primitive ------------------------------------

T = TypeVar("T")


class StableValue(Generic[T]):
    """Only reports a new value after it has been the raw reading for
    `hold_ms` continuously. This is what stops a noisy per-frame signal
    (finger count flipping 4/5/4/5, or a boolean gesture flag) from making
    the visuals flicker.
    """

    def __init__(self, hold_ms: float, initial: T) -> None:
        self.hold_ms = hold_ms
        self._stable: T = initial
        self._pending: T | None = None
        self._pending_since: float | None = None

    @property
    def value(self) -> T:
        return self._stable

    def update(self, raw: T, now_ms: float) -> T:
        if raw == self._stable:
            self._pending = None
            self._pending_since = None
            return self._stable

        if self._pending != raw:
            self._pending = raw
            self._pending_since = now_ms
            return self._stable

        assert self._pending_since is not None
        if now_ms - self._pending_since >= self.hold_ms:
            self._stable = raw
            self._pending = None
            self._pending_since = None
        return self._stable


GESTURE_HOLD_MS = 400  # default hold time for finger-count / gesture smoothing


class GestureDebouncer:
    """Boolean-specialised convenience wrapper around StableValue, kept for
    simple on/off style signals (e.g. "is the activation gesture held?")."""

    def __init__(self, hold_ms: int = GESTURE_HOLD_MS) -> None:
        self._stable_value: StableValue[bool] = StableValue(hold_ms, False)

    @property
    def stable(self) -> bool:
        return self._stable_value.value

    def update(self, raw: bool, now_ms: float) -> bool:
        return self._stable_value.update(raw, now_ms)
