"""Temporal smoothing layer on top of gesture_detector's per-frame classification.

A raw finger count can flip (4, 5, 4, 5, ...) for a couple of frames just from
landmark jitter, even when the hand isn't actually moving. `FingerTracker`
holds one `StableValue` per hand (left/right) and only reports a new finger
count / gesture once it has been the raw reading continuously for
`hold_ms` (~300-500ms, matching a natural, deliberate hand pose change).
"""

from __future__ import annotations

from dataclasses import dataclass

from .gesture_detector import (
    CLOSED_HAND,
    GESTURE_HOLD_MS,
    GestureSnapshot,
    HandGesture,
    StableValue,
    evaluate,
    gesture_from_count,
)
from .hand_detector import HandReading


@dataclass(frozen=True)
class StableHand:
    """A hand's gesture state after temporal smoothing."""

    detected: bool
    finger_count: int          # stabilized, 0-5
    gesture: str                # stabilized gesture name
    raw: HandGesture            # this frame's unsmoothed classification (still useful:
                                 # positions/pinch/depth should stay responsive, only the
                                 # discrete finger_count/gesture need debouncing)


@dataclass(frozen=True)
class TrackingSnapshot:
    left: StableHand
    right: StableHand

    @property
    def both_hands(self) -> bool:
        return self.left.detected and self.right.detected


class _HandSmoother:
    def __init__(self, hold_ms: float) -> None:
        self._count = StableValue[int](hold_ms, 0)

    def update(self, hand: HandGesture, now_ms: float) -> StableHand:
        raw_count = hand.finger_count if hand.detected else 0
        stable_count = self._count.update(raw_count, now_ms)
        # "detected" itself is not smoothed - a hand disappearing should drop
        # out immediately (there's nothing meaningful to hold stable on
        # empty tracking), only its finger count/gesture is debounced.
        return StableHand(
            detected=hand.detected,
            finger_count=stable_count if hand.detected else 0,
            gesture=gesture_from_count(stable_count) if hand.detected else CLOSED_HAND,
            raw=hand,
        )


class FingerTracker:
    """Owns the left/right smoothers and turns a raw GestureSnapshot into a
    debounced TrackingSnapshot, frame by frame."""

    def __init__(self, hold_ms: float = GESTURE_HOLD_MS) -> None:
        self._left = _HandSmoother(hold_ms)
        self._right = _HandSmoother(hold_ms)

    def update(self, readings: list[HandReading], now_ms: float) -> TrackingSnapshot:
        snapshot: GestureSnapshot = evaluate(readings)
        return TrackingSnapshot(
            left=self._left.update(snapshot.left, now_ms),
            right=self._right.update(snapshot.right, now_ms),
        )
