"""Two-hand gesture validation + debounce.

Valid gesture rule (deliberately explicit, not `len(hands) == 2`):

    left_hand_detected  and left_thumb_open  and left_index_open
    and right_hand_detected and right_thumb_open and right_index_open

Two hands alone is not enough — each hand's identity (from MediaPipe
handedness) and each hand's thumb+index state are all checked individually.
"""

from __future__ import annotations

from dataclasses import dataclass

from .hand_detector import HandReading

GESTURE_HOLD_MS = 400  # how long the raw gesture must hold before it flips state


@dataclass(frozen=True)
class HandStatus:
    detected: bool = False
    thumb_open: bool = False
    index_open: bool = False

    @property
    def ok(self) -> bool:
        return self.detected and self.thumb_open and self.index_open


@dataclass(frozen=True)
class GestureSnapshot:
    """Per-frame breakdown, used both for the gesture decision and the UI."""

    left: HandStatus
    right: HandStatus

    @property
    def valid(self) -> bool:
        return self.left.ok and self.right.ok


def evaluate(readings: list[HandReading]) -> GestureSnapshot:
    """Reduce this frame's hand readings into left/right status.

    If MediaPipe reports two hands with the same label (rare, low-confidence
    frames), the higher-confidence reading for that label wins.
    """
    best: dict[str, HandReading] = {}
    for r in readings:
        if r.label not in best or r.confidence > best[r.label].confidence:
            best[r.label] = r

    def to_status(reading: HandReading | None) -> HandStatus:
        if reading is None:
            return HandStatus()
        return HandStatus(detected=True, thumb_open=reading.thumb_open,
                          index_open=reading.index_open)

    return GestureSnapshot(left=to_status(best.get("Left")),
                           right=to_status(best.get("Right")))


class GestureDebouncer:
    """Only flips the reported gesture state after `hold_ms` of a steady raw
    reading, in either direction. This is what stops the X-ray display from
    flickering when a landmark jitters across the open/closed threshold for
    a frame or two.
    """

    def __init__(self, hold_ms: int = GESTURE_HOLD_MS) -> None:
        self.hold_ms = hold_ms
        self._stable = False
        self._pending: bool | None = None
        self._pending_since: float | None = None

    @property
    def stable(self) -> bool:
        return self._stable

    def update(self, raw: bool, now_ms: float) -> bool:
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
