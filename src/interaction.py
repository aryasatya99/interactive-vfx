"""Maps stabilized hand/gesture data onto the numeric parameters the visual
engine actually consumes: position, scale, particle intensity, animation
intensity, relative depth, and activate/pause requests.

GESTURE_CONFIG is the single place that decides which finger count controls
which visual channel. Change the values on the right to reassign a control
without touching any other file:

    GESTURE_CONFIG = {
        "one_finger": "position",
        "two_fingers": "scale",
        "three_fingers": "particles",
        "four_fingers": "animation",
        "five_fingers": "activate",
        "fist": "pause",
    }

When both hands are visible, index-to-index distance/midpoint takes over
position and scale (see "Two-hand interaction" in the README) - showing two
hands is a stronger, more deliberate gesture than a single hand's finger
count, so it overrides the per-hand mapping for those two channels.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .finger_tracker import StableHand, TrackingSnapshot
from .gesture_detector import (
    CLOSED_HAND,
    FOUR_FINGERS,
    ONE_FINGER,
    OPEN_HAND,
    THREE_FINGERS,
    TWO_FINGERS,
    StableValue,
)
from .hand_detector import HandReading
from .utils import clamp, ema, ema_point

GESTURE_CONFIG: dict[str, str] = {
    "one_finger": "position",
    "two_fingers": "scale",
    "three_fingers": "particles",
    "four_fingers": "animation",
    "five_fingers": "activate",
    "fist": "pause",
}

# Bridges gesture_detector's gesture-name constants to the human-readable
# keys used in GESTURE_CONFIG above, so re-assigning a channel only ever
# requires editing the dict, not this lookup.
_GESTURE_TO_CONFIG_KEY = {
    ONE_FINGER: "one_finger",
    TWO_FINGERS: "two_fingers",
    THREE_FINGERS: "three_fingers",
    FOUR_FINGERS: "four_fingers",
    OPEN_HAND: "five_fingers",
    CLOSED_HAND: "fist",
}

MIN_SCALE, MAX_SCALE = 0.4, 2.2
DEFAULT_SCALE = 1.0
# A hand filling roughly this much of the frame diagonal counts as "as close
# as this app cares to model" - past this, depth/intensity just clamp at 1.0.
EXPECTED_MAX_RELATIVE_SIZE = 0.55

POSITION_ALPHA = 0.25   # EMA smoothing factors: higher = snappier, lower = smoother
SCALE_ALPHA = 0.15
INTENSITY_ALPHA = 0.12
DEPTH_ALPHA = 0.10


def action_for(hand: StableHand) -> str | None:
    """Which visual channel this hand's current (stabilized) gesture controls,
    per GESTURE_CONFIG - or None if the hand isn't detected or its gesture
    isn't mapped to anything."""
    if not hand.detected:
        return None
    key = _GESTURE_TO_CONFIG_KEY.get(hand.gesture)
    return GESTURE_CONFIG.get(key) if key else None


def _bbox_diag_px(reading: HandReading) -> float:
    x0, y0, x1, y1 = reading.bbox_px
    return math.hypot(x1 - x0, y1 - y0)


def _scale_from_pinch(reading: HandReading) -> float:
    """Thumb<->index distance, normalized by the hand's own bbox diagonal so
    it works at any distance from the camera, mapped onto MIN_SCALE..MAX_SCALE."""
    diag = _bbox_diag_px(reading)
    if diag <= 0:
        return DEFAULT_SCALE
    ratio = reading.thumb_index_distance_px() / diag
    # ratio is roughly 0 (pinched) .. 0.9 (fingers spread wide) in practice.
    return clamp(MIN_SCALE + ratio * (MAX_SCALE - MIN_SCALE) / 0.7, MIN_SCALE, MAX_SCALE)


def _depth_from(reading: HandReading) -> float:
    return clamp(reading.relative_size / EXPECTED_MAX_RELATIVE_SIZE, 0.0, 1.0)


@dataclass(frozen=True)
class InteractionState:
    """Everything the visual engine needs to render one frame."""

    position: tuple[float, float]
    scale: float
    particle_intensity: float
    animation_intensity: float
    depth: float
    hand_count: int
    two_hands_active: bool
    two_hand_points: tuple[tuple[int, int], tuple[int, int]] | None


class InteractionController:
    """Owns the EMA-smoothed state and turns each frame's TrackingSnapshot
    into an InteractionState. All smoothing lives here so main.py and
    visual_engine.py never see raw, jittery values."""

    def __init__(self, frame_w: int, frame_h: int) -> None:
        self._position = (frame_w / 2, frame_h / 2)
        self._scale = DEFAULT_SCALE
        self._particle_intensity = 0.0
        self._animation_intensity = 0.3  # gentle idle motion even with no input
        self._depth = 0.0

    def update(self, snapshot: TrackingSnapshot) -> InteractionState:
        target_position: tuple[float, float] | None = None
        target_scale: float | None = None
        target_particles: float | None = None
        target_animation: float | None = None
        depth_samples: list[float] = []

        for hand in (snapshot.left, snapshot.right):
            if not hand.detected or hand.raw.reading is None:
                continue
            reading = hand.raw.reading
            depth_samples.append(_depth_from(reading))

            action = action_for(hand)
            if action == "position":
                target_position = reading.index_tip
            elif action == "scale":
                target_scale = _scale_from_pinch(reading)
            elif action == "particles":
                target_particles = _depth_from(reading)
            elif action == "animation":
                target_animation = _depth_from(reading)

        two_hands_active = snapshot.both_hands
        two_hand_points: tuple[tuple[int, int], tuple[int, int]] | None = None
        if two_hands_active:
            left_r = snapshot.left.raw.reading
            right_r = snapshot.right.raw.reading
            assert left_r is not None and right_r is not None
            two_hand_points = (left_r.index_tip, right_r.index_tip)
            # Two hands showing intent together is a stronger signal than a
            # single hand's finger count - it overrides position/scale.
            target_position = (
                (left_r.index_tip[0] + right_r.index_tip[0]) / 2,
                (left_r.index_tip[1] + right_r.index_tip[1]) / 2,
            )
            # Pixel span between the two index tips, mapped onto the same
            # scale range as the single-hand pinch control. 900px is roughly
            # "arms comfortably spread" at 1280x720 - tune to taste.
            span = math.hypot(left_r.index_tip[0] - right_r.index_tip[0],
                             left_r.index_tip[1] - right_r.index_tip[1])
            target_scale = clamp(MIN_SCALE + (span / 900.0) * (MAX_SCALE - MIN_SCALE),
                                 MIN_SCALE, MAX_SCALE)

        if target_position is not None:
            self._position = ema_point(self._position, target_position, POSITION_ALPHA)
        if target_scale is not None:
            self._scale = ema(self._scale, target_scale, SCALE_ALPHA)
        if target_particles is not None:
            self._particle_intensity = ema(self._particle_intensity, target_particles,
                                           INTENSITY_ALPHA)
        if target_animation is not None:
            self._animation_intensity = ema(self._animation_intensity, target_animation,
                                            INTENSITY_ALPHA)

        depth_target = max(depth_samples) if depth_samples else 0.0
        self._depth = ema(self._depth, depth_target, DEPTH_ALPHA)

        return InteractionState(
            position=self._position,
            scale=self._scale,
            particle_intensity=self._particle_intensity,
            animation_intensity=self._animation_intensity,
            depth=self._depth,
            hand_count=(int(snapshot.left.detected) + int(snapshot.right.detected)),
            two_hands_active=two_hands_active,
            two_hand_points=two_hand_points,
        )


class ActivationController:
    """Debounces the OPEN_HAND / CLOSED_HAND gestures used to turn the main
    visual effect on and off, independently of the SPACE key. Requires the
    gesture to be held for `hold_ms` - longer than ordinary finger-count
    smoothing - specifically so a passing open hand doesn't false-trigger
    activation."""

    def __init__(self, hold_ms: float = 600) -> None:
        self._activate = StableValue[bool](hold_ms, False)
        self._pause = StableValue[bool](hold_ms, False)

    def update(self, snapshot: TrackingSnapshot, now_ms: float) -> bool | None:
        raw_activate = any(h.detected and h.gesture == OPEN_HAND
                           for h in (snapshot.left, snapshot.right))
        raw_pause = any(h.detected and h.gesture == CLOSED_HAND
                        for h in (snapshot.left, snapshot.right))
        activate_stable = self._activate.update(raw_activate, now_ms)
        pause_stable = self._pause.update(raw_pause, now_ms)
        if activate_stable:
            return True
        if pause_stable:
            return False
        return None


class PresenceFader:
    """Turns "is at least one hand currently detected" into a smooth 0..1
    fade, used to auto fade the VFX in when a hand appears and fade it back
    out (instead of popping off) shortly after both hands leave the frame.

    A short grace period (default 400ms) means a single dropped-detection
    frame doesn't cause a visible flicker - the fade-out only starts once no
    hand has been seen for the full grace period, then eases toward 0 over
    `fade_ms`, independent of the SPACE/gesture ACTIVE-STANDBY toggle."""

    def __init__(self, grace_ms: float = 400, fade_ms: float = 400) -> None:
        self.grace_ms = grace_ms
        self.fade_ms = fade_ms
        self.alpha = 0.0
        self._last_seen_ms = -1e9

    def update(self, hand_present: bool, dt: float, now_ms: float) -> float:
        if hand_present:
            self._last_seen_ms = now_ms
        target = 1.0 if (now_ms - self._last_seen_ms) <= self.grace_ms else 0.0
        # Exponential approach toward target; speed derived from fade_ms so
        # it takes roughly fade_ms to cross most of the gap, regardless of
        # frame rate (dt-scaled, not a fixed per-frame step).
        speed = 1.0 / max(0.05, self.fade_ms / 1000.0)
        self.alpha = clamp(self.alpha + (target - self.alpha) * clamp(dt * speed, 0.0, 1.0), 0.0, 1.0)
        return self.alpha
