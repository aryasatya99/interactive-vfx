"""Visual mode system: NORMAL / FROSTED_BLUR / VFX_FOCUS / GLOW_DREAM.

Each mode is just a dict of numeric parameters in VISUAL_MODES - adding a
new mode later means adding one more dict entry and one more name in
MODE_ORDER, nothing in visual_engine.py has to change. `ModeController`
owns the current mode and smoothly interpolates every parameter toward the
target mode over `transition_speed` ms whenever the mode changes, so
switching never pops or flickers.

This module also owns the frosted-glass background treatment itself
(`apply_background_treatment`), since its strength/opacity/brightness/
contrast are exactly the per-mode parameters below - keeping the "how do we
turn camera + mode params into a soft background" logic next to the params
that drive it.
"""

from __future__ import annotations

import cv2
import numpy as np

from .gesture_detector import StableValue
from .utils import clamp

# --- Mode configuration -------------------------------------------------
# Every value here is safe to tune by hand - nothing downstream hard-codes
# these numbers, they only ever flow through ModeController.
#
#   blur_strength     0..1   how big the Gaussian kernel is (0 = no blur)
#   blur_alpha        0..1   how much of the blurred/dimmed layer is blended
#                             back over the sharp camera frame (0 = fully
#                             sharp camera, 1 = fully replaced by the
#                             blurred layer - this is what keeps "frosted"
#                             from ever becoming an opaque grey screen)
#   brightness        int    additive brightness offset applied to the
#                             blurred layer only (negative = dimmer)
#   contrast          float  multiplicative contrast applied to the
#                             blurred layer only (1.0 = unchanged)
#   vfx_scale         float  multiplier on jellyfish/cube size
#   particle_intensity float multiplier on particle emission rate
#   glow_intensity    float  multiplier on the additive glow blend strength
#   transition_speed  ms     how long switching *into* this mode takes
VISUAL_MODES: dict[str, dict[str, float]] = {
    "normal": {
        "blur_strength": 0.0,
        "blur_alpha": 0.0,
        "brightness": 0,
        "contrast": 1.0,
        "vfx_scale": 1.0,
        "particle_intensity": 1.0,
        "glow_intensity": 1.0,
        "transition_speed": 300,
    },
    "frosted_blur": {
        "blur_strength": 0.35,
        "blur_alpha": 0.55,   # camera stays clearly visible through the frost
        "brightness": -18,
        "contrast": 0.92,
        "vfx_scale": 1.0,
        "particle_intensity": 1.0,
        "glow_intensity": 1.1,
        "transition_speed": 400,
    },
    "vfx_focus": {
        "blur_strength": 0.5,
        "blur_alpha": 0.75,
        "brightness": -30,
        "contrast": 0.85,
        "vfx_scale": 1.15,
        "particle_intensity": 1.3,
        "glow_intensity": 1.25,
        "transition_speed": 400,
    },
    "glow_dream": {
        "blur_strength": 0.4,
        "blur_alpha": 0.6,
        "brightness": -10,
        "contrast": 0.95,
        "vfx_scale": 1.1,
        "particle_intensity": 1.4,
        "glow_intensity": 1.6,
        "transition_speed": 450,
    },
}

MODE_ORDER: list[str] = ["normal", "frosted_blur", "vfx_focus", "glow_dream"]


def apply_background_treatment(frame: np.ndarray, params: dict[str, float]) -> np.ndarray:
    """Camera -> Gaussian blur -> brightness/contrast -> translucent blend.

    Blurring is done on a downsampled copy (1/4 resolution) and scaled back
    up: this is both much cheaper than a full-resolution blur and adds an
    extra soft, glassy quality on top of the Gaussian blur itself, which is
    exactly the "frosted glass" look rather than a uniformly blurred photo.

    When every parameter is at its NORMAL-mode default this is a no-op
    (returns `frame` untouched) so the common case costs nothing.
    """
    blur_strength = params["blur_strength"]
    blur_alpha = params["blur_alpha"]
    brightness = params["brightness"]
    contrast = params["contrast"]

    if blur_strength < 0.01 and blur_alpha < 0.01 and abs(brightness) < 1 and abs(contrast - 1) < 0.01:
        return frame

    h, w = frame.shape[:2]
    small_w, small_h = max(16, w // 4), max(16, h // 4)
    small = cv2.resize(frame, (small_w, small_h), interpolation=cv2.INTER_AREA)

    kernel = int(clamp(blur_strength, 0.0, 1.0) * 24) | 1  # odd kernel size >= 1
    blurred_small = cv2.GaussianBlur(small, (kernel, kernel), 0) if kernel > 1 else small
    blurred = cv2.resize(blurred_small, (w, h), interpolation=cv2.INTER_LINEAR)

    if abs(brightness) >= 1 or abs(contrast - 1) >= 0.01:
        blurred = cv2.convertScaleAbs(blurred, alpha=contrast, beta=brightness)

    alpha = clamp(blur_alpha, 0.0, 1.0)
    return cv2.addWeighted(blurred, alpha, frame, 1 - alpha, 0)


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


class ModeController:
    """Owns the current visual mode and interpolates its parameters smoothly
    across a mode change instead of popping instantly."""

    def __init__(self, start_mode: str = "normal") -> None:
        if start_mode not in VISUAL_MODES:
            start_mode = MODE_ORDER[0]
        self._index = MODE_ORDER.index(start_mode)
        self._target_index = self._index
        self._current: dict[str, float] = dict(VISUAL_MODES[start_mode])
        self._transition_from: dict[str, float] = dict(self._current)
        self._transition_start_ms = 0.0

    @property
    def mode_name(self) -> str:
        return MODE_ORDER[self._target_index]

    def _goto(self, index: int, now_ms: float) -> None:
        index %= len(MODE_ORDER)
        if index == self._target_index:
            return
        self._transition_from = dict(self._current)
        self._target_index = index
        self._transition_start_ms = now_ms

    def next_mode(self, now_ms: float) -> None:
        self._goto(self._target_index + 1, now_ms)

    def previous_mode(self, now_ms: float) -> None:
        self._goto(self._target_index - 1, now_ms)

    def set_mode(self, name: str, now_ms: float) -> bool:
        """Jump directly to a named mode. Returns False (no-op, no crash) if
        `name` isn't a real mode - invalid input never raises."""
        if name not in VISUAL_MODES:
            return False
        self._goto(MODE_ORDER.index(name), now_ms)
        return True

    def update(self, now_ms: float) -> dict[str, float]:
        """Advance the transition and return the current, possibly still
        interpolating, parameter dict. Safe to call every frame."""
        target = VISUAL_MODES[MODE_ORDER[self._target_index]]
        duration = max(1.0, target.get("transition_speed", 350))
        t = clamp((now_ms - self._transition_start_ms) / duration, 0.0, 1.0)
        for key in self._current:
            from_value = self._transition_from.get(key, target[key])
            self._current[key] = _lerp(from_value, target[key], t)
        if t >= 1.0:
            self._index = self._target_index
        return self._current


class GestureModeSwitcher:
    """Debounced, edge-triggered gesture-to-mode-switch abstraction.

    A raw boolean "is the switch gesture currently being held" signal is
    smoothed with the same StableValue hold-time primitive used everywhere
    else (500-700ms typical), and only fires *once* per hold - the gesture
    must be released (go back to not-held) before it can fire again. This
    is what prevents one long hold from cycling through several modes.
    """

    def __init__(self, hold_ms: float = 600) -> None:
        self._stable = StableValue[bool](hold_ms, False)
        self._armed = True

    def update(self, raw_trigger: bool, now_ms: float) -> bool:
        stable = self._stable.update(raw_trigger, now_ms)
        fired = False
        if stable and self._armed:
            fired = True
            self._armed = False
        elif not stable:
            self._armed = True  # gesture released - re-arm for next trigger
        return fired
