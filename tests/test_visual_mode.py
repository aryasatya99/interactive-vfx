"""Tests for the visual mode system: cycling, transitions, invalid-mode
handling, blur configuration, gesture-switch debounce, and hand-presence
auto activation with its grace period. All pure/synthetic - no camera,
model file, or window required.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.interaction import PresenceFader  # noqa: E402
from src.visual_mode import (  # noqa: E402
    MODE_ORDER,
    VISUAL_MODES,
    GestureModeSwitcher,
    ModeController,
    apply_background_treatment,
)

# --- Mode cycling -------------------------------------------------------

def test_mode_order_matches_spec() -> None:
    assert MODE_ORDER == ["normal", "frosted_blur", "vfx_focus", "glow_dream"]


def test_next_mode_cycles_forward_and_wraps() -> None:
    mc = ModeController("normal")
    assert mc.mode_name == "normal"
    mc.next_mode(now_ms=0)
    assert mc.mode_name == "frosted_blur"
    mc.next_mode(now_ms=0)
    assert mc.mode_name == "vfx_focus"
    mc.next_mode(now_ms=0)
    assert mc.mode_name == "glow_dream"
    mc.next_mode(now_ms=0)  # wraps back to normal
    assert mc.mode_name == "normal"


def test_previous_mode_cycles_backward_and_wraps() -> None:
    mc = ModeController("normal")
    mc.previous_mode(now_ms=0)  # wraps to the last mode
    assert mc.mode_name == "glow_dream"
    mc.previous_mode(now_ms=0)
    assert mc.mode_name == "vfx_focus"


def test_invalid_mode_handling_does_not_crash() -> None:
    mc = ModeController("normal")
    ok = mc.set_mode("does_not_exist", now_ms=0)
    assert ok is False
    assert mc.mode_name == "normal"  # unchanged, no crash, no partial state


def test_invalid_start_mode_falls_back_to_first() -> None:
    mc = ModeController("nonsense")
    assert mc.mode_name == MODE_ORDER[0]


# --- Transitions ----------------------------------------------------------

def test_transition_interpolates_over_time_not_instantly() -> None:
    mc = ModeController("normal")
    mc.next_mode(now_ms=0)  # -> frosted_blur, which has nonzero blur_strength
    target_blur = VISUAL_MODES["frosted_blur"]["blur_strength"]

    just_started = mc.update(now_ms=1)
    assert 0.0 <= just_started["blur_strength"] < target_blur * 0.3

    halfway = mc.update(now_ms=200)  # frosted_blur transition_speed is 400ms
    assert 0.0 < halfway["blur_strength"] < target_blur

    finished = mc.update(now_ms=10_000)
    assert finished["blur_strength"] == target_blur


def test_transition_does_not_flicker_back_and_forth() -> None:
    """Once a transition is running, intermediate values must move
    monotonically toward the target, not oscillate."""
    mc = ModeController("normal")
    mc.next_mode(now_ms=0)
    values = [mc.update(now_ms=t)["blur_strength"] for t in range(0, 401, 40)]
    assert values == sorted(values)  # non-decreasing all the way through


def test_switching_mid_transition_starts_a_new_smooth_transition() -> None:
    mc = ModeController("normal")
    mc.next_mode(now_ms=0)          # heading to frosted_blur
    mc.update(now_ms=150)            # partway there
    mc.next_mode(now_ms=150)         # change our mind -> vfx_focus, from wherever we are now
    assert mc.mode_name == "vfx_focus"
    params = mc.update(now_ms=151)
    # Should not have snapped straight to vfx_focus's value in one frame.
    assert params["blur_strength"] != VISUAL_MODES["vfx_focus"]["blur_strength"]


# --- Blur configuration ----------------------------------------------------

def test_normal_mode_is_a_pure_no_op() -> None:
    frame = np.random.randint(0, 255, (60, 80, 3), dtype=np.uint8)
    out = apply_background_treatment(frame, VISUAL_MODES["normal"])
    assert np.array_equal(out, frame)


def test_frosted_blur_keeps_camera_visible_not_opaque() -> None:
    """The frosted look must blend, never fully replace, the sharp frame -
    blur_alpha < 1 guarantees some of the original frame always shows."""
    assert VISUAL_MODES["frosted_blur"]["blur_alpha"] < 1.0
    frame = np.random.randint(0, 255, (60, 80, 3), dtype=np.uint8)
    out = apply_background_treatment(frame, VISUAL_MODES["frosted_blur"])
    assert out.shape == frame.shape
    assert not np.array_equal(out, frame)  # something changed...
    # ...but it's a blend, not a solid fill: output still has meaningful variance.
    assert out.std() > 5


def test_all_modes_define_the_minimum_required_parameters() -> None:
    required = {"blur_strength", "blur_alpha", "brightness", "contrast",
                "vfx_scale", "particle_intensity", "glow_intensity", "transition_speed"}
    for name, params in VISUAL_MODES.items():
        assert required.issubset(params.keys()), f"{name} is missing parameters"


# --- Gesture-based mode switching ------------------------------------------

def test_gesture_switch_fires_once_after_hold() -> None:
    switcher = GestureModeSwitcher(hold_ms=600)
    assert switcher.update(True, now_ms=0) is False
    assert switcher.update(True, now_ms=300) is False
    assert switcher.update(True, now_ms=650) is True   # fires exactly once


def test_gesture_switch_requires_release_before_firing_again() -> None:
    switcher = GestureModeSwitcher(hold_ms=300)
    switcher.update(True, now_ms=0)
    assert switcher.update(True, now_ms=350) is True
    # Still holding the same gesture - must NOT fire again immediately,
    # no matter how long it's held.
    assert switcher.update(True, now_ms=400) is False
    assert switcher.update(True, now_ms=1000) is False

    # Release: this itself must debounce for hold_ms before counting as a
    # real release (a one-frame dropout shouldn't re-arm the switch either).
    assert switcher.update(False, now_ms=1050) is False
    assert switcher.update(False, now_ms=1200) is False   # only 150ms released so far
    assert switcher.update(False, now_ms=1360) is False    # now released >= 300ms - re-armed

    # Hold again - now it may fire exactly once more.
    switcher.update(True, now_ms=1360)
    assert switcher.update(True, now_ms=1600) is False    # only 240ms held
    assert switcher.update(True, now_ms=1700) is True      # >= 300ms held - fires


# --- Hand-presence auto activation / grace period --------------------------

def test_presence_rises_when_hand_detected() -> None:
    fader = PresenceFader(grace_ms=400, fade_ms=200)
    alpha = 0.0
    for _ in range(30):
        alpha = fader.update(hand_present=True, dt=0.03, now_ms=0)
    assert alpha > 0.9


def test_presence_holds_through_grace_period_then_fades_out() -> None:
    fader = PresenceFader(grace_ms=300, fade_ms=200)
    for t in range(0, 200, 20):
        fader.update(hand_present=True, dt=0.02, now_ms=t)
    assert fader.alpha > 0.5

    # Hand disappears - within the grace period, presence must stay high.
    alpha_during_grace = fader.update(hand_present=False, dt=0.02, now_ms=250)
    assert alpha_during_grace > 0.5

    # Well past the grace period, presence should have faded most of the way out.
    alpha = alpha_during_grace
    for t in range(600, 1600, 30):
        alpha = fader.update(hand_present=False, dt=0.03, now_ms=t)
    assert alpha < 0.15


def test_presence_manual_standby_is_independent() -> None:
    """PresenceFader only tracks hand visibility - it has no notion of the
    manual SPACE/gesture ACTIVE-STANDBY toggle, so it must keep reporting
    accurate presence even while that separate system is "off"."""
    fader = PresenceFader(grace_ms=300, fade_ms=200)
    alpha = 0.0
    for t in range(0, 300, 20):
        alpha = fader.update(hand_present=True, dt=0.02, now_ms=t)
    assert alpha > 0.5  # presence itself doesn't know or care about system_active
