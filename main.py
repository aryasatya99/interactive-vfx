"""Interactive VFX - real-time, hand-controlled 3D visual effects.

Pipeline
    Webcam -> OpenCV -> MediaPipe HandLandmarker -> handedness (Left/Right)
    -> per-hand finger count + gesture (finger_tracker, debounced)
    -> GESTURE_CONFIG mapping -> position/scale/particle/animation controls
    -> visual mode (NORMAL/FROSTED_BLUR/VFX_FOCUS/GLOW_DREAM) background
       treatment -> interactive VFX: glowing jellyfish + wireframe cube +
       particles, faded in/out automatically by hand presence.

Render order: camera -> mode background treatment -> wireframe cube ->
jellyfish -> particles -> hand tracking indicators -> minimal status text.

All processing is local. No frame ever leaves this machine, no network
access is required while running, and no video is saved to disk.

Controls
    SPACE      toggle ACTIVE / STANDBY
    M          next visual mode
    SHIFT + M  previous visual mode
    ESC/Q      quit
Gesture controls (see GESTURE_CONFIG in src/interaction.py)
    OPEN_HAND (5 fingers) held ~600ms   -> activate
    CLOSED_HAND (fist) held ~600ms      -> pause
    Both hands PINCHing, held ~650ms    -> next visual mode
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from src.camera import Camera, CameraError
from src.finger_tracker import FingerTracker, TrackingSnapshot
from src.hand_detector import HandDetector, ModelNotFoundError
from src.interaction import ActivationController, InteractionController, PresenceFader
from src.utils import FPSCounter, put_text, text_width
from src.visual_engine import CYAN, VisualEngine
from src.visual_mode import GestureModeSwitcher, ModeController

ROOT = Path(__file__).parent
WINDOW = "Interactive VFX"
DEFAULT_MODEL = ROOT / "models" / "hand_landmarker.task"

_HAND_INDICATOR_COLORS = {"Left": (255, 210, 40), "Right": (140, 255, 170)}


def draw_hand_indicators(frame: np.ndarray, snapshot: TrackingSnapshot) -> None:
    """Small glowing markers at each detected hand's thumb/index tips - just
    enough to confirm what's being tracked, not a full skeleton overlay."""
    for hand in (snapshot.left, snapshot.right):
        if not hand.detected or hand.raw.reading is None:
            continue
        reading = hand.raw.reading
        color = _HAND_INDICATOR_COLORS.get(reading.label, (230, 230, 230))
        for point in (reading.thumb_tip, reading.index_tip):
            cv2.circle(frame, point, 5, color, 1, cv2.LINE_AA)
            cv2.circle(frame, point, 2, color, -1, cv2.LINE_AA)


def draw_minimal_indicators(frame: np.ndarray, active: bool, mode_name: str,
                            left_fingers: int, right_fingers: int, fps: float) -> None:
    """Small, unobtrusive status text - the visual is the experience, this
    is just enough to confirm the system is tracking, not a dashboard."""
    color = (120, 255, 170) if active else (60, 190, 255)
    put_text(frame, "INTERACTIVE VFX", (16, 26), 0.55, (230, 230, 230), 1)
    put_text(frame, "ACTIVE" if active else "STANDBY", (16, 48), 0.5, color, 2)
    put_text(frame, f"MODE: {mode_name.replace('_', ' ').upper()}", (16, 68), 0.45, CYAN, 1)

    h = frame.shape[0]
    put_text(frame, f"LEFT {left_fingers}", (16, h - 46), 0.48, (230, 230, 230), 1)
    put_text(frame, f"RIGHT {right_fingers}", (16, h - 24), 0.48, (230, 230, 230), 1)

    fps_text = f"FPS {fps:.0f}"
    w = frame.shape[1]
    put_text(frame, fps_text, (w - text_width(fps_text, 0.48, 1) - 16, h - 24),
             0.48, (230, 230, 230), 1)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Interactive VFX (educational)")
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    p.add_argument("--hold-ms", type=float, default=400,
                   help="finger-count/gesture debounce hold time in ms (300-500 typical)")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    try:
        detector = HandDetector(args.model)
    except ModelNotFoundError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1

    try:
        cam = Camera(args.camera, args.width, args.height)
    except CameraError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        detector.close()
        return 1

    w, h = cam.resolution
    finger_tracker = FingerTracker(hold_ms=args.hold_ms)
    interaction = InteractionController(w, h)
    activation = ActivationController(hold_ms=600)
    presence_fader = PresenceFader(grace_ms=400, fade_ms=400)
    mode_controller = ModeController(start_mode="normal")
    mode_switch_gesture = GestureModeSwitcher(hold_ms=650)
    visuals = VisualEngine()
    fps_counter = FPSCounter()

    system_active = True

    cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
    print(f"[INFO] camera online @ {w}x{h}")
    print("[INFO] SPACE = ACTIVE/STANDBY, M = next mode, SHIFT+M = previous mode, ESC/Q = quit")

    start = time.perf_counter()
    prev_time = start

    with cam, detector:
        while True:
            try:
                frame = cam.read()
            except CameraError as exc:
                print(f"[ERROR] {exc}", file=sys.stderr)
                return 1
            if frame is None:
                continue

            now = time.perf_counter()
            dt = max(0.0, now - prev_time)
            prev_time = now
            timestamp_ms = (now - start) * 1000

            readings = detector.detect(frame, int(timestamp_ms))
            snapshot = finger_tracker.update(readings, timestamp_ms)
            state = interaction.update(snapshot)

            request = activation.update(snapshot, timestamp_ms)
            if request is not None:
                system_active = request

            # Both hands pinching, held ~650ms and then released, advances
            # the visual mode - a gesture chosen because PINCH isn't already
            # claimed by GESTURE_CONFIG, so it can't collide with position/
            # scale/particle/animation/activate/pause.
            pinch_trigger = (snapshot.left.detected and snapshot.right.detected
                             and snapshot.left.raw.pinch and snapshot.right.raw.pinch)
            if mode_switch_gesture.update(pinch_trigger, timestamp_ms):
                mode_controller.next_mode(timestamp_ms)
                print(f"[INFO] MODE -> {mode_controller.mode_name} (gesture)")

            presence = presence_fader.update(state.hand_count > 0, dt, timestamp_ms)
            mode_params = mode_controller.update(timestamp_ms)

            visuals.update(state, dt, system_active, mode_params, presence)
            frame = visuals.render(frame, state, system_active, mode_params, presence)

            draw_hand_indicators(frame, snapshot)

            fps = fps_counter.tick()
            draw_minimal_indicators(frame, system_active, mode_controller.mode_name,
                                    snapshot.left.finger_count, snapshot.right.finger_count, fps)

            cv2.imshow(WINDOW, frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):  # ESC or Q
                break
            if key == 32:  # SPACE
                system_active = not system_active
                print(f"[INFO] SYSTEM -> {'ACTIVE' if system_active else 'STANDBY'}")
            if key == ord("m"):
                mode_controller.next_mode(timestamp_ms)
                print(f"[INFO] MODE -> {mode_controller.mode_name}")
            if key == ord("M"):  # SHIFT+M
                mode_controller.previous_mode(timestamp_ms)
                print(f"[INFO] MODE -> {mode_controller.mode_name}")

            if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                break

    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
