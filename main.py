"""AURA — AI Gesture Interface v1.

A futuristic gesture-control HUD: raise both hands with thumb + index open
on each, and AURA activates its visual effects. Everything — camera capture,
hand tracking, gesture logic, and rendering — runs locally; no frame ever
leaves this machine and no network access is required at runtime.

Pipeline
    Camera -> OpenCV -> MediaPipe HandLandmarker -> handedness (Left/Right)
    -> per-hand thumb/index open state -> two-hand gesture validation
    -> debounce -> HUD + particle rendering.

Controls
    SPACE  toggle SYSTEM: ACTIVE / STANDBY
    Q/ESC  quit
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from src import hud
from src.camera import Camera, CameraError
from src.gesture_detector import GestureDebouncer, evaluate
from src.hand_detector import HandDetector, HandReading, ModelNotFoundError
from src.particles import ParticleSystem
from src.utils import FPSCounter, put_text, text_width

ROOT = Path(__file__).parent
WINDOW = "AURA - AI Gesture Interface"
CAPTURE_DIR = ROOT / "captures"
DEFAULT_MODEL = ROOT / "models" / "hand_landmarker.task"

HAND_COLORS = {"Left": hud.CYAN, "Right": hud.GREEN}
ACCEPT_TOAST_FRAMES = 45
PARTICLES_PER_TIP_PER_FRAME = 2


def draw_title(frame: np.ndarray, w: int) -> None:
    title = "AURA"
    subtitle = "AI GESTURE INTERFACE"
    tx = (w - text_width(title, 1.1, 3)) // 2
    put_text(frame, title, (tx, 46), 1.1, hud.CYAN, 3)
    sx = (w - text_width(subtitle, 0.5, 1)) // 2
    put_text(frame, subtitle, (sx, 68), 0.5, hud.WHITE, 1)


def draw_system_badge(frame: np.ndarray, w: int, active: bool) -> None:
    label = "SYSTEM: ACTIVE" if active else "SYSTEM: STANDBY"
    color = hud.GREEN if active else hud.AMBER
    tw = text_width(label, 0.55, 2)
    x, y = w - tw - 34, 20
    hud.draw_panel(frame, x - 10, y - 20, tw + 20, 32, alpha=0.5, border_color=color)
    put_text(frame, label, (x, y + 2), 0.55, color, 2)


def draw_hand_panel(frame: np.ndarray, x: int, y: int, w: int, h: int,
                    name: str, status, color: tuple[int, int, int]) -> None:
    hud.draw_panel(frame, x, y, w, h, alpha=0.5, border_color=color)
    put_text(frame, name, (x + 12, y + 22), 0.5, hud.WHITE, 2)
    hud.status_row(frame, x + 12, y + 46, "HAND", status.detected)
    hud.status_row(frame, x + 12, y + 68, "THUMB", status.thumb_open)
    hud.status_row(frame, x + 12, y + 90, "INDEX", status.index_open)


def draw_bottom_bar(frame: np.ndarray, w: int, h: int, gesture_open: bool,
                    system_active: bool, fps: float, hand_count: int) -> None:
    bar_h = 40
    y = h - bar_h
    hud.draw_panel(frame, 0, y, w, bar_h, alpha=0.55, border_color=hud.CYAN_DIM)
    x = 18
    x = hud.status_row(frame, x, y + 26, "GESTURE:", gesture_open, "VALID", "WAITING")
    put_text(frame, "SYSTEM:", (x, y + 26), 0.5, hud.WHITE, 1)
    x += text_width("SYSTEM:", 0.5, 1) + 10
    put_text(frame, "ACTIVE" if system_active else "STANDBY", (x, y + 26), 0.5,
             hud.GREEN if system_active else hud.AMBER, 2)
    x += text_width("ACTIVE", 0.5, 2) + 30
    put_text(frame, f"HANDS: {hand_count}", (x, y + 26), 0.5, hud.WHITE, 1)
    x += text_width(f"HANDS: {hand_count}", 0.5, 1) + 30
    put_text(frame, f"FPS: {fps:4.1f}", (x, y + 26), 0.5, hud.WHITE, 1)

    hint = "SPACE = ACTIVE/STANDBY    Q = QUIT"
    put_text(frame, hint, (w - text_width(hint, 0.42, 1) - 16, y + 26), 0.42, hud.WHITE, 1)


def draw_accept_toast(frame: np.ndarray, w: int) -> None:
    msg = "GESTURE ACCEPTED"
    tx = (w - text_width(msg, 0.7, 2)) // 2
    put_text(frame, msg, (tx, 100), 0.7, hud.GREEN, 2)


def fingertip_positions(readings: list[HandReading]) -> list[tuple[int, int, tuple[int, int, int]]]:
    """(x, y, color) for each hand's thumb tip and index tip — particle sources."""
    points = []
    for r in readings:
        color = HAND_COLORS.get(r.label, hud.WHITE)
        for idx in hud.TIP_INDICES:
            x, y = r.landmarks_px[idx]
            points.append((x, y, color))
    return points


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="AURA - AI Gesture Interface (educational)")
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    p.add_argument("--hold-ms", type=int, default=400,
                   help="debounce hold time in ms before gesture state flips (300-500 typical)")
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

    debouncer = GestureDebouncer(hold_ms=args.hold_ms)
    particles = ParticleSystem()
    fps_counter = FPSCounter()

    system_active = True
    prev_effect_active = False
    toast_frames = 0
    scan_angle = 0.0

    cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
    print(f"[INFO] camera online @ {cam.resolution[0]}x{cam.resolution[1]}")
    print("[INFO] SPACE = toggle ACTIVE/STANDBY, Q = quit")

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
            timestamp_ms = int((now - start) * 1000)

            readings = detector.detect(frame, timestamp_ms)
            snap = evaluate(readings)
            gesture_open = debouncer.update(snap.valid, timestamp_ms)
            effect_active = system_active and gesture_open

            if effect_active and not prev_effect_active:
                toast_frames = ACCEPT_TOAST_FRAMES
            prev_effect_active = effect_active
            if toast_frames > 0:
                toast_frames -= 1

            fps = fps_counter.tick()
            h, w = frame.shape[:2]

            # --- particle emission (only when the main effect is active) ---
            particles.update(dt)
            if effect_active:
                for x, y, color in fingertip_positions(readings):
                    particles.emit(x, y, color, PARTICLES_PER_TIP_PER_FRAME)

            # --- glow layer: brackets, rings, hand skeletons -----------------
            glow = hud.new_glow_layer(frame)
            bottom_bar_h = 40
            hud.draw_corner_brackets(glow, 10, 10, w - 20, h - bottom_bar_h - 20, size=34,
                                     color=hud.GREEN if effect_active else hud.CYAN)
            center = (w // 2, h // 2)
            scan_angle = (scan_angle + dt * (140 if effect_active else 60)) % 360
            hud.draw_scanning_ring(glow, center, 100, scan_angle,
                                   hud.GREEN if effect_active else hud.CYAN,
                                   active=effect_active)
            for r in readings:
                color = HAND_COLORS.get(r.label, hud.WHITE)
                hud.draw_hand_skeleton(frame, glow, r.landmarks_px, r.label, color)

            frame = hud.blend_glow(frame, glow, intensity=0.9 if effect_active else 0.7)
            particles.render(frame)

            # --- crisp overlays: title, panels, status text ------------------
            draw_title(frame, w)
            draw_system_badge(frame, w, system_active)
            draw_hand_panel(frame, 16, h - 200, 230, 120, "LEFT HAND", snap.left, hud.CYAN)
            draw_hand_panel(frame, w - 246, h - 200, 230, 120, "RIGHT HAND", snap.right, hud.GREEN)
            draw_bottom_bar(frame, w, h, gesture_open, system_active, fps, len(readings))
            if toast_frames > 0:
                draw_accept_toast(frame, w)

            cv2.imshow(WINDOW, frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == 32:  # SPACE
                system_active = not system_active
                print(f"[INFO] SYSTEM -> {'ACTIVE' if system_active else 'STANDBY'}")

            if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                break

    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
