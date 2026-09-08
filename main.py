"""Gesture X-Ray - realtime two-hand gesture gate for a static X-ray image.

DISCLAIMER
    The webcam captures visible light only; it never produces an X-ray image
    itself. `assets/xray.jpg` is a static asset you supply (a photo of an
    X-ray print/film, or any image you choose). This program is an
    educational computer-vision + gesture-control demo. It is NOT a medical
    device and makes NO diagnostic claims of any kind.

Pipeline
    Camera -> OpenCV -> MediaPipe HandLandmarker -> handedness (Left/Right)
    -> per-hand thumb/index open state -> two-hand gesture validation
    -> debounce -> X-ray display gate.

Controls
    SPACE  capture the current camera frame into captures/
    R      reload assets/xray.jpg from disk
    Q/ESC  quit
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from src.camera import Camera, CameraError
from src.gesture_detector import GestureDebouncer, GestureSnapshot, HandStatus, evaluate
from src.hand_detector import HandDetector, HandReading, ModelNotFoundError
from src.utils import (
    COLOR_ACCENT,
    COLOR_BAD,
    COLOR_BG,
    COLOR_MUTED,
    COLOR_OK,
    COLOR_PANEL,
    COLOR_TEXT,
    COLOR_WARN,
    FPSCounter,
    fit_into,
    put_text,
    save_capture,
    text_width,
)
from src.xray_display import XRayImage

ROOT = Path(__file__).parent
WINDOW = "Gesture X-Ray"
CAPTURE_DIR = ROOT / "captures"
DEFAULT_MODEL = ROOT / "models" / "hand_landmarker.task"
DEFAULT_XRAY = ROOT / "assets" / "xray.jpg"

# --- UI geometry ------------------------------------------------------------
W = 1280
MARGIN = 14
GAP = 14
HEADER_H = 52
PANEL_H = 468
HANDS_H = 64
STATUS_H = 96
FOOTER_H = 34
PANEL_W = (W - 2 * MARGIN - GAP) // 2
H = (HEADER_H + GAP + PANEL_H + GAP + HANDS_H + GAP
     + STATUS_H + GAP + FOOTER_H + MARGIN)

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),          # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),          # index
    (5, 9), (9, 10), (10, 11), (11, 12),     # middle
    (9, 13), (13, 14), (14, 15), (15, 16),   # ring
    (13, 17), (17, 18), (18, 19), (19, 20),  # pinky
    (0, 17),                                 # palm base
]


def draw_landmarks(frame: np.ndarray, readings: list[HandReading]) -> np.ndarray:
    view = frame.copy()
    for r in readings:
        color = COLOR_ACCENT if r.is_left else COLOR_OK
        for a, b in HAND_CONNECTIONS:
            cv2.line(view, r.landmarks_px[a], r.landmarks_px[b], color, 2, cv2.LINE_AA)
        for x, y in r.landmarks_px:
            cv2.circle(view, (x, y), 3, COLOR_TEXT, -1, cv2.LINE_AA)
        wx, wy = r.landmarks_px[0]
        put_text(view, r.label.upper(), (wx - 20, wy + 24), 0.5, color, 2)
    return view


def xray_panel(xray: XRayImage, gesture_open: bool, panel_w: int, panel_h: int) -> np.ndarray:
    if not xray.available:
        canvas = fit_into(None, panel_w, panel_h)
        msg = "X-RAY IMAGE NOT FOUND"
        put_text(canvas, msg, ((panel_w - text_width(msg, 0.6, 2)) // 2, panel_h // 2),
                 0.6, COLOR_WARN, 2)
        hint = "letakkan file di assets/xray.jpg lalu tekan R"
        put_text(canvas, hint, ((panel_w - text_width(hint, 0.45, 1)) // 2, panel_h // 2 + 28),
                 0.45, COLOR_MUTED, 1)
        return canvas
    if not gesture_open:
        canvas = fit_into(None, panel_w, panel_h)
        msg = "LOCKED"
        put_text(canvas, msg, ((panel_w - text_width(msg, 1.0, 2)) // 2, panel_h // 2),
                 1.0, COLOR_MUTED, 2)
        return canvas
    return fit_into(xray.image, panel_w, panel_h)


def compose_ui(
    camera_view: np.ndarray,
    xray_view: np.ndarray,
    snap: GestureSnapshot,
    xray_open: bool,
    xray_available: bool,
    fps: float,
    toast: str | None,
) -> np.ndarray:
    ui = np.full((H, W, 3), COLOR_BG, dtype=np.uint8)

    # Header
    cv2.rectangle(ui, (0, 0), (W, HEADER_H), COLOR_PANEL, -1)
    title = "GESTURE X-RAY"
    put_text(ui, title, ((W - text_width(title, 0.85, 2)) // 2, 34), 0.85, COLOR_TEXT, 2)
    put_text(ui, "educational demo - not a medical device", (MARGIN, 34), 0.42, COLOR_MUTED, 1)

    # Camera / X-ray panels
    top = HEADER_H + GAP
    lx, rx = MARGIN, MARGIN + PANEL_W + GAP
    ui[top:top + PANEL_H, lx:lx + PANEL_W] = fit_into(camera_view, PANEL_W, PANEL_H)
    ui[top:top + PANEL_H, rx:rx + PANEL_W] = fit_into(xray_view, PANEL_W, PANEL_H)
    for x, label in ((lx, "CAMERA"), (rx, "X-RAY")):
        cv2.rectangle(ui, (x, top), (x + PANEL_W, top + PANEL_H), (70, 66, 62), 1)
        cv2.rectangle(ui, (x, top), (x + 140, top + 24), COLOR_PANEL, -1)
        put_text(ui, label, (x + 10, top + 17), 0.48, COLOR_ACCENT, 1)

    # Hand status row
    hy = top + PANEL_H + GAP
    cv2.rectangle(ui, (MARGIN, hy), (W - MARGIN, hy + HANDS_H), COLOR_PANEL, -1)

    def draw_hand_row(y: int, name: str, status: HandStatus) -> None:
        x = MARGIN + 18
        put_text(ui, name, (x, y), 0.55, COLOR_TEXT, 2)
        x += text_width(name, 0.55, 2) + 14
        mark, color = ("✓", COLOR_OK) if status.detected else ("✗", COLOR_BAD)
        put_text(ui, mark, (x, y), 0.55, color, 2)
        x += 40
        for label, ok in (("Thumb", status.thumb_open), ("Index", status.index_open)):
            mark, color = ("✓", COLOR_OK) if ok else ("✗", COLOR_BAD)
            put_text(ui, f"{label}", (x, y), 0.5, COLOR_MUTED, 1)
            x += text_width(label, 0.5, 1) + 8
            put_text(ui, mark, (x, y), 0.5, color, 2)
            x += 46

    draw_hand_row(hy + 26, "LEFT HAND", snap.left)
    draw_hand_row(hy + 52, "RIGHT HAND", snap.right)

    # Status block
    sy = hy + HANDS_H + GAP
    cv2.rectangle(ui, (MARGIN, sy), (W - MARGIN, sy + STATUS_H), COLOR_PANEL, -1)
    gesture_color = COLOR_OK if snap.valid else COLOR_WARN
    xray_label, xray_color = (
        ("OPEN", COLOR_OK) if xray_open else
        ("LOCKED", COLOR_WARN) if xray_available else
        ("UNAVAILABLE", COLOR_BAD)
    )
    lines = [
        ("GESTURE:", "VALID" if snap.valid else "INVALID", gesture_color),
        ("X-RAY:", xray_label, xray_color),
        ("CAMERA:", "OK", COLOR_OK),
        ("FPS:", f"{fps:4.1f}", COLOR_TEXT),
    ]
    for i, (name, value, color) in enumerate(lines):
        y = sy + 22 + i * 20
        put_text(ui, name, (MARGIN + 18, y), 0.5, COLOR_MUTED, 1)
        put_text(ui, value, (MARGIN + 18 + text_width(name, 0.5, 1) + 10, y), 0.5, color, 2)

    # Footer
    fy = sy + STATUS_H + GAP
    hint = "SPACE = CAPTURE    R = RELOAD X-RAY    Q = QUIT"
    put_text(ui, hint, (MARGIN + 18, fy + 22), 0.48, COLOR_MUTED, 1)
    if toast:
        put_text(ui, toast, (W - MARGIN - 18 - text_width(toast, 0.48, 1), fy + 22),
                 0.48, COLOR_OK, 1)
    return ui


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Gesture X-Ray (educational)")
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    p.add_argument("--xray", type=Path, default=DEFAULT_XRAY)
    p.add_argument("--hold-ms", type=int, default=400,
                   help="debounce hold time in ms before gesture state flips")
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

    xray = XRayImage(args.xray)
    debouncer = GestureDebouncer(hold_ms=args.hold_ms)
    fps_counter = FPSCounter()
    toast: str | None = None
    toast_frames = 0

    cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
    print(f"[INFO] kamera aktif @ {cam.resolution[0]}x{cam.resolution[1]}")
    print("[INFO] SPACE = capture, R = reload x-ray, Q = quit")
    start = time.perf_counter()

    with cam, detector:
        while True:
            try:
                frame = cam.read()
            except CameraError as exc:
                print(f"[ERROR] {exc}", file=sys.stderr)
                return 1
            if frame is None:
                continue

            timestamp_ms = int((time.perf_counter() - start) * 1000)
            readings = detector.detect(frame, timestamp_ms)
            snap = evaluate(readings)
            xray_open = debouncer.update(snap.valid, timestamp_ms)

            fps = fps_counter.tick()
            if toast_frames > 0:
                toast_frames -= 1
            else:
                toast = None

            camera_view = draw_landmarks(frame, readings)
            xr_view = xray_panel(xray, xray_open, PANEL_W, PANEL_H)
            ui = compose_ui(camera_view, xr_view, snap, xray_open, xray.available,
                            fps, toast)
            cv2.imshow(WINDOW, ui)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("r"):
                ok = xray.reload()
                toast = "X-RAY RELOADED" if ok else "X-RAY IMAGE NOT FOUND"
                toast_frames = 60
            if key == 32:  # SPACE
                path = save_capture(frame, CAPTURE_DIR)
                toast = f"SAVED {path.name}"
                toast_frames = 60
                print(f"[SAVE] {path}")

            if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                break

    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
