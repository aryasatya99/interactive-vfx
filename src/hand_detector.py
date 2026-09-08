"""Hand landmark detection and per-hand finger-state logic.

Uses MediaPipe's Tasks API (`mediapipe.tasks.vision.HandLandmarker`), which is
the current, non-deprecated MediaPipe API — the old `mp.solutions.hands` API
has been removed in recent MediaPipe releases (1.0+).

The geometric helpers below (`thumb_open`, `index_open`, `classify_hand`) work
on plain landmark coordinates and have no MediaPipe dependency themselves, so
they can be unit-tested with synthetic points without a camera or a model
file.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# --- MediaPipe Hand landmark indices (21-point model) ----------------------
WRIST = 0
THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP = 1, 2, 3, 4
INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP = 5, 6, 7, 8
MIDDLE_MCP = 9
PINKY_MCP = 17

Point = tuple[float, float]


class ModelNotFoundError(RuntimeError):
    """Raised when the HandLandmarker .task model file is missing."""


def _dist(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _is_extended(landmarks: list[Point], tip_idx: int, mid_idx: int,
                  wrist_idx: int = WRIST, margin: float = 1.05) -> bool:
    """A finger is "open" when its tip sits farther from the wrist than its
    middle joint does. This ratio-based rule is orientation-independent (it
    does not assume the hand is upright or facing any particular way), which
    matters because a mirrored, tilted, or rotated hand is the normal case
    for a webcam, not the exception.

    `margin` adds a small buffer above 1.0 so a hand that is only barely
    curled does not flicker between open/closed on landmark jitter alone.
    """
    wrist = landmarks[wrist_idx]
    return _dist(wrist, landmarks[tip_idx]) > _dist(wrist, landmarks[mid_idx]) * margin


def thumb_open(landmarks: list[Point]) -> bool:
    """Thumb = OPEN when the tip is extended away from the wrist relative to
    the thumb's IP joint (its middle knuckle)."""
    return _is_extended(landmarks, THUMB_TIP, THUMB_IP)


def index_open(landmarks: list[Point]) -> bool:
    """Index = OPEN when the tip is extended away from the wrist relative to
    the index PIP joint (its middle knuckle)."""
    return _is_extended(landmarks, INDEX_TIP, INDEX_PIP)


@dataclass(frozen=True)
class HandReading:
    """One detected hand for one frame."""

    label: str                      # "Left" or "Right", from MediaPipe handedness
    confidence: float
    thumb_open: bool
    index_open: bool
    landmarks_px: list[tuple[int, int]]  # for drawing, in frame pixel coords

    @property
    def is_left(self) -> bool:
        return self.label == "Left"

    @property
    def is_right(self) -> bool:
        return self.label == "Right"


def build_reading(
    label: str,
    confidence: float,
    landmarks: list[Point],
    frame_w: int,
    frame_h: int,
) -> HandReading:
    """Turn raw normalized landmarks + handedness into a HandReading.

    IMPORTANT: `label` must come from MediaPipe's handedness classifier, never
    from which hand was detected first or which side of the screen it is on.
    MediaPipe's classifier is trained to report "Left"/"Right" from the
    subject's own point of view on a mirrored (selfie-style) image, which is
    exactly the kind of frame `Camera.read()` produces (see camera.py) — so
    the label it returns is used as-is, unmodified.
    """
    pixels = [(int(x * frame_w), int(y * frame_h)) for x, y in landmarks]
    return HandReading(
        label=label,
        confidence=confidence,
        thumb_open=thumb_open(landmarks),
        index_open=index_open(landmarks),
        landmarks_px=pixels,
    )


class HandDetector:
    """Wraps mediapipe.tasks.vision.HandLandmarker for realtime video frames."""

    def __init__(self, model_path: str | Path, max_hands: int = 2,
                 min_detection_confidence: float = 0.5) -> None:
        model_path = Path(model_path)
        if not model_path.exists():
            raise ModelNotFoundError(
                f"Model tidak ditemukan: {model_path}\n"
                "Jalankan: scripts/download_model.sh"
            )

        # Imported lazily so importing this module for the pure geometry
        # helpers (used heavily in tests) never requires mediapipe to have a
        # working camera/GPU backend available.
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        # CPU delegate is forced explicitly: MediaPipe's GPU delegate needs a
        # live Metal/window-server session and crashes hard (not a Python
        # exception - a native abort) when run headless, e.g. over SSH or in
        # some sandboxed shells. CPU is slightly slower but works everywhere
        # and is plenty fast for a 2-hand landmark model at webcam framerate.
        base_options = mp_python.BaseOptions(
            model_asset_path=str(model_path),
            delegate=mp_python.BaseOptions.Delegate.CPU,
        )
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_hands=max_hands,
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=min_detection_confidence,
            min_tracking_confidence=min_detection_confidence,
        )
        self._landmarker = vision.HandLandmarker.create_from_options(options)
        self._mp = mp

    def detect(self, frame_bgr: np.ndarray, timestamp_ms: int) -> list[HandReading]:
        """Run detection on one BGR frame. Returns 0-2 HandReadings."""
        # BGR (OpenCV) -> RGB (MediaPipe). ascontiguousarray is required: the
        # reversed-channel view has a negative stride, which MediaPipe's
        # pybind Image constructor rejects.
        rgb = np.ascontiguousarray(frame_bgr[:, :, ::-1])
        mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect_for_video(mp_image, timestamp_ms)

        h, w = frame_bgr.shape[:2]
        readings: list[HandReading] = []
        for hand_landmarks, handedness in zip(result.hand_landmarks, result.handedness):
            points = [(lm.x, lm.y) for lm in hand_landmarks]
            top = handedness[0]  # best-scoring handedness category for this hand
            readings.append(build_reading(top.category_name, top.score, points, w, h))
        return readings

    def close(self) -> None:
        self._landmarker.close()

    def __enter__(self) -> "HandDetector":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()
