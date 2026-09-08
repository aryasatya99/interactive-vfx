"""Hand landmark detection and per-hand raw geometry extraction.

Uses MediaPipe's Tasks API (`mediapipe.tasks.vision.HandLandmarker`), which is
the current, non-deprecated MediaPipe API — the old `mp.solutions.hands` API
has been removed in recent MediaPipe releases (1.0+).

This module only extracts raw, per-frame facts from landmarks (which fingers
are extended, where the fingertips are, how big the hand is on screen). It
intentionally does not decide what a "gesture" means, and it does not smooth
anything over time — that temporal/semantic layer lives in
`gesture_detector.py` and `finger_tracker.py`. The geometric helpers below
have no MediaPipe dependency themselves, so they can be unit-tested with
synthetic points without a camera or a model file.
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
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP = 9, 10, 11, 12
RING_MCP, RING_PIP, RING_DIP, RING_TIP = 13, 14, 15, 16
PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP = 17, 18, 19, 20

# (tip, middle-joint) pairs used by the extension test, one per finger.
FINGER_JOINTS: dict[str, tuple[int, int]] = {
    "thumb": (THUMB_TIP, THUMB_IP),
    "index": (INDEX_TIP, INDEX_PIP),
    "middle": (MIDDLE_TIP, MIDDLE_PIP),
    "ring": (RING_TIP, RING_PIP),
    "pinky": (PINKY_TIP, PINKY_PIP),
}
PALM_LANDMARKS = (WRIST, INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP)

Point = tuple[float, float]
PixelPoint = tuple[int, int]


class ModelNotFoundError(RuntimeError):
    """Raised when the HandLandmarker .task model file is missing."""


def _dist(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def is_finger_extended(landmarks: list[Point], tip_idx: int, mid_idx: int,
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
    return is_finger_extended(landmarks, THUMB_TIP, THUMB_IP)


def index_open(landmarks: list[Point]) -> bool:
    return is_finger_extended(landmarks, INDEX_TIP, INDEX_PIP)


def finger_states(landmarks: list[Point]) -> dict[str, bool]:
    """Extended/curled state for all five fingers, keyed by name."""
    return {name: is_finger_extended(landmarks, tip, mid)
            for name, (tip, mid) in FINGER_JOINTS.items()}


def count_open_fingers(landmarks: list[Point]) -> int:
    """Raw (unsmoothed) count of extended fingers, 0-5, for a single frame."""
    return sum(finger_states(landmarks).values())


def palm_center(landmarks: list[Point]) -> Point:
    """Average of wrist + the four finger MCP joints - a stable hand anchor
    that moves far less than any single fingertip."""
    xs = [landmarks[i][0] for i in PALM_LANDMARKS]
    ys = [landmarks[i][1] for i in PALM_LANDMARKS]
    return (sum(xs) / len(xs), sum(ys) / len(ys))


def bounding_box(landmarks: list[Point]) -> tuple[float, float, float, float]:
    """(min_x, min_y, max_x, max_y) in the same coordinate space as landmarks."""
    xs = [p[0] for p in landmarks]
    ys = [p[1] for p in landmarks]
    return (min(xs), min(ys), max(xs), max(ys))


def relative_hand_size(landmarks: list[Point]) -> float:
    """Bounding-box diagonal in normalized (0..1) coordinates.

    This is a *relative* size only - how much of the frame the hand spans -
    used as a rough proxy for "closer to the camera = bigger on screen". It
    is not a physical distance measurement of any kind.
    """
    x0, y0, x1, y1 = bounding_box(landmarks)
    return math.hypot(x1 - x0, y1 - y0)


@dataclass(frozen=True)
class HandReading:
    """One detected hand's raw geometry for one frame."""

    label: str                      # "Left" or "Right", from MediaPipe handedness
    confidence: float
    landmarks_px: list[PixelPoint]        # all 21 points, frame pixel coords
    fingers_open: dict[str, bool]         # {"thumb": bool, "index": bool, ...}
    finger_count: int                     # 0-5, raw/unsmoothed for this frame
    thumb_tip: PixelPoint
    index_tip: PixelPoint
    middle_tip: PixelPoint
    ring_tip: PixelPoint
    pinky_tip: PixelPoint
    palm_center_px: PixelPoint
    bbox_px: tuple[int, int, int, int]    # x0, y0, x1, y1
    relative_size: float                  # 0..~0.5, bigger = hand fills more of frame

    @property
    def is_left(self) -> bool:
        return self.label == "Left"

    @property
    def is_right(self) -> bool:
        return self.label == "Right"

    @property
    def thumb_open(self) -> bool:
        return self.fingers_open["thumb"]

    @property
    def index_open(self) -> bool:
        return self.fingers_open["index"]

    def thumb_index_distance_px(self) -> float:
        return math.hypot(self.thumb_tip[0] - self.index_tip[0],
                          self.thumb_tip[1] - self.index_tip[1])


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
    def to_px(p: Point) -> PixelPoint:
        return (int(p[0] * frame_w), int(p[1] * frame_h))

    pixels = [to_px(p) for p in landmarks]
    x0, y0, x1, y1 = bounding_box(landmarks)
    fingers = finger_states(landmarks)
    return HandReading(
        label=label,
        confidence=confidence,
        landmarks_px=pixels,
        fingers_open=fingers,
        finger_count=sum(fingers.values()),
        thumb_tip=to_px(landmarks[THUMB_TIP]),
        index_tip=to_px(landmarks[INDEX_TIP]),
        middle_tip=to_px(landmarks[MIDDLE_TIP]),
        ring_tip=to_px(landmarks[RING_TIP]),
        pinky_tip=to_px(landmarks[PINKY_TIP]),
        palm_center_px=to_px(palm_center(landmarks)),
        bbox_px=(int(x0 * frame_w), int(y0 * frame_h), int(x1 * frame_w), int(y1 * frame_h)),
        relative_size=relative_hand_size(landmarks),
    )


class HandDetector:
    """Wraps mediapipe.tasks.vision.HandLandmarker for realtime video frames."""

    def __init__(self, model_path: str | Path, max_hands: int = 2,
                 min_detection_confidence: float = 0.5) -> None:
        model_path = Path(model_path)
        if not model_path.exists():
            raise ModelNotFoundError(
                f"Model not found: {model_path}\n"
                "Run: scripts/download_model.sh"
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
