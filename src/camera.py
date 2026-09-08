"""Mac webcam access with explicit, friendly error handling."""

from __future__ import annotations

import sys
from types import TracebackType

import cv2
import numpy as np


class CameraError(RuntimeError):
    """Raised when the camera cannot be opened or keeps failing to deliver frames."""


class Camera:
    """Thin wrapper around cv2.VideoCapture.

    On macOS we explicitly request the AVFoundation backend; the default
    auto-backend sometimes picks a stub and returns black frames.
    """

    def __init__(
        self,
        index: int = 0,
        width: int = 1280,
        height: int = 720,
    ) -> None:
        self.index = index
        backend = cv2.CAP_AVFOUNDATION if sys.platform == "darwin" else cv2.CAP_ANY
        self._cap = cv2.VideoCapture(index, backend)
        if not self._cap.isOpened():
            raise CameraError(
                "Camera permission required. Enable camera access for VS Code "
                "in System Settings > Privacy & Security > Camera.\n"
                f"(Tried camera index {index}. If permission is already "
                "granted, another app may be holding the camera, or try "
                "--camera 1 for a different device.)"
            )
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self._failures = 0

    @property
    def resolution(self) -> tuple[int, int]:
        return (
            int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )

    def read(self) -> np.ndarray | None:
        """Return the next BGR frame, already mirrored for a natural preview.

        NOTE ON MIRRORING: the frame returned here is flipped horizontally so
        the on-screen preview behaves like a mirror (raise your right hand,
        see it on the right side of the screen) - this matches every video
        call app and feels natural. Flipping is purely cosmetic for display.
        Left/right hand identity is NEVER derived from screen position; it
        comes from MediaPipe's handedness classifier (see hand_detector.py),
        which is run on this same flipped frame and correctly reports
        "Left"/"Right" from the subject's own point of view because MediaPipe
        expects a selfie-style (mirrored) image and accounts for it
        internally. Never mix a mirrored frame for display with an
        unmirrored one for detection - that mismatch is what would produce
        the wrong left/right answer, not the flip itself.
        """
        ok, frame = self._cap.read()
        if not ok or frame is None:
            self._failures += 1
            if self._failures > 30:
                raise CameraError("Kamera berhenti mengirim frame (30x gagal berturut).")
            return None
        self._failures = 0
        return cv2.flip(frame, 1)

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()

    def __enter__(self) -> "Camera":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.release()
