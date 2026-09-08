"""Loads and serves the static X-ray asset shown when the gesture is valid.

This module never touches the camera or a network. `assets/xray.jpg` is
expected to be a plain image file you place yourself — a photo of an X-ray
film/print, or any image you want to gate behind the two-hand gesture. The
webcam does not, and cannot, produce this image itself.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


class XRayImage:
    """Holds the current X-ray image, reloadable at runtime with the R key."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._image: np.ndarray | None = None
        self.reload()

    def reload(self) -> bool:
        """(Re)load the image from disk. Returns True on success.

        Never raises: a missing or unreadable file just leaves `image` as
        None, and the caller (main.py) renders a NOT FOUND placeholder
        instead of crashing.
        """
        if not self.path.exists():
            self._image = None
            return False
        image = cv2.imread(str(self.path), cv2.IMREAD_COLOR)
        self._image = image  # cv2.imread returns None on decode failure too
        return image is not None

    @property
    def available(self) -> bool:
        return self._image is not None

    @property
    def image(self) -> np.ndarray | None:
        return self._image
