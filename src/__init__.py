"""Interactive VFX - real-time, hand-controlled visual effects.

Webcam -> hand tracking -> finger/gesture data -> real-time visual control
-> 3D interactive VFX (glowing jellyfish + wireframe cube + particles).
All processing is local; no network access is required at runtime.
"""

__all__ = [
    "camera",
    "hand_detector",
    "gesture_detector",
    "finger_tracker",
    "interaction",
    "jellyfish",
    "wireframe_cube",
    "particles",
    "visual_engine",
    "utils",
]
