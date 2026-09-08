# Interactive VFX

Interactive real-time visual system controlled by hand gestures. Your Mac's
webcam feed becomes the canvas for a glowing, procedural jellyfish inside a
rotating 3D wireframe cube, both driven live by your hands — no mouse, no
keyboard, no controller.

```text
WEBCAM -> HAND TRACKING -> FINGER/GESTURE DATA -> REAL-TIME VISUAL CONTROL -> 3D INTERACTIVE VFX
```

This is a hand-controlled visual instrument, not a dashboard: the camera is
the background, the VFX is the experience, and on-screen text is kept to a
handful of small, corner-anchored numbers.

## Overview

MediaPipe tracks up to two hands and reports true left/right handedness for
each. For every hand, this app counts how many fingers are extended (0-5),
classifies that into a named gesture, and smooths it over ~300-500ms so
natural landmark jitter doesn't make the gesture flicker. A configurable
mapping (`GESTURE_CONFIG`) turns each hand's stable gesture into a control
channel — position, scale, particle intensity, or animation intensity —
which an exponential-smoothing interaction layer turns into calm, jitter-free
motion for the jellyfish, the cube, and a capped particle system. Two hands
together unlock a second mode: the distance and midpoint between your two
index fingers directly control scale and position.

## Features

- Realtime Mac camera capture, mirrored like a normal camera app
- MediaPipe HandLandmarker tracking, up to 2 hands, true LEFT/RIGHT from
  MediaPipe's own handedness classifier (never assumed from detection order)
- Stable 0-5 finger counting per hand, debounced (~300-500ms) against jitter
- A configurable gesture system (`GESTURE_CONFIG`) — CLOSED_HAND, ONE..FOUR
  fingers, OPEN_HAND, and PINCH — mapped to visual control channels
- Index-finger position control with exponential smoothing (no jitter)
- Thumb<->index pinch distance as a live scale controller, normalized by the
  hand's own size so it works at any distance from the camera
- Relative hand-size depth estimate (bigger hand on screen = "closer") that
  modulates visual size, glow, and particle count
- A procedurally generated, real-time glowing jellyfish (translucent bell,
  independently swaying tentacles) — no static image or asset involved
- A rotating 3D wireframe cube with glowing edges, scaling with gesture input
- A lightweight, capped (500) particle system with trails and fade-out
- Two-hand interaction: a connection line between both index fingers, whose
  distance/midpoint drives scale and position
- ACTIVE / STANDBY system state — via SPACE or a debounced (600ms)
  OPEN_HAND / CLOSED_HAND gesture, resistant to accidental false triggers
- Minimal on-screen indicators only — no dashboard, no panels covering the visual
- 100% local processing — no cloud, no external API, no telemetry

## Architecture

```text
main.py                  application loop, input handling, state, rendering pipeline
src/
├── camera.py             camera init, mirrored frame capture, cleanup, error handling
├── hand_detector.py       MediaPipe HandLandmarker wrapper, raw per-frame hand geometry
├── gesture_detector.py    pure gesture classification (finger count -> name), pinch test,
│                          StableValue - the generic "hold for N ms" debounce primitive
├── finger_tracker.py      stateful per-hand smoothing built on StableValue
├── interaction.py         GESTURE_CONFIG mapping, EMA smoothing, two-hand override,
│                          ActivationController (debounced gesture on/off)
├── jellyfish.py           procedural glowing jellyfish (bell + tentacles)
├── wireframe_cube.py      rotating 3D cube, manual rotation + perspective projection
├── particles.py           capped, lifetime-based particle system
├── visual_engine.py       composites cube + jellyfish + particles + glow each frame
└── utils.py               FPS counter, text drawing, EMA/clamp math, glow compositing
```

```text
Camera
  |
OpenCV (mirrored frame capture)
  |
MediaPipe HandLandmarker
  |
Handedness Detection (Left / Right, from MediaPipe - never assumed order)
  |
Finger Detection (0-5 open fingers per hand, thumb/index/middle/ring/pinky)
  |
finger_tracker.py (debounce ~300-500ms against jitter)
  |
interaction.py (GESTURE_CONFIG -> position / scale / particles / animation,
  |               EMA smoothing, two-hand distance/midpoint override)
  |
visual_engine.py (jellyfish + wireframe cube + particles, glow composite)
```

Finger extension uses a simple, orientation-independent rule: a finger is
"open" when its tip sits farther from the wrist than its middle joint does.
This holds up under mirroring, tilting, and rotation, which matters because a
webcam rarely gets a perfectly upright hand.

## Requirements

- macOS (Apple Silicon)
- Python 3.12
- A camera and camera permission granted to your terminal / VS Code

## Installation

```bash
cd ~/Projects/interactive-vfx

python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt

# One-time download of MediaPipe's public HandLandmarker model into models/
# (~8MB; the app itself never makes a network call while running)
./scripts/download_model.sh
```

### Dependency compatibility

Pinned to a combination verified to install without conflicts **and** run
correctly on macOS + Apple Silicon + Python 3.12:

```text
opencv-python==4.10.0.84
mediapipe==0.10.21
numpy==1.26.4
```

`mediapipe==0.10.21` requires `numpy<2` — do not upgrade NumPy to 2.x. Do not
upgrade MediaPipe past this pin without re-testing: a newer release (1.0.1,
at time of writing) crashes on macOS with a Metal/GPU-related `Service is
unavailable` error regardless of CPU delegate settings.

Verify after installing:

```bash
python -c "import numpy, cv2, mediapipe; print('NumPy:', numpy.__version__); print('OpenCV:', cv2.__version__); print('MediaPipe:', mediapipe.__version__)"
```

## Running

```bash
python -m pip install -r requirements.txt
pytest -q
python main.py
```

Optional flags:

```bash
python main.py --camera 1        # use a different camera device
python main.py --hold-ms 500     # longer gesture debounce hold (300-500ms typical)
```

## Gesture controls

Each hand's currently held (debounced) finger count maps to a control
channel via `GESTURE_CONFIG` in `src/interaction.py`:

```python
GESTURE_CONFIG = {
    "one_finger": "position",
    "two_fingers": "scale",
    "three_fingers": "particles",
    "four_fingers": "animation",
    "five_fingers": "activate",
    "fist": "pause",
}
```

| Gesture | Effect |
|---|---|
| ☝️ One finger | That hand's index-tip position drives the jellyfish/cube position |
| ✌️ Two fingers | That hand's thumb<->index (pinch) distance drives scale |
| 🤟 Three fingers | Drives particle emission intensity |
| 🖐️✳️ Four fingers | Drives jellyfish/cube animation intensity |
| ✋ Open hand, held ~600ms | Activates the system (same as SPACE) |
| ✊ Closed fist, held ~600ms | Pauses the system (same as SPACE) |
| 🤏 Pinch (thumb touches index) | Detected as a modifier flag alongside the finger count |
| 🙌 Two hands | Overrides single-hand position/scale: the midpoint and distance between both index fingers control the visual directly, plus draws a connecting line |

Reassign any of these by editing the dict — nothing else needs to change.
The finger-count-to-gesture-name mapping itself is separate
(`_GESTURE_TO_CONFIG_KEY` in `interaction.py`) if you want to remap which
finger count *name* feeds into `GESTURE_CONFIG`.

## Finger controls

Every hand reports (in `src/hand_detector.py`'s `HandReading`): all five
fingertip positions, which of the five fingers are extended, a 0-5 finger
count, palm center, bounding box, and a relative on-screen size used for
depth estimation.

## Keyboard controls

```text
SPACE  = toggle ACTIVE / STANDBY
ESC/Q  = quit
```

In `STANDBY`, the camera and hand tracking keep running exactly as before,
but the jellyfish/cube render dimmed and small, and no new particles are
emitted — switch back with SPACE or an open-hand gesture.

## Camera permissions

If the camera can't be opened, the app prints:

```text
Camera access is disabled.
Enable camera permission for the application/VS Code in:
  System Settings -> Privacy & Security -> Camera
```

Steps to fix:

1. Open **System Settings → Privacy & Security → Camera**.
2. Enable access for **VS Code** (or Terminal, if you run from there).
3. **Fully quit** VS Code / Terminal (Cmd+Q) and reopen it — a permission
   change does not apply to an already-running process.
4. Run `python main.py` again.

## Performance

Target is 30-60 FPS. To stay there:

- MediaPipe runs on the CPU delegate (the GPU/Metal delegate crashes
  headless on this stack — see the dependency note above) at video-mode,
  single-frame-at-a-time inference.
- Particles are hard-capped at `MAX_PARTICLES = 500`, regardless of
  emission rate.
- "Glow" is faked with a cheap `cv2.add` composite instead of a full-frame
  Gaussian blur pass.
- No frame is ever buffered, queued, or saved — every frame is processed and
  discarded immediately, so memory use stays flat over an arbitrarily long
  run.
- The camera is always released via a context manager (`with cam, detector:`
  in `main.py`), including on error paths, so a crash or Ctrl-C doesn't
  leave the camera device locked.

## Privacy

All processing — camera capture, hand tracking, gesture logic, and
rendering — runs locally on your Mac. There is no cloud API, no remote
server, no telemetry, no analytics, and no external AI service call of any
kind. The camera feed is read frame-by-frame and never written to disk or
sent anywhere; the only network access this project ever makes is the
one-time, explicit `scripts/download_model.sh` you run yourself during
setup.

## Troubleshooting

| Symptom | Fix |
|---|---|
| Camera permission error | See [Camera permissions](#camera-permissions) above. |
| Model not found on startup | Run `./scripts/download_model.sh`. |
| Hands not detected | Improve lighting; keep the whole hand in frame. |
| Finger count flickers | Increase `--hold-ms` (default 400). |
| Wrong camera opens (e.g. iPhone) | Try `--camera 1`, or disable Continuity Camera on your iPhone. |
| Low FPS | Close other apps using the camera/GPU; try `--width 960 --height 540`. |
| Gesture ON/OFF triggers by accident | Increase `hold_ms` in `ActivationController` (`main.py`'s construction of it) beyond 600ms. |

## Project structure

```text
interactive-vfx/
├── main.py
├── requirements.txt
├── README.md
├── LICENSE
├── .gitignore
├── assets/
├── captures/
├── src/
│   ├── __init__.py
│   ├── camera.py
│   ├── hand_detector.py
│   ├── gesture_detector.py
│   ├── finger_tracker.py
│   ├── visual_engine.py
│   ├── jellyfish.py
│   ├── particles.py
│   ├── wireframe_cube.py
│   ├── interaction.py
│   └── utils.py
└── tests/
    ├── __init__.py
    ├── test_detector.py
    └── test_gestures.py
```

## Roadmap

Working v1 ships with plain OpenCV 2D drawing doing all of the "3D" and
glow work by hand (manual rotation/projection matrices, additive-blend
glow). It's real, it's runnable, and it holds 30-60 FPS on Apple Silicon —
but there's real room to grow:

- Swap the hand-rolled cube projection / glow compositing for a proper
  renderer (e.g. `moderngl`, `pyglet`, or a native OpenGL/Metal context) for
  true depth-tested 3D, bloom, and anti-aliasing.
- TouchDesigner bridge: stream `InteractionState` (position, scale,
  particle/animation intensity, two-hand data) over OSC or a local socket so
  the same hand tracking can drive a TouchDesigner patch instead of the
  built-in Python renderer — `src/interaction.py`'s `InteractionState` is
  already a clean, serializable hand-off point for this.
- Smarter pinch-based scale that also accounts for finger curl, not just
  thumb-index distance.
- Multi-jellyfish or multi-cube scenes, one per detected hand.
- A proper depth model (e.g. stereo or ML-based) instead of the relative
  bounding-box-size heuristic, which is explicitly an approximation, not a
  physical distance measurement.
