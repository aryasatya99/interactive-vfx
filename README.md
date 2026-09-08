# AURA — AI Gesture Interface

A futuristic, HUD-style gesture-control demo for the Mac webcam. Raise both
hands with thumb and index open on each, and AURA's visual effects come
alive — scanning ring brightens, particles bloom from your fingertips, and
the interface flashes `GESTURE ACCEPTED`.

Built with OpenCV, MediaPipe, and NumPy. All processing runs locally — no
cloud, no paid API, no external assets.

## Overview

AURA reads your Mac's camera, tracks up to two hands with MediaPipe's
HandLandmarker, and validates a strict two-hand gesture (thumb + index open
on **both** the left and right hand, judged by MediaPipe's own handedness
classifier — never by detection order or screen position). A short debounce
window keeps the gesture status from flickering on momentary landmark noise.
Everything is drawn as a sci-fi HUD directly over the camera feed: corner
brackets, a scanning ring, glowing fingertips, and lightweight particles.

## Features

- Realtime Mac camera capture (mirrored, like a normal camera app)
- MediaPipe HandLandmarker hand tracking, up to 2 hands
- True LEFT/RIGHT detection from MediaPipe handedness (not detection order)
- Per-hand thumb + index "open" recognition
- Strict two-hand gesture validation, not just "2 hands present"
- 400ms debounce so the gesture status doesn't flicker
- Futuristic HUD: title, corner brackets, scanning ring, crosshairs, glowing
  fingertips, semi-transparent status panels
- Lightweight, capped particle effect that follows fingertip positions
- ACTIVE / STANDBY system state (tracking always runs; the main visual
  effect only fires in ACTIVE)
- FPS counter and detected-hand count
- 100% local processing — nothing ever leaves your machine, no internet
  needed while running

## Requirements

- macOS (Apple Silicon)
- Python 3.12
- A camera and camera permission granted to your terminal / VS Code

## Installation

```bash
cd ~/Projects/gesture-xray

python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt

# One-time download of MediaPipe's public HandLandmarker model into models/
# (~8MB; the app itself never makes a network call while running)
./scripts/download_model.sh
```

### Dependencies

Pinned to a combination verified to install without conflicts **and** run
correctly on macOS + Apple Silicon + Python 3.12:

```text
opencv-python==4.10.0.84
mediapipe==0.10.21
numpy==1.26.4
```

`mediapipe==0.10.21` requires `numpy<2`; do not upgrade NumPy to 2.x, and
do not upgrade MediaPipe past this pin without re-testing — a newer release
(1.0.1, at time of writing) crashes on macOS with a Metal/GPU-related
`Service is unavailable` error regardless of CPU delegate settings.

## Running

```bash
cd ~/Projects/gesture-xray
source .venv/bin/activate
python main.py
```

Optional flags:

```bash
python main.py --camera 1        # use a different camera device
python main.py --hold-ms 500     # longer debounce hold (300-500ms typical)
```

## macOS camera permission

The first run prompts macOS for camera access. If the camera fails to open,
AURA prints:

```text
Camera permission required. Enable camera access for VS Code
in System Settings > Privacy & Security > Camera.
```

Steps to fix:

1. Open **System Settings → Privacy & Security → Camera**.
2. Enable access for **VS Code** (or Terminal, if you run from there).
3. **Fully quit** VS Code / Terminal (Cmd+Q) and reopen it — a permission
   change does not apply to an already-running process.
4. Run `python main.py` again.

## Gesture instructions

Hold both hands up so MediaPipe can see them clearly, with your thumb and
index finger extended (a loose "L" shape) on **each** hand:

```text
LEFT  hand: thumb OPEN + index OPEN
RIGHT hand: thumb OPEN + index OPEN
```

The gesture is valid only when **all six conditions** are true at once:

```python
left_hand_detected  and left_thumb_open  and left_index_open
and right_hand_detected and right_thumb_open and right_index_open
```

Two hands in frame is not enough by itself — each hand's identity comes from
MediaPipe's handedness classifier, and each hand's thumb and index must
individually be open. Hold the pose steady for the debounce window (~400ms)
to see `GESTURE: VALID` and, if the system is `ACTIVE`, the particle/ring
effect and the `GESTURE ACCEPTED` message.

## Keyboard controls

```text
SPACE = toggle SYSTEM: ACTIVE / STANDBY
Q     = quit
```

In `STANDBY`, hand tracking and the HUD status panels keep working, but the
main visual effect (bright ring, particles, acceptance message) is
suppressed even on a valid gesture. Switch back to `ACTIVE` with SPACE.

## Architecture

```text
main.py                 application loop, input handling, state, rendering pipeline
src/
├── camera.py            camera init, frame capture (mirrored), cleanup, error handling
├── hand_detector.py      MediaPipe HandLandmarker wrapper, handedness, thumb/index geometry
├── gesture_detector.py   left/right validation, GESTURE_HOLD_MS debounce
├── particles.py          capped, lifetime-based particle system
├── hud.py                HUD primitives: panels, brackets, rings, crosshair, glow
└── utils.py              FPS counter, drawing helpers, capture-save helper
```

```text
Camera
  |
OpenCV (frame capture, mirroring, drawing)
  |
MediaPipe HandLandmarker
  |
Handedness Detection (Left / Right, from MediaPipe — not screen position)
  |
Finger Detection (thumb + index open/closed, per hand)
  |
Gesture Validation (both hands present AND both fingers open, per hand)
  |
Debounce (400ms hold before the state flips)
  |
HUD + Particle Rendering
```

`thumb_open` / `index_open` use a simple, orientation-independent rule: a
finger is "open" when its tip sits farther from the wrist than its middle
joint does. This holds up under mirroring, tilting, and rotation, which
matters because a webcam rarely gets a perfectly upright hand.

## Testing

```bash
source .venv/bin/activate
pytest -q
```

Covers thumb/index open-closed detection, handedness independence from
position, every valid/invalid two-hand gesture combination, the
`len(hands) == 2` trap (two hands with closed fingers must stay invalid),
gesture debounce timing, and the particle system's lifetime/cap behaviour —
all with synthetic landmarks, so no camera or model file is required to run
the suite.

## Troubleshooting

| Symptom | Fix |
|---|---|
| Camera permission error | See [macOS camera permission](#macos-camera-permission) above. |
| Model not found on startup | Run `./scripts/download_model.sh`. |
| Hands not detected | Improve lighting; keep both hands fully in frame. |
| Gesture flickers valid/invalid | Increase `--hold-ms` (default 400). |
| Wrong camera opens (e.g. iPhone) | Try `--camera 1`, or disable Continuity Camera on your iPhone. |
| Low FPS | Close other apps using the camera or GPU; try a lower `--width`/`--height`. |

## Privacy

All processing — camera capture, hand tracking, gesture logic, and HUD
rendering — happens locally on your Mac. No frame, landmark, or image is
ever sent to a server or cloud service, and the app makes no network
requests while running (the HandLandmarker model is downloaded once, ahead
of time, via `scripts/download_model.sh`).
