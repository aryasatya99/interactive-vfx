# Gesture X-Ray

A two-hand gesture gate for a static X-ray image, built with OpenCV and
MediaPipe on a Mac webcam. Show your left **and** right hand with thumb and
index extended on both, and the X-ray image unlocks in real time.

This is a computer-vision and gesture-control demo, not a medical tool — see
[Medical disclaimer](#medical-disclaimer).

## Features

- Realtime Mac camera capture via OpenCV (AVFoundation backend)
- MediaPipe HandLandmarker (Tasks API) hand detection, up to 2 hands
- True left/right hand detection from MediaPipe handedness classification —
  never guessed from screen position
- Per-hand thumb + index "open" recognition
- Strict two-hand gesture validation (not just "2 hands detected")
- 400ms debounce so the X-ray doesn't flicker on landmark jitter
- Live X-ray display, with graceful "locked" / "not found" states
- FPS counter
- Frame capture to disk with non-overwriting, timestamped filenames
- 100% local processing — nothing ever leaves your machine

## How it works

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
X-Ray Display (assets/xray.jpg shown, or LOCKED / NOT FOUND)
```

The camera never "sees" an X-ray — it only unlocks a static image
(`assets/xray.jpg`) that you supply yourself.

## Requirements

- macOS
- Python 3.12+
- A camera and camera permission granted to your terminal app

## Installation

```bash
git clone https://github.com/aryasatya99/gesture-xray.git
cd gesture-xray

python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt

# Downloads MediaPipe's public HandLandmarker model into models/
# (a one-time ~8MB download; the app never fetches anything at runtime)
./scripts/download_model.sh
```

Then put an image at `assets/xray.jpg` (any image works — a photo of an
X-ray print/film, or a placeholder while you experiment). If it's missing,
the app shows `X-RAY IMAGE NOT FOUND` instead of crashing.

## Run

```bash
python main.py
```

Useful flags:

```bash
python main.py --camera 1          # use a different camera (e.g. iPhone Continuity)
python main.py --hold-ms 600       # longer debounce hold
python main.py --xray assets/other.jpg
```

## Controls

```text
SPACE = Capture
R     = Reload X-ray
Q     = Quit
```

## Gesture rule

The X-ray unlocks only when **all** of the following are true at once:

```python
left_hand_detected  and left_thumb_open  and left_index_open
and right_hand_detected and right_thumb_open and right_index_open
```

Two hands in frame is not sufficient by itself — each hand's identity comes
from MediaPipe's handedness classifier, and each hand's thumb and index must
individually be open.

## Privacy

All processing — camera capture, hand detection, gesture logic, and image
display — runs locally on your Mac. No frame, landmark, or image is ever
sent to a server or cloud service, and no paid API is used.

## Medical disclaimer

This project does not diagnose, detect, or analyze medical conditions of any
kind. A webcam cannot capture X-ray radiation — it only detects visible
light. `assets/xray.jpg` is a plain image file you provide; the gesture only
controls whether it is displayed. Do not use this software, or anything
derived from it, for medical decision-making.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `Tidak bisa membuka kamera index 0` | Grant camera access: System Settings → Privacy & Security → Camera → enable for your terminal app, then fully quit and reopen the terminal. |
| Model not found error on startup | Run `./scripts/download_model.sh`. |
| Hands not detected | Improve lighting and keep both hands fully in frame; MediaPipe needs a reasonably clear view of the hand. |
| Gesture flickers open/closed | Increase `--hold-ms` (default 400). |
| Wrong camera opens (e.g. iPhone) | Try `--camera 1`, or disable Continuity Camera on your iPhone. |

## Future development

- Better gesture recognition (more fingers, more shapes)
- Configurable gestures (define your own unlock condition)
- UI improvements (theming, resizable layout)
- X-ray image *region* detection on top of a photographed film (see the
  companion `xray-camera-detector` project)
- Optional ML model for more complex gesture sets
- Performance optimization (GPU delegate once MediaPipe's macOS GPU path is
  stable, batched inference)
