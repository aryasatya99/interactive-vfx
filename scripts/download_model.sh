#!/usr/bin/env bash
# Downloads Google's public MediaPipe HandLandmarker model into models/.
# This is a one-time setup step, not something the app does silently at
# runtime, so there is never a surprise network call while the camera is on.
set -euo pipefail

DEST_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/models"
DEST="$DEST_DIR/hand_landmarker.task"
URL="https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"

mkdir -p "$DEST_DIR"
echo "Downloading HandLandmarker model to $DEST ..."
curl -sL -o "$DEST" "$URL"
echo "Done: $(du -h "$DEST" | cut -f1) saved."
