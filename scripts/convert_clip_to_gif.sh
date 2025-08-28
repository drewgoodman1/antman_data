#!/usr/bin/env bash
# Convert an MP4 screen recording to a web-optimized GIF using ffmpeg
# Usage: ./scripts/convert_clip_to_gif.sh input.mp4 output.gif

set -euo pipefail
if [ "$#" -lt 2 ]; then
  echo "Usage: $0 input.mp4 output.gif" >&2
  exit 2
fi
INPUT="$1"
OUTPUT="$2"

# intermediate palette
PALETTE="/tmp/palette.png"

# scale and fps for GIF
SCALE="-vf scale=800:-1:flags=lanczos"
FPS=15

ffmpeg -y -i "$INPUT" -vf "fps=$FPS,scale=800:-1:flags=lanczos,palettegen" -palettegen_max_colors 256 "$PALETTE"
ffmpeg -y -i "$INPUT" -i "$PALETTE" -lavfi "fps=$FPS,scale=800:-1:flags=lanczos [x]; [x][1:v] paletteuse" -gifflags -transdiff "$OUTPUT"

rm -f "$PALETTE"

echo "Wrote $OUTPUT"
