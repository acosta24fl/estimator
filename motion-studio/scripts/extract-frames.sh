#!/usr/bin/env bash
# Extract frames + cut list from a reference video so Claude can write a style guide.
# Usage: bash scripts/extract-frames.sh references/ref.mp4 [fps]
set -euo pipefail
src="${1:?usage: extract-frames.sh <video> [fps]}"
fps="${2:-2}"
dir="$(cd "$(dirname "$0")/.." && pwd)/references/frames"
rm -rf "$dir" && mkdir -p "$dir"

ffmpeg -loglevel error -i "$src" -vf "fps=$fps,scale=960:-2" "$dir/f_%04d.jpg"
ffmpeg -loglevel error -i "$src" -vf "fps=1,scale=384:-2,tile=6x6" -frames:v 1 "$dir/contact-sheet.jpg"

echo "Scene cuts (seconds):"
ffmpeg -hide_banner -i "$src" -vf "select='gt(scene,0.3)',showinfo" -f null - 2>&1 \
  | grep -o 'pts_time:[0-9.]*' | cut -d: -f2 | tee "$dir/cuts.txt"
echo "Wrote $(ls "$dir" | wc -l) files to $dir"
