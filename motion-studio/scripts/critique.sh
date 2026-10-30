#!/usr/bin/env bash
# Critique loop step: render stills + contact sheet for review against critique/RUBRIC.md.
# Usage: bash scripts/critique.sh [seconds-between-stills] [film-slug]
set -euo pipefail
cd "$(dirname "$0")/.."
every="${1:-0.5}"
node scripts/render.mjs --stills "$every" ${2:+--film "$2"}
ffmpeg -y -loglevel error -pattern_type glob -i 'critique/frames/*.jpg' \
  -vf "scale=480:-2,drawtext=text='%{eif\\:n*$every*1000\\:d} ms':x=8:y=8:fontsize=22:fontcolor=white:box=1:boxcolor=black@0.6,tile=5x6:padding=4" \
  critique/contact-%02d.jpg
ls critique/contact-*.jpg
echo "Now score each category in critique/RUBRIC.md, log it in critique/LOG.md, fix the lowest, re-render."
