#!/usr/bin/env bash
# Start the dashboard with the green/red band always committing.
#
# Same as start.sh, but with MNQ_SIGNAL_MIN_RATIO=0, so the page colours itself
# on every call however small the projected move is.
#
# Read the colour with that in mind: at this setting a 0.4 point projection
# against a 23 point typical move paints the screen just as hard as a real one.
export MNQ_SIGNAL_MIN_RATIO=0
exec "$(dirname "${BASH_SOURCE[0]}")/start.sh"
