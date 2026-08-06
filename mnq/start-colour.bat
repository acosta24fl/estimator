@echo off
REM Start the dashboard with the green/red band always committing.
REM
REM Same as start.bat, but with MNQ_SIGNAL_MIN_RATIO=0, so the page colours
REM itself on every call however small the projected move is. Double-click
REM this instead of editing start.bat.
REM
REM Read the colour with that in mind: at this setting a 0.4 point projection
REM against a 23 point typical move paints the screen just as hard as a real
REM one. The text beside the badge still reports the true projected size, and
REM the metrics panel still reports measured skill. Use those, not the colour.
REM
REM This file must keep Windows (CRLF) line endings - see start.bat.

set "MNQ_SIGNAL_MIN_RATIO=0"
call "%~dp0start.bat"
