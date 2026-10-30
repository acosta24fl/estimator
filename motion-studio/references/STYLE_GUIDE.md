# Style Guide

The default look for this studio: a modern SaaS launch-film style (dark UI, hard
cuts on the beat, real product on screen). To match a specific reference video
instead, follow "Building a style guide from a reference" below and replace the
sections under "Visual grammar".

## Building a style guide from a reference

1. Drop the reference into `references/` (e.g. `references/ref.mp4`).
2. `bash scripts/extract-frames.sh references/ref.mp4` writes one frame every
   0.5 s plus a contact sheet to `references/frames/`, and prints the
   scene-cut timestamps.
3. Ask Claude: *"Read every frame in references/frames and the cut list. Rewrite
   STYLE_GUIDE.md: palette, type, layout grid, shot lengths, cut rhythm, camera
   moves, easing, transitions, how UI is shown, what is never done."*
4. Keep the brand colours from `brand/brand.json`. Borrow the reference's
   grammar, not its brand.

## Visual grammar (current)

**Palette:** near-black `#0f1117` stage, `#1a1d27` cards, one blue accent
(`#4a8cff` / `#7eb8ff`), money green `#4aff8c` reserved for final prices, amber
`#ffaa4a` for "problem" words. Never more than 2 accent colours in one shot.

**Type:** Inter Display 800 for headlines, 96–160 px, tracking −0.035em. Inter
500 for supporting lines at 36–44 px in `#888`. Every number in JetBrains Mono.

**Layout:** 12-column grid, 120 px safe margin. Headlines sit left-aligned on
the left third or optically centred, never both in one shot.

**Rhythm:** a cut or hit on every downbeat in the busy sections, every bar in the
calm ones. Shots last 2–6 s. The busiest section is about 60% of the way through.

**Camera:** slow constant push-in (1.00 → 1.03) on every shot. One bigger
tilt-and-push on the product hero shot. No handheld shake.

**Transitions:** hard cut (default), mask wipe in the accent colour (once), scale
punch on a downbeat. No crossfades between scenes.

**UI presentation:** real screenshots on a floating card with a 1 px border,
large soft shadow and a slight 3D tilt (≤8°). A cursor shows each click, with a
press-scale of 0.92 that lands on the beat.

**Texture:** 3% film grain, radial vignette, faint 48 px grid behind the opening
titles.

**Sound look:** tight kick, off-beat hats, warm sub bass, airy pad. UI clicks
are crisp ticks. The numbers rolling up get a rising tick sequence.
