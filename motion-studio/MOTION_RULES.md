# Motion Studio Rules

These rules are the harness. Read them before writing or editing any composition.
The prompt is ~10% of the result; these rules, the brand kit, the style guide and
the brief are the other 90%.

## 1. Render contract

- **Time is the only input.** Every visual property is a pure function of `t`
  (seconds). The composition exposes `window.seek(t)`; no CSS transitions, no
  `setTimeout`, no `requestAnimationFrame` loops, no `Math.random()` without a seed.
  Rendering frame 312 twice must give the same pixels.
- **Canvas:** 1920×1080, 30 fps, `yuv420p` H.264 + AAC 48 kHz. Duration comes from
  `beats.json`, never hard-coded in more than one place.
- **One clock.** Animation timings are expressed in beats (`b(n)` helper), not
  seconds. At 120 BPM one beat = 0.5 s, one bar = 2 s.
- **Local assets only.** Fonts, screenshots and icons live in `brand/`. No network
  fetches at render time.
- **Render must finish unattended:** `node scripts/render.mjs` produces
  `out/film.mp4` with no manual steps.

## 2. Visual look

Use:
- Brand tokens from `brand/brand.json` only. No new hues.
- Real product UI (`brand/screens/*.png`), not mock-ups drawn from scratch.
- Inter / Inter Display for type, JetBrains Mono for numbers. Tight tracking on
  headlines (-0.03em), weights 600–800.
- Easing: `expoOut` for entrances, `expoInOut` for camera moves, short
  overshoot (`backOut`, ≤1.4) only for hits that land on a downbeat.
- Generous negative space; one idea per shot; max 6 words on screen at once.
- Subtle depth: soft shadows, a slight grain/vignette, slow camera drift (≤3%
  scale) so no frame is ever fully static.

Banned (the "generic AI video" tells):
- Purple-to-pink gradients, lens flares, glowing particle swarms, bokeh blobs.
- Spinning 3D logos, typewriter text for headlines, bouncing everything.
- Stock emoji as decoration, centred text on every shot, linear easing.
- Fade-to-black between every shot. Prefer cuts on the beat, push-ins, masks.
- Made-up product UI, fake metrics, fake testimonials.

## 3. Sound

- Music is scored to the tempo in `beats.json` (default 120 BPM, 4/4).
- Every visible hit (cut, slam, click, number landing) has a cue in
  `beats.json` `events[]` and a matching sound in the mix. If you can see it,
  you can hear it.
- UI sounds are short (≤120 ms), quiet (−18 dB under music peak), and land
  exactly on the event frame.
- Voiceover (optional, `audio/vo.wav`) ducks music by ~8 dB while speaking.
  Generate it with the Fish Audio MCP when connected (see `README.md`).
- Final mix peaks ≤ −1 dBFS. No clipping.

## 4. Process

1. Brand kit → `brand/` (run `scripts/capture-brand.mjs`).
2. Style guide → `references/STYLE_GUIDE.md` (from a reference video via
   `scripts/extract-frames.sh`, or the defaults already there).
3. Director's brief → `briefs/*.md`. No code before the brief has: film in one
   line, references, beat sheet, on-screen text.
4. `beats.json` from the beat sheet.
5. Composition → `composition/index.html`.
6. Render → `scripts/render.mjs`.
7. Critique loop → `scripts/critique.sh`, score with `critique/RUBRIC.md`,
   fix the lowest-scoring item, re-render. Ship only when every category ≥ 8.
