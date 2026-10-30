---
name: motion-studio
description: Make a professional, beat-synced motion-graphics launch film for this product (or a new feature) using the motion-studio harness - brand capture, style guide, director's brief, beats.json, deterministic HTML composition, scored soundtrack, critique loop, MP4 render. Use when asked for a launch video, promo, explainer animation, or motion graphics.
---

# Motion Studio

Everything lives in `motion-studio/`. Work from that directory.

## Steps

1. **Read the harness first:** `MOTION_RULES.md`, `references/STYLE_GUIDE.md`,
   `brand/brand.json`, and an existing brief in `briefs/` as a template. Obey
   the rules; they override any habit.
2. **Brand:** run `node scripts/capture-brand.mjs` if the app UI changed or the
   film needs new screens. Edit that script to capture other states. Never
   draw fake UI or invent numbers; use `brand/sample-estimate.json`.
3. **Reference (optional):** if the user gives a reference video, run
   `bash scripts/extract-frames.sh <video>`, read the frames and contact sheet,
   and rewrite the "Visual grammar" part of `STYLE_GUIDE.md`.
4. **Director's brief:** write `briefs/<slug>.md` with the film in one line,
   references, characters (if any), tempo, a beat sheet table (bars → shot →
   action → sound), on-screen text (≤6 words each), an optional VO script, and
   "must not". Show the brief to the user and get a yes before writing code.
5. **beats.json:** turn the beat sheet into `events[]` (id, beat, sfx) and
   `sections[]`. Every visible hit needs an event.
6. **Composition:** edit `composition/index.html`. Add a scene function per
   section; time everything with `ev('<event id>')` and `beat(n)`; keep it
   deterministic (no transitions, timers or unseeded randomness).
7. **Voiceover (optional):** if the Fish Audio MCP connector is available,
   generate the VO from the brief and save it to `audio/vo.wav`.
8. **Critique loop:** run `bash scripts/critique.sh`, read the contact sheets
   and the full-size frame at every event, score `critique/RUBRIC.md`, append
   the scores to `critique/LOG.md`, fix the lowest category, and repeat until
   every category is ≥ 8. Expect several passes.
9. **Render:** run `node scripts/render.mjs --rescore`, check `out/film.mp4`
   with ffprobe, and check that the soundtrack peaks ≤ −1 dBFS.
10. Report the film path, the final scores, and what you would improve next.
