# Motion Studio

A harness for making launch films with Claude Code: rules, brand kit, style
guide, director's brief, beat-synced sound, and a critique loop. The first
film is a 25-second launch video for the Job Estimate Generator
(`out/film.mp4`).

```
motion-studio/
  MOTION_RULES.md          render contract, visual look, sound rules   (read first)
  brand/                   brand.json, icons, real UI screenshots, real sample numbers
  references/STYLE_GUIDE.md  visual grammar (rebuild from a reference video)
  briefs/estimator-launch.md director's brief: one-liner, beat sheet, on-screen text, VO
  beats.json               tempo + every timed event (drives visuals AND sound)
  composition/index.html   the film: deterministic, seek(t)-driven HTML
  scripts/
    capture-brand.mjs      screenshots the real app + extracts real numbers
    extract-frames.sh      reference video -> frames, contact sheet, cut list
    score.py               music + SFX scored to beats.json (+ optional VO with ducking)
    render.mjs             frame-by-frame render -> out/film.mp4  (--serve to preview)
    critique.sh            stills + contact sheets for the critique loop
  critique/RUBRIC.md, LOG.md
```

## Requirements
Node 18+, Python 3 + numpy, ffmpeg, and Playwright with Chromium:

```bash
cd motion-studio && npm install && npx playwright install chromium
```

```bash
pip install numpy
```

## Run it

```bash
node scripts/capture-brand.mjs
```

```bash
node scripts/render.mjs --serve
```
Open http://localhost:4173/composition/ to scrub and play the film with sound
(once `out/soundtrack.wav` exists).

```bash
bash scripts/critique.sh
```

```bash
node scripts/render.mjs --rescore
```

## Workflow (as in the video)

1. **Environment and rules:** open the repo in Claude Code (desktop app or
   CLI). `MOTION_RULES.md` keeps the output from looking generic.
2. **Brand:** `capture-brand.mjs` pulls real UI and numbers from the app. Put
   any extra logos or fonts in `brand/`.
3. **Reference style:** put a video you like in `references/`, run
   `extract-frames.sh`, and ask Claude to rewrite `STYLE_GUIDE.md` from the
   frames.
4. **Voiceover (Fish Audio MCP):** in Claude, go to Settings → Connectors → *Add
   custom connector* and paste the MCP server URL from Fish Audio's docs, plus
   your Fish Audio API key. Set its tools to *Always allow*. Then ask: *"Using
   Fish Audio, voice the VO line in briefs/estimator-launch.md with my cloned
   voice and save it as motion-studio/audio/vo.wav."* `score.py` mixes it in
   and ducks the music automatically.
5. **Sound sync:** change `bpm` or events in `beats.json`. Visuals and audio both
   read from it, so they stay locked together.
6. **Director's brief:** copy `briefs/estimator-launch.md` for each new film.
7. **Critique loop:** follow `critique/RUBRIC.md` and keep going until every
   category scores ≥ 8.
8. **Skill:** all of the above is packaged as the `motion-studio` skill in
   `.claude/skills/`, so `/motion-studio <idea>` runs the whole pipeline.
