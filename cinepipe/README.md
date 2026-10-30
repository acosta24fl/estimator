# CinePipe

A 4-stage pre-production pipeline that turns a film idea into **validated,
ready-to-paste prompts** for AI image and video tools:

```
Story Bible  →  Castkit  →  Scenecraft  →  Shotcaller
(story + world)  (locked faces)  (empty sets)   (16-slot shot prompts)
```

Claude (`claude-opus-5-5`) writes each stage as structured output. Pydantic
guardrails reject anything that breaks the rules, and the rejection goes
back to Claude for a fix (up to 2 repair turns). A stage won't run on a
missing, invalid, or out-of-date upstream stage.

**What it does not do:** CinePipe makes the *prompts and specs*. You still
generate the reference images (Midjourney / Flux) and the video clips
(Seedance / Kling / Runway / Higgsfield) in those tools, with your own
accounts and credits.

## Install

```bash
cd cinepipe
```

```bash
pip install -e .
```

Claude stages need an Anthropic API key (`ANTHROPIC_API_KEY`). On Windows:

```bash
setx ANTHROPIC_API_KEY "sk-ant-..."
```

Then open a new terminal.

## Try it with no API key

```bash
cinepipe demo
```

This runs all four stages on the built-in handyman example
(`examples/handyman/`) and writes everything to
`workspace/handyman-demo/`.

## Make your own film

```bash
cinepipe new myfilm
```

```bash
cinepipe bible myfilm --idea "A handyman prices a job on the spot with his phone"
```

Claude interviews you one question at a time until all 6 pillars are locked:
premise, timeline, factions, world rules, locations, characters.

```bash
cinepipe cast myfilm
```

```bash
cinepipe plates myfilm
```

```bash
cinepipe shots myfilm --beats-file beats.md
```

Or run all of it in one go:

```bash
cinepipe run myfilm --beats-file beats.md
```

Then:

1. Generate each character sheet from `castkit_prompts.md`, and each set from
   `scenecraft_prompts.md`.
2. For every `shots/SH###.txt`, upload the listed reference images, paste the
   prompt, and render a **480p preview**.
3. When a preview is good, unlock full quality for that shot:

```bash
cinepipe approve myfilm SH040
```

Other commands: `cinepipe status myfilm` shows what's done or stale, and
`cinepipe brief myfilm` writes a motion-studio director's brief.

Any stage can skip Claude with `--from-file some.json`. The file is
validated against the same guardrails.

## What's in a project folder

| File | Stage | Use |
|---|---|---|
| `story_bible.json` / `story_bible_skill.md` | 1 | Canonical world. The `.md` can go in a Claude Project or `CLAUDE.md` as permanent memory |
| `castkit_registry.json` / `castkit_prompts.md` | 2 | @CharID → locked face, body, wardrobe; 3-panel sheet prompts |
| `scenecraft_plates.json` / `scenecraft_prompts.md` | 3 | @PlateID → empty set + lens, aperture, shutter, Kelvin |
| `shotcaller_shots.json` / `shots/SH###.txt` | 4 | 16-slot prompts plus which references to upload |
| `director_brief.md` | - | Story + characters + beat sheet for `motion-studio` |
| `state.json` | - | Hashes used to detect stale stages |

## The guardrails (enforced in code, `cinepipe/models.py` + `shotcaller.py`)

| Rule | What's checked |
|---|---|
| **18% neutral gray references** | The sheet spec must say 18%, #808080, neutral gray and diffuse; rim, back or dramatic light is rejected |
| **3-panel turnaround** | Exactly: full-body front orthographic, full-body back orthographic, ECU portrait |
| **Face lock before wardrobe** | The face and body locks may not mention clothing; the image prompt is ordered face → body → wardrobe → sheet |
| **Clean plates** | No people words (man, people, silhouette, customer, he/she…), including with punctuation attached |
| **Real optics** | Focal length from a real lens set, aperture f/1.8–f/8, shutter angle, Kelvin, and ≥2 depth cues (aerial perspective, dust, halation, grain, haze) |
| **16-slot token isolation** | No 4-word phrase in two slots; clothing only in slot 09 (resolved from Castkit, never retyped); set description only in slot 07 |
| **ONE MOUTH SPEAKS AT A TIME** | Dialogue segments can't overlap, must fit the shot, the speaker must be bound, and listeners must be "mouth closed" |
| **Timecoded beats** | Every action beat starts with `[start-end s]` inside the shot length |
| **480p first** | New shots are always `480p_preview_draft`; `1080p_master_render` needs `cinepipe approve` with the preview's seed |
| **Stage gating** | Editing the Story Bible marks Castkit, Scenecraft and Shotcaller stale until they're re-run |

## With motion-studio

The phone screen in `SH030` is a flat tracking plate. Composite the real
Estimator UI from a `motion-studio` render over it, because video models
can't draw your real app. `cinepipe brief` turns the pipeline into a
director's brief that motion-studio's skill can build an animated film from.

## Tests

```bash
pip install -e ".[dev]"
```

```bash
pytest
```

The 31 tests cover every guardrail, stage gating, and the Claude repair loop
(with a fake client, so no API key is needed).
