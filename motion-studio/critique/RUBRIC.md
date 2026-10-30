# Critique Rubric

Score each category 1–10 against the brief, from the contact sheet
(`bash scripts/critique.sh`) plus full-size frames at each `beats.json` event.
Log scores in `LOG.md`. Fix the **lowest** category first, re-render, re-score.
Ship only when every category is **≥ 8**.

| # | Category | 10 looks like |
|---|---|---|
| 1 | **Brief fidelity** | Every beat-sheet row happens at its bar; on-screen text matches the brief word for word. |
| 2 | **Sync** | The frame on each event already shows the hit; nothing moves without a sound; no sound without motion. |
| 3 | **Brand** | Only `brand.json` colours; real UI screenshots; numbers match `sample-estimate.json`. |
| 4 | **Typography** | Max 6 words per shot; headlines readable at phone size; no widows or clipped descenders. |
| 5 | **Composition** | One focal point per shot; balanced negative space; nothing stuck in a corner by accident. |
| 6 | **Motion quality** | Eased (never linear) entrances; no dead frames; one camera move per shot; no jitter. |
| 7 | **Banned list** | Zero items from MOTION_RULES §2 "Banned". |
| 8 | **Sound** | Peaks ≤ −1 dBFS; SFX under the music; the music resolves on the end card. |

Prompt to run the loop with Claude:

> Run `bash scripts/critique.sh`. Read every contact sheet and the full-size
> frame at each event in beats.json. Score each rubric category with one
> sentence of evidence and append the result to critique/LOG.md. Fix the
> lowest-scoring issue in composition/index.html (or score.py), re-run, and
> repeat until every category is ≥ 8. Then run `node scripts/render.mjs`.
