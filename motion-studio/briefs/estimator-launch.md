# Director's Brief: Estimator launch film

## Film in one line
A tradesperson stops guessing what to charge: real costs go in, three prices you
can defend come out, in 24 seconds.

## Audience & goal
Solo contractors, handymen and freelancers who price jobs on a napkin. Goal:
install the free app.

## References
- `references/STYLE_GUIDE.md`: dark SaaS launch grammar, cuts on the beat.
- Real UI: `brand/screens/*.png`. Real numbers: `brand/sample-estimate.json`
  (bathroom tile job: $331 materials, 10 h at $45, 34 mi, standard price $1,360.87).

## Characters
None. The product UI and the cursor are the only characters. The cursor is calm
and precise. It never wiggles.

## Tempo
120 BPM, 4/4, 12 bars = 24 s. See `beats.json`.

## Beat sheet
| Bars | Time | Shot | What happens | Sound |
|---|---|---|---|---|
| 1–2 | 0–4 s | **Cold open** | "You finished the job." lands on beat 1. Bar 2: "Did you charge enough?" with *enough?* in amber. | Kick + pad, sparse |
| 3–4 | 4–8 s | **The mess** | Hidden costs hit one per beat in a grid: Materials, Labor, Gas, Overhead, Contingency, Taxes, plus mono numbers. Beat 15: everything collapses to the centre. | Hats join, one tick per word, riser into bar 5 |
| 5 | 8–10 s | **Reveal** | Hard cut to a blue flash. App icon punches in, then the product name. | Impact + sub drop |
| 6–8 | 10–16 s | **Product** | Tilted card of the real UI. The cursor presses "Generate Estimate" on beat 22. The breakdown wipes in and the rows highlight on beats. TOTAL rolls up to **$1,360.87**. | Full groove, click, rising ticks, chime on the total |
| 9–10 | 16–20 s | **Three prices** | Budget / Standard / Premium slam in on beats 32–34. Standard gets the RECOMMENDED glow on beat 36. Caption: "Three prices. Zero guesswork." | Hits per card, chime |
| 11–12 | 20–24 s | **End card** | Icon + "Know your number." Line 2: "Free · Works offline · On your phone". Music resolves; last hit on beat 46. | Pad swell, final hit, tail |

## On-screen text (max 6 words each)
1. You finished the job.
2. Did you charge **enough?**
3. Job Estimate Generator
4. Three prices. Zero guesswork.
5. Know your number.
6. Free · Works offline · On your phone

## Voiceover (optional; generate with Fish Audio MCP → `audio/vo.wav`)
> "You finished the job. But did you charge enough? Materials, labor, gas,
> overhead, taxes... Estimator adds it all up and hands you three prices you
> can stand behind. Know your number."

## Must not
- Show any price that isn't in `sample-estimate.json`.
- Use colours outside `brand.json`.
- Break any rule in `MOTION_RULES.md`.
