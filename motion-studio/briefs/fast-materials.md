# Director's Brief: "Price it on the driveway"

## Film in one line
A handyman types in two materials and their hours, taps once, and has a price
to quote, all before the customer finishes asking.

## Audience & goal
Handymen and small trades who quote on the spot (driveway, kitchen, phone
call). Goal: show it's fast and simple, then install the app.

## References
- `references/STYLE_GUIDE.md` (same dark SaaS grammar as the launch film),
  pushed faster: more cuts, shorter holds, typing on 8th notes.
- Real UI: the Materials and Labor sections of `index.html`.
- Real numbers: `brand/sample-estimate.json` (same bathroom tile job:
  Porcelain tile $4.25 × 60, Thinset + grout $38 × 2, 10 h at $45 →
  standard $1,360.87).

## Characters
None. The cursor and the typing caret are the "hands". They're quick and confident
and never correct a mistake.

## Tempo
128 BPM, 4/4, upbeat (four-on-the-floor, offbeat open hats, plucky bass).
1 beat = 0.469 s, 1 bar = 1.875 s. 10 bars = 18.75 s + 1.25 s tail = **20 s**.

## Beat sheet
| Bars | Time | Shot | What happens | Sound |
|---|---|---|---|---|
| 1 | 0–1.9 s | **Hook** | "Customer wants a price." slams in on beat 1. | Kick + clap, filtered |
| 2 | 1.9–3.8 s | **Pressure** | It cuts to "Right now." on beat 4. Beat 6: *now* flashes amber. | Filter opens, riser |
| 3–4 | 3.8–7.5 s | **Material 1** | Hard cut to the real Materials card (tilted, push-in). The caret types "Porcelain tile" in 8th notes, then 4.25, then 60. Each field lands on a beat. | Full groove; one key-tick per 8th |
| 5 | 7.5–9.4 s | **Material 2** | The cursor taps "+ Add Material" (beat 16). The new row slides in, and "Thinset + grout · 38 · 2" types fast. | Click, ticks |
| 6 | 9.4–11.3 s | **Labor** | Whip-pan to the Labor card: 45 $/hr and 10 hours type in on beats 21 and 22. | Whoosh, ticks |
| 7 | 11.3–13.1 s | **Tap** | The cursor presses **Generate Estimate** on beat 24. Blue flash cut. The total rolls up and lands **$1,360.87** in green on beat 27. | Click, roll, chime |
| 8–9 | 13.1–16.9 s | **Options** | Budget / Standard / Premium slam in on beats 28–30. Standard glows RECOMMENDED on beat 32. Caption: "Three prices. One tap." | Hits, chime |
| 10 | 16.9–18.8 s | **End card** | App icon + "Price it on the driveway." Sub line: "Free · Works offline · On your phone". Final hit on beat 38. | Drop out, final hit, tail |

## On-screen text (max 6 words each)
1. Customer wants a price.
2. Right **now.**
3. Three prices. One tap.
4. Price it on the driveway.
5. Free · Works offline · On your phone

## Voiceover (optional)
> "Customer wants a price, right now. Tile, grout, your hours, tap. Three
> prices, ready to quote. Price it on the driveway."

## Must not
- Claim a specific time ("in 10 seconds") or show a stopwatch. We haven't
  measured real-world speed, so the film *shows* speed and doesn't state it.
- Show any number that isn't from the real app with this sample job.
- Reuse the launch film's opening lines. This one is about speed, not
  undercharging.
- Break `MOTION_RULES.md`.

## Production note
This is the second film, so the studio is moving to one folder per film
(`films/<slug>/beats.json` + `composition.html`). The launch film is kept
as-is and can be re-rendered.
