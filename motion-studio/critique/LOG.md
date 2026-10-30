# Critique Log

## Pass 1
| Category | Score | Evidence |
|---|---|---|
| Brief fidelity | 8 | All six sections land on their bars. |
| Sync | 5 | Frames exactly on 4 s, 10 s, 16 s and 20 s are empty: entrances start *on* the beat, so the hit frame is blank. |
| Brand | 9 | Real screenshots; tier prices match the app ($1,156.74 / $1,360.87 / $1,633.04). |
| Typography | 8 | Clean. Contact-sheet timestamps were wrong (tool bug, not film). |
| Composition | 5 | Form shot stuck in the top-left with the right 50% empty; breakdown card too high. |
| Motion quality | 4 | **Cursor clicks ~370 px below the Generate button** (hard-coded target). |
| Banned list | 10 | None. |
| Sound | 8 | Peak −1 dBFS, every cue present. |

**Fixes:** cursor aims at the button's measured on-screen rect each frame;
visual `LEAD` of 0.1 s so beat frames show the hit; form shot centred and
enlarged; breakdown card lowered to align with the panel; contact-sheet labels
fixed.

## Pass 2
| Category | Score | Evidence |
|---|---|---|
| Brief fidelity | 9 | Beat sheet followed. |
| Sync | 9 | Every event frame shows motion; soundtrack onset at every cue is +6 to +24 dB. |
| Brand | 9 | Unchanged. |
| Typography | 8 | Readable at 960 px wide. |
| Composition | 8 | Form centred; breakdown and panel balanced. |
| Motion quality | 8 | Cursor lands on the button at 11.0 s; total rolls up and lands green at 15.5 s. |
| Banned list | 10 | None. |
| Sound | 8 | Peak −1.00 dBFS. |

All categories ≥ 8 → rendered `out/film.mp4`.

**Next ideas (not yet done):** add a voiceover via Fish Audio; mask the
TOTAL row in the breakdown screenshot until the roll-up lands; give the mess
section a stronger collapse transition.

---

# Film 2: fast-materials ("Price it on the driveway"), 128 BPM, 20 s

## Pass 1
| Category | Score | Evidence |
|---|---|---|
| Brief fidelity | 9 | Every beat-sheet row lands on its bar; no stopwatch or time claim. |
| Sync | 8 | Typing ticks on 16ths line up with characters; clicks land on the button frames. |
| Brand | 9 | Cards rebuilt with the app's exact CSS; all numbers real (4.25×60 + 38×2 = $331; $1,360.87). |
| Typography | 8 | Typed fields readable at 1080p. |
| Composition | 8 | Materials card + subtotal balanced. |
| Motion quality | 6 | Second material typed too late: $331 is visible for only ~0.1 s before the cut. |
| Banned list | 10 | None. |
| Sound | 8 | Peak −1 dBFS. |

**Fix:** pulled row-2 typing forward (m2.sum 19.5 → 18.75), so $331 lands and holds before the whip-pan.

## Pass 2
Motion quality 8: subtotal lands $331 at 9.1 s and holds. Cursor hits Generate Estimate at 11.0 s.
All categories ≥ 8 → rendered `out/fast-materials.mp4`.

**Next ideas:** a vertical 9:16 cut for Reels/TikTok; a VO via Fish Audio; a different sample job (drywall, deck).
