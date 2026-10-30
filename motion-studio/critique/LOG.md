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

---

# Film 3: tiktok-onsite ("Quote it before you leave"), 9:16, 128 BPM, 20 s

## Pass 1
| Category | Score | Evidence |
|---|---|---|
| Brief fidelity | 8 | All beats present: hook bubble on frame 1, phone pull, typing, tap, turn, handshake. |
| Sync | 8 | Every tap lands on its click; typing ticks on 16ths. |
| Brand | 9 | App rebuilt with the real CSS (mobile layout); all numbers real. |
| Typography | 5 | Phone UI too small to read on a phone (~28 px text). |
| Composition | 5 | Caption wraps into the tier cards; cards crowd the handyman's head. |
| Motion quality | 6 | Thumb too fat and covers the fields; crack reads as an "X" close icon. |
| Banned list | 10 | None. |
| Sound | 8 | Peak −1 dBFS. |

**Fixes:** camera zooms 1.5× onto the fields being typed, then 1.35× onto labor, then out for the result; slimmer thumb on an unscaled layer; larger crack at the handyman's eye line; caption on two lines; cards moved down and characters framed lower.

## Pass 2
Typography 8 (fields ~42 px), Composition 7: materials chip collided with the app subtitle.
**Fix:** moved the zoom origin so the app header leaves the frame, and gave the chip a solid fade.

## Pass 3
All categories ≥ 8 → rendered `out/tiktok-onsite.mp4` (1080×1920).

**Next ideas:** a Fish Audio voiceover (TikTok rewards voice); a 3-second loop-friendly ending; burned-in captions for sound-off viewers.

---

# Film 4: cabinet-promo ("Level"), 16:9, 96 BPM, 40 s

## Pass 1
| Category | Score | Evidence |
|---|---|---|
| Brief fidelity | 8 | Story reads shot by shot: crooked DIY job, pro arrives, laser, studs, together, square, gaps, reveal, soft-close, CTA. |
| Sync | 8 | Knock, drill, laser click, bubble chime, light clicks and soft-close thud land on their frames. |
| Brand | 8 | Warm greige/white/brass palette held throughout. |
| Typography | 5 | Kicker labels unreadable; hard-edged scrim rectangle visible behind titles. |
| Composition | 6 | Black edge on the opening push-in; "Square." collides with the shim. |
| Motion quality | 6 | Her hands not on the cabinet; drill sparks far from his hand; flat white swinging doors. |
| Realism | 6 | Textures/lighting/grain good; measurement callouts invisible (white on white); cartoon hand. |
| Sound | 8 | Peak −1 dBFS. |

**Fixes:** full-frame radial scrim; larger bright kickers; callouts as dark chips with red/green status dots; wall/backsplash bleed past frame; title moved for the shim shot; shaker panel + pull drawn on swinging doors; sculpted hand with tapered finger and knuckles; arms and spark positions corrected.

## Pass 2
All categories ≥ 8 except human realism (7 - illustrated figures can't read as photographic). Rendered `out/cabinet-promo.mp4`.
Photoreal path: `cinepipe/examples/cabinet-install` (10 guardrail-validated 16-slot shots for Seedance/Kling).
