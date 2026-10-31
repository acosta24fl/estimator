# Autopilot plan: photoreal film from a CinePipe project, no manual steps

Goal: `python -m cinepipe produce <project> --budget <USD>` turns a validated
CinePipe project into a finished MP4 with real-looking footage.

## Prerequisites (owner does once, in the cloud environment settings)
- A fal.ai account with credits and an API key, stored as the environment
  secret `FAL_KEY` (never pasted into chat).
- Network access to fal.ai's hosts (docs, API, upload and media CDN).

## Pipeline (to build once docs are reachable; verify every endpoint/field against fal.ai docs first)
1. **Character sheets**: castkit `image_prompt()` → photoreal text-to-image model → `refs/<Tag>.png`.
   Write the paths into `castkit_registry.json` `reference_image_paths`.
2. **Clean plates**: scenecraft `image_prompt()` → same model → `refs/<Plate>.png`.
3. **Keyframes**: for each shot, a multi-reference image model gets the bound
   character sheets + plate + the storyboard frame (composition only) + slots
   04/07/08/09/10 → `keyframes/SH0x0.png`.
4. **Clips**: image-to-video (Kling or Seedance) with the keyframe + slots
   06/12/13/15 as motion prompt, the shot's seed, 480p first → `clips/`.
5. **Automatic critique**: Claude looks at sampled frames of each clip against
   the shot's 16 slots (face matches castkit, no extra people, no text, correct
   action); re-roll a shot (new seed) up to 2 times if it scores < 8.
6. **Upscale pass**: approved shots re-rendered at 1080p with the same seed.
7. **Assemble**: ffmpeg concat in shot order, motion-studio soundtrack,
   end card from `motion-studio/films/cabinet-promo` (render 35-40 s), captions
   for dialogue lines.
8. **Budget guard**: estimate cost from the provider's published prices before
   each paid call; stop before exceeding `--budget`; write `costs.json`.

## Known limits
- Dialogue is shown as captions; lip-synced speech needs a model with audio/lip-sync.
- Faces stay consistent only as well as the provider's reference support allows;
  step 5 catches drift.
