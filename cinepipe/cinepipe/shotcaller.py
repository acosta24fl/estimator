"""Stage 4: Shotcaller - compile scene beats into 16-slot video prompts."""
from __future__ import annotations

import re

from . import skills_prompts
from .llm import Engine
from .models import CastkitRegistry, ScenecraftPlates, Shotcaller16SlotPrompt, ShotList, StoryBible
from .story_bible import context_block

NGRAM = 4
# phrases that legitimately repeat (tags are referenced everywhere; slot 16 is boilerplate)
SKIP_SLOTS = {"ASSET_BINDINGS", "NEGATIVE_CONSTRAINTS", "SHOT_ID & DURATION", "RENDER_TIER"}


def _grams(text: str) -> set[str]:
    words = re.findall(r"[a-z0-9']+", re.sub(r"@\w+", " ", text.lower()))
    return {" ".join(words[i:i + NGRAM]) for i in range(len(words) - NGRAM + 1)}


def isolation_errors(shot: Shotcaller16SlotPrompt, cast: CastkitRegistry, plates: ScenecraftPlates) -> list[str]:
    """Token isolation: each fact lives in exactly one slot."""
    sid = shot.slot_01_shot_id_duration.shot_id
    problems = []
    slots = dict(shot.slots())
    # repeated 4-word phrases across slots
    seen: dict[str, str] = {}
    for name, text in slots.items():
        if name in SKIP_SLOTS:
            continue
        for g in _grams(text):
            if g in seen and seen[g] != name:
                problems.append(f"{sid}: '{g}' appears in both {seen[g]} and {name}.")
            seen.setdefault(g, name)
    # wardrobe may only live in slot 09
    bound_chars = [t for t in shot.slot_03_asset_bindings if not t.endswith("_Plate")]
    for tag in bound_chars:
        try:
            c = cast.get(tag)
        except KeyError:
            problems.append(f"{sid}: {tag} is not in the castkit registry.")
            continue
        wardrobe = _grams(" ".join([c.wardrobe_default] + [w.outfit for w in c.wardrobe_layers]))
        for name, text in slots.items():
            if name not in ("WARDROBE_LOCK", *SKIP_SLOTS) and (leak := wardrobe & _grams(text)):
                problems.append(f"{sid}: wardrobe of {tag} leaks into {name} ('{sorted(leak)[0]}'). Clothing lives only in slot 09.")
        if tag not in shot.slot_09_wardrobe_lock:
            problems.append(f"{sid}: slot 09 must name {tag}'s locked wardrobe.")
        elif not re.search(re.escape(tag) + r":\s*castkit\s+(wardrobe_default|layer\s+\")", shot.slot_09_wardrobe_lock):
            problems.append(f"{sid}: slot 09 must reference {tag}'s outfit as '{tag}: castkit wardrobe_default' or '{tag}: castkit layer \"<context>\"' - never retype clothing.")
    try:
        resolve_wardrobe(shot, cast)
    except KeyError as e:
        problems.append(f"{sid}: {e.args[0]}")
    # plate binding must exist, and the set description stays in slot 07
    for tag in [t for t in shot.slot_03_asset_bindings if t.endswith("_Plate")]:
        try:
            plate = plates.get(tag)
        except KeyError:
            problems.append(f"{sid}: {tag} is not a Scenecraft plate.")
            continue
        geo = _grams(plate.clean_plate_geometry)
        for name, text in slots.items():
            if name not in ("GEOMETRY & SPATIAL MATTE", *SKIP_SLOTS) and (leak := geo & _grams(text)):
                problems.append(f"{sid}: set description of {tag} leaks into {name} ('{sorted(leak)[0]}').")
    # every speaker and every listener mentioned must be bound
    for seg in shot.slot_14_serialized_dialogue_gate:
        for tag in re.findall(r"@\w+", seg.silent_listeners_state):
            if tag not in shot.slot_03_asset_bindings:
                problems.append(f"{sid}: listener {tag} isn't bound in slot 03.")
    return problems


def check(shots: ShotList, cast: CastkitRegistry, plates: ScenecraftPlates) -> list[str]:
    problems = []
    ids = [s.slot_01_shot_id_duration.shot_id for s in shots.shots]
    if len(ids) != len(set(ids)):
        problems.append("Duplicate shot ids.")
    for s in shots.shots:
        if s.slot_02_render_tier != "480p_preview_draft" or s.approved_preview_seed is not None:
            problems.append(f"{s.slot_01_shot_id_duration.shot_id}: new shots must be 480p_preview_draft with no approval.")
        problems += isolation_errors(s, cast, plates)
    return problems


def build(engine: Engine, bible: StoryBible, cast: CastkitRegistry, plates: ScenecraftPlates, beats: str) -> ShotList:
    assets = "\n".join(
        [f"CASTKIT {c.char_id}: wardrobe_default = {c.wardrobe_default}; layers = "
         f"{[(w.context, w.outfit) for w in c.wardrobe_layers]}" for c in cast.characters]
        + [f"PLATE {p.plate_id} ({p.location_ref}): optics = {p.optics()}; practicals = {p.practical_lights}; "
           f"geometry = {p.clean_plate_geometry}" for p in plates.plates])
    ask = f"Locked assets:\n{assets}\n\nCompile these scene beats into shots:\n{beats}"
    return engine.structured(skills_prompts.SHOTCALLER, [{"role": "user", "content": ask}], ShotList,
                             context=context_block(bible), check=lambda s: check(s, cast, plates))


WARDROBE_REF = re.compile(r"(@\w+):\s*castkit\s+(wardrobe_default|layer\s+\"([^\"]+)\")")


def resolve_wardrobe(shot: Shotcaller16SlotPrompt, cast: CastkitRegistry) -> str:
    """Swap `@Tag: castkit wardrobe_default` / `@Tag: castkit layer "Act 2"` for the locked outfit text."""
    def sub(m: re.Match) -> str:
        c = cast.get(m.group(1))
        if m.group(3):
            outfit = next((w.outfit for w in c.wardrobe_layers if w.context == m.group(3)), None)
            if outfit is None:
                raise KeyError(f"{m.group(1)} has no wardrobe layer '{m.group(3)}'")
        else:
            outfit = c.wardrobe_default
        return f"{m.group(1)}: {outfit}"
    return WARDROBE_REF.sub(sub, shot.slot_09_wardrobe_lock)


def references(shot: Shotcaller16SlotPrompt, cast: CastkitRegistry) -> list[str]:
    out = []
    for tag in shot.slot_03_asset_bindings:
        if tag.endswith("_Plate"):
            out.append(f"{tag}: {tag[1:]}.png (clean plate)")
        else:
            paths = cast.get(tag).reference_image_paths
            out.append(f"{tag}: {', '.join(paths) if paths else '(no reference sheet yet - generate it from castkit_prompts.md)'}")
    return out


def shot_file(shot: Shotcaller16SlotPrompt, cast: CastkitRegistry) -> str:
    refs = "\n".join(f"  {r}" for r in references(shot, cast))
    return (f"# {shot.slot_01_shot_id_duration}  |  {shot.slot_02_render_tier}  |  seed {shot.seed}\n"
            f"# Upload these references and bind them to their @tags:\n{refs}\n\n{shot.compile(resolve_wardrobe(shot, cast))}\n")
