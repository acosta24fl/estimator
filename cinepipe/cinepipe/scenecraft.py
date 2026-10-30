"""Stage 3: Scenecraft - actor-free clean plates with real optics."""
from __future__ import annotations

from . import skills_prompts
from .llm import Engine
from .models import ScenecraftPlates, StoryBible
from .story_bible import context_block


def check(bible: StoryBible, plates: ScenecraftPlates) -> list[str]:
    locs = {l.loc_id for l in bible.locations}
    problems = [f"{p.plate_id} points at unknown location {p.location_ref}." for p in plates.plates if p.location_ref not in locs]
    if uncovered := locs - {p.location_ref for p in plates.plates}:
        problems.append(f"No clean plate for {sorted(uncovered)}.")
    problems += [f"{p.plate_id} must end in _Plate." for p in plates.plates if not p.plate_id.endswith("_Plate")]
    ids = [p.plate_id for p in plates.plates]
    if len(ids) != len(set(ids)):
        problems.append("Duplicate plate ids.")
    return problems


def build(engine: Engine, bible: StoryBible) -> ScenecraftPlates:
    ask = ("Build the clean plates: at least one per Story Bible location "
           f"({', '.join(l.loc_id for l in bible.locations)}).")
    return engine.structured(skills_prompts.SCENECRAFT, [{"role": "user", "content": ask}], ScenecraftPlates,
                             context=context_block(bible), check=lambda p: check(bible, p))


def prompts_md(plates: ScenecraftPlates) -> str:
    L = ["# Scenecraft: clean-plate prompts", "",
         "Generate each plate (Midjourney v7, then Magnific/Topaz for grain), save it, and keep the file",
         "name = plate id (e.g. Bathroom_Plate.png) so Shotcaller can list it as a reference.", ""]
    for p in plates.plates:
        L += [f"## {p.plate_id}  (location {p.location_ref})", "", "```", p.image_prompt(), "```", ""]
    return "\n".join(L)
