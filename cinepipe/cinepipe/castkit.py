"""Stage 2: Castkit - face-locked, 18% gray turnaround sheets for every character."""
from __future__ import annotations

from . import skills_prompts
from .llm import Engine
from .models import CastkitRegistry, StoryBible
from .story_bible import context_block


def check(bible: StoryBible, reg: CastkitRegistry) -> list[str]:
    want = {c.char_id for c in bible.characters}
    got = [c.char_id for c in reg.characters]
    problems = []
    if missing := want - set(got):
        problems.append(f"Missing castkit entries for {sorted(missing)}.")
    if extra := set(got) - want:
        problems.append(f"Castkit invents characters not in the bible: {sorted(extra)}.")
    if len(got) != len(set(got)):
        problems.append("Duplicate castkit entries.")
    return problems


def build(engine: Engine, bible: StoryBible) -> CastkitRegistry:
    ask = ("Build the castkit registry: one entry per Story Bible character "
           f"({', '.join(c.char_id for c in bible.characters)}).")
    return engine.structured(skills_prompts.CASTKIT, [{"role": "user", "content": ask}], CastkitRegistry,
                             context=context_block(bible), check=lambda r: check(bible, r))


def prompts_md(reg: CastkitRegistry) -> str:
    L = ["# Castkit: reference-sheet prompts", "",
         "Generate each sheet in Midjourney v7 / Flux Pro, save the image, and add its path to",
         "`reference_image_paths` in castkit_registry.json (then re-run `cinepipe shots`).", ""]
    for c in reg.characters:
        L += [f"## {c.char_id}", "", "```", c.image_prompt(), "```", ""]
        for layer in c.wardrobe_layers:
            L += [f"Wardrobe for {layer.context}:", "", "```", c.image_prompt(layer.outfit), "```", ""]
    return "\n".join(L)
