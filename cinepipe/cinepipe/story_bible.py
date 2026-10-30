"""Stage 1: Story Bible - interview until the 6 pillars lock, then compile."""
from __future__ import annotations

import json
from typing import Callable

from . import skills_prompts
from .llm import Engine
from .models import InterviewStep, StoryBible

MAX_TURNS = 15


def interview(engine: Engine, idea: str, ask: Callable[[str, list[str]], str], max_turns: int = MAX_TURNS) -> StoryBible:
    """Multi-turn interview. `ask(question, locked_pillars)` returns the user's answer."""
    messages: list[dict] = [{"role": "user", "content": f"Here's my idea for a short film:\n{idea}"}]
    for _ in range(max_turns):
        step = engine.structured(skills_prompts.STORY_BIBLE, messages, InterviewStep)
        if step.bible is not None:
            return step.bible
        answer = ask(step.next_question or "", step.locked_pillars)
        messages += [
            {"role": "assistant", "content": json.dumps(step.model_dump(mode="json", exclude_none=True))},
            {"role": "user", "content": answer},
        ]
    messages[-1]["content"] += "\n\nWe're out of interview turns: lock every pillar now with your best choices and return the compiled bible."
    step = engine.structured(skills_prompts.STORY_BIBLE, messages, InterviewStep)
    if step.bible is None:
        raise RuntimeError("Interview ended without a compiled bible.")
    return step.bible


def context_block(bible: StoryBible) -> str:
    """Canonical bible text injected (and cached) into every downstream stage."""
    return ("STORY BIBLE - canonical. Never contradict it; use its @tags exactly.\n"
            + json.dumps(bible.model_dump(mode="json"), indent=1, sort_keys=True))


def skill_md(bible: StoryBible) -> str:
    """story_bible_skill.md: drop into a Claude Project / CLAUDE.md as persistent universe memory."""
    L = [f"# Story Bible: {bible.title}", "", "Canonical universe for this project. Never contradict it; use the @tags exactly.", "",
         f"**Logline:** {bible.logline}", "", f"**Premise:** {bible.premise}", "", "## Timeline"]
    L += [f"{i}. {e}" for i, e in enumerate(bible.era.timeline, 1)]
    L += ["", "## Era rules"] + [f"- {r}" for r in bible.era.era_rules]
    L += ["", "## Factions & power"] + [f"- **{f.name}**: wants {f.goal}. Leverage: {f.power_over}" for f in bible.factions]
    L += ["", "## World rules (never broken)"] + [f"- {r}" for r in bible.world_rules]
    L += ["", "## Locations"]
    for loc in bible.locations:
        L += [f"### {loc.loc_id}: {loc.name}", f"- Motifs: {', '.join(loc.visual_motifs)}",
              f"- Palette: {', '.join(loc.color_palette)}", f"- Architecture: {loc.architecture}", ""]
    L += ["## Characters"]
    for c in bible.characters:
        L += [f"### {c.char_id}: {c.name} ({c.age})", f"- Archetype: {c.archetype}", f"- Flaw: {c.flaw}", f"- Wants: {c.want}",
              f"- Voice: {c.vocal_cadence}", f"- Ticks: {'; '.join(c.behavioral_ticks)}", ""]
    return "\n".join(L)
