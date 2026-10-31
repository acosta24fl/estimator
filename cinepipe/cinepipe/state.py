"""Workspace + stage gating.

A stage only runs on upstream artifacts that exist, pass their schema, and
haven't changed since the stage last ran (sha256). Editing the Story Bible
marks Castkit, Scenecraft and Shotcaller stale until they're re-run.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from .models import CastkitRegistry, ProjectState, ScenecraftPlates, ShotList, StageRecord, StoryBible

T = TypeVar("T", bound=BaseModel)

STAGES: dict[str, tuple[str, type[BaseModel]]] = {
    "story_bible": ("story_bible.json", StoryBible),
    "castkit": ("castkit_registry.json", CastkitRegistry),
    "scenecraft": ("scenecraft_plates.json", ScenecraftPlates),
    "shotcaller": ("shotcaller_shots.json", ShotList),
}
UPSTREAM = {
    "story_bible": [],
    "castkit": ["story_bible"],
    "scenecraft": ["story_bible"],
    "shotcaller": ["story_bible", "castkit", "scenecraft"],
}
COMMAND = {"story_bible": "bible", "castkit": "cast", "scenecraft": "plates", "shotcaller": "shots"}


class GateError(RuntimeError):
    pass


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Workspace:
    def __init__(self, root: Path, name: str):
        self.dir = root / name
        self.name = name
        self.state_path = self.dir / "state.json"

    def create(self) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        if not self.state_path.exists():
            self._write_state(ProjectState(name=self.name))

    def state(self) -> ProjectState:
        if not self.state_path.exists():
            raise GateError(f"No project '{self.name}' in {self.dir.parent}. Run: cinepipe new {self.name}")
        return ProjectState.model_validate_json(self.state_path.read_text(encoding="utf-8"))

    def _write_state(self, st: ProjectState) -> None:
        self.state_path.write_text(st.model_dump_json(indent=2), encoding="utf-8")

    def path(self, stage: str) -> Path:
        return self.dir / STAGES[stage][0]

    def save(self, stage: str, artifact: BaseModel) -> Path:
        """Write a validated artifact and record the upstream hashes it was built from."""
        self.require_upstream(stage)
        p = self.path(stage)
        p.write_text(json.dumps(artifact.model_dump(mode="json"), indent=2, sort_keys=False) + "\n", encoding="utf-8")
        st = self.state()
        st.stages[stage] = StageRecord(
            path=p.name, sha256=_sha(p),
            upstream_sha256={u: _sha(self.path(u)) for u in UPSTREAM[stage]},
        )
        self._write_state(st)
        return p

    def load(self, stage: str, schema: type[T] | None = None) -> T:
        """Load a stage artifact, refusing if it's missing, invalid, or built on stale upstream."""
        p = self.path(stage)
        if not p.exists():
            raise GateError(f"{stage} has not been run yet. Run: cinepipe {COMMAND[stage]} {self.name}")
        try:
            artifact = (schema or STAGES[stage][1]).model_validate_json(p.read_text(encoding="utf-8"))
        except ValidationError as e:
            raise GateError(f"{p.name} fails its schema (was it hand-edited?):\n{e}") from e
        stale = self.stale_upstream(stage)
        if stale:
            raise GateError(f"{stage} is stale: {', '.join(stale)} changed since it ran. Re-run: cinepipe {COMMAND[stage]} {self.name}")
        return artifact  # type: ignore[return-value]

    def require_upstream(self, stage: str) -> None:
        for u in UPSTREAM[stage]:
            self.load(u)

    def stale_upstream(self, stage: str) -> list[str]:
        rec = self.state().stages.get(stage)
        if not rec:
            return []
        return [u for u, h in rec.upstream_sha256.items() if not self.path(u).exists() or _sha(self.path(u)) != h]

    def status(self) -> list[tuple[str, str]]:
        out = []
        st = self.state()
        for stage in STAGES:
            if stage not in st.stages or not self.path(stage).exists():
                out.append((stage, "not run"))
                continue
            try:
                self.load(stage)
                out.append((stage, "ok"))
            except GateError as e:
                out.append((stage, str(e).splitlines()[0]))
        return out
