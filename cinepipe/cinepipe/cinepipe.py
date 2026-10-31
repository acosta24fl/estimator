"""CinePipe orchestrator: Story Bible -> Castkit -> Scenecraft -> Shotcaller."""
from __future__ import annotations

import shutil
from pathlib import Path
from typing import Optional

import typer
from pydantic import ValidationError
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table

from . import castkit, scenecraft, shotcaller, story_bible, storyboard as storyboard_mod
from .llm import MODEL, Engine, EngineError
from .models import CastkitRegistry, ScenecraftPlates, ShotList, StoryBible
from .state import STAGES, GateError, Workspace

app = typer.Typer(help="CinePipe: a 4-stage pre-production + prompt compiler for AI video.", no_args_is_help=True)
con = Console()
EXAMPLES = Path(__file__).resolve().parent.parent / "examples"

WS = typer.Option(Path("workspace"), "--workspace", "-w", help="Folder that holds all projects.")
FROM = typer.Option(None, "--from-file", help="Skip Claude: import and validate this JSON instead.")


def _ws(root: Path, name: str) -> Workspace:
    ws = Workspace(root, name)
    ws.state()  # fails clearly if the project doesn't exist
    return ws


def _engine(model: str) -> Engine:
    return Engine(model=model)


def _import(path: Path, schema):
    try:
        return schema.model_validate_json(path.read_text(encoding="utf-8"))
    except ValidationError as e:
        raise GateError(f"{path} fails the {schema.__name__} guardrails:\n{e}") from e


def _done(ws: Workspace, stage: str, extra: list[Path] = ()) -> None:
    files = [ws.path(stage), *extra]
    con.print(f"[green]✓ {stage}[/] → " + ", ".join(str(f) for f in files))


def _usage(engine: Engine | None) -> None:
    if engine and any(engine.usage.values()):
        u = engine.usage
        con.print(f"[dim]tokens: in {u['input']:,} · out {u['output']:,} · cache read {u['cache_read']:,} · cache write {u['cache_write']:,}[/]")


# ---------------------------------------------------------------- stages
def run_bible(ws: Workspace, idea: str | None, from_file: Path | None, model: str) -> Engine | None:
    engine = None
    if from_file:
        bible = _import(from_file, StoryBible)
    else:
        if not idea:
            idea = Prompt.ask("[bold]Your film idea in a sentence or two[/]")
        engine = _engine(model)

        def ask(q: str, locked: list[str]) -> str:
            con.print(f"[dim]locked: {', '.join(locked) or 'none'} ({len(locked)}/6)[/]")
            return Prompt.ask(f"[bold cyan]{q}[/]")

        bible = story_bible.interview(engine, idea, ask)
    ws.save("story_bible", bible)
    md = ws.dir / "story_bible_skill.md"
    md.write_text(story_bible.skill_md(bible), encoding="utf-8")
    _done(ws, "story_bible", [md])
    return engine


def run_cast(ws: Workspace, from_file: Path | None, model: str) -> Engine | None:
    bible = ws.load("story_bible")
    engine = None
    if from_file:
        reg = _import(from_file, CastkitRegistry)
        if problems := castkit.check(bible, reg):
            raise GateError("\n".join(problems))
    else:
        engine = _engine(model)
        reg = castkit.build(engine, bible)
    ws.save("castkit", reg)
    md = ws.dir / "castkit_prompts.md"
    md.write_text(castkit.prompts_md(reg), encoding="utf-8")
    _done(ws, "castkit", [md])
    return engine


def run_plates(ws: Workspace, from_file: Path | None, model: str) -> Engine | None:
    bible = ws.load("story_bible")
    engine = None
    if from_file:
        plates = _import(from_file, ScenecraftPlates)
        if problems := scenecraft.check(bible, plates):
            raise GateError("\n".join(problems))
    else:
        engine = _engine(model)
        plates = scenecraft.build(engine, bible)
    ws.save("scenecraft", plates)
    md = ws.dir / "scenecraft_prompts.md"
    md.write_text(scenecraft.prompts_md(plates), encoding="utf-8")
    _done(ws, "scenecraft", [md])
    return engine


def run_shots(ws: Workspace, beats: str | None, from_file: Path | None, model: str) -> Engine | None:
    bible, cast, plates = ws.load("story_bible"), ws.load("castkit"), ws.load("scenecraft")
    engine = None
    if from_file:
        shots = _import(from_file, ShotList)
        if problems := shotcaller.check(shots, cast, plates):
            raise GateError("\n".join(problems))
    else:
        if not beats:
            raise GateError("Shotcaller needs scene beats: --beats 'text' or --beats-file beats.md")
        engine = _engine(model)
        shots = shotcaller.build(engine, bible, cast, plates, beats)
    ws.save("shotcaller", shots)
    _export_shots(ws, shots, cast)
    return engine


def _export_shots(ws: Workspace, shots: ShotList, cast: CastkitRegistry) -> None:
    out = ws.dir / "shots"
    if out.exists():
        shutil.rmtree(out)
    out.mkdir()
    for s in shots.shots:
        (out / f"{s.slot_01_shot_id_duration.shot_id}.txt").write_text(shotcaller.shot_file(s, cast), encoding="utf-8")
    _done(ws, "shotcaller", [out])


def _beats(beats: str | None, beats_file: Path | None) -> str | None:
    return beats_file.read_text(encoding="utf-8") if beats_file else beats


# ---------------------------------------------------------------- commands
@app.command()
def new(name: str, workspace: Path = WS):
    """Create a project folder."""
    ws = Workspace(workspace, name)
    ws.create()
    con.print(f"Created [bold]{ws.dir}[/]. Next: [cyan]cinepipe bible {name}[/]")


@app.command()
def bible(name: str, idea: Optional[str] = typer.Option(None, help="Seed idea for the interview."),
          from_file: Optional[Path] = FROM, workspace: Path = WS, model: str = MODEL):
    """Stage 1: interview until all 6 pillars lock, then compile the Story Bible."""
    _guard(lambda: _usage(run_bible(_ws(workspace, name), idea, from_file, model)))


@app.command()
def cast(name: str, from_file: Optional[Path] = FROM, workspace: Path = WS, model: str = MODEL):
    """Stage 2: face-locked 18% gray turnaround specs for every character."""
    _guard(lambda: _usage(run_cast(_ws(workspace, name), from_file, model)))


@app.command()
def plates(name: str, from_file: Optional[Path] = FROM, workspace: Path = WS, model: str = MODEL):
    """Stage 3: actor-free clean plates with real optics."""
    _guard(lambda: _usage(run_plates(_ws(workspace, name), from_file, model)))


@app.command()
def shots(name: str, beats: Optional[str] = typer.Option(None, help="Scene beats as text."),
          beats_file: Optional[Path] = typer.Option(None, help="Scene beats from a file."),
          from_file: Optional[Path] = FROM, workspace: Path = WS, model: str = MODEL):
    """Stage 4: compile beats into 16-slot prompts (480p preview tier)."""
    _guard(lambda: _usage(run_shots(_ws(workspace, name), _beats(beats, beats_file), from_file, model)))


@app.command()
def run(name: str, idea: Optional[str] = typer.Option(None), beats: Optional[str] = typer.Option(None),
        beats_file: Optional[Path] = typer.Option(None), workspace: Path = WS, model: str = MODEL):
    """Run all four stages in order (creates the project if needed)."""
    def go():
        ws = Workspace(workspace, name)
        ws.create()
        for e in (run_bible(ws, idea, None, model), run_cast(ws, None, model), run_plates(ws, None, model)):
            _usage(e)
        b = _beats(beats, beats_file) or Prompt.ask("[bold]Scene beats (what happens, in order)[/]")
        _usage(run_shots(ws, b, None, model))
    _guard(go)


@app.command()
def approve(name: str, shot_id: str, seed: Optional[int] = typer.Option(None, help="Seed of the preview you approved (defaults to the shot's seed)."),
            workspace: Path = WS):
    """Approve a 480p preview and unlock 1080p_master_render for that shot."""
    def go():
        ws = _ws(workspace, name)
        shots_, cast_ = ws.load("shotcaller"), ws.load("castkit")
        for i, s in enumerate(shots_.shots):
            if s.slot_01_shot_id_duration.shot_id == shot_id:
                data = s.model_dump()
                if seed is not None:
                    data["seed"] = seed
                data["approved_preview_seed"] = data["seed"]
                data["slot_02_render_tier"] = "1080p_master_render"
                shots_.shots[i] = type(s).model_validate(data)
                break
        else:
            raise GateError(f"No shot {shot_id}.")
        ws.save("shotcaller", shots_)
        _export_shots(ws, shots_, cast_)
        con.print(f"[green]{shot_id} unlocked for 1080p (seed {shots_.shots[i].seed}).[/]")
    _guard(go)


@app.command()
def status(name: str, workspace: Path = WS):
    """Show which stages are done, stale, or missing."""
    def go():
        t = Table(title=f"CinePipe: {name}")
        t.add_column("stage"); t.add_column("status")
        for stage, st in _ws(workspace, name).status():
            t.add_row(stage, f"[green]{st}[/]" if st == "ok" else f"[yellow]{st}[/]")
        con.print(t)
    _guard(go)


@app.command()
def brief(name: str, workspace: Path = WS):
    """Write a motion-studio director's brief (story + characters + beat sheet) from the pipeline."""
    def go():
        ws = _ws(workspace, name)
        b, c, s = ws.load("story_bible"), ws.load("castkit"), ws.load("shotcaller")
        p = ws.dir / "director_brief.md"
        p.write_text(director_brief(b, c, s), encoding="utf-8")
        con.print(f"[green]✓[/] {p}")
    _guard(go)


@app.command()
def storyboard(name: str, frames: Optional[Path] = typer.Option(None, help="Folder of SH010.jpg, SH020.jpg... storyboard frames."),
               workspace: Path = WS):
    """Write production_pack.html: every prompt with copy buttons, plus storyboard frames."""
    def go():
        ws = _ws(workspace, name)
        b, c, p, s = ws.load("story_bible"), ws.load("castkit"), ws.load("scenecraft"), ws.load("shotcaller")
        fr = frames or next((d for d in (ws.dir / "frames", EXAMPLES / name / "frames") if d.is_dir()), None)
        out = ws.dir / "production_pack.html"
        out.write_text(storyboard_mod.build(b, c, p, s, fr), encoding="utf-8")
        con.print(f"[green]✓[/] {out}" + ("" if fr else "  [dim](no frames folder found)[/]"))
    _guard(go)


@app.command()
def demo(workspace: Path = WS):
    """Run the whole pipeline offline on the built-in handyman example (no API key needed)."""
    def go():
        src = EXAMPLES / "handyman"
        ws = Workspace(workspace, "handyman-demo")
        ws.create()
        run_bible(ws, None, src / "story_bible.json", MODEL)
        run_cast(ws, src / "castkit_registry.json", MODEL)
        run_plates(ws, src / "scenecraft_plates.json", MODEL)
        run_shots(ws, None, src / "shotcaller_shots.json", MODEL)
        (ws.dir / "director_brief.md").write_text(director_brief(ws.load("story_bible"), ws.load("castkit"), ws.load("shotcaller")), encoding="utf-8")
        first = sorted((ws.dir / "shots").glob("*.txt"))[0]
        con.print(Panel(first.read_text(encoding="utf-8"), title=f"{first.name}: compiled 16-slot prompt"))
    _guard(go)


def director_brief(b: StoryBible, c: CastkitRegistry, s: ShotList) -> str:
    L = [f"# Director's Brief: {b.title}", "", "## Film in one line", b.logline, "", "## Premise", b.premise, "", "## Character bible"]
    for ch in b.characters:
        ck = c.get(ch.char_id)
        L += [f"**{ch.char_id} {ch.name}** ({ch.age}, {ch.archetype}). Wants: {ch.want.rstrip('.')}. Flaw: {ch.flaw.rstrip('.')}.",
              f"- Face: {ck.craniofacial_lock}", f"- Body: {ck.body_lock}", f"- Wardrobe: {ck.wardrobe_default}",
              f"- Voice: {ch.vocal_cadence}", f"- Ticks: {'; '.join(ch.behavioral_ticks)}", ""]
    L += ["## Beat sheet", "| Time | Shot | Action | Dialogue |", "|---|---|---|---|"]
    t = 0.0
    for sh in s.shots:
        d = sh.slot_01_shot_id_duration.duration_s
        dlg = " / ".join(f"{x.active_speaker_id}: \"{x.spoken_line}\"" for x in sh.slot_14_serialized_dialogue_gate) or "-"
        L.append(f"| {t:.1f}-{t + d:.1f}s | {sh.slot_01_shot_id_duration.shot_id} {sh.slot_04_shot_scale_framing} | "
                 f"{' '.join(sh.slot_12_timecoded_action_beats)} | {dlg} |")
        t += d
    L += ["", "## World rules (must not break)"] + [f"- {r}" for r in b.world_rules]
    return "\n".join(L) + "\n"


def _guard(fn):
    try:
        fn()
    except (GateError, EngineError) as e:
        con.print(f"[red]✗ {e}[/]")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
