"""Join downloaded clips (SH010.mp4 ...) into one finished film with ffmpeg."""
from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path

from .models import ShotList

VIDEO_EXT = (".mp4", ".mov", ".webm", ".mkv")
DIALOGUE_EXTRA_S = 2.0  # spoken lines rarely land exactly where planned; keep a little more


class AssembleError(RuntimeError):
    pass


def _need(tool: str) -> str:
    path = shutil.which(tool)
    if not path:
        raise AssembleError(f"{tool} not found. Install it (Windows: winget install Gyan.FFmpeg), then open a new terminal.")
    return path


def _probe(ffprobe: str, f: Path) -> tuple[float, bool]:
    r = subprocess.run([ffprobe, "-v", "error", "-show_entries", "format=duration:stream=codec_type", "-of", "json", str(f)],
                       capture_output=True, text=True)
    if r.returncode:
        raise AssembleError(f"Can't read {f.name}: {r.stderr.strip()}")
    info = json.loads(r.stdout)
    return float(info["format"]["duration"]), any(s.get("codec_type") == "audio" for s in info.get("streams", []))


def find_clips(clips_dir: Path, shot_ids: list[str]) -> list[tuple[str, Path]]:
    files = sorted(p for p in clips_dir.iterdir() if p.suffix.lower() in VIDEO_EXT) if clips_dir.is_dir() else []
    found = []
    for sid in shot_ids:
        match = next((p for p in files if p.stem.upper().startswith(sid)), None)
        if match:
            found.append((sid, match))
    return found


def assemble(shots: ShotList, clips_dir: Path, out: Path, endcard: Path | None, full: bool = False,
             lengths: list[float] | None = None, width: int = 1920, height: int = 1080, fps: int = 30) -> tuple[Path, list[str]]:
    ffmpeg, ffprobe = _need("ffmpeg"), _need("ffprobe")
    plan = {s.slot_01_shot_id_duration.shot_id: s for s in shots.shots}
    clips = find_clips(clips_dir, list(plan))
    if not clips:
        raise AssembleError(f"No clips found in {clips_dir}. Save them as SH010.mp4, SH020.mp4, ...")
    missing = [sid for sid in plan if sid not in {c[0] for c in clips}]
    vf = (f"scale={width}:{height}:force_original_aspect_ratio=decrease,pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:color=black,"
          f"setsar=1,fps={fps},format=yuv420p")
    log = []
    with tempfile.TemporaryDirectory() as tmp:
        parts = []
        items = [(sid, f, plan[sid]) for sid, f in clips] + ([("END", endcard, None)] if endcard and endcard.exists() else [])
        for n, (sid, f, shot) in enumerate(items):
            dur, has_audio = _probe(ffprobe, f)
            if shot is not None and lengths and n < len(lengths):
                keep = min(dur, lengths[n])
            elif shot is None or full:
                keep = dur
            else:
                want = shot.slot_01_shot_id_duration.duration_s + (DIALOGUE_EXTRA_S if shot.slot_14_serialized_dialogue_gate else 0)
                keep = min(dur, want)
            fade = min(0.06, keep / 4)
            af = f"aresample=48000,afade=t=in:d={fade},afade=t=out:st={keep - fade:.3f}:d={fade}"
            part = Path(tmp) / f"{n:03d}.mp4"
            cmd = [ffmpeg, "-y", "-loglevel", "error", "-i", str(f)]
            if not has_audio:
                cmd += ["-f", "lavfi", "-i", "anullsrc=r=48000:cl=stereo"]
            cmd += ["-t", f"{keep:.3f}", "-map", "0:v:0", "-map", "0:a:0" if has_audio else "1:a:0",
                    "-vf", vf, "-af", af, "-c:v", "libx264", "-preset", "medium", "-crf", "18",
                    "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-ac", "2", str(part)]
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode:
                raise AssembleError(f"ffmpeg failed on {f.name}: {r.stderr.strip()[-400:]}")
            parts.append(part)
            log.append(f"{sid}: {f.name} → {keep:.1f}s" + ("" if has_audio else " (no audio, silence added)"))
        lst = Path(tmp) / "list.txt"
        lst.write_text("".join(f"file '{p.as_posix()}'\n" for p in parts), encoding="utf-8")
        out.parent.mkdir(parents=True, exist_ok=True)
        r = subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(lst),
                            "-c", "copy", "-movflags", "+faststart", str(out)], capture_output=True, text=True)
        if r.returncode:
            raise AssembleError(f"Joining failed: {r.stderr.strip()[-400:]}")
    if missing:
        log.append(f"Missing (skipped): {', '.join(missing)}")
    return out, log
