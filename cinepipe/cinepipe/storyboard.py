"""Production pack: one self-contained HTML page with every prompt (copy buttons) and storyboard frames."""
from __future__ import annotations

import base64
import html
from pathlib import Path

from . import shotcaller
from .models import CastkitRegistry, ScenecraftPlates, ShotList, StoryBible

CSS = """
:root{--bg:#14110e;--card:#1e1a16;--line:#3a332b;--text:#efe9df;--muted:#a89f92;--brass:#d9bb7c;--ok:#7fc77a}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:16px/1.55 Inter,'Segoe UI',system-ui,sans-serif}
main{max-width:1100px;margin:0 auto;padding:32px 16px 80px}h1{font-size:40px;margin:0 0 4px;letter-spacing:-.02em}
h2{font-size:26px;margin:48px 0 6px;border-top:1px solid var(--line);padding-top:28px}h3{margin:0 0 6px;font-size:20px}
.muted{color:var(--muted)}.kicker{color:var(--brass);font-weight:700;letter-spacing:.14em;text-transform:uppercase;font-size:13px}
.card{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:18px;margin:16px 0}
.shot{display:grid;grid-template-columns:minmax(0,1.1fr) minmax(0,1fr);gap:18px}@media(max-width:800px){.shot{grid-template-columns:1fr}}
.shot img{width:100%;border-radius:10px;display:block;border:1px solid var(--line)}
pre{white-space:pre-wrap;word-break:break-word;background:#0e0c0a;border:1px solid var(--line);border-radius:10px;padding:12px;font:13px/1.5 'JetBrains Mono',Consolas,monospace;max-height:340px;overflow:auto;margin:8px 0 0}
button{background:var(--brass);color:#1a140d;border:0;border-radius:8px;padding:8px 14px;font-weight:700;cursor:pointer}
button.done{background:var(--ok)}ol li{margin:6px 0}.tag{font-family:'JetBrains Mono',monospace;color:var(--brass)}
.meta{display:grid;grid-template-columns:auto 1fr;gap:4px 12px;font-size:14px}.meta b{color:var(--muted);font-weight:600}
.say{border-left:3px solid var(--brass);padding-left:10px;margin:6px 0}
"""
JS = """document.addEventListener('click',e=>{const b=e.target.closest('button[data-copy]');if(!b)return;
const t=document.getElementById(b.dataset.copy).textContent;
const ok=()=>{b.textContent='Copied';b.classList.add('done');setTimeout(()=>{b.textContent='Copy';b.classList.remove('done')},1500)};
if(navigator.clipboard&&window.isSecureContext)navigator.clipboard.writeText(t).then(ok);else{const a=document.createElement('textarea');a.value=t;document.body.appendChild(a);a.select();document.execCommand('copy');a.remove();ok()}});"""


def _img(path: Path) -> str:
    if not path.exists():
        return '<p class="muted">(no storyboard frame)</p>'
    data = base64.b64encode(path.read_bytes()).decode()
    return f'<img alt="{html.escape(path.stem)} storyboard frame" src="data:image/jpeg;base64,{data}">'


def _prompt(pid: str, text: str) -> str:
    return f'<button data-copy="{pid}">Copy</button><pre id="{pid}">{html.escape(text)}</pre>'


def build(bible: StoryBible, cast: CastkitRegistry, plates: ScenecraftPlates, shots: ShotList, frames: Path | None) -> str:
    e = html.escape
    total = sum(s.slot_01_shot_id_duration.duration_s for s in shots.shots)
    out = [f"<!doctype html><html lang=en><head><meta charset=utf-8><meta name=viewport content='width=device-width,initial-scale=1'>"
           f"<title>{e(bible.title)} Production Pack</title><style>{CSS}</style></head><body><main>",
           f"<div class=kicker>CinePipe production pack</div><h1>{e(bible.title)}</h1><p class=muted>{e(bible.logline)}</p>",
           f"<p>{len(shots.shots)} shots · {total:.1f} s of footage · characters {', '.join(f'<span class=tag>{e(c.char_id)}</span>' for c in cast.characters)}</p>",
           "<div class=card><h3>How to use this pack</h3><ol>"
           "<li><b>Character sheets:</b> paste each prompt in Section 1 into Midjourney v7 / Flux / Ideogram. Keep the best sheet per character and name it after the tag (e.g. <span class=tag>Installer.png</span>).</li>"
           "<li><b>Clean plates:</b> do the same for Section 2 (e.g. <span class=tag>Kitchen_Plate.png</span>). No people should appear. Regenerate if any do.</li>"
           "<li><b>Shots:</b> in Seedance / Kling / Runway / Higgsfield, upload the reference images listed on each shot, paste its prompt, and render at <b>480p</b> first.</li>"
           "<li>When a preview looks right, approve it (<span class=tag>python -m cinepipe approve &lt;project&gt; SH0x0</span>) and render that shot at 1080p with the same seed.</li>"
           "<li>Edit the clips in order, add music, and finish with the end card from the motion-studio film.</li></ol>"
           "<p class=muted>The storyboard frames are composition guides from the animated version, not the final look. Aim for real photography in the same layout.</p></div>"]

    out.append("<h2>1 · Character reference sheets</h2><p class=muted>Flat 18% gray background, 3 panels: front, back, close-up portrait.</p>")
    for i, c in enumerate(cast.characters):
        prof = next(p for p in bible.characters if p.char_id == c.char_id)
        out.append(f"<div class=card><h3><span class=tag>{e(c.char_id)}</span> {e(prof.name)}, {prof.age}</h3>"
                   f"<p class=muted>{e(prof.archetype)}. {e(prof.want)}</p>{_prompt(f'c{i}', c.image_prompt())}</div>")

    out.append("<h2>2 · Clean plates (empty sets)</h2>")
    for i, p in enumerate(plates.plates):
        out.append(f"<div class=card><h3><span class=tag>{e(p.plate_id)}</span></h3><p class=muted>{e(p.optics())}</p>{_prompt(f'p{i}', p.image_prompt())}</div>")

    out.append("<h2>3 · Shots</h2>")
    t = 0.0
    for i, s in enumerate(shots.shots):
        h = s.slot_01_shot_id_duration
        dlg = "".join(f'<div class=say><span class=tag>{e(d.active_speaker_id)}</span> "{e(d.spoken_line)}" '
                      f'<span class=muted>{d.start_sec:.1f}-{d.end_sec:.1f}s</span></div>' for d in s.slot_14_serialized_dialogue_gate)
        refs = "".join(f"<li>{e(r)}</li>" for r in shotcaller.references(s, cast))
        frame = _img(frames / f"{h.shot_id}.jpg") if frames else ""
        out.append(
            f"<div class=card><div class=kicker>{h.shot_id} · {t:.1f}-{t + h.duration_s:.1f} s · seed {s.seed}</div>"
            f"<div class=shot><div>{frame}</div><div><div class=meta>"
            f"<b>Framing</b><span>{e(s.slot_04_shot_scale_framing)}</span><b>Lens</b><span>{e(s.slot_05_camera_optics)}</span>"
            f"<b>Camera</b><span>{e(s.slot_06_camera_rig_motion)}</span><b>Action</b><span>{e(' '.join(s.slot_12_timecoded_action_beats))}</span></div>"
            f"{dlg}<p class=muted style='margin:10px 0 2px'>Upload:</p><ul style='margin:0;padding-left:20px'>{refs}</ul></div></div>"
            f"{_prompt(f's{i}', s.compile(shotcaller.resolve_wardrobe(s, cast)))}</div>")
        t += h.duration_s
    out.append(f"</main><script>{JS}</script></body></html>")
    return "".join(out)
