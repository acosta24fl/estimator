"""Google Flow (Veo) pack: step-by-step page for making the clips by hand with minimal work."""
from __future__ import annotations

import html
import json
from pathlib import Path

from .models import ShotList
from .storyboard import CSS, _img

EXTRA_CSS = """
.step{display:flex;gap:12px;align-items:flex-start}.step input{width:22px;height:22px;margin-top:4px;accent-color:#d9bb7c;flex:none}
.chips span{display:inline-block;background:#2a241d;border:1px solid var(--line);border-radius:999px;padding:3px 12px;margin:3px 6px 3px 0;font-weight:600}
.big{font-size:18px}.warn{border-left:3px solid var(--brass);padding:8px 12px;background:#221c16;border-radius:0 8px 8px 0}
"""
JS = """
const KEY='flowpack-done';let done={};try{done=JSON.parse(localStorage.getItem(KEY)||'{}')}catch(e){}
document.querySelectorAll('input[type=checkbox][data-id]').forEach(c=>{c.checked=!!done[c.dataset.id];
 c.addEventListener('change',()=>{done[c.dataset.id]=c.checked;try{localStorage.setItem(KEY,JSON.stringify(done))}catch(e){}})});
document.addEventListener('click',e=>{const b=e.target.closest('button[data-copy]');if(!b)return;
 const t=document.getElementById(b.dataset.copy).textContent;
 const ok=()=>{b.textContent='Copied';b.classList.add('done');setTimeout(()=>{b.textContent='Copy';b.classList.remove('done')},1500)};
 if(navigator.clipboard&&window.isSecureContext)navigator.clipboard.writeText(t).then(ok);else{const a=document.createElement('textarea');a.value=t;document.body.appendChild(a);a.select();document.execCommand('copy');a.remove();ok()}});
"""


def _copy(pid: str, text: str) -> str:
    return f'<button data-copy="{pid}">Copy</button><pre id="{pid}">{html.escape(text)}</pre>'


def build(title: str, veo: dict, shots: ShotList, frames: Path | None, project: str) -> str:
    e = html.escape
    order = [s.slot_01_shot_id_duration.shot_id for s in shots.shots]
    out = [f"<!doctype html><html lang=en><head><meta charset=utf-8><meta name=viewport content='width=device-width,initial-scale=1'>"
           f"<title>{e(title)} Flow Pack</title><style>{CSS}{EXTRA_CSS}</style></head><body><main>",
           f"<div class=kicker>Google Flow · Veo</div><h1>{e(title)}: make the real film</h1>",
           "<p class=muted>About 30-45 minutes. Tick each box as you go; your ticks are remembered in this browser.</p>",
           "<div class='card warn'><b>Save every file with the exact name shown</b> (e.g. <span class=tag>SH010.mp4</span>) "
           f"into the folder <span class=tag>cinepipe\\workspace\\{e(project)}\\clips</span>. The last step joins them automatically.</div>",
           "<h2>Step 1 · Open Flow</h2>",
           "<div class=card><div class=step><input type=checkbox data-id=open><div>Go to <b>labs.google/flow</b>, sign in with your Ultra account, and create a new project called "
           f"<b>{e(title)}</b>. Set the aspect ratio to <b>16:9</b>.</div></div></div>",
           "<h2>Step 2 · Make the 4 reference images</h2>",
           "<p class=muted>These keep the same faces and the same kitchen in every shot. Create each one with Flow's image generation (or the Gemini app → Create image). "
           "Generate, pick the most natural-looking result, and download it with the name shown.</p>"]
    for i, (name, prompt) in enumerate(veo["ingredients"].items()):
        out.append(f"<div class=card><div class=step><input type=checkbox data-id=img-{e(name)}><div style='flex:1;min-width:0'>"
                   f"<h3>{e(name)}.png</h3>{_copy(f'i{i}', prompt)}</div></div></div>")
    out.append("<h2>Step 3 · Make the 10 shots</h2>"
               "<div class=card><ol class=big>"
               "<li>In Flow choose <b>Ingredients to Video</b>.</li>"
               "<li>Add the reference images listed on the shot.</li>"
               "<li>Paste the prompt and generate. Use <b>Veo Fast</b> while trying things out (cheaper), and switch to <b>Veo Quality</b> for the take you keep.</li>"
               "<li>Pick the best take. Check: same faces, no extra people, no text on screen, and lines spoken clearly.</li>"
               "<li>Download it at the highest resolution offered and rename it to the shot name.</li></ol>"
               "<p class=muted>If Ingredients to Video isn't offered for the model you picked, use <b>Frames to Video</b> with the storyboard frame below as the "
               "start frame, and keep the same prompt. Clips are up to 8 seconds; the action happens at the start of each one, and the joining step trims them.</p></div>")
    for i, sid in enumerate(order):
        v = veo["shots"][sid]
        s = shots.shots[i]
        dlg = "".join(f'<div class=say>"{e(d.spoken_line)}"</div>' for d in s.slot_14_serialized_dialogue_gate)
        chips = "".join(f"<span>{e(n)}.png</span>" for n in v["ingredients"])
        frame = _img(frames / f"{sid}.jpg") if frames else ""
        out.append(f"<div class=card><div class=step><input type=checkbox data-id={sid}><div style='flex:1;min-width:0'>"
                   f"<div class=kicker>{sid} · save as {sid}.mp4</div>"
                   f"<div class=shot><div>{frame}</div><div><p style='margin:0 0 4px'>Add these images:</p><div class=chips>{chips}</div>"
                   f"<p class=muted style='margin:10px 0 0'>{e(s.slot_04_shot_scale_framing)}</p>{dlg}</div></div>"
                   f"{_copy(f's{i}', v['prompt'])}</div></div></div>")
    out.append("<h2>Step 4 · Join everything (one command)</h2>"
               "<div class=card><div class=step><input type=checkbox data-id=assemble><div style='flex:1;min-width:0'>"
               "<p>Open Command Prompt and run:</p>"
               + _copy("a0", "cd C:\\Users\\Lex\\estimator\\cinepipe") +
               _copy("a1", f"python -m cinepipe assemble {project}") +
               f"<p>It trims each clip, joins them in order, adds the end card, and saves <span class=tag>workspace\\{e(project)}\\final.mp4</span>.</p>"
               "</div></div></div>")
    out.append(f"</main><script>{JS}</script></body></html>")
    return "".join(out)


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))
