"""Scores the film to beats.json: music at the set BPM + one SFX per event.

Writes out/soundtrack.wav (48 kHz stereo). If audio/vo.wav exists it is mixed
in and the music ducks under it (Rule 3).
Usage: python3 scripts/score.py [beats.json] [out.wav]
"""
import json
import os
import sys
import wave

import numpy as np

SR = 48000
STUDIO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BEATS_PATH = sys.argv[1] if len(sys.argv) > 1 else os.path.join(STUDIO, "beats.json")
OUT_PATH = sys.argv[2] if len(sys.argv) > 2 else os.path.join(STUDIO, "out", "soundtrack.wav")
B = json.load(open(BEATS_PATH))
# arrangement knobs (defaults = the launch film)
M = {"dropBeats": [15], "stopBeat": 46, "grooveFromBar": 2, "fullFromBar": 4, "arpUntilBar": 11, "padGain": 0.22}
M.update(B.get("music", {}))
SPB = 60 / B["bpm"]
TOTAL_BEATS = B["bars"] * B["beatsPerBar"]
DUR = TOTAL_BEATS * SPB + B["tailSeconds"]
N = int(DUR * SR) + SR  # room for reverb tails
rng = np.random.default_rng(7)

music = np.zeros(N)
sfx = np.zeros(N)


def t(n):
    return np.arange(n) / SR


def env(n, attack=0.002, decay=0.2):
    x = t(n)
    a = np.clip(x / max(attack, 1e-4), 0, 1)
    return a * np.exp(-x / decay)


def place(buf, sig, at_sec, gain=1.0):
    i = int(round(at_sec * SR))
    j = min(len(buf), i + len(sig))
    if i < len(buf):
        buf[i:j] += sig[: j - i] * gain


def lowpass(x, cutoff):
    # one-pole low-pass, good enough for warmth
    a = np.exp(-2 * np.pi * cutoff / SR)
    y = np.zeros_like(x)
    acc = 0.0
    for k in range(len(x)):
        acc = (1 - a) * x[k] + a * acc
        y[k] = acc
    return y


def noise(n):
    return rng.uniform(-1, 1, n)


# ---------- instruments ----------
def kick():
    n = int(0.45 * SR)
    x = t(n)
    f = 45 + 110 * np.exp(-x / 0.03)
    ph = 2 * np.pi * np.cumsum(f) / SR
    return np.sin(ph) * env(n, 0.001, 0.16) + 0.15 * noise(n) * env(n, 0.0005, 0.004)


def hat():
    n = int(0.08 * SR)
    h = noise(n)
    h = h - lowpass(h, 7000)
    return h * env(n, 0.0005, 0.02)


def clap():
    n = int(0.25 * SR)
    c = noise(n)
    c = lowpass(c - lowpass(c, 900), 5000)
    e = env(n, 0.001, 0.06)
    for d in (0.01, 0.02):
        e += np.roll(env(n, 0.001, 0.012), int(d * SR)) * 0.6
    return c * e


def note(freq, length, wave_="saw", decay=0.4, attack=0.01):
    n = int(length * SR)
    x = t(n)
    if wave_ == "sine":
        s = np.sin(2 * np.pi * freq * x)
    else:
        s = sum(np.sin(2 * np.pi * freq * k * x) / k for k in range(1, 12)) * 0.6
    return s * env(n, attack, decay)


def pad(freqs, length):
    n = int(length * SR)
    x = t(n)
    s = np.zeros(n)
    for f in freqs:
        for det in (-0.12, 0, 0.12):
            s += np.sin(2 * np.pi * f * (1 + det / 100) * x + rng.uniform(0, 6))
    a = np.clip(x / 0.6, 0, 1) * np.clip((length - x) / 0.8, 0, 1)
    return lowpass(s / (len(freqs) * 3), 2200) * a


def midi(m):
    return 440 * 2 ** ((m - 69) / 12)


# A minor-ish progression, one chord per bar: Am - F - C - G
CHORDS = [[57, 60, 64], [53, 57, 60], [48, 52, 55], [55, 59, 62]]
ROOTS = [45, 41, 36, 43]

STOP = M["stopBeat"]
for bar in range(B["bars"]):
    t0 = bar * 4 * SPB
    ch = CHORDS[bar % 4]
    place(music, pad([midi(m) for m in ch], 4 * SPB + 0.6), t0, M["padGain"] if bar < B["bars"] - 2 else 0.3)
    for b in range(4):
        bt = t0 + b * SPB
        beat_no = bar * 4 + b
        if beat_no >= STOP:
            continue
        groove = bar >= M["grooveFromBar"] and beat_no not in M["dropBeats"]
        full = bar >= M["fullFromBar"]
        if beat_no not in M["dropBeats"]:
            place(music, kick(), bt, 0.9 if groove else 0.7)
        if groove:
            place(music, hat(), bt + SPB / 2, 0.25)
            if full:
                place(music, hat(), bt + SPB / 4, 0.1)
                place(music, hat(), bt + 3 * SPB / 4, 0.1)
            # sub bass on the 8ths
            place(music, note(midi(ROOTS[bar % 4]), SPB / 2, "sine", 0.18), bt + SPB / 2, 0.45)
            place(music, note(midi(ROOTS[bar % 4]), SPB / 2, "sine", 0.18), bt, 0.3)
        if full and b in (1, 3):
            place(music, clap(), bt, 0.35)
    if M["fullFromBar"] <= bar < M["arpUntilBar"]:
        # arpeggio on 16ths, low-passed
        arp = [m + 12 for m in ch] + [ch[0] + 24]
        seq = np.concatenate([note(midi(arp[k % 4]), SPB / 4, "saw", 0.08) for k in range(16)])
        place(music, lowpass(seq, 2600), t0, 0.06)

# final chord ring-out
place(music, pad([midi(m) for m in [45, 57, 60, 64, 69]], 3.0), STOP * SPB, 0.35)


# ---------- SFX ----------
def sfx_hit():
    k = kick()
    n = len(k)
    return k * 0.6 + lowpass(noise(n), 1800) * env(n, 0.001, 0.05) * 0.6


def sfx_impact():
    n = int(1.6 * SR)
    x = t(n)
    sub = np.sin(2 * np.pi * np.cumsum(30 + 60 * np.exp(-x / 0.12)) / SR) * env(n, 0.001, 0.6)
    crack = lowpass(noise(n), 4000) * env(n, 0.0005, 0.08)
    return sub * 0.9 + crack * 0.5


def sfx_tick():
    n = int(0.05 * SR)
    return np.sin(2 * np.pi * 2400 * t(n)) * env(n, 0.0005, 0.008) + 0.3 * noise(n) * env(n, 0.0002, 0.003)


def sfx_click():
    n = int(0.09 * SR)
    a = np.sin(2 * np.pi * 1800 * t(n)) * env(n, 0.0003, 0.006)
    b = np.roll(np.sin(2 * np.pi * 1200 * t(n)) * env(n, 0.0003, 0.01), int(0.035 * SR))
    return a + 0.7 * b


def sfx_whoosh(length=0.5):
    n = int(length * SR)
    x = t(n)
    w = noise(n)
    out = np.zeros(n)
    seg = n // 16
    for k in range(16):  # sweeping band
        c = 400 + 5000 * np.sin(np.pi * k / 16)
        s = w[k * seg:(k + 1) * seg]
        out[k * seg:(k + 1) * seg] = lowpass(s, c) - lowpass(s, c / 3)
    return out * np.sin(np.pi * np.clip(x / length, 0, 1)) ** 2 * 1.4


def sfx_riser(beats):
    n = int(beats * SPB * SR)
    x = t(n)
    f = 200 * (8 ** (x / x[-1]))
    tone = np.sin(2 * np.pi * np.cumsum(f) / SR) * 0.25
    nz = noise(n)
    nz = nz - lowpass(nz, 1500)
    return (tone + nz * 0.5) * (x / x[-1]) ** 2


def sfx_chime():
    return sum(note(midi(m), 1.5, "sine", 0.5, 0.002) * g for m, g in [(81, 1), (88, 0.6), (93, 0.35)])


def sfx_roll(beats):
    out = np.zeros(int(beats * SPB * SR))
    steps = int(beats * 8)
    for k in range(steps):
        tk = sfx_tick() * (0.4 + 0.6 * k / steps)
        tk = np.interp(np.arange(len(tk)) * (1 + k / steps), np.arange(len(tk)), tk)
        place(out, tk, k * beats * SPB / steps)
    return out


GAIN = {"type": 0.18, "hit": 0.55, "impact": 0.8, "tick": 0.25, "click": 0.4, "whoosh": 0.3, "riser": 0.35, "chime": 0.22, "roll": 0.3}
for e in B["events"]:
    kind, at = e["sfx"], e["beat"] * SPB
    if kind == "riser":
        sig = sfx_riser(e.get("lengthBeats", 4))
    elif kind == "type":
        n16 = int(e.get("lengthBeats", 1) * 4)
        sig = np.zeros(int((n16 + 1) * SPB / 4 * SR))
        for k in range(n16):
            place(sig, sfx_tick() * (0.7 + 0.3 * (k % 2)), k * SPB / 4)
    elif kind == "roll":
        sig = sfx_roll(e.get("lengthBeats", 2))
    elif kind == "whoosh":
        sig = sfx_whoosh()
        at -= 0.25  # whoosh peaks on the event
    else:
        sig = {"hit": sfx_hit, "impact": sfx_impact, "tick": sfx_tick, "click": sfx_click, "chime": sfx_chime}[kind]()
    place(sfx, sig, max(at, 0), GAIN[kind])

# ---------- mix ----------
mix = music * 0.8 + sfx
# launch film: audio/vo.wav; other films: films/<slug>/vo.wav
vo_path = os.path.join(os.path.dirname(BEATS_PATH), "vo.wav") if len(sys.argv) > 1 else os.path.join(STUDIO, "audio", "vo.wav")
if os.path.exists(vo_path):
    with wave.open(vo_path) as w:
        raw = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(float) / 32768
        if w.getnchannels() == 2:
            raw = raw.reshape(-1, 2).mean(axis=1)
        vo = np.interp(np.arange(int(len(raw) * SR / w.getframerate())) * w.getframerate() / SR, np.arange(len(raw)), raw)
    vo_buf = np.zeros(N)
    place(vo_buf, vo, 0.0)
    level = np.convolve(np.abs(vo_buf), np.ones(4800) / 4800, mode="same")
    duck = 1 - 0.6 * np.clip(level * 20, 0, 1)  # ~ -8 dB under speech
    mix = music * 0.8 * duck + sfx + vo_buf
    print("Mixed voiceover from audio/vo.wav")

mix = mix[: int(DUR * SR)]
fade = int(B["tailSeconds"] * SR)
mix[-fade:] *= np.linspace(1, 0, fade) ** 2
mix = np.tanh(mix * 1.1)  # soft clip
mix *= 10 ** (-1 / 20) / np.max(np.abs(mix))  # peak -1 dBFS

# subtle stereo width: music slightly delayed on the right
stereo = np.stack([mix, np.roll(mix, 12) * 0.98 + mix * 0.02], axis=1)
os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
with wave.open(OUT_PATH, "wb") as w:
    w.setnchannels(2)
    w.setsampwidth(2)
    w.setframerate(SR)
    w.writeframes((stereo * 32767).astype(np.int16).tobytes())
print(f"Wrote {os.path.relpath(OUT_PATH, STUDIO)} ({DUR:.1f}s @ {B['bpm']} BPM, {len(B['events'])} cues)")
