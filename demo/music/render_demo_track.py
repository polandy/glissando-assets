"""Renders the demo slideshow's music: an uplifting pop/corporate piece, synthesised from scratch.

Every sound is computed here, so the track has no third-party rights. Output: a 44.1 kHz stereo WAV.
"""

import sys
import wave

import numpy as np

RATE = 44100
BPM = 112
BEAT = 60 / BPM
BAR = 4 * BEAT
STEP = BEAT / 4
RNG = np.random.default_rng(11)
TABLE_SIZE = 4096

# Chords as (bass root, voicing) in MIDI notes: warm add9/maj7 voicings in the middle register.
CADD9 = (36, [55, 60, 62, 64, 67])
G_B = (35, [55, 59, 62, 67, 69])
AM7 = (33, [55, 57, 60, 64, 67])
FMAJ7 = (29, [53, 57, 60, 64, 65])
DM7 = (38, [53, 57, 60, 62, 65])
GSUS = (31, [55, 60, 62, 67, 69])
VERSE = [CADD9, G_B, AM7, FMAJ7]
BREAK = [AM7, FMAJ7, CADD9, GSUS]
BUILD = [FMAJ7, GSUS, DM7, GSUS]

# (bars, layers, pad brightness from→to in Hz)
SECTIONS = [
    (VERSE, {"pad", "piano", "riser_end", "roll_end"}, (500, 3500)),
    (VERSE * 2, {"pad", "piano", "bass", "kick", "hat", "crash"}, (3500, 3500)),
    (VERSE * 4, {"pad", "piano", "bass", "kick", "snare", "hat", "lead", "crash"}, (5000, 5000)),
    (BREAK, {"pad", "piano"}, (1500, 1500)),
    (BUILD, {"pad", "piano", "bass", "kick", "hat", "riser_end", "roll_end"}, (1500, 5000)),
    (VERSE * 4, {"pad", "piano", "bass", "kick", "snare", "hat", "lead", "crash"}, (5500, 5500)),
    ([CADD9], {"final"}, (3000, 3000)),
]

# Lead hook over the four verse chords, in 16th steps: (step, midi note, steps long).
HOOK = [
    [(0, 72, 3), (3, 74, 3), (6, 76, 4), (10, 79, 2), (12, 76, 4)],
    [(0, 74, 3), (3, 71, 3), (6, 74, 6), (12, 67, 4)],
    [(0, 72, 3), (3, 71, 3), (6, 72, 4), (10, 76, 2), (12, 79, 4)],
    [(0, 77, 3), (3, 76, 3), (6, 72, 4), (10, 74, 6)],
]
# 3-3-2 pop rhythm for the piano chords, in 16th steps.
STABS = [0, 3, 6, 8, 11, 14]


def hz(midi):
    return 440.0 * 2 ** ((midi - 69) / 12)


def t_axis(seconds):
    return np.arange(int(seconds * RATE)) / RATE


def saw_table(harmonics):
    phase = np.arange(TABLE_SIZE) / TABLE_SIZE * 2 * np.pi
    return sum(np.sin(k * phase) / k for k in range(1, harmonics + 1)) * 0.6


def saw(freq, t, brightness, phase0=0.0):
    table = saw_table(max(1, min(40, int(brightness / freq))))
    idx = ((freq * t + phase0) * TABLE_SIZE).astype(np.int64) % TABLE_SIZE
    return table[idx]


def envelope(t, seconds, attack, release):
    return np.minimum(1, t / attack) * np.clip((seconds - t) / release, 0, 1)


def pad(chord, seconds, brightness):
    """Five detuned saws per note, spread across the stereo field."""
    t = t_axis(seconds)
    out = np.zeros((len(t), 2))
    for n in chord:
        for v, detune in enumerate((-0.12, -0.05, 0.0, 0.05, 0.12)):
            wave_ = saw(hz(n + detune), t, brightness, RNG.random())
            pan = (v - 2) / 2 * 0.7
            out[:, 0] += wave_ * np.sqrt(0.5 - pan / 2)
            out[:, 1] += wave_ * np.sqrt(0.5 + pan / 2)
    return out * envelope(t, seconds, 0.05, 0.25)[:, None] * 0.012


def piano(midi, seconds, velocity):
    t = t_axis(seconds)
    f = hz(midi)
    out = np.zeros_like(t)
    for k in range(1, 7):
        partial = f * k * np.sqrt(1 + 0.0003 * k * k)
        out += np.sin(2 * np.pi * partial * t) * np.exp(-(2 + 1.4 * k) * t) / k**1.6
    return out * envelope(t, seconds, 0.003, 0.04) * velocity


def lead(midi, seconds):
    """A plucked saw doubled an octave down, its brightness falling after the attack."""
    t = t_axis(seconds + 0.15)
    bright = saw(hz(midi), t, 6000) * np.exp(-t * 6)
    body = saw(hz(midi), t, 1800) + 0.5 * saw(hz(midi - 12), t, 1200)
    return (bright * 0.5 + body * 0.5) * envelope(t, seconds + 0.15, 0.005, 0.15) * 0.09


def bass(midi, seconds):
    t = t_axis(seconds)
    f = hz(midi)
    tone = np.sin(2 * np.pi * f * t) * 1.0 + saw(f, t, 700) * 0.5
    return tone * envelope(t, seconds, 0.004, 0.03) * np.exp(-t * 1.5) * 0.28


def kick():
    t = t_axis(0.4)
    freq = 48 + 110 * np.exp(-t * 32)
    body = np.sin(2 * np.pi * np.cumsum(freq) / RATE) * np.exp(-t * 6)
    click = RNG.uniform(-1, 1, len(t)) * np.exp(-t * 300) * 0.3
    return (body + click) * 0.9


def snare():
    t = t_axis(0.3)
    tone = np.sin(2 * np.pi * 185 * t) * np.exp(-t * 25)
    noise = highpass(RNG.uniform(-1, 1, len(t)), 1500) * np.exp(-t * 14)
    return (tone * 0.5 + noise * 0.6) * 0.45


def hat(accent):
    t = t_axis(0.05)
    return highpass(RNG.uniform(-1, 1, len(t)), 8000) * np.exp(-t * 70) * (0.09 if accent else 0.05)


def crash():
    t = t_axis(2.5)
    return highpass(RNG.uniform(-1, 1, len(t)), 5000) * np.exp(-t * 1.8) * 0.12


def riser(seconds):
    t = t_axis(seconds)
    noise = highpass(RNG.uniform(-1, 1, len(t)), 2500)
    return noise * (t / seconds) ** 2.5 * 0.18


def highpass(x, cutoff):
    spectrum = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(len(x), 1 / RATE)
    spectrum *= 1 / np.sqrt(1 + (cutoff / np.maximum(freqs, 1)) ** 4)
    return np.fft.irfft(spectrum, len(x))


def place(track, sound, at_seconds, pan=0.0):
    start = int(at_seconds * RATE)
    end = min(len(track), start + len(sound))
    if end <= start:
        return
    if sound.ndim == 2:
        track[start:end] += sound[: end - start]
        return
    track[start:end, 0] += sound[: end - start] * np.sqrt(0.5 - pan / 2)
    track[start:end, 1] += sound[: end - start] * np.sqrt(0.5 + pan / 2)


def reverb(dry, seconds=1.8, mix=0.25):
    t = t_axis(seconds)
    out = np.empty_like(dry)
    for ch in range(2):
        ir = RNG.normal(0, 1, len(t)) * np.exp(-t * 6.5 / seconds)
        size = 1 << (len(dry) + len(ir) - 1).bit_length()
        wet = np.fft.irfft(np.fft.rfft(dry[:, ch], size) * np.fft.rfft(ir, size), size)[: len(dry)]
        gain = np.sqrt(np.mean(dry[:, ch] ** 2) / max(np.mean(wet**2), 1e-12))
        out[:, ch] = dry[:, ch] + mix * wet * gain
    return out


def render():
    total_bars = sum(len(bars) for bars, _, _ in SECTIONS)
    length = total_bars * BAR + 3
    shape = (int(length * RATE), 2)
    music, pumped, drums = np.zeros(shape), np.zeros(shape), np.zeros(shape)
    pump = np.ones(shape[0])
    kick_s, snare_s, crash_s = kick(), snare(), crash()
    hats = (hat(False), hat(True))
    bar_at = 0
    for bars, layers, (bright_from, bright_to) in SECTIONS:
        for i, (root, chord) in enumerate(bars):
            t0 = bar_at * BAR
            last = i == len(bars) - 1
            bright = bright_from + (bright_to - bright_from) * i / max(1, len(bars) - 1)
            if "pad" in layers:
                place(pumped, pad(chord, BAR + 0.05, bright), t0)
            if "piano" in layers:
                for s in STABS:
                    for n in chord[1:]:
                        place(music, piano(n, STEP * 2.5, 0.07), t0 + s * STEP, -0.2)
            if "bass" in layers:
                for e in range(8):
                    place(pumped, bass(root + (12 if e % 2 else 0), BEAT / 2), t0 + e * BEAT / 2)
            if "kick" in layers:
                for b in range(4):
                    place(drums, kick_s, t0 + b * BEAT)
                    start = int((t0 + b * BEAT) * RATE)
                    dip = 1 - 0.55 * np.exp(-t_axis(BEAT) * 14)
                    pump[start : start + len(dip)] = dip[: len(pump) - start]
            if "snare" in layers:
                for b in (1, 3):
                    place(drums, snare_s, t0 + b * BEAT, 0.05)
            if "hat" in layers:
                for s in range(2, 16, 2):
                    place(drums, hats[s % 4 == 2], t0 + s * STEP, 0.35)
            if "crash" in layers and i == 0:
                place(drums, crash_s, t0, -0.3)
            if "lead" in layers:
                for s, note, steps in HOOK[i % 4]:
                    place(music, lead(note, steps * STEP), t0 + s * STEP, 0.15)
            if "riser_end" in layers and last:
                place(drums, riser(BAR), t0)
            if "roll_end" in layers and last:
                for s in range(8, 16):
                    place(drums, snare_s * (0.3 + 0.07 * (s - 8)), t0 + s * STEP)
            if "final" in layers:
                place(drums, kick_s, t0)
                place(drums, crash_s, t0, -0.3)
                place(music, bass(root, 2.5) * 1.2, t0)
                for n in chord:
                    place(music, piano(n, 3, 0.12), t0)
                place(music, pad(chord, 2.8, 2500) * 1.3, t0)
            bar_at += 1
    mix = reverb(music + pumped * pump[:, None]) + drums
    mix = np.tanh(mix * 1.6) / np.tanh(1.6)
    return mix / np.max(np.abs(mix)) * 0.89


def write_wav(path, audio):
    pcm = (audio * 32767).astype("<i2")
    with wave.open(path, "wb") as out:
        out.setnchannels(2)
        out.setsampwidth(2)
        out.setframerate(RATE)
        out.writeframes(pcm.tobytes())


if __name__ == "__main__":
    write_wav(sys.argv[1], render())
