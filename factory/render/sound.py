"""Deterministic original suspense ambience, kept well below the narration."""
import math
from pathlib import Path

import numpy as np

from ..providers.tts.base import write_wav

# Role → layered cue kinds. All tones are synthesized locally (no borrowed music).
_ROLE_CUES = {
    "hook": ("door_pulse",),
    "escalation": ("heartbeat", "whoosh"),
    "twist": ("whoosh", "hit", "heartbeat"),
    "payoff": ("hit", "door_pulse"),
    "爆点": ("whoosh", "hit"),
    "转折": ("whoosh", "hit", "heartbeat"),
}


def make_soundbed(timeline, path, level_db=-29.0):
    sr = 24000
    duration = float(timeline["source_duration"])
    t = np.arange(math.ceil(duration * sr), dtype=np.float64) / sr
    rng = np.random.default_rng(20261008)
    # Low dissonant drone, slow movement, softened air, and a duckable pad.
    noise = rng.normal(0, 1, len(t))
    noise = np.convolve(noise, np.ones(80) / 80, mode="same")
    bed = (0.32 * np.sin(2 * np.pi * 55 * t + .3 * np.sin(t * .7))
           + .15 * np.sin(2 * np.pi * 58.3 * t)
           + .09 * np.sin(2 * np.pi * 110 * t)
           + .07 * np.sin(2 * np.pi * 41 * t + .2 * np.sin(t * .45))
           + noise * .55)
    # Very low pad (BGM-like), kept quieter than the drone so VO stays dominant.
    bed += .07 * np.sin(2 * np.pi * 73 * t) * (.55 + .45 * np.sin(t * .21) ** 2)
    bed *= .65 + .35 * np.sin(t * .32) ** 2
    cues = []
    for item in timeline.get("sentence_timings", []):
        role = str(item.get("role", ""))
        kinds = _ROLE_CUES.get(role)
        if not kinds:
            continue
        at = float(item["start"])
        rel = t - at
        for kind in kinds:
            if kind == "whoosh":
                env = np.exp(-np.maximum(rel, 0) * 4.5) * ((rel >= -0.05) & (rel < 0.55))
                bed += .28 * env * rng.normal(0, 1, len(t))
                bed += .18 * env * np.sin(2 * np.pi * (180 * np.maximum(rel, 0) + 90))
            elif kind == "hit":
                env = np.exp(-np.maximum(rel, 0) * 9.0) * ((rel >= 0) & (rel < 0.45))
                bed += .55 * env * np.sin(2 * np.pi * (38 * rel - 8 * rel ** 2))
                bed += .22 * env * np.sin(2 * np.pi * 90 * rel)
            elif kind == "heartbeat":
                for beat in (0.0, 0.32, 0.78):
                    br = rel - beat
                    env = np.exp(-np.maximum(br, 0) * 14.0) * ((br >= 0) & (br < 0.22))
                    bed += .20 * env * np.sin(2 * np.pi * (52 * br))
            elif kind == "door_pulse":
                env = np.exp(-np.maximum(rel, 0) * 2.8) * ((rel >= 0) & (rel < 1.4))
                bed += .38 * env * np.sin(2 * np.pi * (47 * rel - 5 * rel ** 2))
            cues.append({"time": at, "role": role, "sound": kind})
    fade = np.minimum(np.minimum(t / .6, (duration - t) / .5), 1).clip(0, 1)
    bed *= fade
    rms = np.sqrt(np.mean(bed * bed)) or 1
    bed *= 10 ** (float(level_db) / 20) / rms
    # Soft ceiling so layered cues cannot clip before the VO limiter.
    peak = np.max(np.abs(bed)) or 1
    if peak > 0.55:
        bed *= 0.55 / peak
    write_wav(path, bed, sr)
    return {"path": str(Path(path).resolve()), "rms_db": float(level_db), "cues": cues,
            "origin": "locally_synthesized_original", "duration": duration,
            "layers": ["drone", "air", "pad"] + sorted({c["sound"] for c in cues})}
