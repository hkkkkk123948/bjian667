import hashlib
from pathlib import Path

import numpy as np

from ...utils import PUNCT_BREAK, PUNCT_PAUSE, estimate_speech_seconds, ensure_dir
from .base import TTSProvider, distribute_timings, write_wav

SR = 44100


def _unit_seed(token, salt=0):
    raw = hashlib.md5((token + str(salt)).encode("utf-8")).digest()
    return int.from_bytes(raw[:4], "big")


class MockTTS(TTSProvider):
    kind = "mock"

    def synth(self, text, out_path, voice=None, speed=None, emotion=None):
        text = (text or "").strip()
        speed = float(speed or self.speed or 1.0)
        target = estimate_speech_seconds(text) / max(0.5, min(2.0, speed))
        samples = self._render(text, target)
        out_path = Path(out_path)
        ensure_dir(out_path.parent)
        write_wav(out_path, samples, SR)
        return {
            "audio_path": str(out_path),
            "duration": round(len(samples) / SR, 3),
            "word_timings": distribute_timings(text, len(samples) / SR),
            "voice": voice or "mock",
            "provider": "mock",
            "sample_rate": SR,
        }

    def _tokens(self, text):
        tokens = []
        buf = ""
        for ch in text:
            if ch in PUNCT_BREAK or ch in PUNCT_PAUSE:
                if buf:
                    tokens.append((buf, False))
                tokens.append((ch, True))
                buf = ""
            else:
                buf += ch
        if buf:
            tokens.append((buf, False))
        return tokens

    def _render(self, text, target_seconds):
        tokens = self._tokens(text)
        if not tokens:
            tokens = [("嗯", False)]
        weights = []
        for tok, is_punct in tokens:
            if is_punct:
                weights.append(0.45 if tok in PUNCT_BREAK else 0.22)
            else:
                weights.append(max(0.18, estimate_speech_seconds(tok)))
        total = sum(weights) or 1.0
        scale = target_seconds / total
        total_samples = int(target_seconds * SR)
        out = np.zeros(total_samples, dtype=np.float64)
        cursor = 0.0
        for (tok, is_punct), weight in zip(tokens, weights):
            dur = weight * scale
            n = int(dur * SR)
            if cursor + n > total_samples:
                n = max(0, total_samples - int(cursor * SR))
            start = int(cursor * SR)
            if n <= 0:
                break
            if is_punct:
                t = np.arange(n) / SR
                tail = np.maximum(0.0, 1.0 - (t / max(1e-6, dur)) * 2.5)
                out[start:start + n] += 0.015 * tail
            else:
                out[start:start + n] += self._syllable(tok, n)
            cursor += dur
        noise = np.random.default_rng(7).normal(0.0, 0.004, total_samples)
        out = out + noise
        peak = float(np.max(np.abs(out))) or 1.0
        out = out / peak * 0.72
        return out

    def _syllable(self, token, n):
        seed = _unit_seed(token)
        f0 = 96.0 + (seed % 90)
        formant = 420.0 + ((seed >> 7) % 1400)
        t = np.arange(n) / SR
        env = np.sin(np.pi * np.clip(np.arange(n) / max(1, n), 0, 1)) ** 0.55
        vib = 1.0 + 0.02 * np.sin(2 * np.pi * 5.5 * t)
        phase = 2 * np.pi * np.cumsum(f0 * vib) / SR
        tone = (
            1.00 * np.sin(phase)
            + 0.45 * np.sin(2 * phase)
            + 0.22 * np.sin(3 * phase)
            + 0.14 * np.sin(4 * phase)
        )
        formant_mod = 1.0 + 0.35 * np.sin(2 * np.pi * formant * t)
        body = tone * formant_mod * env
        breath = np.random.default_rng(seed % 9973).normal(0, 0.05, n) * env
        return (body * 0.42 + breath * 0.5)
