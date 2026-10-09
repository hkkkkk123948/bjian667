import wave
from pathlib import Path

from ...utils import estimate_speech_seconds, ensure_dir, ffprobe_bin, run_cmd
from ...config import DEFAULT_TTS_SPEED


class TTSProvider:
    kind = "base"

    def __init__(self, conf=None):
        self.conf = dict(conf or {})
        self.name = self.conf.get("type", self.kind)

    @property
    def voice(self):
        return self.conf.get("voice") or ""

    @property
    def speed(self):
        try:
            return float(self.conf.get("speed") or DEFAULT_TTS_SPEED)
        except (TypeError, ValueError):
            return DEFAULT_TTS_SPEED

    def synth(self, text, out_path, voice=None, speed=None, emotion=None):
        raise NotImplementedError

    def test_connection(self):
        try:
            out = Path(out_path_stub())
            self.synth("测试", out)
            return True, "ok"
        except Exception as exc:
            return False, str(exc)[:200]

    def probe_duration(self, path):
        try:
            out, _, _ = run_cmd([ffprobe_bin(), "-v", "error", "-show_entries", "format=duration",
                                 "-of", "default=noprint_wrappers=1:nokey=1", str(path)], check=False)
            return float(out.strip() or 0.0)
        except Exception:
            return 0.0


def out_path_stub():
    import tempfile
    return Path(tempfile.gettempdir()) / "ytsf_tts_test.wav"


def write_wav(path, samples, sample_rate=44100):
    ensure_dir(Path(path).parent)
    import numpy as np

    data = np.clip(np.asarray(samples, dtype=np.float64), -1.0, 1.0)
    pcm = (data * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())


def distribute_timings(text, total_seconds):
    from ...utils import PUNCT_BREAK, PUNCT_PAUSE

    tokens = []
    buf = ""
    for ch in text:
        if ch in PUNCT_BREAK or ch in PUNCT_PAUSE:
            if buf.strip():
                tokens.append(buf.strip())
            tokens.append(ch)
            buf = ""
        else:
            buf += ch
    if buf.strip():
        tokens.append(buf.strip())
    weights = [max(0.15, estimate_speech_seconds(tok)) for tok in tokens]
    total_weight = sum(weights) or 1.0
    scale = max(0.0, total_seconds) / total_weight
    timings = []
    cursor = 0.0
    for tok, weight in zip(tokens, weights):
        dur = weight * scale
        timings.append({"text": tok, "start": round(cursor, 3), "end": round(cursor + dur, 3)})
        cursor += dur
    return timings
