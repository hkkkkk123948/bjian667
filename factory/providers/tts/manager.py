from ...log import get_logger
from .base import TTSProvider
from .bailian import BailianTTS
from .edge import EdgeTTS
from .mock import MockTTS
from .rest import ElevenLabsTTS, GeminiTTS, OpenAITTS

_REGISTRY = {
    "mock": MockTTS,
    "edge": EdgeTTS,
    "bailian": BailianTTS,
    "elevenlabs": ElevenLabsTTS,
    "openai": OpenAITTS,
    "gemini": GeminiTTS,
}


class FallbackTTS(TTSProvider):
    kind = "fallback"

    def __init__(self, primary, fallback=None):
        self.primary = primary
        self.fallback = fallback or MockTTS(primary.conf)
        self.conf = primary.conf

    @property
    def kind_name(self):
        return self.primary.kind

    def synth(self, text, out_path, voice=None, speed=None, emotion=None):
        try:
            result = self.primary.synth(text, out_path, voice=voice, speed=speed, emotion=emotion)
            result["provider"] = self.primary.kind
            return result
        except Exception as exc:
            get_logger().warning("TTS %s 失败(%s)，降级为本地 Mock 语音", self.primary.kind, exc)
            result = self.fallback.synth(text, out_path, voice=voice, speed=speed, emotion=emotion)
            result["provider"] = "mock"
            result["fallback_reason"] = str(exc)
            return result


def create_tts(tts_conf, allow_fallback=True):
    conf = dict(tts_conf or {})
    kind = (conf.get("type") or "edge").lower()
    cls = _REGISTRY.get(kind, EdgeTTS)
    primary = cls(conf)
    if kind == "mock":
        return primary
    return FallbackTTS(primary, MockTTS(conf)) if allow_fallback else primary


class TTSRouter:
    def __init__(self, config):
        self.config = config
        self._provider = None

    @property
    def provider(self):
        if self._provider is None:
            self._provider = create_tts(self.config.tts)
        return self._provider

    def invalidate(self):
        self._provider = None

    def set_conf(self, tts_conf):
        self.config.data["tts"] = dict(tts_conf)
        self.invalidate()

    def synth(self, text, out_path, **kwargs):
        seed = kwargs.pop("seed", None)
        if seed is not None:
            conf = dict(self.config.tts, seed=int(seed))
            return create_tts(conf, allow_fallback=False).synth(text, out_path, **kwargs)
        return self.provider.synth(text, out_path, **kwargs)

    def test(self):
        import tempfile
        from pathlib import Path

        try:
            out = Path(tempfile.gettempdir()) / "ytsf_tts_probe.wav"
            result = self.provider.synth("这是一段语音合成测试。", out)
            return True, "%s / %.2fs" % (result.get("provider"), result.get("duration", 0))
        except Exception as exc:
            return False, str(exc)[:200]
