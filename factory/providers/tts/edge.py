import asyncio
import urllib.request
from pathlib import Path

from ...utils import ensure_dir
from .base import TTSProvider


class EdgeTTS(TTSProvider):
    kind = "edge"

    def synth(self, text, out_path, voice=None, speed=None, emotion=None):
        import edge_tts

        text = (text or "").strip()
        out_path = Path(out_path)
        ensure_dir(out_path.parent)
        voice = voice or self.voice or "zh-CN-YunxiNeural"
        speed = float(speed or self.speed or 1.0)
        rate = "%+d%%" % int(round((speed - 1.0) * 100))
        boundaries = []
        proxy = self.conf.get("proxy") or urllib.request.getproxies().get("https") or urllib.request.getproxies().get("http")

        async def _run():
            communicate = edge_tts.Communicate(text, voice, rate=rate, boundary="WordBoundary",
                                              proxy=proxy, connect_timeout=10, receive_timeout=40)
            with open(out_path, "wb") as fh:
                async for chunk in communicate.stream():
                    kind = chunk.get("type")
                    if kind == "audio":
                        fh.write(chunk.get("data") or b"")
                    elif kind == "WordBoundary":
                        boundaries.append({
                            "text": chunk.get("text", ""),
                            "start": float(chunk.get("offset", 0)) / 1e7,
                            "end": float(chunk.get("offset", 0) + chunk.get("duration", 0)) / 1e7,
                        })

        async def _bounded():
            await asyncio.wait_for(_run(), timeout=float(self.conf.get("timeout") or 60))

        asyncio.run(_bounded())
        if not out_path.exists() or out_path.stat().st_size < 128:
            raise RuntimeError("edge-tts 输出为空")
        duration = self.probe_duration(out_path) or 0.0
        timings = [{"text": b["text"], "start": b["start"], "end": min(duration, b["end"])} for b in boundaries]
        return {
            "audio_path": str(out_path),
            "duration": round(duration, 3),
            "word_timings": timings,
            "voice": voice,
            "provider": "edge",
            "sample_rate": 48000,
            "playback_speed": speed,
            "timing_source": "provider_word_timestamps" if timings else "estimated",
        }
