import base64
import json
import wave
from pathlib import Path

import requests

from ...utils import ensure_dir
from .base import TTSProvider, distribute_timings


class ElevenLabsTTS(TTSProvider):
    kind = "elevenlabs"

    def synth(self, text, out_path, voice=None, speed=None, emotion=None):
        api_key = self.conf.get("api_key") or ""
        if not api_key:
            raise RuntimeError("ElevenLabs api_key 未配置")
        voice_id = voice or self.voice or "21m00Tcm4TlvDq8ikWAM"
        model = self.conf.get("model") or "eleven_multilingual_v2"
        url = "https://api.elevenlabs.io/v1/text-to-speech/%s" % voice_id
        headers = {"xi-api-key": api_key, "Content-Type": "application/json", "Accept": "audio/mpeg"}
        body = {"text": text, "model_id": model, "voice_settings": {"stability": 0.5, "similarity_boost": 0.75}}
        resp = requests.post(url, headers=headers, json=body, timeout=float(self.conf.get("timeout") or 180))
        if resp.status_code >= 400:
            raise RuntimeError("elevenlabs http %s: %s" % (resp.status_code, resp.text[:300]))
        out_path = Path(out_path)
        ensure_dir(out_path.parent)
        out_path.write_bytes(resp.content)
        duration = self.probe_duration(out_path)
        return {
            "audio_path": str(out_path),
            "duration": round(duration, 3),
            "word_timings": distribute_timings(text, duration),
            "voice": voice_id,
            "provider": "elevenlabs",
            "sample_rate": 44100,
        }


class OpenAITTS(TTSProvider):
    kind = "openai"

    def synth(self, text, out_path, voice=None, speed=None, emotion=None):
        base = (self.conf.get("base_url") or "https://api.openai.com/v1").rstrip("/")
        api_key = self.conf.get("api_key") or ""
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = "Bearer " + api_key
        body = {
            "model": self.conf.get("model") or "gpt-4o-mini-tts",
            "voice": voice or self.voice or "alloy",
            "input": text,
            "response_format": "mp3",
            "speed": float(speed or self.speed or 1.0),
        }
        resp = requests.post(base + "/audio/speech", headers=headers, json=body,
                             timeout=float(self.conf.get("timeout") or 180))
        if resp.status_code >= 400:
            raise RuntimeError("openai tts http %s: %s" % (resp.status_code, resp.text[:300]))
        out_path = Path(out_path)
        ensure_dir(out_path.parent)
        out_path.write_bytes(resp.content)
        duration = self.probe_duration(out_path)
        return {
            "audio_path": str(out_path),
            "duration": round(duration, 3),
            "word_timings": distribute_timings(text, duration),
            "voice": voice or "alloy",
            "provider": "openai",
            "sample_rate": 44100,
        }


class GeminiTTS(TTSProvider):
    kind = "gemini"

    def synth(self, text, out_path, voice=None, speed=None, emotion=None):
        api_key = self.conf.get("api_key") or ""
        if not api_key:
            raise RuntimeError("Gemini api_key 未配置")
        model = self.conf.get("model") or "gemini-2.5-flash-preview-tts"
        base = (self.conf.get("base_url") or "https://generativelanguage.googleapis.com").rstrip("/")
        url = "%s/v1beta/models/%s:generateContent" % (base, model)
        prompt = text
        if emotion:
            prompt = "用%s的语气说：%s" % (emotion, text)
        body = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {"responseModalities": ["AUDIO"]},
        }
        headers = {"Content-Type": "application/json", "x-goog-api-key": api_key}
        resp = requests.post(url, headers=headers, json=body, timeout=float(self.conf.get("timeout") or 180))
        if resp.status_code >= 400:
            raise RuntimeError("gemini tts http %s: %s" % (resp.status_code, resp.text[:300]))
        data = resp.json()
        blob = None
        for part in data.get("candidates", [{}])[0].get("content", {}).get("parts", []):
            inline = part.get("inlineData") or part.get("inline_data") or {}
            if inline.get("data"):
                blob = inline["data"]
                break
        if not blob:
            raise RuntimeError("gemini tts 未返回音频: %s" % json.dumps(data)[:300])
        raw = base64.b64decode(blob)
        out_path = Path(out_path)
        ensure_dir(out_path.parent)
        if out_path.suffix.lower() == ".wav":
            with wave.open(str(out_path), "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(24000)
                wf.writeframes(raw)
        else:
            import subprocess
            import tempfile
            tmp = Path(tempfile.gettempdir()) / "ytsf_gemini_tts.raw"
            tmp.write_bytes(raw)
            subprocess.run(["ffmpeg", "-y", "-f", "s16le", "-ar", "24000", "-ac", "1", "-i", str(tmp),
                            str(out_path)], check=True, capture_output=True)
        duration = self.probe_duration(out_path)
        return {
            "audio_path": str(out_path),
            "duration": round(duration, 3),
            "word_timings": distribute_timings(text, duration),
            "voice": voice or "gemini",
            "provider": "gemini",
            "sample_rate": 24000,
        }
