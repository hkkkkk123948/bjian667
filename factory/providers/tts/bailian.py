"""阿里云百炼（DashScope）语音合成。

支持 qwen-tts / 通义千问 TTS 的 HTTP 接口，base_url / api_key / model / voice 均可在配置中填写。
"""

from pathlib import Path

import requests

from ...utils import ensure_dir
from .base import TTSProvider, distribute_timings

DEFAULT_PATH = "/api/v1/services/aigc/multimodal-generation/generation"
DEFAULT_BASE = "https://dashscope.aliyuncs.com"
CUSTOMIZATION_PATH = "/api/v1/services/audio/tts/customization"
DEFAULT_ENROLL_MODEL = "cosyvoice-v2"
WS_URL = "wss://dashscope.aliyuncs.com/api-ws/v1/inference"


def create_voice(api_key, audio_path, prefix="myvoice", target_model=DEFAULT_ENROLL_MODEL,
                 base_url=DEFAULT_BASE, timeout=180):
    """上传参考音频并创建克隆音色，返回 voice_id（同步接口，data URI 方式）。

    prefix 只能包含英文字母和数字（百炼限制），此处自动清洗。
    """
    import base64
    import re
    from pathlib import Path

    if not api_key:
        raise RuntimeError("百炼 api_key 未配置")
    prefix = re.sub(r"[^A-Za-z0-9]", "", str(prefix or "")) or "myvoice"
    if len(prefix) > 16:
        prefix = prefix[:16]
    path = Path(audio_path)
    if not path.exists():
        raise FileNotFoundError("参考音频不存在: %s" % audio_path)
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    url = (base_url or DEFAULT_BASE).rstrip("/") + CUSTOMIZATION_PATH
    body = {"model": "voice-enrollment", "input": {
        "action": "create_voice", "target_model": target_model, "prefix": prefix,
        "url": "data:audio/wav;base64," + data}}
    resp = requests.post(url, headers={"Authorization": "Bearer " + api_key,
                                       "Content-Type": "application/json"}, json=body, timeout=timeout)
    if resp.status_code >= 400:
        detail = resp.text[:300]
        if "prefix" in detail:
            detail += "（音色前缀只能用英文字母和数字）"
        raise RuntimeError("百炼音色克隆失败 http %s: %s" % (resp.status_code, detail))
    voice_id = (resp.json().get("output") or {}).get("voice_id")
    if not voice_id:
        raise RuntimeError("百炼音色克隆未返回 voice_id: %s" % resp.text[:300])
    return voice_id


def delete_voice(api_key, voice_id, base_url=DEFAULT_BASE, target_model=DEFAULT_ENROLL_MODEL, timeout=60):
    """删除已克隆音色（尽力而为；失败不阻塞本地删除）。"""
    if not api_key or not voice_id:
        return False
    url = (base_url or DEFAULT_BASE).rstrip("/") + CUSTOMIZATION_PATH
    body = {"model": "voice-enrollment", "input": {
        "action": "delete_voice", "target_model": target_model, "voice_id": voice_id}}
    try:
        resp = requests.post(url, headers={"Authorization": "Bearer " + api_key,
                                           "Content-Type": "application/json"}, json=body, timeout=timeout)
        return resp.status_code < 400
    except Exception:
        return False


class BailianTTS(TTSProvider):
    kind = "bailian"

    def _endpoint(self):
        base = (self.conf.get("base_url") or DEFAULT_BASE).rstrip("/")
        if "/api/" in base or base.endswith("/generation"):
            return base
        return base + DEFAULT_PATH

    def synth(self, text, out_path, voice=None, speed=None, emotion=None):
        api_key = self.conf.get("api_key") or ""
        if not api_key:
            raise RuntimeError("百炼 api_key 未配置")
        text = (text or "").strip()
        if not text:
            raise RuntimeError("百炼 TTS 文本为空")
        model = self.conf.get("model") or "qwen3-tts-flash"
        voice = self._resolve_voice(voice or self.voice or self.conf.get("voice") or "Cherry")
        model = self._voice_model(voice, model)
        if model.startswith("cosyvoice") or str(voice).startswith("cosyvoice"):
            return self._synth_cosyvoice(text, voice, out_path, speed=speed)
        return self._synth_http(text, voice, model, out_path, speed, emotion)

    def _synth_http(self, text, voice, model, out_path, speed, emotion):
        api_key = self.conf.get("api_key") or ""
        headers = {"Authorization": "Bearer " + api_key, "Content-Type": "application/json"}
        body = {"model": model, "input": {"text": text, "voice": voice}}
        params = {}
        for src, dst in (("rate", "rate"), ("pitch", "pitch"), ("volume", "volume"),
                         ("language", "language_type"), ("emotion", "emotion")):
            val = self.conf.get(src)
            if val not in (None, ""):
                params[dst] = val
        if speed and "rate" not in params:
            params["rate"] = round(float(speed), 2)
        if emotion and "emotion" not in params:
            params["emotion"] = emotion
        if params:
            body["parameters"] = params
        resp = requests.post(self._endpoint(), headers=headers, json=body,
                             timeout=float(self.conf.get("timeout") or 180))
        if resp.status_code >= 400:
            raise RuntimeError("bailian tts http %s: %s" % (resp.status_code, resp.text[:300]))
        data = resp.json()
        kind, value = _extract_audio(data)
        if not value:
            raise RuntimeError("bailian tts 未返回音频: %s" % str(data)[:300])
        out_path = Path(out_path)
        if kind == "url":
            audio = requests.get(value, timeout=float(self.conf.get("timeout") or 180)).content
        else:
            import base64

            audio = base64.b64decode(value)
        target = out_path
        if out_path.suffix.lower() == ".mp3" and audio[:4] == b"RIFF":
            target = out_path.with_suffix(".wav")
        ensure_dir(target.parent)
        target.write_bytes(audio)
        duration = self.probe_duration(target)
        return {
            "audio_path": str(target),
            "duration": round(duration, 3),
            "word_timings": distribute_timings(text, duration),
            "voice": voice,
            "provider": "bailian",
            "sample_rate": 24000,
        }

    def _synth_cosyvoice(self, text, voice, out_path, speed=None):
        import json
        import uuid

        import websocket

        api_key = self.conf.get("api_key") or ""
        ws_url = self.conf.get("ws_url") or WS_URL
        out_path = Path(out_path)
        if out_path.suffix.lower() != ".wav":
            out_path = out_path.with_suffix(".wav")
        ensure_dir(out_path.parent)
        ws = websocket.create_connection(ws_url, header=["Authorization: Bearer " + api_key],
                                         timeout=float(self.conf.get("timeout") or 180))
        task_id = uuid.uuid4().hex
        try:
            ws.send(json.dumps({
                "header": {"action": "run-task", "task_id": task_id, "streaming": "duplex"},
                "payload": {"task_group": "audio", "task": "tts", "function": "SpeechSynthesizer",
                            "model": self.conf.get("cosyvoice_model") or DEFAULT_ENROLL_MODEL,
                            "parameters": {"text_type": "PlainText", "voice": voice,
                                           "format": "wav", "sample_rate": 24000, "rate": float(speed or self.speed),
                                           "word_timestamp_enabled": True,
                                           "seed": int(self.conf.get("seed", 0))},
                            "input": {}}}))
            audio = bytearray()
            timestamp_events = []
            while True:
                msg = ws.recv()
                if isinstance(msg, (bytes, bytearray)):
                    audio.extend(msg)
                    continue
                event = json.loads(msg)
                name = event.get("header", {}).get("event")
                if name == "task-started":
                    ws.send(json.dumps({"header": {"action": "continue-task", "task_id": task_id,
                                                   "streaming": "duplex"},
                                        "payload": {"input": {"text": text}}}))
                    ws.send(json.dumps({"header": {"action": "finish-task", "task_id": task_id,
                                                   "streaming": "duplex"}, "payload": {"input": {}}}))
                elif name == "result-generated":
                    timestamp_events.append(event.get("payload", {}).get("output", {}))
                elif name == "task-finished":
                    break
                elif name == "task-failed":
                    raise RuntimeError("cosyvoice 合成失败: %s" % json.dumps(event, ensure_ascii=False)[:300])
        finally:
            try:
                ws.close()
            except Exception:
                pass
        if not audio:
            raise RuntimeError("cosyvoice 未返回音频")
        out_path.write_bytes(bytes(audio))
        duration = self.probe_duration(out_path)
        return {
            "audio_path": str(out_path),
            "duration": round(duration, 3),
            "word_timings": _cosy_word_timings(timestamp_events) or distribute_timings(text, duration),
            "timestamp_events": timestamp_events,
            "timing_source": "provider_word_timestamps" if _cosy_word_timings(timestamp_events) else "estimated",
            "playback_speed": float(speed or self.speed),
            "seed": int(self.conf.get("seed", 0)),
            "voice": voice,
            "provider": "bailian-cosyvoice",
            "sample_rate": 24000,
        }

    def _resolve_voice(self, voice):
        """把音色列表里的“名称”解析成真正的 voice_id。"""
        for item in self.conf.get("voices") or []:
            if isinstance(item, dict) and item.get("name") == voice and item.get("voice_id"):
                return item["voice_id"]
        return voice

    def _voice_model(self, voice, default_model):
        """自定义音色自带模型；命中 voices 列表时用其 model。"""
        for item in self.conf.get("voices") or []:
            if isinstance(item, dict) and (item.get("voice_id") == voice or item.get("name") == voice):
                return item.get("model") or default_model
        if isinstance(voice, str) and voice.startswith("cosyvoice"):
            return self.conf.get("cosyvoice_model") or DEFAULT_ENROLL_MODEL
        if isinstance(voice, str) and voice.startswith("qwen-tts-vc-"):
            return self.conf.get("voice_model") or "qwen3-tts-vc-2026-01-22"
        return default_model


def _extract_audio(data):
    """在响应 JSON 中查找音频 URL 或 base64 数据。"""
    url = _find_url(data)
    if url:
        return "url", url
    b64 = _find_base64(data)
    if b64:
        return "b64", b64
    return None, None


def _find_url(node):
    if isinstance(node, dict):
        for key, value in node.items():
            if isinstance(value, str) and value.startswith("http"):
                if key.lower() in ("url", "audio_url", "audio"):
                    return value
                if any(value.lower().split("?")[0].endswith(ext) for ext in (".wav", ".mp3", ".m4a", ".ogg")):
                    return value
            found = _find_url(value)
            if found:
                return found
    elif isinstance(node, list):
        for item in node:
            found = _find_url(item)
            if found:
                return found
    return None


def _find_base64(node):
    if isinstance(node, dict):
        for key, value in node.items():
            if isinstance(value, str) and len(value) > 512 and "http" not in value[:8]:
                if key.lower() in ("data", "audio", "audio_data", "b64", "base64", "audio_base64"):
                    return value
            found = _find_base64(value)
            if found:
                return found
    elif isinstance(node, list):
        for item in node:
            found = _find_base64(item)
            if found:
                return found
    return None


def _cosy_word_timings(events):
    # Streaming synthesis repeats provisional timestamps. Only the final event
    # for each sentence is authoritative; never union provisional revisions.
    final = {}
    for event in events:
        if event.get("type") == "sentence-end":
            final[int((event.get("sentence") or {}).get("index", 0))] = event
    words = []
    for index in sorted(final):
        sentence = final[index].get("sentence") or {}
        for word in sorted(sentence.get("words", []), key=lambda w: w.get("begin_index", 0)):
            if word.get("begin_time") is not None and word.get("end_time") is not None:
                words.append({"text": str(word.get("text", "")), "start": float(word["begin_time"]) / 1000,
                              "end": float(word["end_time"]) / 1000})
    return words
