import base64
import json
import os
import time
from pathlib import Path

import requests

from ..log import get_logger
from .base import LLMProvider, ProviderBlockedError, ProviderError

BLOCK_PATTERNS = ("prohibited_content", "prompt_blocked", "content_filter", "safety", "blocked")


def _transient_provider_error(exc):
    text = str(exc).lower()
    return any(code in text for code in ("http 429", "http 500", "http 502", "http 503", "http 504"))


def _is_blocked(text):
    low = (text or "").lower()
    return any(p in low for p in BLOCK_PATTERNS)


class GeminiProvider(LLMProvider):
    kind = "gemini"
    BASE = "https://generativelanguage.googleapis.com/v1beta"

    def endpoint(self, model=None):
        model = model or self.model or "gemini-2.5-flash"
        base = (self.conf.get("base_url") or self.BASE).rstrip("/")
        if base.endswith("v1beta") or base.endswith("v1"):
            return "%s/models/%s:generateContent" % (base, model)
        return "%s/v1beta/models/%s:generateContent" % (base, model)

    def _models(self):
        backups = self.conf.get("backup_models") or []
        return [self.model] + [m for m in backups if m and m != self.model] or ["gemini-2.5-flash"]

    def build_body(self, prompt, system, images, json_mode, audios=None):
        parts = []
        if system:
            parts.append({"text": system + "\n\n" + prompt})
        else:
            parts.append({"text": prompt})
        for image in images or []:
            path = image
            suffix = os.path.splitext(str(path))[1].lower().lstrip(".") or "jpeg"
            mime = {"jpg": "jpeg", "jpeg": "jpeg", "png": "png", "webp": "webp"}.get(suffix, "jpeg")
            data = base64.b64encode(Path(path).read_bytes()).decode("ascii")
            parts.append({"inline_data": {"mime_type": "image/jpeg" if mime == "jpeg" else "image/" + mime, "data": data}})
        for audio in audios or []:
            path = str(audio)
            suffix = os.path.splitext(path)[1].lower().lstrip(".") or "wav"
            mime = {"mp3": "audio/mp3", "wav": "audio/wav", "ogg": "audio/ogg"}.get(suffix, "audio/wav")
            data = base64.b64encode(Path(path).read_bytes()).decode("ascii")
            parts.append({"inline_data": {"mime_type": mime, "data": data}})
        body = {"contents": [{"role": "user", "parts": parts}]}
        gen = {"temperature": 0.7, "maxOutputTokens": 8192}
        if json_mode:
            gen["responseMimeType"] = "application/json"
        if self.conf.get("youtube_url"):
            body["contents"][0]["parts"].append({"file_data": {"file_uri": self.conf["youtube_url"]}})
        body["generationConfig"] = gen
        return body

    def _request(self, model, prompt, system, images, json_mode, timeout, audios=None):
        headers = {"Content-Type": "application/json", "x-goog-api-key": self.api_key}
        body = self.build_body(prompt, system, images, json_mode, audios)
        resp = requests.post(self.endpoint(model), headers=headers, json=body, timeout=timeout or self.timeout)
        if resp.status_code in (429, 500, 502, 503, 504):
            raise ProviderError("gemini http %s: %s" % (resp.status_code, resp.text[:300]))
        if resp.status_code >= 400:
            if _is_blocked(resp.text):
                raise ProviderBlockedError("gemini %s blocked: %s" % (model, resp.text[:300]))
            raise ProviderError("gemini http %s: %s" % (resp.status_code, resp.text[:500]))
        data = resp.json()
        self.last_usage = data.get("usageMetadata") or {}
        candidates = data.get("candidates") or []
        if not candidates:
            raise ProviderError("gemini empty candidates: %s" % str(data)[:300])
        texts = []
        for part in candidates[0].get("content", {}).get("parts", []):
            if "text" in part:
                texts.append(part["text"])
        if not texts:
            raise ProviderError("gemini no text part")
        return "".join(texts)

    def _complete(self, prompt, system=None, images=None, audios=None, json_mode=False, timeout=None):
        if not self.api_key:
            raise ProviderError("gemini api_key 未配置")
        models = self._models()
        last = None
        for model in models:
            for attempt in range(max(1, self.max_retry)):
                try:
                    return self._request(model, prompt, system, images, json_mode, timeout, audios=audios)
                except ProviderBlockedError as exc:
                    if len(models) == 1:
                        raise
                    last = exc
                    get_logger().warning("gemini %s 被内容策略拒绝，切换后备模型: %s", model, str(exc)[:120])
                    break
                except ProviderError as exc:
                    if not _transient_provider_error(exc):
                        raise
                    last = exc
                    time.sleep(2.0 * (attempt + 1))
                    get_logger().warning("gemini %s 暂时失败，重试 %s: %s", model, attempt + 1, str(exc)[:160])
                except Exception as exc:
                    last = exc
                    message = str(exc).lower()
                    ssl_like = "ssl" in message or "unexpected_eof" in message or "connection" in message
                    time.sleep((3.0 if ssl_like else 2.0) * (attempt + 1))
                    if ssl_like or isinstance(exc, json.JSONDecodeError):
                        get_logger().warning("gemini 请求网络/JSON异常，重试 %s: %s", attempt + 1, str(exc)[:160])
        if isinstance(last, ProviderBlockedError):
            raise last
        raise ProviderError(str(last))
