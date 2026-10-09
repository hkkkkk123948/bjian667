import base64
import os
import time
from pathlib import Path

import requests

from ..log import get_logger
from .base import (LLMProvider, ProviderBlockedError, ProviderEmptyResponseError,
                   ProviderError, ProviderQuotaError, ProviderTimeoutError)

BLOCK_PATTERNS = {
    "prompt_blocked",
    "content_filter",
    "prohibited_content",
    "categoried as banned",
    "safety",
    "content management policy",
}


def _is_blocked(text):
    low = (text or "").lower()
    return any(p in low for p in BLOCK_PATTERNS)


class OpenAICompatProvider(LLMProvider):
    kind = "openai"

    def endpoint(self):
        base = (self.conf.get("base_url") or "https://api.openai.com/v1").rstrip("/")
        if base.endswith("/chat/completions"):
            return base
        return base + "/chat/completions"

    def build_messages(self, prompt, system, images, audios=None):
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        if not images and not audios:
            messages.append({"role": "user", "content": prompt})
            return messages
        content = [{"type": "text", "text": prompt}]
        for image in images or []:
            path = str(image)
            suffix = os.path.splitext(path)[1].lower().lstrip(".") or "jpeg"
            mime = {"jpg": "jpeg", "jpeg": "jpeg", "png": "png", "webp": "webp"}.get(suffix, "jpeg")
            b64 = base64.b64encode(Path(path).read_bytes()).decode("ascii")
            content.append({"type": "image_url", "image_url": {"url": "data:image/%s;base64,%s" % (mime, b64)}})
        for audio in audios or []:
            path = str(audio)
            suffix = os.path.splitext(path)[1].lower().lstrip(".") or "wav"
            fmt = {"mp3": "mp3", "wav": "wav", "m4a": "mp3", "aac": "mp3", "ogg": "ogg"}.get(suffix, "wav")
            b64 = base64.b64encode(Path(path).read_bytes()).decode("ascii")
            content.append({"type": "input_audio", "input_audio": {"data": b64, "format": fmt}})
        messages.append({"role": "user", "content": content})
        return messages

    def _models(self):
        backups = self.conf.get("backup_models") or []
        models = [self.model] + [m for m in backups if m and m != self.model]
        return models or ["gpt-4o-mini"]

    def _request(self, model, prompt, system, images, json_mode, timeout, audios=None):
        if self.api_key:
            headers = {"Authorization": "Bearer " + self.api_key, "Content-Type": "application/json"}
        else:
            headers = {"Content-Type": "application/json"}
        body = {
            "model": model,
            "messages": self.build_messages(prompt, system, images, audios),
            "temperature": float(self.conf.get("temperature", 0.7)),
        }
        for option in ("max_tokens", "thinking", "enable_thinking", "reasoning_effort"):
            if option in self.conf:
                body[option] = self.conf[option]
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        resp = requests.post(self.endpoint(), headers=headers, json=body, timeout=timeout or self.timeout)
        if resp.status_code in (400, 402, 403, 429) and any(marker in resp.text.lower() for marker in (
                "insufficient_user_quota", "insufficient_quota", "额度不足", "余额不足", "credit balance", "arrearage", "overdue-payment")):
            raise ProviderQuotaError("模型账户额度不足，请补充额度或使用render-plan导入已审阅的故事方案")
        if resp.status_code in (429, 500, 502, 503, 504):
            raise ProviderError("openai http %s: %s" % (resp.status_code, resp.text[:300]))
        if resp.status_code >= 400:
            if _is_blocked(resp.text):
                raise ProviderBlockedError("openai %s blocked: %s" % (model, resp.text[:300]))
            raise ProviderError("openai http %s: %s" % (resp.status_code, resp.text[:500]))
        data = resp.json()
        self.record_response_usage(model, data.get("usage") or {})
        choice = (data.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        finish = choice.get("finish_reason") or "unknown"
        details = self.last_usage.get("completion_tokens_details") or {}
        reasoning_tokens = int(details.get("reasoning_tokens") or self.last_usage.get("reasoning_tokens") or 0)
        reasoning_present = bool(message.get("reasoning_content"))
        self.last_response_diagnostics = {
            "provider": self.kind, "model": model, "finish_reason": finish,
            "usage_reported": bool(data.get("usage")),
            "prompt_tokens": int(self.last_usage.get("prompt_tokens") or 0),
            "completion_tokens": int(self.last_usage.get("completion_tokens") or 0),
            "total_tokens": int(self.last_usage.get("total_tokens") or 0),
            "reasoning_tokens": reasoning_tokens, "max_tokens": self.conf.get("max_tokens"),
            "reasoning_effort": self.conf.get("reasoning_effort"),
            "reasoning_present": reasoning_present,
            "budget_exhausted": finish == "length" and (reasoning_tokens > 0 or reasoning_present),
        }
        if finish == "content_filter" or message.get("refusal"):
            raise ProviderBlockedError("模型 %s 未提供最终内容（finish_reason=%s，内容策略拒绝）" % (model, finish))
        content = message.get("content", "")
        if isinstance(content, list):
            content = "".join(p.get("text", "") for p in content if isinstance(p, dict) and isinstance(p.get("text"), str))
        if not isinstance(content, str) or not content.strip():
            raise ProviderEmptyResponseError(self.last_response_diagnostics)
        return content

    def _complete(self, prompt, system=None, images=None, audios=None, json_mode=False, timeout=None):
        models = self._models()
        last = None
        budget_retries = 0
        for model in models:
            attempts = max(1, int(self.max_retry)) + 2  # extra room for SSL / length
            for attempt in range(attempts):
                try:
                    return self._request(model, prompt, system, images, json_mode, timeout, audios=audios)
                except ProviderQuotaError:
                    raise
                except ProviderEmptyResponseError as exc:
                    # Reasoning models often burn a tiny max_tokens budget on thinking.
                    if exc.budget_exhausted and budget_retries < 2:
                        budget_retries += 1
                        current = int(self.conf.get("max_tokens") or 512)
                        boosted = min(8192, max(current * 2, current + 512, 1024))
                        self.conf["max_tokens"] = boosted
                        self.conf["enable_thinking"] = False
                        if "thinking" in self.conf:
                            self.conf["thinking"] = {"type": "disabled"}
                        get_logger().warning(
                            "模型 %s 输出预算耗尽(finish=length)，提高 max_tokens %s→%s 并关闭思考后重试",
                            model, current, boosted)
                        last = exc
                        continue
                    raise
                except requests.Timeout as exc:
                    raise ProviderTimeoutError("模型接口等待响应超时（%.0f秒）。已保存的内容可继续使用，请续跑未完成部分。" %
                                               float(timeout or self.timeout)) from exc
                except ProviderBlockedError as exc:
                    if len(models) == 1:
                        raise
                    last = exc
                    get_logger().warning("模型 %s 被内容策略拒绝，切换后备模型: %s", model, str(exc)[:120])
                    break
                except Exception as exc:
                    last = exc
                    msg = str(exc).lower()
                    ssl_like = "ssl" in msg or "unexpected_eof" in msg or "connection" in msg
                    time.sleep((3.0 if ssl_like else 2.0) * (attempt + 1))
                    if ssl_like:
                        get_logger().warning("模型请求网络/SSL异常，重试 %s/%s: %s", attempt + 1, attempts, str(exc)[:160])
        if isinstance(last, ProviderBlockedError):
            raise last
        raise ProviderError(str(last))
