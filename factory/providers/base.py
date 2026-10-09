import base64
import hashlib
import json
import os
import re
import time
from pathlib import Path

from ..log import get_logger
from ..utils import clamp

CODE_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.S)

_CACHE = {"dir": None, "enabled": False}
_USAGE = {"models": {}, "requests": 0, "cache_hits": 0, "cache_misses": 0}


def configure_llm_cache(cache_dir=None, enabled=True):
    """配置 LLM 响应磁盘缓存（doc §90：prompt+model+input 命中则直接取缓存）。"""
    _CACHE["dir"] = Path(cache_dir) if cache_dir else None
    _CACHE["enabled"] = bool(enabled and cache_dir)
    if _CACHE["enabled"]:
        try:
            _CACHE["dir"].mkdir(parents=True, exist_ok=True)
        except OSError:
            _CACHE["enabled"] = False


def llm_cache_enabled():
    return bool(_CACHE["enabled"])


def _images_signature(images):
    h = hashlib.sha256()
    for item in images or []:
        h.update(str(item).encode("utf-8"))
        try:
            st = os.stat(item)
            h.update(("%d:%d" % (st.st_size, int(st.st_mtime))).encode())
        except OSError:
            pass
    return h.hexdigest()


def _cache_key(provider, prompt, system, images, json_mode, audios=None):
    if not _CACHE["enabled"]:
        return None
    from ..versioning import prompt_salt

    h = hashlib.sha256()
    parts = [str(provider.model), str(provider.conf.get("base_url", "")), str(json_mode), str(provider.conf.get("temperature", "")),
             json.dumps({key: provider.conf[key] for key in ("max_tokens", "thinking", "enable_thinking", "reasoning_effort")
                         if key in provider.conf}, sort_keys=True, ensure_ascii=False),
             prompt or "", system or "", _images_signature(images), _images_signature(audios), prompt_salt()]
    for part in parts:
        h.update(part.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()


def _cache_path(key):
    return _CACHE["dir"] / key[:2] / (key + ".json")


def _cache_load(key):
    path = _cache_path(key)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _cache_save(key, payload):
    path = _cache_path(key)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    except OSError:
        pass


def record_usage(model, usage):
    """累计 token 用量（doc §56：每个任务记录 token/模型/成本）。"""
    if not usage:
        return
    slot = _USAGE["models"].setdefault(str(model or "unknown"),
                                       {"requests": 0, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})
    prompt_tokens = int(usage.get("prompt_tokens") or usage.get("promptTokenCount") or 0)
    completion_tokens = int(usage.get("completion_tokens") or usage.get("candidatesTokenCount") or 0)
    total = int(usage.get("total_tokens") or usage.get("totalTokenCount") or (prompt_tokens + completion_tokens))
    slot["requests"] += 1
    slot["prompt_tokens"] += prompt_tokens
    slot["completion_tokens"] += completion_tokens
    slot["total_tokens"] += total
    _USAGE["requests"] += 1


def usage_summary():
    models = {k: dict(v) for k, v in _USAGE["models"].items()}
    totals = {"requests": _USAGE["requests"], "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    for slot in models.values():
        for field in ("prompt_tokens", "completion_tokens", "total_tokens"):
            totals[field] += slot[field]
    return {"models": models, "totals": totals,
            "cache_hits": _USAGE["cache_hits"], "cache_misses": _USAGE["cache_misses"]}


def reset_usage():
    _USAGE["models"].clear()
    _USAGE["requests"] = 0
    _USAGE["cache_hits"] = 0
    _USAGE["cache_misses"] = 0


class ProviderError(RuntimeError):
    pass


class ProviderQuotaError(ProviderError):
    """A billing/quota failure must not trigger retries or substitute models."""
    pass


class ProviderTimeoutError(ProviderError):
    """An unanswered request must be resumed explicitly, not retried blindly."""
    pass


class ProviderEmptyResponseError(ProviderError):
    """A completed, possibly billed request returned no final message content."""

    def __init__(self, diagnostics=None):
        self.diagnostics = dict(diagnostics or {})
        self.budget_exhausted = bool(self.diagnostics.get("budget_exhausted"))
        super().__init__("模型未返回最终内容（模型=%s，finish_reason=%s，输出token=%s，推理token=%s）。已保留进度。" %
                         (self.diagnostics.get("model", "unknown"), self.diagnostics.get("finish_reason", "unknown"),
                          self.diagnostics.get("completion_tokens", 0), self.diagnostics.get("reasoning_tokens", 0)))


class ProviderBlockedError(ProviderError):
    """模型端因内容安全策略拒绝请求（如 PROHIBITED_CONTENT）。

    此类错误重试无意义，应立即降级/切换后备模型。
    """


def extract_json(text):
    if text is None:
        return None
    if isinstance(text, (dict, list)):
        return text
    raw = str(text).strip()
    if not raw:
        return None
    fenced = CODE_FENCE_RE.findall(raw)
    candidates = fenced + [raw]
    for chunk in candidates:
        chunk = chunk.strip()
        start = min([i for i in (chunk.find("{"), chunk.find("[")) if i >= 0], default=-1)
        if start < 0:
            continue
        for end in range(len(chunk) - 1, start, -1):
            if chunk[end] not in "}]" :
                continue
            try:
                return json.loads(chunk[start:end + 1])
            except json.JSONDecodeError:
                continue
    return None


def encode_image(path):
    data = Path(path).read_bytes()
    return base64.b64encode(data).decode("ascii")


def image_part(path):
    suffix = Path(path).suffix.lower().lstrip(".") or "jpeg"
    mime = {"jpg": "jpeg", "jpeg": "jpeg", "png": "png", "webp": "webp", "bmp": "bmp"}.get(suffix, "jpeg")
    return {"mime": mime, "data": encode_image(path)}


class LLMProvider:
    kind = "base"

    def __init__(self, conf):
        self.conf = dict(conf or {})
        self.name = self.conf.get("type", "base")
        self.model = self.conf.get("model") or ""
        self.timeout = float(self.conf.get("timeout") or 180)
        self.max_retry = int(self.conf.get("max_retry") or 3)
        self.api_key = self.conf.get("api_key") or ""
        self.last_usage = {}
        self.last_call_usage = {}
        self.last_call_requests = 0
        self.last_response_diagnostics = {}

    @property
    def is_mock(self):
        return self.name == "mock"

    def _complete(self, prompt, system=None, images=None, audios=None, json_mode=False, timeout=None):
        raise NotImplementedError

    def complete(self, prompt, system=None, images=None, audios=None, json_mode=False, timeout=None):
        self.last_usage = {}
        self.last_call_usage = {}
        self.last_call_requests = 0
        self.last_response_diagnostics = {}
        key = _cache_key(self, prompt, system, images, json_mode, audios)
        if key:
            cached = _cache_load(key)
            if (isinstance(cached, dict) and cached.get("text") is not None
                    and not (isinstance(cached.get("text"), str) and not cached["text"].strip())
                    and (not json_mode or extract_json(cached["text"]) is not None)):
                _USAGE["cache_hits"] += 1
                self.last_usage = cached.get("usage") or {}
                return cached.get("text", "")
            _USAGE["cache_misses"] += 1
        try:
            text = self._complete(prompt, system=system, images=images, audios=audios,
                                  json_mode=json_mode, timeout=timeout)
            if text is None or isinstance(text, str) and not text.strip():
                raise ProviderEmptyResponseError({"provider": self.kind, "model": self.model})
        except ProviderError:
            self._record_fallback_usage()
            raise
        if key and (not json_mode or extract_json(text) is not None):
            _cache_save(key, {"text": text, "usage": self.last_usage or {}})
        self._record_fallback_usage()
        return text

    def record_response_usage(self, model, usage):
        """Count each HTTP response once, including empty results and recovery attempts."""
        self.last_usage = dict(usage or {})
        self.last_call_requests += 1
        accounted = self.last_usage or {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        record_usage(model, accounted)
        prompt = int(accounted.get("prompt_tokens") or accounted.get("promptTokenCount") or 0)
        completion = int(accounted.get("completion_tokens") or accounted.get("candidatesTokenCount") or 0)
        values = {"prompt_tokens": prompt, "completion_tokens": completion,
                  "total_tokens": int(accounted.get("total_tokens") or accounted.get("totalTokenCount") or prompt + completion)}
        for key, value in values.items():
            self.last_call_usage[key] = self.last_call_usage.get(key, 0) + value

    def _record_fallback_usage(self):
        # Other adapters still expose one usage object per completion. OpenAI
        # compatible adapters record their individual HTTP responses directly.
        if not self.last_call_requests:
            record_usage(self.model, self.last_usage)
            self.last_call_usage = dict(self.last_usage)
            self.last_call_requests = 1 if self.last_usage else 0

    def complete_json(self, prompt, system=None, images=None, audios=None, json_mode=True, timeout=None, default=None):
        last_err = None
        modes = [True, False] if json_mode else [False]
        for mode in modes:
            for attempt in range(max(1, self.max_retry)):
                try:
                    text = self.complete(prompt, system=system, images=images, audios=audios,
                                         json_mode=mode, timeout=timeout)
                    data = extract_json(text)
                    if data is not None:
                        return data
                    last_err = ProviderError("model returned non-json: %s" % str(text)[:200])
                except ProviderQuotaError:
                    raise
                except ProviderTimeoutError:
                    raise
                except ProviderEmptyResponseError:
                    # A higher-level caller may make one changed, justified
                    # request; nested JSON/model retries must not replay this one.
                    raise
                except ProviderBlockedError as exc:
                    last_err = exc
                    get_logger().warning("llm 内容被拒(含后备模型): %s", str(exc)[:200])
                    break
                except Exception as exc:
                    last_err = exc
                    get_logger().warning("llm attempt %d failed (json_mode=%s): %s", attempt + 1, mode, exc)
                    # response_format failures should immediately try plain text JSON
                    if mode and any(x in str(exc).lower() for x in ("response_format", "invalid_parameter", "abnormal", "invalid json")):
                        break
            else:
                continue
            # blocked: stop all modes
            if isinstance(last_err, ProviderBlockedError):
                break
        if default is not None:
            return default
        raise last_err or ProviderError("complete_json failed")


    def test_connection(self):
        try:
            out = self.complete("reply with OK", timeout=30)
            return True, str(out)[:80]
        except Exception as exc:
            return False, str(exc)[:200]


def with_timeout(conf, timeout):
    if timeout:
        return float(timeout)
    return float(conf.get("timeout") or 180)


def temperature(conf, default=0.7):
    try:
        return clamp(float(conf.get("temperature", default)), 0.0, 2.0)
    except (TypeError, ValueError):
        return default
