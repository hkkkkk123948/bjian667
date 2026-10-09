import json
import time

import requests

from .base import LLMProvider, ProviderError


def resolve_path(data, path, default=None):
    if not path:
        return data
    cur = data
    for part in str(path).split("."):
        if isinstance(cur, list):
            try:
                cur = cur[int(part)]
                continue
            except (ValueError, IndexError):
                return default
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur if cur is not None else default


class CustomHTTPProvider(LLMProvider):
    kind = "custom_http"

    def _complete(self, prompt, system=None, images=None, audios=None, json_mode=False, timeout=None):
        url = self.conf.get("base_url") or ""
        if not url:
            raise ProviderError("custom_http url 未配置")
        headers = {"Content-Type": "application/json"}
        for key, value in (self.conf.get("headers") or {}).items():
            headers[key] = str(value).replace("{api_key}", self.api_key)
        template = self.conf.get("body_template") or ""
        if template:
            rendered = template.replace("{{prompt}}", json.dumps(prompt, ensure_ascii=False))
            rendered = rendered.replace("{{system}}", json.dumps(system or "", ensure_ascii=False))
            rendered = rendered.replace("{{model}}", json.dumps(self.model or ""))
            rendered = rendered.replace("{{api_key}}", self.api_key)
            try:
                body = json.loads(rendered)
            except json.JSONDecodeError as exc:
                raise ProviderError("custom_http body_template 非法: %s" % exc)
        else:
            body = {"model": self.model, "messages": [], "temperature": 0.7}
            msgs = []
            if system:
                msgs.append({"role": "system", "content": system})
            msgs.append({"role": "user", "content": prompt})
            body["messages"] = msgs
            if json_mode:
                body["response_format"] = {"type": "json_object"}
        last = None
        for attempt in range(max(1, self.max_retry)):
            try:
                method = (self.conf.get("method") or "POST").upper()
                resp = requests.request(method, url, headers=headers, json=body, timeout=timeout or self.timeout)
                if resp.status_code in (429, 500, 502, 503, 504):
                    last = ProviderError("custom http %s: %s" % (resp.status_code, resp.text[:300]))
                    time.sleep(2.0 * (attempt + 1))
                    continue
                if resp.status_code >= 400:
                    raise ProviderError("custom http %s: %s" % (resp.status_code, resp.text[:500]))
                data = resp.json()
                path = self.conf.get("response_path") or "choices.0.message.content"
                text = resolve_path(data, path, None)
                if isinstance(text, (dict, list)):
                    return json.dumps(text, ensure_ascii=False)
                if text is None:
                    raise ProviderError("custom response_path 未命中: %s" % path)
                return str(text)
            except ProviderError:
                raise
            except Exception as exc:
                last = exc
                time.sleep(2.0 * (attempt + 1))
        raise ProviderError(str(last))

    def test_connection(self):
        return super().test_connection()
