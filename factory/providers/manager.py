from .base import configure_llm_cache
from .custom_http import CustomHTTPProvider
from .gemini import GeminiProvider
from .openai_compat import OpenAICompatProvider

from ..config import ROLES

_REGISTRY = {
    "gemini": GeminiProvider,
    "openai": OpenAICompatProvider,
    "custom_http": CustomHTTPProvider,
}


class MockProvider:
    kind = "mock"
    is_mock = True
    name = "mock"

    def __init__(self, conf=None):
        self.conf = dict(conf or {})
        self.model = "mock"

    def complete(self, prompt, system=None, images=None, json_mode=False, timeout=None):
        return "{}"

    def complete_json(self, prompt, system=None, images=None, timeout=None, default=None):
        return default if default is not None else {}

    def test_connection(self):
        return True, "本地规则引擎（无需网络）"


class LLMRouter:
    def __init__(self, config):
        self.config = config
        self._cache = {}
        try:
            enabled = str(config.param("llm_cache", "on")).lower() not in ("off", "0", "false")
            cache_root = config.workspace_path() / ".cache" / "llm"
            configure_llm_cache(cache_root, enabled)
        except Exception:
            configure_llm_cache(None, False)

    def invalidate(self, role=None):
        if role:
            self._cache.pop(role, None)
        else:
            self._cache.clear()

    def set_provider(self, role, conf):
        self.config.set_provider(role, conf)
        self.invalidate(role)

    def get(self, role):
        if role not in ROLES:
            role = "story_analysis"
        if role in self._cache:
            return self._cache[role]
        conf = self.config.provider(role)
        kind = (conf.get("type") or "mock").lower()
        cls = _REGISTRY.get(kind, MockProvider)
        provider = MockProvider(conf) if cls is MockProvider else cls(conf)
        if provider.kind in ("openai", "gemini") and not (conf.get("api_key") or ""):
            provider = MockProvider(conf)
        self._cache[role] = provider
        return provider

    def is_mock(self, role):
        return self.get(role).kind == "mock"

    def test(self, role):
        return self.get(role).test_connection()
