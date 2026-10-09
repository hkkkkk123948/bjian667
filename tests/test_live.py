import os
from pathlib import Path

import pytest

from factory.config import Config
from factory.providers.manager import LLMRouter

RUN_LIVE = os.environ.get("RUN_LIVE_TESTS") == "1"


@pytest.mark.skipif(not RUN_LIVE, reason="设置 RUN_LIVE_TESTS=1 才运行真实 API 集成测试")
def test_live_provider_roundtrip():
    if not Path("workspace/config.json").exists():
        pytest.skip("缺少 workspace/config.json")
    config = Config.load("workspace/config.json")
    router = LLMRouter(config)
    provider = router.get("story_analysis")
    if provider.is_mock:
        pytest.skip("story_analysis 未配置 api_key（当前为 mock）")
    text = provider.complete("用一个词回答：你好", json_mode=False, timeout=30)
    assert text and isinstance(text, str)
