"""Official direct-API profiles. Keys belong to their own providers."""
PROFILES = {
    "seedance_deepseek": {"type": "openai", "base_url": "https://api.seedance.nz/v1",
                          "model": "deepseek/deepseek-v4.1-flash", "max_retry": 1, "timeout": 120,
                          "max_tokens": 2000, "thinking": {"type": "disabled"}, "backup_models": []},
    "seedance_writer": {"type": "openai", "base_url": "https://api.seedance.nz/v1",
                        "model": "zhenzhen/g5.6-luna", "max_retry": 1, "timeout": 120,
                        "max_tokens": 2300, "reasoning_effort": "low", "backup_models": []},
    "qwen_budget": {"type": "openai", "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
                    "model": "qwen3-vl-flash", "max_retry": 1, "timeout": 120,
                    "max_tokens": 2000, "enable_thinking": False, "backup_models": []},
    "deepseek_budget": {"type": "openai", "base_url": "https://api.deepseek.com",
                        "model": "deepseek-flash", "max_retry": 1, "timeout": 120,
                        "max_tokens": 2000, "thinking": {"type": "disabled"}, "backup_models": []},
    "gemini_budget": {"type": "gemini", "base_url": "https://generativelanguage.googleapis.com",
                      "model": "gemini-2.5-flash-lite", "max_retry": 1, "timeout": 120},
}
