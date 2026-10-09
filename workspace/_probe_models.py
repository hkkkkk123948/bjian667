"""Probe relay models without printing secrets. Writes workspace/model_probe_20261010.json"""
from __future__ import annotations
import json
import time
from pathlib import Path
from copy import deepcopy

from factory.config import Config
from factory.providers.openai_compat import OpenAICompatProvider
from factory.providers.base import ProviderEmptyResponseError, ProviderBlockedError, ProviderQuotaError, ProviderError

OUT = Path("workspace/model_probe_20261010.json")
IMG = Path("workspace/jobs/20261007_170609_2e8211/eval/ours_frames/frame_00.jpg")

TEXT_CANDIDATES = [
    "zhenzhen/g6-sol",
    "deepseek/deepseek-v4-pro",
    "zhenzhen/g5.6-sol",
    "qwen/qwen3.8-max",
    "kimi-k3",
    "glm-5.3",
    "bytedance/doubao-seed-2.1-pro",
    "deepseek/deepseek-v4.1-flash",
]
VISION_CANDIDATES = [
    "qwen/qwen3.8-max",
    "qwen/qwen3.7-max",
    "qwen/qwen3.6-max-preview",
    "qwen/qwen3.6-plus",
    "glm-5v-turbo",
    "zhenzhen/g6-sol",
]

def scrub(o):
    if isinstance(o, dict):
        return {k: ("***" if any(x in k.lower() for x in ("key", "token", "secret", "password", "authorization")) else scrub(v))
                for k, v in o.items()}
    if isinstance(o, list):
        return [scrub(i) for i in o]
    return o

def base_conf(cfg: Config) -> dict:
    # Prefer script_generation / video_analysis relay credentials
    for role in ("script_generation", "video_analysis", "story_analysis", "match_judge"):
        conf = cfg.provider(role) or {}
        if conf.get("api_key") and conf.get("base_url"):
            return {
                "type": "openai",
                "base_url": conf["base_url"],
                "api_key": conf["api_key"],
                "timeout": 90,
                "max_retry": 1,
                "temperature": 0.1,
                "enable_thinking": False,
                "max_tokens": 256,
            }
    raise RuntimeError("no relay credentials in config")

def probe_text(conf_template: dict, model: str) -> dict:
    conf = deepcopy(conf_template)
    conf["model"] = model
    conf["max_tokens"] = 256
    conf["enable_thinking"] = False
    p = OpenAICompatProvider(conf)
    t0 = time.time()
    try:
        text = p.complete("只回复JSON：{\"ok\":true,\"n\":1}", system="你是探针。只输出JSON。", json_mode=True)
        elapsed = round(time.time() - t0, 2)
        data = None
        try:
            data = json.loads(text) if isinstance(text, str) else text
        except Exception:
            data = {"raw_len": len(str(text or ""))}
        return {
            "model": model, "ok": True, "elapsed_s": elapsed,
            "finish": (p.last_response_diagnostics or {}).get("finish_reason"),
            "completion_tokens": (p.last_response_diagnostics or {}).get("completion_tokens"),
            "reasoning_tokens": (p.last_response_diagnostics or {}).get("reasoning_tokens"),
            "sample": str(data)[:120],
            "error": None,
        }
    except Exception as exc:
        return {
            "model": model, "ok": False, "elapsed_s": round(time.time() - t0, 2),
            "error_type": type(exc).__name__, "error": str(exc)[:240],
            "diag": scrub(getattr(exc, "diagnostics", None) or getattr(p, "last_response_diagnostics", None) or {}),
        }

def probe_vision(conf_template: dict, model: str, image: Path) -> dict:
    conf = deepcopy(conf_template)
    conf["model"] = model
    conf["max_tokens"] = 256
    conf["enable_thinking"] = False
    p = OpenAICompatProvider(conf)
    t0 = time.time()
    try:
        text = p.complete(
            "看图，只回复JSON：{\"ok\":true,\"has_person\":true/false}",
            system="你是视觉探针。只输出JSON。",
            images=[str(image)],
            json_mode=True,
        )
        elapsed = round(time.time() - t0, 2)
        return {
            "model": model, "ok": True, "elapsed_s": elapsed, "vision": True,
            "finish": (p.last_response_diagnostics or {}).get("finish_reason"),
            "completion_tokens": (p.last_response_diagnostics or {}).get("completion_tokens"),
            "reasoning_tokens": (p.last_response_diagnostics or {}).get("reasoning_tokens"),
            "sample": str(text)[:120],
            "error": None,
        }
    except Exception as exc:
        return {
            "model": model, "ok": False, "vision": True, "elapsed_s": round(time.time() - t0, 2),
            "error_type": type(exc).__name__, "error": str(exc)[:240],
            "diag": scrub(getattr(exc, "diagnostics", None) or getattr(p, "last_response_diagnostics", None) or {}),
        }

def pick_first_ok(rows):
    for row in rows:
        if row.get("ok"):
            return row["model"]
    return None

def main():
    cfg = Config.load("workspace/config.json")
    template = base_conf(cfg)
    if not IMG.exists():
        raise RuntimeError("missing probe image")
    text_results = []
    for model in TEXT_CANDIDATES:
        print("TEXT", model, flush=True)
        text_results.append(probe_text(template, model))
        time.sleep(0.4)
    vision_results = []
    for model in VISION_CANDIDATES:
        print("VISION", model, flush=True)
        vision_results.append(probe_vision(template, model, IMG))
        time.sleep(0.4)
    best_text = pick_first_ok(text_results)
    best_vision = pick_first_ok(vision_results)
    report = {
        "text_results": text_results,
        "vision_results": vision_results,
        "best_text": best_text,
        "best_vision": best_vision,
        "note": "order is preference; first ok wins",
    }
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("WROTE", OUT)
    print("BEST_TEXT", best_text)
    print("BEST_VISION", best_vision)

if __name__ == "__main__":
    main()
