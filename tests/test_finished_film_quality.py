"""Quality gates for causal narration, action-frame picture match, and flaky APIs."""
import inspect

import pytest

from factory.matching.initial_plan import (
    anchor_window, build_initial_timeline, phrase_overlap, place_shot_windows, resolve_frame_ids,
)
from factory.providers.base import ProviderError, extract_json
from factory.providers.gemini import GeminiProvider
from factory.providers.openai_compat import apply_model_budget
from factory.render.exporter import _segment_role
from factory.script.quality import causal_chain_issues, narration_spine
from factory.script.writer import FLAVORS, _llm_variants


def test_causal_chain_rejects_scare_list_and_reversed_order():
    scare = ["怪人盯着女孩。", "怪人扑向门口。", "怪人抓住门缝。", "门后露出怪脸。", "女孩吓得不敢动。", "怪人还在门外。"]
    issues = causal_chain_issues([{"text": text} for text in scare])
    assert any("结果" in item or "升级" in item for item in issues)
    reversed_story = [
        "女孩已经安全到家。", "她回想起刚才的事。", "女人带她走了一段。",
        "她才跟着女人下车。", "男人突然追了上来。", "一个陌生男人一直盯着她。",
    ]
    issues = causal_chain_issues([{"text": text} for text in reversed_story])
    assert any("太晚" in item or "顺序" in item or "过早" in item for item in issues)


def test_causal_chain_accepts_ordered_short():
    beats = [
        "女孩等公交时，一个陌生男人一直盯着她。",
        "她怕男人再靠近，就跟着下车的女人走。",
        "女孩拉住女人，说有人在跟踪她。",
        "男人还是跟了上来。",
        "女人没有拒绝，带着女孩一起走。",
        "跟踪的人停住了，女孩的父母把她接回了家。",
    ]
    assert causal_chain_issues([{"text": text} for text in beats]) == []


def test_narration_spine_names_visible_facts_without_inventing_events():
    spine = narration_spine([
        {"id": "F01", "visible": "男人盯着等车的女孩"},
        {"id": "F02", "visible": "女孩拉住女人"},
        {"id": "F03", "visible": "男人跟上来"},
        {"id": "F04", "visible": "父母把女孩接上车"},
    ])
    assert "F01" in spine and "F04" in spine and "遇险" in spine and "结果" in spine
    assert "玫瑰" not in spine


def test_writer_prompt_keeps_full_causal_rules():
    assert "疑问句结尾" not in FLAVORS[0]["tone"]
    assert "STORY_RULES[:220]" not in inspect.getsource(_llm_variants)


def test_phrase_overlap_prefers_the_action_frame_not_the_middle():
    shot = {
        "frames": [
            {"frame_id": "start", "time": 10.2},
            {"frame_id": "mid", "time": 15.0},
            {"frame_id": "end", "time": 19.4},
        ],
        "actions": [{"text": "男孩赶紧跑出巷子", "frame_ids": ["end"]}],
        "observations": [{"frame_id": "mid", "visible": "巷子全景"}],
    }
    assert resolve_frame_ids("男孩赶紧跑出巷子", shot) == ["end"]
    assert phrase_overlap("男孩冲到门口", "男孩冲向门口") >= 0.45


def test_action_windows_stay_inside_the_shot_and_follow_the_anchor():
    start = anchor_window(40, 70, 2.4, 68)
    center = anchor_window(40, 70, 2.4, 55)
    assert start > center
    assert 40 <= start <= 70 - 2.4
    placed = place_shot_windows(40, 70, [
        {"length": 2.0, "anchor": 42},
        {"length": 2.0, "anchor": 68},
    ])
    assert placed[0] < placed[1]
    assert placed[0] >= 40 and placed[1] + 2 <= 70 + 1e-6
    assert placed[0] + 2 <= placed[1] + 1e-6


def test_spoken_action_uses_its_frame_instead_of_the_shot_center(tmp_path):
    from test_initial_plan import fixture, speech

    pipeline, candidate, variant, shots, _answer, _calls, _provider = fixture(tmp_path)
    run = shots[3]
    run["end"] = 70
    run["frames"] = [
        {"frame_id": "RUN_start", "time": 41.0, "position": "start"},
        {"frame_id": "RUN_end", "time": 68.0, "position": "end"},
    ]
    run["actions"] = [{"text": "男孩赶紧跑走", "frame_ids": ["RUN_end"]}]
    run["observations"] = [{"frame_id": "RUN_start", "visible": "巷口"}]
    timeline = build_initial_timeline(pipeline, candidate, variant, speech(variant), shots)
    segment = next(item for item in timeline["segments"] if item["shot_id"] == "RUN")
    center = anchor_window(40, 70, segment["src_duration"], 55)
    assert segment["src_start"] > center
    assert segment["evidence_start"] <= segment["src_start"]
    assert segment["src_start"] + segment["src_duration"] <= segment["evidence_end"] + 1e-6


def test_segment_role_uses_one_based_sentence_index():
    timeline = {"sentence_timings": [
        {"index": 1, "role": "hook"},
        {"index": 2, "role": "payoff"},
    ]}
    assert _segment_role({"sentence_index": 1}, timeline) == "hook"
    assert _segment_role({"sentence_index": 2, "role": "twist"}, timeline) == "twist"


def test_extract_json_repairs_trailing_commas():
    assert extract_json('{"a":1,}') == {"a": 1}
    assert extract_json('```json\n{"a": [1, 2,],}\n```') == {"a": [1, 2]}
    assert extract_json("not json") is None


def test_deepseek_budget_disables_thinking_without_raising_explicit_limits():
    body = {"temperature": 0.7}
    apply_model_budget(body, "deepseek/deepseek-v4.1-flash")
    assert body["enable_thinking"] is False
    assert body["thinking"] == {"type": "disabled"}
    assert body["max_tokens"] == 4096
    explicit = {"max_tokens": 256, "enable_thinking": True}
    apply_model_budget(explicit, "deepseek-flash")
    assert explicit["max_tokens"] == 256 and explicit["enable_thinking"] is True
    other = {"temperature": 0.2}
    apply_model_budget(other, "qwen/qwen3.6-plus")
    assert "thinking" not in other


def test_gemini_retries_transient_http_and_not_bad_requests(monkeypatch):
    monkeypatch.setattr("factory.providers.gemini.time.sleep", lambda *_args: None)
    provider = GeminiProvider({"api_key": "fixture", "model": "gemini-2.5-flash-lite", "max_retry": 3})
    calls = {"n": 0}

    def flaky(*_args, **_kwargs):
        calls["n"] += 1
        if calls["n"] < 3:
            raise ProviderError("gemini http 503: unavailable")
        return "ok"

    monkeypatch.setattr(provider, "_request", flaky)
    assert provider._complete("hi") == "ok"
    assert calls["n"] == 3

    def bad(*_args, **_kwargs):
        calls["n"] += 1
        raise ProviderError("gemini http 400: bad")

    calls["n"] = 0
    monkeypatch.setattr(provider, "_request", bad)
    with pytest.raises(ProviderError):
        provider._complete("hi")
    assert calls["n"] == 1
