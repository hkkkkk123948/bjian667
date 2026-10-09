import copy
import json
from types import SimpleNamespace

import pytest

from factory.config import Config
from factory.models import Evidence, MediaIndex, SceneSegment, ScriptSentence, ScriptVariant, StoryCandidate, VideoMeta
from factory.pipeline import ProductionPipeline
from factory.providers.base import (ProviderEmptyResponseError, configure_llm_cache, reset_usage, usage_summary)
from factory.providers.openai_compat import OpenAICompatProvider
from factory.utils import read_json, write_json


@pytest.fixture(autouse=True)
def isolated_usage_and_cache():
    configure_llm_cache(None, enabled=False)
    reset_usage()
    yield
    configure_llm_cache(None, enabled=False)
    reset_usage()


def response(content=None, finish="stop", reasoning=0, output=20, prompt=100):
    return SimpleNamespace(status_code=200, text="", json=lambda: {
        "choices": [{"finish_reason": finish, "message": {"content": content,
            "reasoning_content": '{"shots":["private chain of thought is not final JSON"]}'}}],
        "usage": {"prompt_tokens": prompt, "completion_tokens": output, "total_tokens": prompt+output,
                  "completion_tokens_details": {"reasoning_tokens": reasoning}}})


def reader():
    return OpenAICompatProvider({"type": "openai", "model": "fixture-model", "base_url": "https://fixture.invalid/v1",
        "api_key": "fixture-private-key", "max_tokens": 777, "reasoning_effort": "high", "max_retry": 4})


def test_empty_response_is_typed_billed_once_not_reasoning_json_or_blind_retry(tmp_path, monkeypatch):
    configure_llm_cache(tmp_path / "cache", enabled=True)
    calls = []
    monkeypatch.setattr("factory.providers.openai_compat.requests.post", lambda *a, **k:
                        calls.append(k["json"]) or response(finish="length", reasoning=1900, output=1900))
    provider = reader()
    with pytest.raises(ProviderEmptyResponseError) as caught:
        provider.complete_json("fixture")
    assert len(calls) == 1 and caught.value.budget_exhausted
    diagnostic = caught.value.diagnostics
    assert diagnostic["model"] == "fixture-model" and diagnostic["finish_reason"] == "length"
    assert diagnostic["reasoning_tokens"] == 1900 and diagnostic["completion_tokens"] == 1900
    assert "fixture-private-key" not in json.dumps(diagnostic) and "private chain" not in str(caught.value)
    assert usage_summary()["totals"]["requests"] == 1
    assert usage_summary()["totals"]["total_tokens"] == 2000
    assert not list((tmp_path / "cache").rglob("*.json"))


def test_success_usage_not_double_counted_and_generation_controls_change_cache_key(tmp_path, monkeypatch):
    configure_llm_cache(tmp_path / "cache", enabled=True)
    calls = []
    monkeypatch.setattr("factory.providers.openai_compat.requests.post", lambda *a, **k:
                        calls.append(k["json"]) or response('{"ok":true}', output=20))
    provider = reader()
    assert provider.complete_json("fixture") == {"ok": True}
    assert provider.complete_json("fixture") == {"ok": True}
    assert len(calls) == 1 and usage_summary()["totals"]["requests"] == 1
    assert provider.last_call_requests == 0 and provider.last_call_usage == {}
    provider.conf["max_tokens"] = 3800
    assert provider.complete_json("fixture") == {"ok": True}
    assert len(calls) == 2 and usage_summary()["totals"]["requests"] == 2
    assert usage_summary()["totals"]["total_tokens"] == 240


def local_shots(tmp_path, count=6):
    shots = []
    for number in range(1, count+1):
        identity = "SH%04d" % number
        frames = []
        for suffix, position, fraction in (("A", "first", .08), ("B", "middle", .5), ("C", "last", .92)):
            path = tmp_path / (identity+"_"+suffix+".jpg")
            path.write_bytes(b"fixture image")
            frames.append({"frame_id": identity+"_"+suffix, "position": position,
                           "time": (number-1)*4+fraction*4, "path": str(path)})
        shots.append({"shot_id": identity, "start": (number-1)*4, "end": number*4, "frames": frames})
    return {"settings": {"source_signature": "fixture-source", "source_start": 0, "source_end": count*4}, "shots": shots}


def observed_json(body):
    prompt = body["messages"][-1]["content"][0]["text"]
    mapping = json.loads(prompt.split("\n真实图片映射：", 1)[1])
    result = {"characters": [], "shots": []}
    for identity in dict.fromkeys(item["shot_id"] for item in mapping):
        result["shots"].append({"shot_id": identity, "visible": "女孩站在门前", "characters": [],
            "observations": [{"frame_id": frame["frame_id"], "visible": "当前真实姿态"}
                             for frame in mapping if frame["shot_id"] == identity], "actions": [], "uncertain": []})
    return json.dumps(result, ensure_ascii=False)


def vision_pipeline(tmp_path, provider):
    messages = []
    return SimpleNamespace(config=Config({"workspace": str(tmp_path / "ws")}),
        router=SimpleNamespace(get=lambda role: provider),
        state=SimpleNamespace(artifacts={}), job_dir=tmp_path / 'job',
        emitter=SimpleNamespace(log=lambda *args: messages.append(args)), _check_cancel=lambda: None)


def test_failed_batch_progress_and_usage_survive_and_only_unfinished_batches_are_read(tmp_path, monkeypatch):
    import factory.mining.shot_evidence as shots
    local = local_shots(tmp_path)
    monkeypatch.setattr(shots, "build_local_index", lambda *a, **k: local)
    provider = reader()
    p = vision_pipeline(tmp_path, provider)
    calls = []

    def request(*args, **kwargs):
        body = kwargs["json"]
        calls.append(copy.deepcopy(body))
        return response(output=900) if len(calls) == 2 else response(observed_json(body), output=900)

    monkeypatch.setattr("factory.providers.openai_compat.requests.post", request)
    folder = tmp_path / "story"
    with pytest.raises(ProviderEmptyResponseError):
        shots.build_and_observe(p, "fixture.mp4", 0, 24, folder)
    progress = read_json(folder / "shot_evidence/progress.json")
    assert progress["shots_observed"] == 2 and progress["complete"] is False
    assert progress["pending_shot_ids"] == ["SH0003", "SH0004"]
    assert progress["usage"]["calls"] == 2 and progress["usage"]["total_tokens"] == 2000
    assert not (folder / "shot_evidence/catalog.json").exists()
    cache_root = p.config.workspace_path() / ".cache" / shots.POLICY
    assert len(list(cache_root.glob("*.json"))) == 1
    final = shots.build_and_observe(p, "fixture.mp4", 0, 24, folder)
    assert len(final) == 6 and len(calls) == 4
    progress = read_json(folder / "shot_evidence/progress.json")
    assert progress["complete"] and progress["usage"]["calls"] == 4
    assert progress["usage"]["total_tokens"] == usage_summary()["totals"]["total_tokens"] == 4000
    assert len(list(cache_root.glob("*.json"))) == 3
    assert provider.conf["max_tokens"] == 777 and provider.conf["reasoning_effort"] == "high"
    assert provider.max_retry == 4


def test_proven_reasoning_budget_exhaustion_gets_one_changed_request(tmp_path, monkeypatch):
    import factory.mining.shot_evidence as shots
    local = local_shots(tmp_path, count=2)
    monkeypatch.setattr(shots, "build_local_index", lambda *a, **k: local)
    calls = []

    def request(*args, **kwargs):
        body = kwargs["json"]
        calls.append(copy.deepcopy(body))
        if len(calls) == 1:
            return response(finish="length", reasoning=1900, output=1900)
        return response(observed_json(body), output=500)

    monkeypatch.setattr("factory.providers.openai_compat.requests.post", request)
    provider = reader()
    p = vision_pipeline(tmp_path, provider)
    assert len(shots.build_and_observe(p, "fixture.mp4", 0, 8, tmp_path / "story")) == 2
    assert [body["max_tokens"] for body in calls] == [1900, 3800]
    assert [body["reasoning_effort"] for body in calls] == ["high", "low"]
    progress = read_json(tmp_path / "story/shot_evidence/progress.json")
    assert progress["usage"]["budget_recoveries"] == 1
    assert progress["usage"]["total_tokens"] == usage_summary()["totals"]["total_tokens"] == 2600
    assert usage_summary()["totals"]["requests"] == 2
    assert provider.conf["max_tokens"] == 777 and provider.conf["reasoning_effort"] == "high"


def test_configured_visual_budget_is_not_shrunk_before_sending(tmp_path, monkeypatch):
    import factory.mining.shot_evidence as shots
    local = local_shots(tmp_path, count=2)
    monkeypatch.setattr(shots, 'build_local_index', lambda *a, **k: local)
    calls = []
    monkeypatch.setattr('factory.providers.openai_compat.requests.post', lambda *a, **k:
                        calls.append(k['json']) or response(observed_json(k['json']), output=700))
    provider = reader()
    provider.conf['max_tokens'] = 3800
    assert len(shots.build_and_observe(vision_pipeline(tmp_path, provider), 'fixture.mp4', 0, 8,
                                      tmp_path / 'story')) == 2
    assert len(calls) == 1 and calls[0]['max_tokens'] == 3800


def test_failed_budget_recovery_is_not_repeated_automatically_on_resume(tmp_path, monkeypatch):
    import factory.mining.shot_evidence as shots
    local = local_shots(tmp_path, count=2)
    monkeypatch.setattr(shots, "build_local_index", lambda *a, **k: local)
    calls = []
    monkeypatch.setattr("factory.providers.openai_compat.requests.post", lambda *a, **k:
        calls.append(copy.deepcopy(k["json"])) or response(finish="length", reasoning=k["json"]["max_tokens"], output=k["json"]["max_tokens"]))
    p = vision_pipeline(tmp_path, reader())
    folder = tmp_path / "story"
    for expected in (2, 3):
        with pytest.raises(ProviderEmptyResponseError):
            shots.build_and_observe(p, "fixture.mp4", 0, 8, folder)
        assert len(calls) == expected
    assert [call["max_tokens"] for call in calls] == [1900, 3800, 3800]
    assert read_json(folder / "shot_evidence/progress.json")["usage"]["budget_recoveries"] == 1
    assert not (folder / "shot_evidence/catalog.json").exists()


def test_provider_error_in_fourth_story_preserves_three_edited_drafts_and_pending_state(tmp_path, monkeypatch):
    import factory.mining.visual_story as director
    source = tmp_path / "source.mp4"
    source.write_bytes(b"fixture source")
    p = ProductionPipeline(Config({"workspace": str(tmp_path / "ws"), "params": {
        "max_clips": 5, "review_mode": "semi", "comment_hook_text": ""}}), str(source))
    p.state.completed_stages = ["URL_ANALYZING"]
    p.state.artifacts.update(meta={"duration": 500}, video_path=str(source))
    p.index = MediaIndex(video=VideoMeta(path=str(source), duration=500), scenes=[SceneSegment(scene_id=1, start=0, end=20)])
    p.candidates = [StoryCandidate(candidate_id="C%03d" % index, title="已修改稿%d" % index,
        source_start=(index-1)*100, source_end=(index-1)*100+60, status="review_required", selected_script="USER%d" % index,
        scripts=[ScriptVariant(variant_id="USER%d" % index, sentences=[ScriptSentence(index=1,
            text="保留我的第%d段修改文案。" % index, source_evidence=Evidence(start=0, end=20))])]) for index in range(1, 4)]
    p._save_candidates()
    selection = {"mode": "limit", "requested_limit": 5, "candidates": [{"title": "故事%d" % index,
        "source_start": (index-1)*100, "source_end": (index-1)*100+60, "story_summary": "完整章节"} for index in range(1, 6)]}
    write_json(p.job_dir / "analysis/director/selection.json", selection)
    provider = SimpleNamespace(is_mock=False)
    p.router = SimpleNamespace(get=lambda role: provider, is_mock=lambda role: False)
    monkeypatch.setattr(director, "media_duration", lambda _: 500)
    monkeypatch.setattr(director, "storyboard", lambda *a, **k: pytest.fail("Saved selection must be reused"))
    calls = []

    def fail(pipeline, video, lo, hi, folder):
        calls.append(folder.name)
        raise ProviderEmptyResponseError({"model": "fixture", "finish_reason": "stop", "total_tokens": 1900})

    monkeypatch.setattr("factory.mining.shot_evidence.build_and_observe", fail)
    state = p.run()
    assert state.status == "CANDIDATE_FILTERED" and state.error == ""
    assert calls == ["story_04"] and not p.completed("CANDIDATE_FILTERED")
    assert [item["number"] for item in state.artifacts["pending_stories"]] == [4, 5]
    saved = read_json(p.job_dir / "analysis/story_candidates.json")
    assert len(saved) == 3
    assert [item["selected_script"] for item in saved] == ["USER1", "USER2", "USER3"]
    assert all("保留我的第%d段修改文案" % index in saved[index-1]["scripts"][0]["full_text"] for index in range(1, 4))
    assert not p.completed("TTS_GENERATED")
