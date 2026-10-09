import copy
from types import SimpleNamespace

import numpy as np
import pytest

from factory.config import Config
from factory.models import Evidence, ScriptSentence, ScriptVariant, StoryCandidate, to_dict, variant_from_dict
from factory.script.comment_hook import apply_comment_hook, story_text
from factory.providers.tts.base import write_wav
from factory.providers.tts.continuous import synth_continuous
from factory.render.timeline import build_timeline
from factory.utils import write_json


TEXTS = ["女孩发现一个怪人在身后跟着她。", "她赶紧跑到仓库门口找人帮忙。",
         "男人听见呼救开门让女孩进来。", "他陪着女孩报警后送她安全回家。"]


def setup(tmp_path, phrase="忘记带玫瑰花了", position="auto"):
    class Provider:
        conf = {"max_tokens": 2000}
        max_retry = 2
        model = "fixture"
        is_mock = False
        calls = 0

        def complete_json(self, prompt, images=None, **kwargs):
            assert images == []
            assert self.conf["max_tokens"] == 256
            self.calls += 1
            return {"after_sentence": 1}

    provider = Provider()
    config = Config({"workspace": str(tmp_path / "ws"), "params": {
        "comment_hook_text": phrase, "comment_hook_position": position}})
    p = SimpleNamespace(config=config, router=SimpleNamespace(get=lambda role: provider),
                        state=SimpleNamespace(artifacts={}), job_dir=tmp_path / "job",
                        emitter=SimpleNamespace(log=lambda *a: None))
    v = ScriptVariant(variant_id="v", candidate_id="c", sentences=[
        ScriptSentence(index=i+1, text=text, role="hook" if i == 0 else "payoff", estimated_seconds=2,
                       source_evidence=Evidence(start=10*i, end=10*i+5)) for i, text in enumerate(TEXTS)])
    v.full_text = "".join(TEXTS)
    return p, v, provider


def test_auto_aside_is_exact_idempotent_cached_and_not_plot(tmp_path):
    p, v, provider = setup(tmp_path)
    original = copy.deepcopy(v)
    apply_comment_hook(p, v)
    assert v.full_text.count("忘记带玫瑰花了") == 1
    assert [story_text(s) for s in v.sentences] == TEXTS
    assert [s.source_evidence for s in v.sentences] == [s.source_evidence for s in original.sentences]
    apply_comment_hook(p, v)
    assert provider.calls == 1
    assert provider.conf["max_tokens"] == 2000 and provider.max_retry == 2
    roundtrip = variant_from_dict(to_dict(v))
    assert roundtrip == v
    p2, v2, provider2 = setup(tmp_path)
    apply_comment_hook(p2, v2)
    assert provider2.calls == 0
    p.config.params["comment_hook_text"] = "男孩忘记点灯了"
    apply_comment_hook(p, v)
    assert "忘记带玫瑰花了" not in v.full_text
    assert v.full_text.count("男孩忘记点灯了") == 1
    p.config.params["comment_hook_text"] = ""
    apply_comment_hook(p, v)
    assert v.full_text == "".join(TEXTS)
    assert provider.calls == 2


def test_manual_position_and_blank_cost_no_requests(tmp_path):
    p, v, provider = setup(tmp_path, position="late")
    apply_comment_hook(p, v)
    assert v.sentences[-2].comment_hook
    p.config.params["comment_hook_position"] = "early"
    apply_comment_hook(p, v)
    assert v.sentences[0].comment_hook and not v.sentences[-2].comment_hook
    assert provider.calls == 0
    p.config.params["comment_hook_position"] = "auto"
    apply_comment_hook(p, v)
    assert provider.calls == 1


def test_invalid_ai_position_uses_explicit_cached_rule_fallback(tmp_path):
    p, v, provider = setup(tmp_path)
    provider.complete_json = lambda *a, **k: {"after_sentence": 99}
    apply_comment_hook(p, v)
    assert v.sentences[0].comment_hook == "忘记带玫瑰花了"
    info = next(iter(p.state.artifacts["comment_hooks"].values()))
    assert info["basis"] == "local_fallback_invalid_ai_position"
    p2, v2, provider2 = setup(tmp_path)
    apply_comment_hook(p2, v2)
    assert provider2.calls == 0
    assert next(iter(p2.state.artifacts["comment_hooks"].values()))["basis"] == info["basis"]


def test_full_tts_contains_aside_and_picture_continues_same_evidence(tmp_path):
    p, v, _ = setup(tmp_path)
    apply_comment_hook(p, v)

    class Recorder:
        config = p.config
        calls = []

        def synth(self, text, out_path, **kwargs):
            self.calls.append(text)
            write_wav(out_path, .1 * np.sin(np.arange(24000*10) * .1), 24000)
            return {"audio_path": str(out_path), "provider": "test_fixture", "duration": 10}

    tts = Recorder()
    result = synth_continuous(tts, v, tmp_path / "clip")
    assert tts.calls == [v.full_text]
    assert result["audio_edits"] == []
    assert result["voice_sha256"] == result["original_audio_sha256"]
    timeline = build_timeline(StoryCandidate(source_start=0, source_end=40), v, result, {})
    assert len([s for s in timeline["segments"] if s.get('visual_kind')=='comment_still']) == 1
    assert timeline["sentence_timings"][0]["narrative_text"] == TEXTS[0]
    assert timeline["sentence_timings"][0]["comment_hook"] == "忘记带玫瑰花了"
    p.config.params["comment_hook_text"] = "换一句话术"
    apply_comment_hook(p, v)
    with pytest.raises(ValueError, match="重新配音"):
        build_timeline(StoryCandidate(source_start=0, source_end=40), v, result, {})


def test_qc_checks_narrative_without_treating_aside_as_prop(tmp_path, monkeypatch):
    from factory.models import ClipResult
    from factory.render.qc import semantic_final_check
    timeline = {"sentence_timings": [{"text": TEXTS[0] + "忘记带玫瑰花了。", "narrative_text": TEXTS[0],
                                     "comment_hook": "忘记带玫瑰花了", "start": 0, "end": 3}]}
    write_json(tmp_path / "timeline.json", timeline)

    class Provider:
        conf = {}
        data = {"score": 95, "mismatches": [], "uncertain_sentences": []}

        def complete_json(self, prompt, **kwargs):
            assert '"text": "' + TEXTS[0] + '"' in prompt
            assert "忘记带玫瑰花了" not in prompt
            assert "不评价文案好坏" in prompt
            assert "causal_complete" not in prompt and "simple_story" not in prompt
            return self.data

    provider = Provider()
    router = SimpleNamespace(is_mock=lambda role: False, get=lambda role: provider)
    monkeypatch.setattr("factory.eval.judge.extract_frames", lambda *a, **k: ["a.jpg", "b.jpg"])
    clip = ClipResult(path=str(tmp_path / "clip.mp4"))
    assert semantic_final_check(clip, "旁白", "", router)["status"] == "pass"
    provider.data.update(causal_complete=False, missing_story_links=["突然出现的男人如何帮助未交代"])
    assert semantic_final_check(clip, "旁白", "", router)["status"] == "pass"


def test_voice_repair_cannot_change_exact_user_phrase(tmp_path):
    from factory.script.voice_repair import repair_for_speech
    p, v, _ = setup(tmp_path)
    apply_comment_hook(p, v)
    with pytest.raises(RuntimeError, match="停止自动改写话术"):
        repair_for_speech(p, StoryCandidate(), v, {"issues": [{"expected": "忘记带玫瑰花了"}]})


def test_rejected_story_reuses_only_source_span_not_old_narrative(tmp_path, monkeypatch):
    import factory.story_library as library
    source = tmp_path / "source.mp4"
    source.write_bytes(b"source fixture")
    db = tmp_path / "library.json"
    write_json(db, {"stories": [{"source_size": source.stat().st_size,
        "source_sha256": library.source_digest(source), "review_status": "rejected_incomplete",
        "script": {"title": "旧故事", "story_logic": "遇险到脱险", "sentences": [
            {"text": "不完整的旧文案", "visual": "女孩走出房间", "evidence_start": 20, "evidence_end": 80}]}}]})
    monkeypatch.setattr(library, "LIBRARY", db)
    assert library.find_reviewed_story(source, {}) is None
    outline = library.find_story_outline(source)
    assert (outline["source_start"], outline["source_end"]) == (20, 80)
    assert "不完整的旧文案" not in str(outline)
    source.write_bytes(b"changed fixture")
    assert library.find_story_outline(source) is None


def test_writer_cannot_override_verified_timestamps_or_reuse_shots():
    from factory.mining.visual_story import _bind_known_shots, _validate
    shots = [{"shot_id": i + 1, "start": i * 10, "end": i * 10 + 5, "visual": "事实%d" % i}
             for i in range(6)]
    data = {"sentences": [{"text": t, "shot_id": i+1, "evidence_start": 999, "evidence_end": 1000,
                           "source_windows": [{"start": 500, "end": 600}]} for i, t in enumerate(TEXTS)]}
    bound = _bind_known_shots(data, shots)
    assert bound["sentences"][0]["evidence_start"] == 0
    assert bound["sentences"][0]["evidence_end"] == 5
    assert "source_windows" not in bound["sentences"][0]
    data["sentences"][1]["shot_id"] = 1
    assert any("重复" in issue for issue in _validate(_bind_known_shots(data, shots), 0, 60))
    data["sentences"][1]["shot_id"] = 999
    assert any("范围无效" in issue for issue in _validate(_bind_known_shots(data, shots), 0, 60))
    by_order = _bind_known_shots({"texts": ["旁白%d。" % i for i in range(6)]}, shots)
    assert [b["evidence_start"] for b in by_order["sentences"]] == [i * 10 for i in range(6)]
    assert _bind_known_shots({"texts": ["漏句。"]}, shots)["sentences"] == []


def test_punctuation_free_continuous_speech_retains_structured_story_beats(tmp_path, monkeypatch):
    import factory.render.qc as qc
    from factory.models import ClipResult
    path = tmp_path / "clip.mp4"
    path.write_bytes(b"fixture")
    monkeypatch.setattr(qc, "_probe", lambda _: {"duration": 20, "width": 1920, "height": 1080,
                                                "has_video": True, "has_audio": True})
    monkeypatch.setattr(qc, "_loudness", lambda _: -16)
    monkeypatch.setattr(qc, "_sample_brightness", lambda *a: [90, 80, 70])
    sentences = [ScriptSentence(text=t.rstrip("。")) for t in TEXTS]
    text = "".join(s.text for s in sentences)
    result = qc.final_qc(ClipResult(path=str(path)), text, {"target_width": 1920, "target_height": 1080},
                         sentences=sentences)
    assert result["status"] == "pass"
    assert qc.final_qc(ClipResult(path=str(path)), text, {"target_width": 1920, "target_height": 1080})["status"] == "pass"


def test_resume_cannot_skip_failed_quality_check(tmp_path, monkeypatch):
    from factory.pipeline import ProductionPipeline
    from factory.models import ClipResult
    p = ProductionPipeline(Config({"workspace": str(tmp_path / "ws")}), "fixture.mp4")
    p.candidates = [StoryCandidate(candidate_id="c", status="approved")]
    p.clips = [ClipResult(candidate_id="c", status="qc_failed")]
    p.state.completed_stages = ["TTS_GENERATED", "TIMELINE_ALIGNED", "RENDERING", "FINAL_MATCH_CHECK", "QUALITY_CHECK"]
    monkeypatch.setattr(p, "_load_index", lambda: None)
    monkeypatch.setattr(p, "_load_clips", lambda: None)
    assert p.produce().status == "REVIEW_REQUIRED"
    assert not p.completed("COMPLETED")
