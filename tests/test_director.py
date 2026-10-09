import pytest

from factory.models import Evidence, ScriptSentence, ScriptVariant, StoryCandidate
from factory.render.timeline import build_timeline
from factory.script.quality import script_issues


def fixture_story():
    texts = ["女孩发现门外一直有人。", "她凑近猫眼，看到一只眼睛。", "女孩后退，把房门锁死。", "窗帘后却伸出了一只手。"]
    sentences = [ScriptSentence(index=i + 1, text=t, estimated_seconds=3,
                  source_evidence=Evidence(start=i * 10 + 2, end=i * 10 + 7)) for i, t in enumerate(texts)]
    variant = ScriptVariant(sentences=sentences, full_text="".join(texts))
    timing = [{"index": i + 1, "text": t, "start": i * 3, "end": (i + 1) * 3} for i, t in enumerate(texts)]
    return StoryCandidate(source_start=0, source_end=45), variant, {
        "duration": 12, "audio_path": "voice.wav", "sentence_timings": timing,
        "timing_source": "measured_sentence_audio"}


def test_measured_boundaries_and_evidence_are_preserved():
    candidate, variant, tts = fixture_story()
    timeline = build_timeline(candidate, variant, tts, {"min_duration": 10, "max_duration": 35})
    assert timeline["voice_path"] == "voice.wav"
    assert timeline["timing_source"] == "measured_sentence_audio"
    cursor = 0
    for seg, sent in zip(timeline["segments"], variant.sentences):
        assert seg["out_start"] == cursor
        cursor += seg["dur"]
        assert seg["src_start"] >= sent.source_evidence.start
        assert seg["src_start"] + seg["src_duration"] <= sent.source_evidence.end
    assert cursor == 12


def test_changed_sentence_metadata_keeps_audio_and_falls_back_to_estimated_clock():
    candidate, variant, tts = fixture_story()
    variant.sentences[0].text = "已经改过的文案。"
    timeline = build_timeline(candidate, variant, tts, {})
    assert timeline['voice_path'] == tts['audio_path']
    assert timeline['narration_duration'] == tts['duration']
    assert timeline['speed_factor'] == 1
    assert timeline['sentence_timings'][0]['text'] == variant.sentences[0].text
    assert timeline['timing_source'] == 'estimated_text_metadata_mismatch'
    assert timeline['alignment_report']['confidence'] == 'low'


def test_long_speech_does_not_spill_into_another_shot():
    candidate, variant, tts = fixture_story()
    variant.sentences[0].source_evidence = Evidence(start=2, end=3.5)
    timeline = build_timeline(candidate, variant, tts, {})
    seg = timeline["segments"][0]
    assert seg["src_duration"] == 1.5
    assert seg["dur"] == 3
    assert seg["playback_speed"] == .5


def test_reused_evidence_is_a_picture_diagnostic_not_a_script_gate():
    candidate, variant, tts = fixture_story()
    variant.sentences[1].source_evidence = variant.sentences[0].source_evidence
    timeline = build_timeline(candidate, variant, tts, {})
    assert timeline['segments'][1]['shared_source_interval'] is True
    assert timeline['voice_path'] == tts['audio_path']


def test_two_shots_cover_one_intact_sentence():
    from factory.eval.judge import audit_timeline
    candidate, variant, tts = fixture_story()
    variant.scene_matches = [{"sentence_index": 1, "source_windows": [
        {"start": 2, "end": 4, "weight": .4}, {"start": 5, "end": 7, "weight": .6}]}]
    timeline = build_timeline(candidate, variant, tts, {})
    pieces = [s for s in timeline["segments"] if s["sentence_index"] == 1]
    assert len(pieces) == 2
    assert sum(s["dur"] for s in pieces) == pytest.approx(3)
    assert pieces[1]["out_start"] == pytest.approx(pieces[0]["dur"])
    assert audit_timeline(timeline)["issues"] == []


def test_repeated_old_caption_is_rejected():
    texts = ["紫红外套男子坐在椅子上，侧头望向左侧，白衣从外套下露出。",
             "紫红外套的男子坐在室内，侧头望向左侧，白色上衣从外套下露出。",
             "女孩打开了房门。", "门后却空无一人。"]
    assert any("内容重复" in issue for issue in script_issues([{"text": t} for t in texts]))


def test_pending_review_is_never_auto_produced(tmp_path):
    from factory.config import Config
    from factory.pipeline import ProductionPipeline
    p = ProductionPipeline(Config({"workspace": str(tmp_path), 'params':{'review_mode':'auto'}}), "fixture.mp4")
    p.candidates = [StoryCandidate(candidate_id="pending", status="review_required"),
                    StoryCandidate(candidate_id="passed", status="gate1_pass")]
    assert [c.candidate_id for c in p.production_queue()] == ["passed"]


def test_quota_error_is_not_retried_or_hidden_by_default():
    from factory.providers.base import LLMProvider, ProviderQuotaError

    class Exhausted(LLMProvider):
        calls = 0

        def _complete(self, *args, **kwargs):
            self.calls += 1
            raise ProviderQuotaError("额度不足")

    p = Exhausted({"max_retry": 3})
    with pytest.raises(ProviderQuotaError):
        p.complete_json("fixture", default={})
    assert p.calls == 1


def test_invalid_judge_score_cannot_pass():
    from factory.eval.judge import audit_timeline
    candidate, variant, tts = fixture_story()
    timeline = build_timeline(candidate, variant, tts, {})
    timeline["segments"][1]["out_start"] += 1.0
    assert any("空隙" in issue for issue in audit_timeline(timeline)["issues"])
