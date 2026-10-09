from types import SimpleNamespace

import pytest

from factory.models import ClipResult
from factory.render.qc import _alignment_review, final_qc, semantic_final_check
from factory.utils import read_json, write_json


TIMINGS = [{"text": "女孩向男人求助。", "narrative_text": "女孩向男人求助。", "start": 1.0, "end": 4.0}]


def test_confirmed_story_structure_is_not_rejudged():
    result = _alignment_review({"score": 96, "mismatches": [], "simple_story": False,
                                "causal_complete": False, "missing_story_links": ["无过渡"]}, TIMINGS, 80)
    assert result["status"] == "pass"
    assert result["basis"] == "approved_script_visual_alignment"
    assert "causal_complete" not in result and "missing_story_links" not in result


def test_concrete_wrong_shot_beats_high_score_and_uses_real_sentence_clock():
    result = _alignment_review({"score": 99, "mismatches": [{"sentence_index": 1,
        "start": 80, "end": 90, "reason": "此处用了林地捡石镜头，未对应求助事件",
        "evidence": "两张图均为女孩独自在林地捡石，求助镜头出现在另一个时间段"}]}, TIMINGS, 80)
    assert result["status"] == "fail" and result["score"] == 99
    assert result["mismatches"][0]["start"] == 1.0 and result["mismatches"][0]["end"] == 4.0
    assert "第1句（1.00–4.00秒）" in result["reason"]


@pytest.mark.parametrize("data", [
    {"score": 99}, {"score": True, "mismatches": []}, {"score": float("nan"), "mismatches": []},
    {"score": 101, "mismatches": []}, {"score": 95, "mismatches": "没有"},
    {"score": 95, "mismatches": [{"sentence_index": 2, "reason": "错位", "evidence": "另一个人"}]},
    {"score": 95, "mismatches": [{"sentence_index": 1, "reason": "抽帧没看到动作"}]},
    {"score": 95, "mismatches": [], "uncertain_sentences": "不清楚"},
    {"consistency": 99, "unsupported_claims": [], "simple_story": True, "causal_complete": True},
])
def test_invalid_or_old_story_review_never_becomes_a_pass(data):
    result = _alignment_review(data, TIMINGS, 80)
    assert result["status"] == "review" and result["score"] is None
    assert result["mismatches"] == []


def test_motion_not_caught_by_sparse_frames_stays_uncertain():
    result = _alignment_review({"score": 98, "mismatches": [], "uncertain_sentences": [
        {"sentence_index": 1, "reason": "两张图未捕获动作过程，需核查实际镜头"}]}, TIMINGS, 80)
    assert result["status"] == "review" and result["mismatches"] == []
    assert "不据此改写已确认文案" in result["reason"]
    assert _alignment_review({"score": 40, "mismatches": []}, TIMINGS, 80)["status"] == "review"


def test_visual_review_cache_tracks_confirmed_text_timeline_and_policy(tmp_path, monkeypatch):
    from factory.versioning import PROMPT_VERSIONS
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"complete fixture")
    write_json(tmp_path / "timeline.json", {"sentence_timings": TIMINGS})
    calls = []

    def complete(prompt, **kwargs):
        calls.append(prompt)
        assert "不评价文案好坏" in prompt and "uncertain_sentences" in prompt
        assert "抽帧没捕获动作过程不等于动作不存在" in prompt
        assert "原片未确认的转写" not in prompt and "作者未确认描述" not in prompt
        return {"score": 99, "mismatches": [{"sentence_index": 1, "reason": "错配厨房镜头",
                "evidence": "图片中的男人在厨房切菜，求助镜头在另一段"}]}

    provider = SimpleNamespace(model="fixture", conf={}, complete_json=complete)
    router = SimpleNamespace(is_mock=lambda role: False, get=lambda role: provider)
    monkeypatch.setattr("factory.eval.judge.extract_frames", lambda *a, **k: ["a.jpg", "b.jpg"])
    clip = ClipResult(path=str(video))
    args = (clip, "女孩向男人求助。", "原片未确认的转写", router)
    result = semantic_final_check(*args, visuals="作者未确认描述")
    assert result["status"] == "fail" and len(calls) == 1
    assert semantic_final_check(*args)["cache_hit"] and len(calls) == 1
    assert semantic_final_check(clip, "女孩走到男人面前。", "", router)["status"] == "fail"
    assert len(calls) == 2
    write_json(tmp_path / "timeline.json", {"sentence_timings": [dict(TIMINGS[0], end=4.5)]})
    assert semantic_final_check(*args)["status"] == "fail" and len(calls) == 3
    monkeypatch.setitem(PROMPT_VERSIONS, "qc_semantic", "future_visual_policy")
    assert semantic_final_check(*args)["status"] == "fail" and len(calls) == 4
    saved = read_json(tmp_path / "semantic_review.json")
    assert saved["review"]["basis"] == "approved_script_visual_alignment"


def test_uncertain_review_is_not_cached_as_confirmed_result(tmp_path, monkeypatch):
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"fixture")
    write_json(tmp_path / "timeline.json", {"sentence_timings": TIMINGS})
    provider = SimpleNamespace(model="fixture", conf={}, complete_json=lambda *a, **k: {
        "score": 98, "mismatches": [], "uncertain_sentences": [{"sentence_index": 1, "reason": "看不清"}]})
    router = SimpleNamespace(is_mock=lambda role: False, get=lambda role: provider)
    monkeypatch.setattr("factory.eval.judge.extract_frames", lambda *a, **k: ["a.jpg", "b.jpg"])
    assert semantic_final_check(ClipResult(path=str(video)), "正文", "", router)["status"] == "review"
    assert not (tmp_path / "semantic_review.json").exists()


def test_export_validation_only_checks_readable_video_not_other_quality_gates(tmp_path, monkeypatch):
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"fixture")
    import factory.render.qc as qc
    monkeypatch.setattr(qc, "_probe", lambda p: {"duration": 12.0, "width": 320, "height": 180,
                                              "has_video": True, "has_audio": False})
    monkeypatch.setattr(qc, "_loudness", lambda p: pytest.fail("No independent loudness review"))
    monkeypatch.setattr(qc, "_sample_brightness", lambda *a: pytest.fail("No independent brightness review"))
    clip = ClipResult(path=str(video))
    result = final_qc(clip, "Help me!", {}, min_duration=35, max_duration=50, source_size=(320, 180))
    assert result["status"] == "pass" and result["basis"] == "export_file_validity"
    assert result["issues"] == []
    monkeypatch.setattr(qc, "_probe", lambda p: {"has_video": False})
    assert final_qc(clip, "Help me!", {})["status"] == "fail"
    video.unlink()
    assert final_qc(clip, "Help me!", {}) == {"status": "fail", "issues": ["输出文件不存在"]}
