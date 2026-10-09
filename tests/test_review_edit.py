import copy

from factory.config import Config
from factory.models import Evidence, ScriptSentence, ScriptVariant, StoryCandidate, ClipResult
from factory.pipeline import ProductionPipeline
from factory.script.comment_hook import apply_comment_hook, story_text
from factory.script.review_edit import COMMENT_MARKER, review_text, edit_variant
from factory.render.timeline import build_timeline


def fixture(tmp_path):
    p = ProductionPipeline(Config({"workspace": str(tmp_path), "params": {"comment_hook_text": "忘记带玫瑰花了",
                           "comment_hook_position": "early"}}), "fixture.mp4")
    texts = ["女孩发现门外一直有人", "她赶紧躲到垃圾箱后面", "怪人却从旁边的门缝探出头", "她向男人求助后等来了警察"]
    v = ScriptVariant(variant_id="v1", candidate_id="c", sentences=[ScriptSentence(index=i+1, text=t,
                      source_evidence=Evidence(start=i*10, end=i*10+5, scene_ids=[i+1]), estimated_seconds=2)
                      for i, t in enumerate(texts)])
    v.full_text = "".join(texts)
    c = StoryCandidate(candidate_id="c", source_start=0, source_end=40, selected_script="v1", scripts=[v], status="approved")
    p.candidates = [c]
    apply_comment_hook(p, v)
    return p, c, v


def test_marked_hook_is_spoken_once_and_preserves_all_picture_evidence(tmp_path):
    p, c, v = fixture(tmp_path)
    old_evidence = [copy.deepcopy(s.source_evidence) for s in v.sentences]
    text = review_text(v).replace("她赶紧躲到垃圾箱后面", "她为了躲开怪人藏到垃圾箱后面")
    fresh, changed = edit_variant(p, v, text)
    assert changed
    assert COMMENT_MARKER not in fresh.full_text
    assert fresh.full_text.count("忘记带玫瑰花了") == 1
    assert [s.source_evidence for s in fresh.sentences] == old_evidence
    assert all(story_text(s).endswith("。") for s in fresh.sentences)
    assert fresh.review_comment_hook == {"text": "忘记带玫瑰花了", "after_sentence": 1}


def test_edit_only_hook_does_not_change_story_or_matching(tmp_path):
    p, c, v = fixture(tmp_path)
    v.scene_matches = [{"sentence_index": 2, "source_windows": [{"start": 10, "end": 12, "weight": 1}]}]
    fresh, changed = edit_variant(p, v, review_text(v).replace("忘记带玫瑰花了", "男孩忘记点灯了"))
    assert not changed
    assert [story_text(s).rstrip("。") for s in fresh.sentences] == [story_text(s).rstrip("。") for s in v.sentences]
    assert fresh.scene_matches == v.scene_matches
    assert "男孩忘记点灯了" in fresh.full_text and "忘记带玫瑰花了" not in fresh.full_text
    apply_comment_hook(p, fresh)
    assert "男孩忘记点灯了" in fresh.full_text


def test_edited_sentence_and_full_duration_use_same_calibrated_clock_as_review(tmp_path):
    from factory.script.duration import estimate_duration, record_duration
    p, c, v = fixture(tmp_path)
    p.config.data["tts"].update(type="bailian", model="cosyvoice-v2", voice="cosyvoice-v2-fixture", speed=1.2)
    for phrase in ("女孩提着灯在走廊里寻找声音" * 8, "男孩发现门外的脚步声越来越近" * 8):
        assert record_duration(p.config, phrase, {"provider": "bailian-cosyvoice", "duration": len(phrase) / 5,
            "voice": "cosyvoice-v2-fixture", "playback_speed": 1.2, "content_check": {"status": "pass"}})
    fresh, _ = edit_variant(p, v, review_text(v))
    assert fresh.estimated_seconds == estimate_duration(p.config, review_text(fresh)).seconds
    for sent in fresh.sentences:
        assert sent.estimated_seconds == estimate_duration(p.config, sent.text).seconds
    assert abs(sum(s.estimated_seconds for s in fresh.sentences) - fresh.estimated_seconds) < 0.03


def test_user_can_move_or_remove_marked_line_without_global_reinsertion(tmp_path):
    p, c, v = fixture(tmp_path)
    lines = review_text(v).splitlines()
    lines[0] = lines[0].replace("（忘记带玫瑰花了）", "")
    lines[2] += "（忘记带玫瑰花了）"
    fresh, _ = edit_variant(p, v, "\n".join(lines))
    assert fresh.sentences[2].comment_hook
    removed, _ = edit_variant(p, fresh, "\n".join(lines).replace("（忘记带玫瑰花了）", ""))
    apply_comment_hook(p, removed)
    assert "忘记带玫瑰花了" not in removed.full_text


def test_edit_invalidates_old_audio_and_requires_user_confirmation(tmp_path):
    p, c, v = fixture(tmp_path)
    p.state.completed_stages = ["CANDIDATE_FILTERED", "TTS_GENERATED", "RENDERING", "QUALITY_CHECK", "COMPLETED"]
    p.clips = [ClipResult(candidate_id="c", status="completed")]
    p.apply_review("c", "edit", new_text=review_text(v))
    assert c.status == "review_required"
    assert "TTS_GENERATED" not in p.state.completed_stages
    assert p.clips[0].status == "outdated_script"
    assert p.production_queue() == []
    p.apply_review("c", "approve")
    assert p.production_queue() == [c]


def test_splitting_a_story_row_keeps_other_shots_and_unique_ranges(tmp_path):
    p, c, v = fixture(tmp_path)
    text = review_text(v).replace("她赶紧躲到垃圾箱后面", "她赶紧往后退。\n然后躲在垃圾箱后面。")
    fresh, _ = edit_variant(p, v, text)
    assert len(fresh.sentences) == 5
    assert fresh.sentences[0].source_evidence == v.sentences[0].source_evidence
    assert fresh.sentences[-1].source_evidence == v.sentences[-1].source_evidence
    timeline = build_timeline(c, fresh, {"duration": 15, "audio_path": "voice.wav"}, {})
    spans = [(s["src_start"], s["src_start"]+s["src_duration"]) for s in timeline["segments"]]
    assert not any(min(b,d)-max(a,c) > .01 for i,(a,b) in enumerate(spans) for c,d in spans[:i])


def test_comment_phrase_is_excluded_from_story_match_scores(tmp_path):
    from factory.matching import score_variant
    from factory.script.comment_hook import strip_hook
    from factory.style.analyzer import _default_suspense_profile
    p, c, v = fixture(tmp_path)
    body = copy.deepcopy(v)
    strip_hook(body)
    a = score_variant(c, body, _default_suspense_profile(), [], params={"content_mode": "creative"})
    b = score_variant(c, v, _default_suspense_profile(), [], params={"content_mode": "creative"})
    assert a.final_score == b.final_score
    for field in ("visual_alignment", "narrative_alignment", "fact_alignment", "information_coverage"):
        assert getattr(a, field) == getattr(b, field)


def test_review_checkpoint_is_after_script_and_before_tts_or_picture_matching(tmp_path, monkeypatch):
    from factory.mining.visual_story import _finish
    from factory.models import SceneSegment
    p, c, v = fixture(tmp_path)
    p.config.params["review_mode"] = "semi"
    c.status = "gate1_pass"
    p.state.artifacts.update(meta={}, video_path="fixture.mp4")
    scenes = [SceneSegment(scene_id=i+1, start=s.source_evidence.start, end=s.source_evidence.end)
              for i,s in enumerate(v.sentences)]
    monkeypatch.setattr(p, "analyze", lambda: _finish(p, [c], scenes))
    calls = []

    def tts(queue):
        calls.append("tts")
        p.clips = [ClipResult(candidate_id="c", status="completed")]

    monkeypatch.setattr(p, "_do_tts", tts)
    monkeypatch.setattr(p, "_do_timeline", lambda q: calls.append("picture_matching"))
    monkeypatch.setattr(p, "_do_render", lambda q: calls.append("render"))
    monkeypatch.setattr(p, "_do_final_check", lambda q: None)
    monkeypatch.setattr(p, "_do_qc", lambda q: True)
    monkeypatch.setattr(p, "_do_draft_export", lambda q: None)
    state = p.run()
    assert p.completed("SCRIPT_GENERATED")
    assert state.status == "CANDIDATE_FILTERED" and state.artifacts["script_review_pending"]
    assert not p.completed("TTS_GENERATED") and not p.completed("MATCHING")
    assert calls == []
    p.apply_review("c", "approve", new_text=review_text(v))
    p.produce()
    assert calls == ["tts", "picture_matching", "render"]
