import hashlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from factory.config import Config
from factory.models import Evidence, ScriptSentence, ScriptVariant, StoryCandidate
from factory.pipeline import ProductionPipeline
from factory.production import RENDER_POLICY, do_tts, reconcile_production_queue
from factory.production_queue import (STAGES, checkpoint_voice, file_digest, picture_key,
                                      quality_key, render_key)
from factory.providers.tts.base import write_wav
from factory.providers.tts.continuous import synthesis_identity
from factory.utils import read_json, write_json


def pipeline(tmp_path):
    config = Config({"workspace": str(tmp_path / "ws"), "params": {
        "review_mode": "semi", "max_clips": 0, "min_duration": 1, "max_duration": 10,
        "comment_hook_text": "", "scene_align": "off"}, "tts": {
        "type": "bailian", "model": "cosyvoice-v2", "voice": "N1", "speed": 1.2,
        "voices": [{"name": "N1", "voice_id": "cosyvoice-v2-fixture", "model": "cosyvoice-v2"}]}})
    p = ProductionPipeline(config, "fixture.mp4")
    p.state.artifacts.update(video_path=str(tmp_path / "source.mp4"), source_size=[1920, 1080])
    (tmp_path / "source.mp4").write_bytes(b"source fixture")
    p.index = SimpleNamespace(video=SimpleNamespace(duration=100), scenes=[], text_in=lambda *args: "")
    p._load_index = lambda: p.index
    return p


def candidate(p, number):
    cid = "C%03d" % number
    texts = ["女孩发现门外站着怪人。", "她锁好门躲在屋里。", "怪人却从窗外探进头。", "女孩逃出房间向邻居求助。"]
    variant = ScriptVariant(candidate_id=cid, variant_id="D1", sentences=[
        ScriptSentence(index=i + 1, text=text, estimated_seconds=1,
                       source_evidence=Evidence(start=number * 10 + i, end=number * 10 + i + 1))
        for i, text in enumerate(texts)], full_text="".join(texts), estimated_seconds=4)
    item = StoryCandidate(candidate_id=cid, source_start=number * 10, source_end=number * 10 + 4,
                          scripts=[variant], selected_script="D1", status="approved")
    p.candidates.append(item)
    return item


def narration(p, item, folder):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    voice = folder / "voice.wav"
    write_wav(voice, .05 * np.sin(np.arange(24000) * .05), 24000)
    text = p._pick_variant(item).full_text
    return {"audio_path": str(voice), "duration": 4, "text": text, "provider": "test_fixture",
            "synthesis_identity": synthesis_identity(p.config.tts), "audio_sha256": file_digest(voice),
            "content_check": {"status": "pass"}, "audio_edits": []}


def finished(p, item):
    reconcile_production_queue(p, [*p.candidates])
    clip = next(clip for clip in p.clips if clip.candidate_id == item.candidate_id)
    variant = p._pick_variant(item)
    folder = p.job_dir / "clips" / clip.clip_id
    result = narration(p, item, folder)
    write_json(folder / "tts.json", result)
    checkpoint_voice(p, clip, item, variant, result)
    write_json(folder / "timeline.json", {"voice_path": result["audio_path"], "source_start": item.source_start,
                                          "source_duration": 4, "sentence_timings": []})
    clip.scores["picture_key"] = picture_key(p, clip, item, variant)
    output = folder / (clip.clip_id + ".mp4")
    output.write_bytes(b"completed media " + item.candidate_id.encode())
    clip.path = str(output)
    clip.status = "completed"
    clip.scores.update(render_policy=RENDER_POLICY, render_key=render_key(p, clip, RENDER_POLICY),
                       render_media_digest=file_digest(output))
    clip.qc, clip.gate2 = {"status": "pass", "issues": []}, {"status": "pass"}
    clip.scores.update(quality_key=quality_key(p, clip, item, variant),
                       final_check_key=quality_key(p, clip, item, variant))
    p._save_clips()
    return clip


def test_later_approval_adds_only_missing_narration_and_picture_stages(tmp_path, monkeypatch):
    import factory.production as production
    p = pipeline(tmp_path)
    a = candidate(p, 1)
    old = finished(p, a)
    old_audio = file_digest(old.scores["tts_path"])
    b = candidate(p, 2)
    p.state.completed_stages = list(STAGES)
    report = reconcile_production_queue(p, [a, b])
    assert report["changed_clip_ids"] == ["clip_002"]
    assert report["reused_clip_ids"] == ["clip_001"]
    assert "TTS_GENERATED" not in p.state.completed_stages
    assert p.clips[0] is old and old.status == "completed"
    calls = []

    def synth(tts, variant, folder):
        calls.append(("tts", variant.candidate_id))
        return narration(p, b, folder)

    monkeypatch.setattr(production, "preflight_scripts", lambda p, q: None)
    monkeypatch.setattr("factory.providers.tts.continuous.synth_continuous", synth)
    do_tts(p, [a, b])
    assert calls == [("tts", b.candidate_id)]
    assert old.status == "completed" and old.qc["status"] == "pass"
    assert file_digest(old.scores["tts_path"]) == old_audio
    monkeypatch.setattr("factory.story_library.refresh_evidence_windows", lambda *a: None)
    monkeypatch.setattr(production, "upgrade_narration_alignment", lambda p, v, result, path: result)

    def timeline(pipeline, item, variant, result, *args, **kwargs):
        calls.append(("timeline", item.candidate_id))
        return {"voice_path": result["audio_path"], "source_start": item.source_start,
                "source_duration": 4, "sentence_timings": []}

    monkeypatch.setattr(production, "_initial_catalog", lambda *a: [])
    monkeypatch.setattr('factory.matching.initial_plan.build_initial_timeline', timeline)
    production.do_timeline(p, [a, b])
    assert ("timeline", a.candidate_id) not in calls
    monkeypatch.setattr("factory.utils.media_size", lambda path: (1920, 1080))
    monkeypatch.setattr("factory.render.clean.detect_black_bars", lambda *a: None)
    monkeypatch.setattr("factory.render.reframing.estimate_focus", lambda *a: .5)
    monkeypatch.setattr("factory.render.reframing.focus_expression", lambda *a: "")
    monkeypatch.setattr("factory.utils.media_duration", lambda path: 4)

    def export(source, voice, subtitle, timeline, params, output, **kwargs):
        calls.append(("render", str(output.parent.name)))
        Path(output).write_bytes(b"new rendered media")

    monkeypatch.setattr(production, "export_clip", export)
    production.do_render(p, [a, b])
    assert ("render", old.clip_id) not in calls
    monkeypatch.setattr(production, "quality_gate", lambda *a, **k: {"status": "pass"})
    production.do_final_check(p, [a, b])
    monkeypatch.setattr(production, "final_qc", lambda *a, **k: {"status": "pass", "issues": []})
    monkeypatch.setattr("factory.eval.media_audit.audit_media", lambda *a: {"status": "pass", "issues": []})

    def semantic(clip, *args, **kwargs):
        calls.append(("semantic", clip.candidate_id))
        return {"status": "pass"}

    monkeypatch.setattr(production, "semantic_final_check", semantic)
    assert production.do_qc(p, [a, b])
    assert ("semantic", a.candidate_id) not in calls
    assert ("semantic", b.candidate_id) in calls
    assert old.status == "completed" and file_digest(old.scores["tts_path"]) == old_audio


def test_shrinking_and_expanding_limit_parks_and_restores_original_outputs(tmp_path):
    p = pipeline(tmp_path)
    a, b = candidate(p, 1), candidate(p, 2)
    first, second = finished(p, a), finished(p, b)
    before = file_digest(second.path)
    report = reconcile_production_queue(p, [a])
    assert p.clips == [first] and report["parked_clip_ids"] == [second.clip_id]
    assert Path(second.path).is_file()
    p._load_clips()
    report = reconcile_production_queue(p, [a, b])
    assert [clip.clip_id for clip in p.clips] == [first.clip_id, second.clip_id]
    assert report["changed_clip_ids"] == [] and report["parked_clip_ids"] == []
    assert "TTS_GENERATED" not in report["reset_stages"]
    assert file_digest(p.clips[1].path) == before and p.clips[1].status == "completed"


@pytest.mark.parametrize("change", ["spoken", "clone", "speed", "corrupt_audio"])
def test_changed_spoken_text_or_actual_voice_never_reuses_old_audio(tmp_path, change):
    p = pipeline(tmp_path)
    item = candidate(p, 1)
    old = finished(p, item)
    if change == "spoken":
        variant = p._pick_variant(item)
        variant.sentences[0].text = "女孩发现窗外站着怪人。"
        variant.full_text = "".join(s.text for s in variant.sentences)
    elif change == "clone":
        p.config.tts["voices"][0]["voice_id"] = "cosyvoice-v2-another"
    elif change == "speed":
        p.config.tts["speed"] = 1.1
    else:
        Path(old.scores["tts_path"]).write_bytes(b"damaged")
    report = reconcile_production_queue(p, [item])
    assert p.clips[0].clip_id != old.clip_id
    assert report["reused_clip_ids"] == []
    assert report["parked_clip_ids"] == [old.clip_id]
    assert Path(old.path).is_file()


def test_registry_alias_changes_with_same_actual_clone_reuse_narration(tmp_path):
    p = pipeline(tmp_path)
    item = candidate(p, 1)
    old = finished(p, item)
    p.config.tts["voice"] = "renamed"
    p.config.tts["voices"][0]["name"] = "renamed"
    p.config.tts["api_key"] = "fixture-current-credential"
    report = reconcile_production_queue(p, [item])
    assert p.clips == [old] and report["changed_clip_ids"] == []
    assert report["reused_clip_ids"] == [old.clip_id]


def test_visual_change_keeps_whole_voice_and_rebuilds_picture_only(tmp_path):
    p = pipeline(tmp_path)
    item = candidate(p, 1)
    old = finished(p, item)
    p._pick_variant(item).comment_visual_mode = "motion"
    p.state.completed_stages = list(STAGES)
    report = reconcile_production_queue(p, [item])
    assert p.clips == [old] and report["reused_clip_ids"] == [old.clip_id]
    assert "TTS_GENERATED" in p.state.completed_stages
    assert "MATCHING" not in p.state.completed_stages


def test_mid_batch_tts_failure_keeps_first_voice_checkpoint_and_reuses_on_retry(tmp_path, monkeypatch):
    import factory.production as production
    p = pipeline(tmp_path)
    a, b = candidate(p, 1), candidate(p, 2)
    monkeypatch.setattr(production, "preflight_scripts", lambda p, q: None)
    calls = []
    fail_second = [True]

    def synth(tts, variant, folder):
        calls.append(variant.candidate_id)
        if variant.candidate_id == b.candidate_id and fail_second[0]:
            raise RuntimeError("fixture timeout")
        return narration(p, a if variant.candidate_id == a.candidate_id else b, folder)

    monkeypatch.setattr("factory.providers.tts.continuous.synth_continuous", synth)
    with pytest.raises(RuntimeError, match="timeout"):
        do_tts(p, [a, b])
    stored = read_json(p.job_dir / "clips" / "clips.json")
    assert stored[0]["candidate_id"] == a.candidate_id and stored[0]["status"] == "tts_done"
    digest = hashlib.sha256(Path(stored[0]["scores"]["tts_path"]).read_bytes()).hexdigest()
    p._load_clips()
    fail_second[0] = False
    do_tts(p, [a, b])
    assert calls.count(a.candidate_id) == 1 and calls.count(b.candidate_id) == 2
    assert file_digest(p.clips[0].scores["tts_path"]) == digest


def test_changed_output_digest_resets_render_and_keeps_unchanged_narration(tmp_path):
    p = pipeline(tmp_path)
    item = candidate(p, 1)
    old = finished(p, item)
    Path(old.path).write_bytes(b"different output")
    p.state.completed_stages = list(STAGES)
    report = reconcile_production_queue(p, [item])
    assert p.clips == [old]
    assert "TTS_GENERATED" in p.state.completed_stages
    assert "RENDERING" not in p.state.completed_stages
    assert report["changed_clip_ids"] == [old.clip_id]


def test_prediction_calibration_does_not_repeat_picture_or_cloud_review(tmp_path):
    p = pipeline(tmp_path)
    item = candidate(p, 1)
    old = finished(p, item)
    for sentence in p._pick_variant(item).sentences:
        sentence.estimated_seconds *= 1.5
    report = reconcile_production_queue(p, [item])
    assert p.clips == [old] and report["changed_clip_ids"] == []


def test_unavailable_semantic_review_remains_retryable(tmp_path):
    p = pipeline(tmp_path)
    item = candidate(p, 1)
    old = finished(p, item)
    old.qc.update(status="review", semantic_check={"status": "review", "reason": "timeout"})
    p.state.completed_stages = list(STAGES)
    report = reconcile_production_queue(p, [item])
    assert p.clips == [old] and report["changed_clip_ids"] == [old.clip_id]
    assert "QUALITY_CHECK" not in p.state.completed_stages
    assert "RENDERING" in p.state.completed_stages


@pytest.mark.parametrize("mode", ["auto", "semi"])
@pytest.mark.parametrize("status", ["approved", "gate1_pass"])
def test_selected_copy_skips_story_critic_and_keeps_words_in_any_mode(tmp_path, monkeypatch, mode, status):
    import factory.production as production
    p = pipeline(tmp_path)
    item = candidate(p, 1)
    item.status = status
    p.config.params.update(review_mode=mode, max_duration=30)
    variant = p._pick_variant(item)
    for sentence in variant.sentences:
        sentence.estimated_seconds = 100
    variant.match_report = {"story_grounding": {"status": "fail", "reason": "old critic"}}
    original = variant.full_text
    monkeypatch.setattr("factory.mining.visual_story.review_candidate_story",
                        lambda *a: pytest.fail("Human-approved copy must not be judged again"))
    monkeypatch.setattr("factory.script.repair.compress_variant",
                        lambda *a: pytest.fail("Approved words must not be compressed"))
    monkeypatch.setattr("factory.render.timeline.shorten_variant_to_fit",
                        lambda *a: pytest.fail("Approved sentences must not be dropped"))
    seen = []

    def synth(tts, selected, folder):
        seen.append(selected.full_text)
        assert selected.user_approved_copy is True
        return narration(p, item, folder)

    monkeypatch.setattr("factory.providers.tts.continuous.synth_continuous", synth)
    production.do_tts(p, [item])
    assert seen == [original] and variant.full_text == original
    assert item.status == status


def test_old_asr_difference_never_rewrites_or_blocks_selected_copy(tmp_path, monkeypatch):
    import factory.production as production
    p = pipeline(tmp_path)
    item = candidate(p, 1)
    p.config.params.update(review_mode="auto", max_duration=30)
    original = p._pick_variant(item).full_text
    monkeypatch.setattr("factory.script.voice_repair.repair_for_speech",
                        lambda *a: pytest.fail("An audio failure cannot rewrite approved copy"))

    def result_with_old_comparison(tts, variant, folder):
        result = narration(p, item, folder)
        result['content_check'] = {'status': 'fail', 'issues': [{'text': 'ASR difference'}]}
        return result

    monkeypatch.setattr("factory.providers.tts.continuous.synth_continuous", result_with_old_comparison)
    production.do_tts(p, [item])
    assert p._pick_variant(item).full_text == original
    assert item.status == 'approved' and p.clips[0].status == 'tts_done'
    assert reconcile_production_queue(p, [item])['reused_clip_ids'] == [p.clips[0].clip_id]


def test_old_story_rejection_rechecks_picture_only_without_resynthesizing(tmp_path, monkeypatch):
    import factory.production as production
    p = pipeline(tmp_path)
    item = candidate(p, 1)
    clip = finished(p, item)
    clip.qc = {"status": "fail", "issues": ["old story incomplete"]}
    clip.gate2 = {"status": "fail", "reason": "old story score"}
    clip.scores.pop("quality_key")
    clip.scores.pop("final_check_key")
    p.state.completed_stages = list(STAGES)
    report = reconcile_production_queue(p, [item])
    assert "TTS_GENERATED" in p.state.completed_stages and "RENDERING" in p.state.completed_stages
    assert report["reset_stages"][0] == "FINAL_MATCH_CHECK"
    monkeypatch.setattr(production, "quality_gate", lambda *a, **k: pytest.fail("Old story scores cannot block approved words"))
    monkeypatch.setattr("factory.utils.media_duration", lambda *a: 4)
    production.do_final_check(p, [item])
    assert clip.gate2["status"] == "skipped"


def test_only_picture_review_can_accept_export_despite_old_text_audio_and_length_flags(tmp_path, monkeypatch):
    import factory.production as production
    p = pipeline(tmp_path)
    item = candidate(p, 1)
    clip = finished(p, item)
    clip.qc = {}
    clip.gate2 = {'status': 'fail', 'reason': 'old literary gate'}
    p.config.params.update(min_duration=35, max_duration=50)
    p.state.artifacts['source_size'] = [320, 180]
    monkeypatch.setattr(production, 'final_qc', lambda *a, **k: pytest.fail('There is no second quality review'))
    monkeypatch.setattr('factory.eval.media_audit.audit_media', lambda *a, **k: pytest.fail('No sound or resolution gate'))
    calls = []
    monkeypatch.setattr(production, 'semantic_final_check', lambda *a, **k: calls.append(a) or {'status': 'pass', 'score': 93})
    assert production.do_qc(p, [item])
    assert len(calls) == 1 and clip.status == 'completed'
    assert clip.qc['review_scope'] == 'picture_matching_only'


def test_missing_export_is_a_real_error_not_a_successful_picture_review(tmp_path, monkeypatch):
    import factory.production as production
    from factory.pipeline import PipelineError
    p = pipeline(tmp_path)
    item = candidate(p, 1)
    clip = finished(p, item)
    Path(clip.path).write_bytes(b'')
    monkeypatch.setattr(production, 'semantic_final_check', lambda *a, **k: pytest.fail('There is no exported picture to review'))
    with pytest.raises(PipelineError, match='输出视频不存在或为空'):
        production.do_qc(p, [item])
    assert item.status == 'approved'
