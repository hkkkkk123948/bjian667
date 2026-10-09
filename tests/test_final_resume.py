from types import SimpleNamespace

from factory.config import Config
from factory.models import ClipResult, ScriptSentence, ScriptVariant, StoryCandidate
from factory.utils import write_json


def test_successful_semantic_review_is_reused_only_for_identical_export(tmp_path, monkeypatch):
    from factory.render.qc import semantic_final_check
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"first export")
    write_json(tmp_path / "timeline.json", {"sentence_timings": [
        {"text": "女孩向男人求助。", "start": 0, "end": 3}]})
    calls = []

    class Provider:
        model = "fixture"
        conf = {}

        def complete_json(self, *a, **k):
            calls.append(True)
            return {"score": 95, "mismatches": [], "uncertain_sentences": [], "reason": "fixture"}

    provider = Provider()
    router = SimpleNamespace(is_mock=lambda role: False, get=lambda role: provider)
    monkeypatch.setattr("factory.eval.judge.extract_frames", lambda *a, **k: ["fixture.jpg"]*2)
    clip = ClipResult(path=str(video))
    assert semantic_final_check(clip, "女孩向男人求助。", "", router)["status"] == "pass"
    assert semantic_final_check(clip, "女孩向男人求助。", "", router)["cache_hit"] is True
    assert len(calls) == 1
    video.write_bytes(b"changed export")
    assert semantic_final_check(clip, "女孩向男人求助。", "", router)["status"] == "pass"
    assert len(calls) == 2


def test_resume_quality_check_restores_source_aspect_ratio(tmp_path, monkeypatch):
    import factory.production as production
    from factory.style.short_story import VERSION
    conf = Config({"params": {"aspect_ratio": "auto"}})
    variant = ScriptVariant(sentences=[ScriptSentence(text="女孩已经安全离开。")])
    video = tmp_path / 'fixture.mp4'
    video.write_bytes(b'export fixture')
    clip = ClipResult(candidate_id="c", path=str(video), script_text="女孩已经安全离开。", gate2={"status": "pass"})
    candidate = StoryCandidate(candidate_id="c")
    p = SimpleNamespace(config=conf, clips=[clip], index=None,
        state=SimpleNamespace(artifacts={"source_size": [1920,1080], "reviewed_source_sha256":"fixture", "reviewed_story_policy":VERSION}),
        _check_cancel=lambda:None, _pick_variant=lambda c:variant, _by_id=lambda c:candidate,
        _save_clips=lambda:None)

    def qc(c, text, window, router, params, **kwargs):
        assert (params["target_width"], params["target_height"]) == (1920,1080)
        return {"status": "pass", "issues": []}

    monkeypatch.setattr(production, "semantic_final_check", qc)
    assert production.do_qc(p, [candidate])
    assert clip.status == "completed"


def test_missing_review_frames_do_not_spend_tokens_or_claim_story_mismatch(tmp_path,monkeypatch):
    from factory.render.qc import semantic_final_check
    video=tmp_path/'clip.mp4'
    video.write_bytes(b'fixture')
    write_json(tmp_path/'timeline.json',{'sentence_timings':[{'text':'正文。','start':0,'end':3}]})
    calls=[]
    provider=SimpleNamespace(model='fixture',conf={},complete_json=lambda *a,**k:calls.append(True))
    router=SimpleNamespace(is_mock=lambda role:False,get=lambda role:provider)
    monkeypatch.setattr('factory.eval.judge.extract_frames',lambda *a,**k:['one.jpg'])
    result=semantic_final_check(ClipResult(path=str(video)),'正文。','',router)
    assert result['status']=='review' and result['score'] is None
    assert result['basis']=='incomplete_visual_evidence' and calls==[]


def test_old_truncated_export_resumes_picture_stages_and_keeps_tts(tmp_path,monkeypatch):
    from factory.production import resume_export_repairs
    video=tmp_path/'old.mp4'
    video.write_bytes(b'fixture')
    clip=ClipResult(path=str(video),scores={'tts_duration':26})
    completed=['TTS_GENERATED','MATCHING','TIMELINE_ALIGNED','RENDERING','FINAL_MATCH_CHECK','QUALITY_CHECK']
    state=SimpleNamespace(completed_stages=completed)
    p=SimpleNamespace(clips=[clip],state=state,completed=lambda s:s in state.completed_stages,
                      emitter=SimpleNamespace(log=lambda *a:None),save_state=lambda:None)
    monkeypatch.setattr('factory.utils.run_cmd',lambda *a,**k:('{"streams":[{"duration":"4"}]}','',0))
    resume_export_repairs(p)
    assert 'TTS_GENERATED' in state.completed_stages
    assert 'RENDERING' not in state.completed_stages and 'QUALITY_CHECK' not in state.completed_stages
