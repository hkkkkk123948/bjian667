from pathlib import Path

import pytest

from factory.config import Config
from factory.pipeline import ProductionPipeline
from factory.utils import read_json

ASSET = Path(__file__).resolve().parent / "assets" / "source_video.mp4"
REF = Path(__file__).resolve().parent / "assets" / "reference_style.txt"


@pytest.mark.skipif(not ASSET.exists(), reason="缺少测试素材，先运行 python tests/make_test_video.py")
def test_full_pipeline(tmp_path, monkeypatch):
    # Explicit fixtures exercise rendering without claiming Mock audio is usable narration.
    from factory.models import ScriptVariant, ScriptSentence, Evidence
    from factory.providers.tts.base import write_wav
    import numpy as np

    def scripts(candidate, *args, **kwargs):
        texts = ["女孩发现门外一直有人。", "她透过猫眼，看见一只眼睛。", "她赶紧后退，把门锁得死死的。",
                 "身后的窗帘突然动了一下。", "女孩回过头，看到藏在窗后的陌生人。", "原来门外的人只是负责引开她。"]
        span = candidate.duration / len(texts)
        sentences = [ScriptSentence(index=i + 1, text=text, estimated_seconds=3,
                     source_evidence=Evidence(start=candidate.source_start + i * span,
                                              end=candidate.source_start + (i + 1) * span))
                     for i, text in enumerate(texts)]
        return [ScriptVariant(variant_id="fixture", candidate_id=candidate.candidate_id,
                              sentences=sentences, full_text="".join(texts), estimated_seconds=18)]

    def synth(self, text, out_path, **kwargs):
        path = Path(out_path).with_suffix(".wav")
        duration = 3.0 * max(1, text.count("。"))
        t = np.arange(int(duration * 24000)) / 24000
        write_wav(path, .1 * np.sin(2 * np.pi * 180 * t), 24000)
        return {"audio_path": str(path), "duration": duration, "provider": "test_fixture"}

    monkeypatch.setattr("factory.pipeline.write_scripts", scripts)
    monkeypatch.setattr("factory.providers.tts.manager.TTSRouter.synth", synth)
    # This synthetic colour-bar video has no real story objects. Supply the
    # explicit fixture shot bindings, and test actual full-audio rendering;
    # the real action matcher is covered separately in test_initial_plan.
    from factory.render.timeline import build_timeline
    monkeypatch.setattr("factory.production._initial_catalog", lambda *a: [])
    monkeypatch.setattr("factory.matching.initial_plan.build_initial_timeline",
                        lambda p, c, v, t, catalog: build_timeline(c, v, t, p.config.params))
    monkeypatch.setattr("factory.production.semantic_final_check",
                        lambda *a, **k: {"status": "pass", "score": 100, "basis": "explicit_test_fixture"})
    config = Config({"workspace": str(tmp_path / "ws"), "params": {
        "review_mode": "auto", "min_match": 40, "max_clips": 1, "prefer_nvenc": "off"}})
    refs = [str(REF)] if REF.exists() else []
    pipe = ProductionPipeline(config, str(ASSET), references=refs)
    state = pipe.run()
    assert state.status == "COMPLETED", state.error

    clips = pipe.clips
    assert 1 <= len(clips) <= int(config.param("max_clips"))
    for clip in clips:
        path = Path(clip.path)
        assert path.exists()
        from factory.utils import media_duration
        duration = media_duration(path)
        assert duration >= 5.0
        assert clip.status == "completed"
        assert clip.qc.get("status") in ("pass", "review")
        assert clip.gate2.get("status") == "skipped"

    report = pipe.report()
    assert report["selected_count"] >= 1
    assert len(report["clips"]) == len(clips)

    candidates = pipe._load_candidates()
    assert any(c.match_report.get("gate1", {}).get("basis") == "selection_only" for c in candidates)

    job_json = read_json(pipe.job_dir / "job.json")
    assert job_json["status"] == "COMPLETED"
    assert "QUALITY_CHECK" in job_json["completed_stages"]
    assert "DRAFT_EXPORT" in job_json["completed_stages"]

    from factory.export.capcut import detect_draft_root
    if detect_draft_root():
        exported = [c for c in clips if c.draft_path]
        assert len(exported) == len(clips)
        from pathlib import Path as _P
        for clip in exported:
            assert (_P(clip.draft_path) / "draft_content.json").exists()
            assert (_P(clip.draft_path) / "draft_meta_info.json").exists()

    resumed = ProductionPipeline(Config({"workspace": str(tmp_path / "ws")}), str(ASSET),
                                 job_id=pipe.job_id)
    assert resumed.completed("RENDERING")
    assert resumed.state.status == "COMPLETED"


@pytest.mark.skipif(not ASSET.exists(), reason="缺少测试素材")
def test_index_only(tmp_path):
    config = Config({"workspace": str(tmp_path / "ws2")})
    pipe = ProductionPipeline(config, str(ASSET))
    pipe._stage("URL_ANALYZING")
    pipe._do_ingest()
    pipe._done("URL_ANALYZING")
    pipe._stage("MEDIA_INDEXING")
    pipe._do_index()
    pipe._done("MEDIA_INDEXING")
    index = pipe._load_index()
    assert index is not None
    assert len(index.scenes) >= 2
    assert index.scenes[0].start == 0
    assert index.scenes[-1].end == pytest.approx(60.0, abs=0.5)
    assert all(a.end == pytest.approx(b.start, abs=0.05) for a, b in zip(index.scenes, index.scenes[1:]))
    assert len(index.transcript) >= 5
    assert index.transcript_quality.get("ok") is True
    assert index.transcript_quality.get("coverage", 0.0) > 0.5
    assert index.video.duration == pytest.approx(60.0, abs=0.5)
    assert any(index.text_in(s.start, s.end) for s in index.scenes)
