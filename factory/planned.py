"""Render a reviewable authored plan through the same TTS/editor/exporter as automatic production."""
from pathlib import Path

from .config import Config
from .eval.judge import audit_timeline
from .models import Evidence, ScriptSentence, ScriptVariant, StoryCandidate, MediaIndex, VideoMeta, SceneSegment, to_dict
from .pipeline import ProductionPipeline
from .render.qc import final_qc
from .utils import read_json, write_json, media_duration


def render_plan(config, plan_path, emitter=None, job_id=None):
    plan = read_json(plan_path)
    if not isinstance(plan, dict) or not plan.get("source") or not plan.get("sentences"):
        raise ValueError("方案必须包含source和sentences")
    source = Path(plan["source"]).resolve()
    duration = media_duration(source)
    sentences = []
    intervals = []
    for i, beat in enumerate(plan["sentences"], 1):
        a, b = float(beat["evidence_start"]), float(beat["evidence_end"])
        if not 0 <= a < b <= duration:
            raise ValueError("第%d句证据超出源片" % i)
        intervals.append((a, b))
        sentences.append(ScriptSentence(index=i, text=beat["text"], role=beat.get("role", "context"),
                          estimated_seconds=len(beat["text"]) / 6.2,
                          source_evidence=Evidence(start=a, end=b, scene_ids=[i], quote=beat.get("visual", ""))))
    config = config.clone() if isinstance(config, Config) else Config(config)
    config.params.update(max_clips=1, scene_align="off", evidence_cut="on")
    p = ProductionPipeline(config, str(source), emitter=emitter, job_id=job_id)
    write_json(p.job_dir / "analysis" / "authored_plan.json", plan)
    if not p.completed("URL_ANALYZING"):
        p._stage("URL_ANALYZING")
        p._do_ingest()
        p._done("URL_ANALYZING")
    variant = ScriptVariant(variant_id="PLAN1", candidate_id="C001", sentences=sentences,
                            scene_matches=[{"sentence_index": i+1, "source_windows": b["source_windows"]}
                                           for i, b in enumerate(plan["sentences"]) if b.get("source_windows")],
                            full_text="".join(s.text for s in sentences),
                            estimated_seconds=sum(s.estimated_seconds for s in sentences))
    candidate = StoryCandidate(candidate_id="C001", title=plan.get("title", "审阅方案"),
                               source_start=min(a for a, b in intervals), source_end=max(b for a, b in intervals),
                               scripts=[variant], selected_script="PLAN1", status="approved",
                               story_summary=plan.get("story_logic", ""))
    p.candidates = [candidate]
    p.index = MediaIndex(video=VideoMeta(**{k: v for k, v in p.state.artifacts["meta"].items() if k in VideoMeta.__annotations__}),
                         scenes=[SceneSegment(scene_id=i + 1, start=a, end=b,
                                  visual=plan["sentences"][i].get("visual", "")) for i, (a, b) in enumerate(intervals)])
    write_json(p.job_dir / "index" / "video_index.json", to_dict(p.index))
    p._save_candidates()
    p.state.artifacts["editorial_review"] = {"mode": "authored_plan", "reviewer": plan.get("reviewer", "unspecified"),
                                             "model_score": None, "note": "导入已审阅的分镜方案，不伪造模型匹配分"}
    p._done("MEDIA_INDEXING")
    p._done("CANDIDATE_FILTERED")
    try:
        for name, operation in [("TTS_GENERATED", p._do_tts), ("TIMELINE_ALIGNED", p._do_timeline), ("RENDERING", p._do_render)]:
            p._stage(name)
            operation([candidate])
            p._done(name)
        for clip in p.clips:
            timeline = read_json(Path(clip.path).parent / "timeline.json")
            validation = final_qc(clip, clip.script_text, p.config.params)
            if validation['status'] != 'pass':
                raise RuntimeError('真实导出失败：'+'；'.join(validation.get('issues') or ['视频无法读取']))
            # Cheap diagnostics remain available to a developer, without
            # becoming a second review or another paid narration/render loop.
            clip.scores['export_validation'] = validation
            clip.scores['timeline_diagnostics'] = dict(audit_timeline(timeline), gate_enabled=False)
            clip.gate1 = {"status": "authored_plan", "model_score": None}
        p._save_clips()
        p._stage('QUALITY_CHECK')
        p._do_qc([candidate])
        p._done("QUALITY_CHECK")
        p.state.status = "COMPLETED" if all(c.status == "completed" for c in p.clips) else "REVIEW_REQUIRED"
        p.save_state()
        write_json(p.job_dir / "quality_report.json", [c.qc for c in p.clips])
        p._write_report()
        return p
    except Exception as exc:
        p._fail(exc)
        raise
