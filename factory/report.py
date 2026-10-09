"""生产报告生成（从 pipeline 拆出，便于维护与单测）。"""

from datetime import datetime

from .utils import fmt_clock, write_json


def write_report(pipeline):
    report = {
        "source_video": pipeline.state.artifacts.get("meta", {}).get("title", ""),
        "source_url": pipeline.url,
        "job_id": pipeline.job_id,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "params": dict(pipeline.config.params),
        "candidate_count": len(pipeline.candidates),
        "selected_count": sum(1 for c in pipeline.candidates if c.status in ("approved", "gate1_pass")),
        "rejected_count": sum(1 for c in pipeline.candidates
                              if c.status.startswith("rejected") or c.status == "not_selected"),
        "draft_root": pipeline.state.artifacts.get("draft_root", ""),
        "llm_usage": _usage(pipeline),
        "token_saving": pipeline.state.artifacts.get("token_saving", {}),
        "stage_times": pipeline.state.artifacts.get("stage_times", {}),
        "schema": _schema(),
        "clips": [],
        "rejected": [],
    }
    for clip in pipeline.clips:
        cand = pipeline._by_id(clip.candidate_id)
        report["clips"].append({
            "clip_id": clip.clip_id,
            "path": clip.path,
            "draft": clip.draft_path,
            "candidate_id": clip.candidate_id,
            "source": "%s - %s" % (fmt_clock(clip.source_start), fmt_clock(clip.source_end)),
            "story_score": cand.story_score,
            "final_match": (clip.scores or {}).get("final_score", cand.final_match),
            "gate2": clip.gate2,
            "qc": clip.qc.get("status"),
            "status": clip.status,
        })
    for cand in pipeline.candidates:
        if cand.status in ("approved", "gate1_pass"):
            continue
        report["rejected"].append({
            "candidate_id": cand.candidate_id,
            "status": cand.status,
            "reason": cand.reject_reason or ("匹配度 %.1f 未达阈值" % cand.final_match if cand.final_match else "未入选"),
            "story_score": cand.story_score,
        })
    write_json(pipeline.job_dir / "reports" / "production_report.json", report)
    pipeline.state.artifacts["report"] = str(pipeline.job_dir / "reports" / "production_report.json")
    pipeline.save_state()
    pipeline.emitter.update(report=report)
    return report


def _schema():
    from .versioning import describe
    return describe()


def _usage(pipeline):
    from .providers.base import usage_summary

    usage = usage_summary()
    params = pipeline.config.params
    price_in = float(params.get("llm_price_input", 0) or 0)
    price_out = float(params.get("llm_price_output", 0) or 0)
    totals = usage.get("totals", {})
    cost = (totals.get("prompt_tokens", 0) / 1e6) * price_in + \
           (totals.get("completion_tokens", 0) / 1e6) * price_out
    usage["estimated_cost"] = round(cost, 4) if (price_in or price_out or not totals.get("total_tokens")) else None
    usage["cost_basis"] = "configured_prices" if price_in or price_out else "unknown_prices" if totals.get("total_tokens") else "no_model_calls"
    usage["price_input_per_m"] = price_in
    usage["price_output_per_m"] = price_out
    return usage
