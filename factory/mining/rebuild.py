"""故事重建（doc §80）：脚本多轮重写仍不达标时，微调故事边界重新生成。"""

import copy

from ..log import get_logger
from ..matching import run_gates, score_variant
from ..script import write_scripts
from .planner import refine_boundaries

DELTAS = (-4.0, 4.0, -8.0, 8.0)


def rebuild_candidate(pipeline, cand, index, style, params, router):
    """在边界附近尝试偏移并重新生成脚本，取更优者。返回是否改善。"""
    logger = get_logger()
    if str(params.get("story_rebuild", "on")).lower() in ("off", "0", "false"):
        return False
    min_dur = float(params.get("min_duration", 15))
    max_dur = float(params.get("max_duration", 35))
    video_dur = index.video.duration if index and index.video else 0.0
    old_score = float(cand.final_match or 0.0)
    best_score = old_score
    best_trial = None
    for delta in DELTAS:
        start = max(0.0, cand.source_start + delta)
        end = cand.source_end + delta
        if video_dur and end > video_dur:
            shift = end - video_dur
            start -= shift
            end = video_dur
        if start < 0 or end - start < min_dur:
            continue
        trial = copy.deepcopy(cand)
        trial.source_start = round(start, 3)
        trial.source_end = round(end, 3)
        refine_boundaries(trial, index, min_dur, max_dur)
        trial.scripts = write_scripts(trial, style, index.transcript if index else [], params,
                                      router, role="script_generation", index=index,
                                      language=params.get("language", "zh"),
                                      reference_text=getattr(pipeline, "_reference_text", lambda: "")())
        for variant in trial.scripts:
            score_variant(trial, variant, style, index.transcript if index else [],
                          index=index, router=router, role="match_judge", params=params)
        gate1, best_variant, passed = run_gates(trial, trial.scripts, style,
                                                index.transcript if index else [], params, router,
                                                index=index, regenerate_fn=None)
        trial.match_report = (best_variant.match_report if best_variant else trial.match_report) or {}
        trial.match_report["gate1"] = gate1
        if trial.final_match > best_score:
            best_score = trial.final_match
            best_trial = trial
        if passed:
            break
    if best_trial is None:
        return False
    for field in ("source_start", "source_end", "scripts", "match_report",
                  "final_match", "hook_visual", "best_visual_moments"):
        setattr(cand, field, getattr(best_trial, field))
    cand.status = best_trial.status
    cand.selected_script = best_trial.selected_script
    cand.reject_reason = best_trial.reject_reason
    logger.info("故事重建改善 %s: %.1f → %.1f", cand.candidate_id, old_score, best_score)
    return True
