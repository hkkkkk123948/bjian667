from ..log import get_logger
from ..models import MatchReport
from ..utils import audio_fit_score, clamp
from .engine import score_variant, select_variant_best


def _active_gates(params):
    if str(params.get("hard_gates", "on")).lower() in ("off", "0", "false"):
        return {}
    if str(params.get("content_mode", "creative")).lower() == "creative":
        return MatchReport.CREATIVE_HARD_GATES
    return MatchReport.HARD_GATES


def _hard_gates_pass(report, gates=None):
    gates = MatchReport.HARD_GATES if gates is None else gates
    for k, v in gates.items():
        if float((report or {}).get(k, 0.0)) < v:
            return False
    return True


def _hard_gate_misses(report, gates=None):
    gates = MatchReport.HARD_GATES if gates is None else gates
    misses = []
    for k, v in gates.items():
        if float((report or {}).get(k, 0.0)) < v:
            misses.append("%s(%.0f<%.0f)" % (k, float((report or {}).get(k, 0.0)), v))
    return misses


def run_gates(candidate, variants, style_profile, transcript, params, router, index=None, regenerate_fn=None):
    logger = get_logger()
    threshold = float(params.get("min_match", 75.0))
    max_attempts = int(params.get("max_rewrite_attempts", 3))
    rewrite_margin = float(params.get("rewrite_margin", 15.0))
    gate1 = {"threshold": threshold, "attempts": [], "status": "pending"}
    best_variant, best_score, passed = select_variant_best(variants, threshold)
    for variant in variants:
        if not variant.match_report:
            score_variant(candidate, variant, style_profile, transcript, index=index, router=router, params=params)
    best_variant, best_score, passed = select_variant_best(variants, threshold)
    reported_score = best_score
    hard = _hard_gate_misses(best_variant.match_report if best_variant else None, _active_gates(params))
    passed = bool(passed and not hard)
    gate1["attempts"].append({"round": 0, "best_score": best_score,
                              "hard_gates": hard or "pass"})
    gate1["hard_gates_enabled"] = bool(_active_gates(params))

    round_no = 1
    while (not passed and round_no <= max_attempts and regenerate_fn
           and reported_score >= threshold - rewrite_margin):
        logger.info("候选 %s 未达标 (总分 %.1f/%.1f, 硬门槛: %s)，第 %d 次重写",
                    candidate.candidate_id, reported_score, threshold, hard or "pass", round_no)
        weak = _weakest_components(best_variant.match_report or {})
        fresh = regenerate_fn(candidate, best_variant, weak, round_no)
        if not fresh:
            break
        for variant in fresh:
            score_variant(candidate, variant, style_profile, transcript, index=index, router=router, params=params)
            variants.append(variant)
        best_variant, best_score, passed = select_variant_best(variants, threshold)
        reported_score = best_score
        hard = _hard_gate_misses(best_variant.match_report if best_variant else None, _active_gates(params))
        passed = bool(passed and not hard)
        gate1["attempts"].append({"round": round_no, "best_score": best_score,
                                  "fixed": weak, "hard_gates": hard or "pass"})
        round_no += 1

    if passed:
        gate1["status"] = "pass"
        candidate.status = "gate1_pass"
        candidate.selected_script = best_variant.variant_id
        candidate.final_match = best_score
        gate1["hard_gates"] = "pass"
    else:
        reason = ""
        if hard:
            reason = "硬门槛未达: " + "; ".join(hard)
        gate1["status"] = "review_required" if reported_score >= threshold - 15 else "reject"
        if gate1["status"] == "reject":
            candidate.status = "rejected_low_match"
            candidate.reject_reason = "匹配度 %.1f 低于阈值 %.1f%s" % (reported_score, threshold, ("；" + reason) if reason else "")
        else:
            candidate.status = "review_required"
            candidate.reject_reason = ("重写 %d 轮仍未达标 (%.1f/%.1f)%s" % (
                max_attempts, reported_score, threshold, ("；" + reason) if reason else ""))
        candidate.final_match = reported_score
    candidate.match_report = (best_variant.match_report if best_variant else {}) or {}
    gate1["best_variant"] = best_variant.variant_id if best_variant else ""
    gate1["best_score"] = reported_score
    candidate.gate1 = gate1
    candidate.match_report["gate1"] = gate1
    logger.info("第一闸门 %s: %s 得分 %.1f", gate1["status"], candidate.candidate_id, reported_score)
    return gate1, best_variant, passed


def quality_gate(candidate, clip, actual_duration, script_text, tts_duration, threshold=None, tolerance=8.0,
                 hard_gates_enabled=True, params=None):
    logger = get_logger()
    threshold = float(threshold if threshold is not None else 75.0)
    report = MatchReport()
    est = tts_duration or actual_duration
    ratio = est / max(1.0, actual_duration) if actual_duration else 1.0
    report.audio_tts_fit = round(audio_fit_score(ratio), 1)
    est_chars = sum(1 for ch in script_text if ch.strip())
    chars_per_sec = est_chars / max(0.5, actual_duration)
    report.reference_style_match = round(clamp(100.0 - abs(chars_per_sec - 4.2) * 18.0, 0.0, 100.0), 1)
    prev = clip.scores or {}
    report.visual_alignment = float(prev.get("visual_alignment", 70.0))
    report.narrative_alignment = float(prev.get("narrative_alignment", 70.0))
    report.temporal_alignment = float(prev.get("temporal_alignment", 70.0))
    report.fact_alignment = float(prev.get("fact_alignment", 70.0))
    report.information_coverage = float(prev.get("information_coverage", 65.0))
    report.story_completeness = float(prev.get("story_completeness", 65.0))
    final = report.compute()
    drop = float(prev.get("final_score", final)) - final
    qc = {
        "final_match": final,
        "previous_match": prev.get("final_score", final),
        "drop": round(drop, 2),
        "duration": round(actual_duration, 2),
        "tts_duration": round(tts_duration, 2),
        "chars_per_sec": round(chars_per_sec, 2),
        "tolerance": tolerance,
        "status": "pass",
    }
    if actual_duration < 5.0 or actual_duration > 120.0:
        qc["status"] = "fail"
        qc["reason"] = "成片时长异常 %.1fs" % actual_duration
    elif drop > tolerance:
        qc["status"] = "fail"
        qc["reason"] = "成片匹配度下降 %.1f 超过容忍度 %.1f" % (drop, tolerance)
    elif final < threshold - 10:
        qc["status"] = "review"
        qc["reason"] = "成片匹配度 %.1f 低于复检线" % final
    else:
        gates = _active_gates(params or {}) if hard_gates_enabled else {}
        hard = _hard_gate_misses({k: prev.get(k, 0.0) for k in gates}, gates)
        if hard:
            qc["status"] = "fail"
            qc["reason"] = "成片硬门槛未达: " + "; ".join(hard)
    if qc["status"] != "pass":
        logger.warning("第二闸门 %s: %s (%s)", qc["status"], clip.clip_id, qc.get("reason", ""))
    else:
        logger.info("第二闸门通过: %s (%.1f)", clip.clip_id, final)
    clip.gate2 = qc
    return qc


def _weakest_components(report, limit=3):
    if not report:
        return ["visual_alignment", "narrative_alignment", "temporal_alignment"]
    keys = [k for k in MatchReport.WEIGHTS if k in report]
    ranked = sorted(keys, key=lambda k: report.get(k, 0.0))
    return ranked[:limit]
