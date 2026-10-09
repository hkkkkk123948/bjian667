
from ..log import get_logger
from ..models import MatchReport
from ..style.analyzer import style_fingerprint
from ..utils import audio_fit_score, clamp, normalize_space, tokenize


def score_variant(candidate, variant, style_profile, transcript, index=None, router=None, role="match_judge",
                  params=None):
    if any(getattr(s, "comment_hook", "") for s in variant.sentences):
        import copy
        from ..script.comment_hook import strip_hook
        body = copy.deepcopy(variant)
        strip_hook(body)
        report = score_variant(candidate, body, style_profile, transcript, index=index, router=router, role=role, params=params)
        variant.match_report = body.match_report
        return report
    logger = get_logger()
    report = MatchReport()
    details = {}
    mode = str((params or {}).get("content_mode", "creative")).lower()

    grounding = _llm_grounding(candidate, variant, index, transcript, router, role, details, mode)
    visual = _visual_alignment(candidate, variant, index, router, role, details, grounding)
    report.visual_alignment = visual
    narrative = _narrative_alignment(candidate, variant, details)
    report.narrative_alignment = narrative
    ref_style = _reference_style_match(variant, style_profile, details)
    report.reference_style_match = ref_style
    temporal = _temporal_alignment(candidate, variant, transcript, details)
    report.temporal_alignment = temporal
    fact = _fact_alignment(candidate, variant, transcript, index, router, role, details, grounding, mode)
    report.fact_alignment = fact
    info_cov = _information_coverage(candidate, variant, details)
    report.information_coverage = info_cov
    tts_fit = _audio_tts_fit(candidate, variant, details)
    report.audio_tts_fit = tts_fit
    story_comp = _story_completeness(candidate, details)
    report.story_completeness = story_comp

    report.details = details
    final = report.compute()
    variant.match_report = report.as_dict()
    logger.info(
        "候选 %s 变体 %s 匹配度 %.1f (画面 %.0f / 时间 %.0f / 事实 %.0f / 叙事 %.0f / 信息 %.0f / 风格 %.0f / 配音 %.0f / 故事 %.0f)",
        candidate.candidate_id, variant.variant_id, final,
        visual, temporal, fact, narrative, info_cov, ref_style, tts_fit, story_comp,
    )
    return report


def _llm_grounding(candidate, variant, index, transcript, router, role, details, mode="creative"):
    """一次 LLM 调用给出画面匹配 + 事实(commentary)/故事完整度(creative)。"""
    if not (router and not router.is_mock(role)) or not variant.sentences:
        return {}
    try:
        from ..script.writer import _window_visuals

        provider = router.get(role)
        lo, hi = candidate.source_start, candidate.source_end
        window = _window_transcript(transcript, lo, hi, margin=6.0)
        visuals = _window_visuals(candidate, index) if index else ""
        lines = ["- %s" % s.text for s in variant.sentences[:8]]
        if mode == "creative":
            prompt = """你是短视频成片评审。解说为「依据原片画面原创的故事」，请判断：
(A) visual_alignment：每条解说与对应时段画面是否协调吻合（0-100）；
(B) story_coherence：整体是否是一个自成起承转合、连贯有爆点的完整小故事（0-100）。
原片画面描述:
%s
解说:
%s
只输出 JSON: {"visual_alignment": 整数, "story_coherence": 整数, "reason": "简短理由"}""" % (
                (visuals or "")[:1800], "\n".join(lines))
            data = provider.complete_json(prompt, json_mode=True, default=None)
            if isinstance(data, dict) and "visual_alignment" in data:
                result = {"visual_alignment": clamp(float(data["visual_alignment"]), 0.0, 100.0)}
                if "story_coherence" in data:
                    result["fact_alignment"] = clamp(float(data["story_coherence"]), 0.0, 100.0)
                details["grounding_llm_reason"] = data.get("reason", "")
                return result
            details["grounding_llm_skipped"] = True
            return {}
        prompt = """你是视频-文案匹配评审。请一次性判断：
(A) 画面匹配 visual_alignment：每条解说与原片对应时段画面是否吻合（0-100）；
(B) 事实匹配：把每条解说拆成"主体/动作/对象/数字"等关键 claim，逐条判断原片是否有依据（supported true/false）。
原片区间 %.1f-%.1f 字幕:
%s
原片画面描述:
%s
解说:
%s
只输出 JSON: {"visual_alignment": 整数, "claims":[{"sentence":"解说原文","supported":true/false}], "reason": "简短理由"}""" % (
            lo, hi, (window or "")[:1500], (visuals or "")[:1500], "\n".join(lines))
        data = provider.complete_json(prompt, json_mode=True, default=None)
        if isinstance(data, dict) and "visual_alignment" in data:
            fact_score = None
            claims = data.get("claims") or []
            supported = [c for c in claims if isinstance(c, dict)]
            if supported:
                good = sum(1 for c in supported if c.get("supported"))
                fact_score = round(good / len(supported) * 100.0, 1)
                details["claim_support"] = [
                    {"sentence": str(c.get("sentence", ""))[:40], "supported": bool(c.get("supported"))}
                    for c in supported
                ]
            if fact_score is None and "fact_alignment" in data:
                fact_score = clamp(float(data["fact_alignment"]), 0.0, 100.0)
            result = {"visual_alignment": clamp(float(data["visual_alignment"]), 0.0, 100.0)}
            if fact_score is not None:
                result["fact_alignment"] = fact_score
            details["grounding_llm_reason"] = data.get("reason", "")
            return result
    except Exception as exc:
        logger = get_logger()
        logger.warning("LLM 匹配评审失败，使用规则值: %s", exc)
    details["grounding_llm_skipped"] = True
    return {}


def _visual_alignment(candidate, variant, index, router, role, details, grounding=None):
    sentences = variant.sentences
    if not sentences:
        return 30.0
    scores = []
    for sent in sentences:
        ev = sent.source_evidence
        if not ev or ev.end <= ev.start:
            scores.append(35.0)
            continue
        text = sent.text
        evidence_text = _evidence_text(ev, index)
        overlap = _text_overlap(text, evidence_text)
        base = 40.0 + overlap * 30.0
        scene = _scene_in(index, ev.start, ev.end)
        if scene:
            base += 10.0
            base += clamp(float(getattr(scene, "speech_ratio", 0.0) or 0.0), 0.0, 1.0) * 20.0
        scores.append(clamp(base, 0.0, 100.0))
    avg = sum(scores) / len(scores)
    details["visual_sentence_scores"] = [round(x, 1) for x in scores]
    if grounding and "visual_alignment" in grounding:
        return round(grounding["visual_alignment"], 1)
    if router and not router.is_mock(role) and index and index.scenes:
        avg = _llm_visual(candidate, variant, index, router, role, avg, details)
    return round(avg, 1)


def _llm_visual(candidate, variant, index, router, role, fallback, details):
    logger = get_logger()
    try:
        provider = router.get(role)
        lines = []
        for sent in variant.sentences[:6]:
            ev = sent.source_evidence
            if not ev:
                continue
            lines.append("- %s (原片 %.1f-%.1f)" % (sent.text, ev.start, ev.end))
        prompt = """你是画面-文案匹配度评审。判断每条解说词与原片对应时间段的画面是否吻合。

文案与时间:
%s

只输出 JSON: {"visual_alignment": 0-100 的整数, "reason": "简短理由"}
""" % "\n".join(lines)
        data = provider.complete_json(prompt, json_mode=True, default=None)
        if data and "visual_alignment" in data:
            return clamp(float(data["visual_alignment"]), 0.0, 100.0)
    except Exception as exc:
        logger.warning("LLM 画面匹配失败，使用规则值: %s", exc)
    details["visual_llm_skipped"] = True
    return fallback


def _fact_alignment(candidate, variant, transcript, index, router, role, details, grounding=None, mode="commentary"):
    sentences = variant.sentences
    if not sentences:
        return 0.0
    if grounding and "fact_alignment" in grounding:
        return round(grounding["fact_alignment"], 1)
    if mode == "creative":
        # 原创故事不与原片台词做字面核对；无 LLM 评分时给中性分
        details["fact_creative_neutral"] = True
        return 72.0
    lo, hi = candidate.source_start, candidate.source_end
    window = _window_transcript(transcript, lo, hi, margin=6.0)
    fallback = _fact_heuristic(sentences, window, details)
    if router and not router.is_mock(role):
        if fallback < 85.0 or fallback >= 95.0:
            return _llm_fact(candidate, variant, index, router, role, fallback, details)
        details["fact_llm_skipped"] = True
    return fallback


def _window_transcript(transcript, lo, hi, margin=6.0):
    if not transcript:
        return ""
    parts = []
    for seg in transcript:
        if not getattr(seg, "text", ""):
            continue
        if seg.end >= lo - margin and seg.start <= hi + margin:
            parts.append(seg.text)
    return normalize_space(" ".join(parts))


def _fact_heuristic(sentences, window_text, details):
    import re as _re
    window_text = normalize_space(window_text)
    w_tokens = set(tokenize(window_text))
    if not w_tokens:
        details["fact_window_empty"] = True
        return 25.0
    scores = []
    misses = []
    for s in sentences:
        text = normalize_space(s.text)
        t_tokens = set(tokenize(text))
        important = set()
        digits = _re.findall(r"\d+", text)
        for tok in t_tokens:
            if len(tok) < 2 or tok in _FACT_STOPWORDS:
                continue
            important.add(tok)
        covered = [t for t in important if t in w_tokens]
        hit = len(covered) / max(1, len(important))
        score = clamp(hit * 100.0, 0.0, 100.0)
        if digits and not any(d in window_text for d in digits):
            score = min(score, 40.0)
            misses.append("数字%s无依据" % ",".join(digits[:2]))
        elif score < 60:
            misses.append(s.text[:24])
        scores.append(score)
    avg = sum(scores) / len(scores)
    details["fact_sentence_scores"] = [round(x, 1) for x in scores]
    details["fact_misses"] = misses[:5]
    return round(clamp(avg, 0.0, 100.0), 1)


def _llm_fact(candidate, variant, index, router, role, fallback, details):
    logger = get_logger()
    try:
        provider = router.get(role)
        lo, hi = candidate.source_start, candidate.source_end
        window = _window_transcript(index.transcript, lo, hi, margin=6.0) if index else ""
        lines = []
        for s in variant.sentences[:8]:
            lines.append("- 解说: %s" % s.text)
        prompt = """你是事实可靠性评审。成片要依据原片字幕转写制作，检查每条解说是不是原片里真实发生/讲过的事实。
原片区间 %.1f-%.1f 的原文字幕:
%s

解说:
%s

严禁放行原片没有依据的内容（AI 幻觉）。只输出 JSON: {"fact_alignment": 0-100 的整数, "reason": "简短理由"}
""" % (lo, hi, window[:1500], "\n".join(lines))
        data = provider.complete_json(prompt, json_mode=True, default=None)
        if data and "fact_alignment" in data:
            score = clamp(float(data["fact_alignment"]), 0.0, 100.0)
            details["fact_llm_reason"] = data.get("reason", "")
            return round(score, 1)
    except Exception as exc:
        logger.warning("LLM 事实校验失败，使用规则值: %s", exc)
    details["fact_llm_skipped"] = True
    return fallback


_FACT_STOPWORDS = {"的", "了", "和", "在", "我", "你", "他", "她", "它", "这", "那", "是", "有", "不", "就", "都", "被",
                   "把", "也", "很", "会", "能", "要", "又", "还", "给", "与", "或", "及", "the", "a", "an", "and", "to",
                   "of", "in", "on", "is", "are", "was", "were", "be", "it", "that", "this", "he", "she", "they", "we"}


def _evidence_text(ev, index):
    if not index:
        return ev.quote or ""
    text = index.text_in(ev.start, ev.end)
    return text or ev.quote or ""


def _text_overlap(text, evidence_text):
    text = normalize_space(text)
    evidence_text = normalize_space(evidence_text)
    if not text or not evidence_text:
        return 0.0
    t_tokens = set(tokenize(text))
    e_tokens = set(tokenize(evidence_text))
    if not t_tokens or not e_tokens:
        return 0.0
    inter = len(t_tokens & e_tokens)
    union = len(t_tokens | e_tokens)
    return inter / union if union else 0.0


def _scene_in(index, start, end):
    if not index or not index.scenes:
        return None
    for scene in index.scenes:
        if scene.start >= start and scene.end <= end:
            return scene
        if start <= scene.start < end or start <= scene.end - 0.5 < end:
            return scene
    return None


def _narrative_alignment(candidate, variant, details):
    sentences = variant.sentences
    if len(sentences) < 2:
        return 70.0
    times = []
    for s in sentences:
        ev = s.source_evidence
        if ev and ev.end > ev.start:
            times.append((ev.start + ev.end) / 2.0)
        else:
            times.append((candidate.source_start + candidate.source_end) / 2.0)
    inversions = 0
    total_pairs = len(times) * (len(times) - 1) / 2
    for i in range(len(times)):
        for j in range(i + 1, len(times)):
            if times[j] < times[i] - 1.0:
                inversions += 1
    mono = 1.0 - (inversions / total_pairs if total_pairs else 0.0)
    window_cov = _window_coverage(candidate, variant)
    score = clamp(mono * 60 + window_cov * 40, 0.0, 100.0)
    details["narrative_inversions"] = inversions
    details["narrative_monotonic"] = round(mono, 3)
    return round(score, 1)


def _window_coverage(candidate, variant):
    lo, hi = candidate.source_start, candidate.source_end
    span = max(1e-6, hi - lo)
    total = 0.0
    for s in variant.sentences:
        ev = s.source_evidence
        if ev and ev.end > ev.start:
            total += max(0.0, min(ev.end, hi) - max(ev.start, lo))
        else:
            total += span * 0.25
    return clamp(total / span, 0.0, 1.0)


def _reference_style_match(variant, style_profile, details):
    fp = style_fingerprint(variant.full_text)
    if not fp:
        return 40.0
    scores = []
    target_cps = style_profile.chars_per_second or 4.4
    ratio = abs(fp.get("chars_per_second", target_cps) - target_cps) / max(0.5, target_cps)
    scores.append(clamp(1.0 - ratio * 1.2, 0.0, 1.0) * 100)

    target_short = style_profile.short_sentence_ratio
    ratio2 = abs(fp.get("short_sentence_ratio", target_short) - target_short)
    scores.append(clamp(1.0 - ratio2 * 1.5, 0.0, 1.0) * 100)

    if fp.get("narration") == style_profile.narration:
        scores.append(100.0)
    elif style_profile.narration == "neutral":
        scores.append(85.0)
    else:
        scores.append(45.0)

    col_ratio = abs(fp.get("colloquial_ratio", 0.0) - style_profile.colloquial_ratio)
    scores.append(clamp(1.0 - col_ratio * 1.8, 0.0, 1.0) * 100)

    target_len = style_profile.avg_sentence_chars or 15
    len_ratio = abs(fp.get("avg_sentence_chars", target_len) - target_len) / max(4.0, target_len)
    scores.append(clamp(1.0 - len_ratio * 1.3, 0.0, 1.0) * 100)

    avg = sum(scores) / len(scores)
    details["style_fp"] = fp
    details["style_component_scores"] = [round(x, 1) for x in scores]
    return round(avg, 1)


def _temporal_alignment(candidate, variant, transcript, details):
    sentences = variant.sentences
    if not sentences:
        return 0.0
    lo, hi = candidate.source_start, candidate.source_end
    in_range = 0
    no_evidence = 0
    precision_sum = 0.0
    for s in sentences:
        ev = s.source_evidence
        if not ev or ev.end <= ev.start:
            no_evidence += 1
            continue
        inter = max(0.0, min(ev.end, hi) - max(ev.start, lo))
        ev_len = max(1e-6, ev.end - ev.start)
        if inter > 0.2:
            in_range += 1
        precision_sum += inter / ev_len
    coverage = in_range / len(sentences)
    precision = precision_sum / len(sentences) if sentences else 0.0
    penalty = no_evidence / len(sentences)
    score = clamp((coverage * 0.6 + precision * 0.4) * 100 - penalty * 40, 0.0, 100.0)
    details["temporal_in_range"] = in_range
    details["temporal_no_evidence"] = no_evidence
    return round(score, 1)


def _information_coverage(candidate, variant, details):
    required = []
    for key in ("conflict", "turning_point", "payoff"):
        val = getattr(candidate, key, "")
        if val and len(val) > 2:
            required.append(val)
    if not required:
        return 65.0
    tokens = tokenize(variant.full_text)
    hit = 0
    for req in required:
        req_tokens = set(tokenize(req))
        if not req_tokens:
            hit += 0.5
            continue
        inter = len(req_tokens & set(tokens))
        if inter >= 2 or (inter >= 1 and len(req_tokens) <= 3):
            hit += 1.0
        elif inter >= 1:
            hit += 0.6
        else:
            hit += 0.15
    score = clamp((hit / len(required)) * 100, 0.0, 100.0)
    details["info_required"] = required
    return round(score, 1)


def _audio_tts_fit(candidate, variant, details):
    est = variant.estimated_seconds or sum(s.estimated_seconds for s in variant.sentences)
    span = max(0.5, candidate.source_end - candidate.source_start)
    ratio = est / span
    return audio_fit_score(ratio)


def _story_completeness(candidate, details):
    base = candidate.story_completeness
    if candidate.conflict:
        base += 2
    if candidate.turning_point:
        base += 2
    if candidate.payoff:
        base += 2
    return clamp(base, 0.0, 100.0)


def select_variant_best(variants, threshold):
    if not variants:
        return None, 0.0, False
    best = max(variants, key=lambda v: (v.match_report or {}).get("final_score", 0.0))
    final = (best.match_report or {}).get("final_score", 0.0)
    return best, final, final >= threshold
