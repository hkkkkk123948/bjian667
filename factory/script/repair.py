"""脚本自动修复（doc §78）：优先只重写最薄弱的几句，失败再整段重写。

从 pipeline 中拆出，保持编排层精简、修复策略可单测。
"""

from ..log import get_logger


def weak_sentence_indices(sentences, details, limit=3):
    """根据逐句画面/事实得分找出最薄弱的句子（1-based）。"""
    n = len(sentences)
    if n <= 2:
        return []
    scores = [0.0] * n
    counted = [False] * n
    for key in ("visual_sentence_scores", "fact_sentence_scores"):
        arr = (details or {}).get(key) or []
        for i in range(min(n, len(arr))):
            try:
                scores[i] += 100.0 - float(arr[i])
                counted[i] = True
            except (TypeError, ValueError):
                continue
    idxs = [i for i in range(n) if counted[i]]
    if not idxs:
        return []
    idxs.sort(key=lambda i: scores[i], reverse=True)
    return [i + 1 for i in idxs[:limit]]


def regenerate_script(pipeline, candidate, weak_variant, weakest, round_no):
    index = pipeline._load_index()
    provider = pipeline.router.get("script_generation")
    if provider.is_mock:
        return None
    sentences = weak_variant.sentences or []
    if not sentences:
        return None
    details = (weak_variant.match_report or {}).get("details") or {}
    targets = weak_sentence_indices(sentences, details, limit=3)
    try:
        if targets:
            repaired = _repair_sentences(pipeline, candidate, weak_variant, targets, round_no, index, provider)
            if repaired:
                return repaired
        return _rewrite_whole(pipeline, candidate, weak_variant, weakest, round_no, index, provider)
    except Exception as exc:
        get_logger().warning("重写脚本失败: %s", exc)
        return None


def compress_variant(pipeline, candidate, variant, target_seconds):
    """配音前预测超长时，让模型压缩脚本而不是硬裁句。"""
    provider = pipeline.router.get("script_generation")
    if provider.is_mock or not variant.sentences:
        return None
    window = pipeline._load_index()
    text = "\n".join("%d. %s" % (i + 1, s.text) for i, s in enumerate(variant.sentences))
    evidence = window.text_in(candidate.source_start, candidate.source_end)[:900] if window else ""
    prompt = (
        "下面这段短视频解说预计超过 %d 秒。请在不改变原意、保留悬念与爆点的前提下压缩，"
        "减少句数或缩短句子，使其配速不超过 %d 秒。严禁编造原片之外的内容。\n"
        "原文:\n%s\n原片台词参考: %s\n"
        '只输出 JSON: {"sentences":[{"text":"...","evidence_start":秒,"evidence_end":秒}]}'
    ) % (int(target_seconds * 1.1), int(target_seconds), text, evidence)
    try:
        data = provider.complete_json(prompt, json_mode=True, default=None)
    except Exception as exc:
        get_logger().warning("脚本压缩失败: %s", exc)
        return None
    if not data:
        return None
    raw = data.get("sentences") if isinstance(data, dict) else data
    from .writer import _assemble, _coerce_sentences
    sentences = _coerce_sentences(raw, candidate, 0, 8)
    if not sentences:
        return None
    new_variant = _assemble(candidate, variant.variant_id, variant.flavor, sentences,
                            float(pipeline.config.param("min_duration", 15)),
                            float(pipeline.config.param("max_duration", 35)))
    new_variant.match_report = variant.match_report
    new_variant.attempts = variant.attempts
    if new_variant.estimated_seconds >= variant.estimated_seconds:
        return None
    get_logger().info("脚本压缩: %.1fs → %.1fs", variant.estimated_seconds, new_variant.estimated_seconds)
    return new_variant


def _repair_sentences(pipeline, candidate, weak_variant, targets, round_no, index, provider):
    """只重写最薄弱的几句，保留其余句子与句序。"""
    import copy

    from .writer import _window_visuals
    from ..utils import estimate_speech_seconds, normalize_space

    lines = []
    for i, sent in enumerate(weak_variant.sentences):
        mark = "  <-- 需重写" if (i + 1) in targets else ""
        lines.append("%d. %s%s" % (i + 1, sent.text, mark))
    window = index.text_in(candidate.source_start, candidate.source_end)[:900] if index else ""
    visuals = _window_visuals(candidate, index, pipeline.config.param("language", "zh"))
    visuals_block = "原片画面描述:\n%s\n" % visuals if visuals else ""
    prompt = (
        "下面是短视频解说全文，其中第 %s 句与原片画面/事实对应较弱，需要重写。\n"
        "要求：只重写标记的句子，其它句子保持原样、句数与顺序不变；"
        "新句子必须依据原片台词与画面，严禁编造。\n"
        "全文:\n%s\n"
        "%s原片台词: %s\n"
        '只输出 JSON: {"rewrites":[{"index": 句号, "text": "新句子"}]}'
    ) % (", ".join(str(t) for t in targets), "\n".join(lines), visuals_block, window)
    data = provider.complete_json(prompt, json_mode=True, default=None)
    if isinstance(data, dict):
        rewrites = data.get("rewrites") or data.get("sentences") or []
    else:
        rewrites = data or []
    if not rewrites:
        return None

    fit = copy.deepcopy(weak_variant)
    changed = 0
    for item in rewrites:
        if not isinstance(item, dict):
            continue
        try:
            idx = int(item.get("index"))
        except (TypeError, ValueError):
            continue
        text = normalize_space(str(item.get("text") or item.get("sentence") or ""))
        if 1 <= idx <= len(fit.sentences) and text:
            fit.sentences[idx - 1].text = text
            fit.sentences[idx - 1].estimated_seconds = estimate_speech_seconds(text)
            changed += 1
    if not changed:
        return None
    fit.variant_id = "%sR%d" % (weak_variant.variant_id, round_no)
    fit.flavor = weak_variant.flavor
    fit.full_text = " ".join(s.text for s in fit.sentences)
    fit.estimated_seconds = round(sum(s.estimated_seconds for s in fit.sentences), 3)
    fit.attempts = round_no
    fit.match_report = None
    get_logger().info("候选 %s 仅重写 %d 句（第 %s 句）", candidate.candidate_id, changed, targets)
    return [fit]


def _rewrite_whole(pipeline, candidate, weak_variant, weakest, round_no, index, provider):
    try:
        prompt = (
            "上一版解说词匹配度不达标，薄弱项: %s。请重写 %d 句解说，强化画面证据对应与风格贴合。\n"
            "候选故事: %s\n上一版全文: %s\n原片台词: %s\n输出 JSON: {\"sentences\":[{\"text\":...,\"evidence_start\":...,\"evidence_end\":...}]}"
        ) % (", ".join(weakest), len(weak_variant.sentences), candidate.story_summary,
             weak_variant.full_text[:600], index.text_in(candidate.source_start, candidate.source_end)[:800] if index else "")
        data = provider.complete_json(prompt, json_mode=True, default=None)
        if not data:
            return None
        raw = data.get("sentences") if isinstance(data, dict) else data
        from .writer import _assemble, _coerce_sentences
        sentences = _coerce_sentences(raw, candidate, len(candidate.scripts), 8)
        if not sentences:
            return None
        variant = _assemble(candidate, "%sR%d" % (weak_variant.variant_id, round_no),
                            weak_variant.flavor, sentences, float(pipeline.config.param("min_duration", 15)),
                            float(pipeline.config.param("max_duration", 35)))
        variant.attempts = round_no
        variant.match_report = None
        return [variant]
    except Exception as exc:
        get_logger().warning("整段重写失败: %s", exc)
        return None
