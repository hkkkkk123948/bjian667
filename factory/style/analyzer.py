import re
from collections import Counter

from ..log import get_logger
from ..models import StyleProfile
from ..providers import LLMRouter
from ..utils import estimate_speech_seconds, normalize_space, split_sentences, tokenize
from .loader import load_references

SUSPENSE_LEXICON = {
    "suspense": ["为什么", "怎么回事", "难道", "谜", "疑点", "疑问", "不对劲", "诡异", "神秘", "真相", "秘密", "监控", "录像", "录音"],
    "horror": ["恐怖", "惊悚", "尖叫", "血", "尸体", "鬼", "黑影", "阴森", "失踪", "报警", "疯", "死", "杀"],
    "tension": ["紧张", "对峙", "冲突", "追逐", "逃跑", "躲", "危险", "来不及", "最后", "终于", "突然"],
    "humor": ["笑", "哈哈", "搞笑", "离谱", "翻车", "绝了", "整活", "吐槽", "笑死", "离谱"],
    "drama": ["没想到", "竟然", "居然", "反转", "结局", "泪目", "感动", "震惊"],
}

COLLOQUIAL_MARKERS = [
    "竟然", "居然", "哇", "绝了", "离谱", "笑死", "直接", "疯", "炸", "顶不住",
    "谁懂", "救命", "破防", "细思极恐", "有点东西", "细思", "真的", "秒",
]

STRUCTURE_MARKERS = {
    "hook": ["你敢信", "你敢相信", "直到", "就在", "下一秒", "直到那一刻", "你敢信"],
    "context": ["事情是这样的", "背景", "一开始", "原来", "故事发生在", "那天"],
    "conflict": ["但是", "可是", "问题", "冲突", "不对", "发现", "却"],
    "escalation": ["越来越", "更加", "开始", "终于", "彻底", "失控"],
    "twist": ["反转", "真相", "竟然", "居然", "原来", "结果是", "没想到"],
    "open_loop": ["？", "悬念", "结尾", "细思", "你还记得", "最后"],
}


def analyze_reference_style(reference_paths, router: LLMRouter, role="reference_analysis") -> StyleProfile:
    logger = get_logger()
    texts, _items = load_references(reference_paths, max_refs=20)
    if not texts:
        logger.warning("未提供有效参考文案，使用默认惊悚风格模板")
        return _default_suspense_profile()
    profile = _measure_profile(texts)
    provider = router.get(role)
    if not provider.is_mock and str(router.config.param("reference_local_only", "on")) != "on":
        try:
            _llm_enrich(profile, texts, provider)
        except Exception as exc:
            logger.warning("LLM 风格增强失败，仅使用规则指标: %s", exc)
    logger.info("参考文案风格分析完成: %d 篇样本, 节奏=%s, 人称=%s, 结构=%s",
                len(texts), profile.pacing, profile.narration, "/".join(profile.structure))
    return profile


def _measure_profile(texts) -> StyleProfile:
    sentences = []
    for text in texts:
        sentences.extend(_split_sentences(normalize_space(text)))
    if not sentences:
        sentences = [normalize_space(" ".join(texts))[:80]]

    lens = [len(s) for s in sentences]
    avg_len = sum(lens) / len(lens)
    short_ratio = sum(1 for n in lens if n <= 16) / len(lens)
    pauses = sum(s.count("，") + s.count("、") + s.count("；") for s in sentences) / max(1, len(sentences))

    colloquial_hits = 0
    for sent in sentences:
        for marker in COLLOQUIAL_MARKERS:
            if marker in sent:
                colloquial_hits += 1
    colloquial_ratio = min(1.0, colloquial_hits / max(1, len(sentences) * 2))

    total_chars = sum(len(s) for s in sentences)
    total_est_seconds = sum(estimate_speech_seconds(s) for s in sentences)
    cps = total_chars / total_est_seconds if total_est_seconds > 0 else 4.4

    hook_seconds = min(6.0, max(0.8, estimate_speech_seconds(sentences[0]) if sentences else 2.0))

    first_person = sum(1 for s in sentences if re.search(r"我(们)?", s))
    third_person = sum(1 for s in sentences if re.search(r"他|她|他们|这个男人|这个女人|老人|女孩|男孩", s))
    if first_person > third_person:
        narration = "first_person"
    elif third_person > 0:
        narration = "third_person"
    else:
        narration = "neutral"

    emotion = {k: _emotion_score(sentences, lex) for k, lex in SUSPENSE_LEXICON.items()}
    structure = _detect_structure(sentences)

    words = [w for s in sentences for w in tokenize(s)]
    counter = Counter(words)
    fingerprint = [w for w, _ in counter.most_common(24)]

    ending = _detect_ending(sentences)

    return StyleProfile(
        name="user_reference",
        sample_count=len(texts),
        hook_length="%.1f-%.1fs" % (max(0.8, hook_seconds * 0.7), hook_seconds * 1.4),
        hook_seconds=round(hook_seconds, 2),
        sentence_length="short" if short_ratio >= 0.45 else ("medium" if short_ratio >= 0.2 else "long"),
        avg_sentence_chars=round(avg_len, 1),
        short_sentence_ratio=round(short_ratio, 3),
        narration=narration,
        pacing="fast" if cps > 5.0 else ("medium" if cps >= 3.8 else "slow"),
        chars_per_second=round(cps, 2),
        emotion=emotion,
        structure=structure,
        ending=ending,
        colloquial_ratio=round(colloquial_ratio, 3),
        pause_density=round(pauses, 3),
        repetition_patterns=_repetition_patterns(sentences),
        lexical_fingerprint=fingerprint,
        raw_metrics={"sentences": len(sentences), "total_chars": total_chars, "est_seconds": round(total_est_seconds, 1)},
        notes="auto-measured from %d reference texts" % len(texts),
    )


def _split_sentences(text):
    return split_sentences(text)


def _emotion_score(sentences, lexicon):
    total = 0.0
    for sent in sentences:
        for word in lexicon:
            if word in sent:
                total += 1.0
    per = total / max(1.0, len(sentences))
    return int(min(100, per * 100))


def _detect_structure(sentences):
    joined = " ".join(sentences)
    found = []
    for name, markers in STRUCTURE_MARKERS.items():
        for m in markers:
            if m in joined:
                found.append(name)
                break
    default_order = ["hook", "context", "conflict", "escalation", "twist"]
    if not found:
        return default_order
    found = [n for n in default_order if n in found]
    if "open_loop" in found:
        found.remove("open_loop")
        found.append("open_loop")
    return found


def _detect_ending(sentences):
    tail = normalize_space(" ".join(sentences[-3:]))
    if re.search(r"真相|原来是|结果是|谜底|揭露|解开", tail):
        return "closed_truth"
    if re.search(r"细思极恐|细思|你还记得|值得反复看|你细品|？", tail):
        return "open_loop"
    return "open_loop"


def _repetition_patterns(sentences, k=4, limit=6):
    phrases = []
    for sent in sentences:
        for i in range(len(sent) - k + 1):
            phrase = sent[i:i + k]
            if re.search(r"[\u4e00-\u9fff]", phrase):
                phrases.append(phrase)
    counter = Counter(phrases)
    return [p for p, c in counter.most_common(limit) if c >= 3]


def _default_suspense_profile() -> StyleProfile:
    return StyleProfile(
        name="builtin_thriller",
        sample_count=0,
        hook_length="1.5-3.0s",
        hook_seconds=2.2,
        sentence_length="short",
        avg_sentence_chars=14.0,
        short_sentence_ratio=0.55,
        narration="third_person",
        pacing="fast",
        chars_per_second=4.6,
        emotion={"suspense": 82, "horror": 70, "tension": 74, "humor": 8, "drama": 80},
        structure=["hook", "conflict", "escalation", "twist"],
        ending="open_loop",
        colloquial_ratio=0.35,
        pause_density=0.6,
        repetition_patterns=[],
        lexical_fingerprint=["监控", "尸体", "真相", "反转", "不对劲", "细思极恐"],
        raw_metrics={},
        notes="默认惊悚短片风格模板",
    )


def _llm_enrich(profile: StyleProfile, texts, provider):
    sample = normalize_space(" ".join(texts))[:2400]
    prompt = (
        "你是文案风格分析师。以下是多条参考文案。请输出 JSON 风格画像，包含字段: "
        "hook_seconds, sentence_length(short/medium/long), narration(first_person/third_person/neutral), "
        "pacing(fast/medium/slow), emotion{suspense,horror,tension,humor,drama:0-100}, "
        "structure[数组], ending(open_loop/closed_truth), colloquial_ratio, structure_notes。"
        "只输出 JSON。\n\n参考文案:\n%s" % sample
    )
    data = provider.complete_json(prompt, json_mode=True, default=None)
    if not data:
        return
    if data.get("hook_seconds"):
        try:
            profile.hook_seconds = float(data["hook_seconds"])
        except (TypeError, ValueError):
            pass
    if data.get("sentence_length"):
        profile.sentence_length = str(data["sentence_length"])
    if data.get("narration"):
        profile.narration = str(data["narration"])
    if data.get("pacing"):
        profile.pacing = str(data["pacing"])
    if isinstance(data.get("emotion"), dict):
        profile.emotion = {k: int(v) for k, v in data["emotion"].items()}
    if data.get("structure"):
        profile.structure = [str(x) for x in data["structure"]]
    if data.get("ending"):
        profile.ending = str(data["ending"])
    if data.get("structure_notes"):
        profile.notes += " | LLM: " + str(data["structure_notes"])[:300]


def style_fingerprint(text) -> dict:
    sentences = _split_sentences(normalize_space(text))
    if not sentences:
        return {}
    lens = [len(s) for s in sentences]
    total_chars = sum(lens)
    cps = total_chars / max(0.5, sum(estimate_speech_seconds(s) for s in sentences))
    colloquial_hits = 0
    for sent in sentences:
        for marker in COLLOQUIAL_MARKERS:
            if marker in sent:
                colloquial_hits += 1
    return {
        "avg_sentence_chars": round(sum(lens) / len(lens), 1),
        "short_sentence_ratio": round(sum(1 for n in lens if n <= 16) / len(lens), 3),
        "chars_per_second": round(cps, 2),
        "narration": "first_person" if any(re.search(r"我", s) for s in sentences) else "third_person",
        "colloquial_ratio": round(min(1.0, colloquial_hits / max(1, len(sentences) * 2)), 3),
    }
