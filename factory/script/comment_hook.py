"""Insert an exact audience aside; never turn it into an event or a prop."""
import hashlib
import json

from ..utils import read_json, write_json
from ..providers.base import ProviderTimeoutError


def story_text(sentence):
    return getattr(sentence,'narrative_text','') or sentence.text


def comment_phrases(sentence):
    hooks = getattr(sentence,'comment_hooks',[])
    return [h["text"] for h in hooks] if hooks else ([sentence.comment_hook] if getattr(sentence,'comment_hook','') else [])


def attach_inline_hooks(sentence, hooks, speed=1.2):
    body = story_text(sentence)
    ordered = sorted(hooks, key=lambda h: h["offset"])
    sentence.narrative_text = body
    sentence.comment_hooks = [{"text": h["text"], "offset": min(len(body), max(0, h["offset"]))} for h in ordered]
    sentence.comment_hook = sentence.comment_hooks[0]["text"] if hooks else ""
    sentence.comment_hook_mode = "user_review_inline"
    spoken, cursor = "", 0
    punctuation = "，,。！？!?；;：:"
    for hook in sentence.comment_hooks:
        offset = hook["offset"]
        spoken += body[cursor:offset]
        if spoken and spoken[-1] not in punctuation:
            spoken += "，"
        from .quality import compact
        hook["spoken_start"] = len(compact(spoken))
        spoken += hook["text"]
        hook["spoken_end"] = len(compact(spoken))
        if offset < len(body) and body[offset] not in punctuation and spoken[-1] not in punctuation:
            spoken += "，"
        cursor = offset
    sentence.text = spoken + body[cursor:]
    if sentence.text and sentence.text[-1] not in "。！？!?":
        sentence.text += "。"
    sentence.estimated_seconds = len(sentence.text) / (6.2 * speed)


def strip_hook(variant, speed=1.2):
    for sent in variant.sentences:
        if getattr(sent,'comment_hook',''):
            sent.text = sent.narrative_text
            sent.narrative_text = ""
            sent.comment_hook = ""
            sent.comment_hook_mode = ""
            sent.comment_hooks = []
            sent.estimated_seconds = len(sent.text) / (6.2 * speed)
    _refresh(variant)


def attach_hook(sentence, phrase, speed=1.2):
    sentence.narrative_text = story_text(sentence)
    sentence.comment_hook = phrase
    sentence.comment_hooks = []
    body = sentence.narrative_text.rstrip("。！？!?，,；; ")
    sentence.text = body + "，" + phrase
    if not sentence.text.endswith(("。", "！", "？", "!", "?")):
        sentence.text += "。"
    sentence.estimated_seconds = len(sentence.text) / (6.2 * speed)


def _refresh(variant):
    variant.full_text = "".join(s.text for s in variant.sentences)
    variant.estimated_seconds = sum(s.estimated_seconds for s in variant.sentences)


def apply_comment_hook(pipeline, variant):
    """Idempotent before preview and before full TTS; blank restores the story."""
    if getattr(variant,'review_comment_hooks',None) is not None:
        speed = float(pipeline.config.tts.get("speed", 1.2))
        strip_hook(variant, speed)
        by_sentence = {}
        for hook in variant.review_comment_hooks:
            index = int(hook["after_sentence"])
            if not 1 <= index <= len(variant.sentences) or not str(hook.get("text", "")).strip():
                raise ValueError("评论话术必须对应一段有效正文")
            sent = variant.sentences[index - 1]
            offset = hook.get("offset", len(sent.text.rstrip("。！？!?")))
            by_sentence.setdefault(index, []).append({"text": hook["text"], "offset": offset})
        for index, hooks in by_sentence.items():
            attach_inline_hooks(variant.sentences[index - 1], hooks, speed)
        _refresh(variant)
        return variant
    phrase = str(pipeline.config.param("comment_hook_text", "") or "").strip()
    position = pipeline.config.param("comment_hook_position", "auto")
    override = getattr(variant,'review_comment_hook',None)
    if override is not None:
        phrase = str(override.get("text", "") or "").strip()
        position = "review:%d" % int(override.get("after_sentence", 0))
    speed = float(pipeline.config.tts.get("speed", 1.2))
    current = [(i, s) for i, s in enumerate(variant.sentences) if s.comment_hook]
    if not variant.sentences:
        return variant
    # Manual placement must also take effect when the phrase has not changed.
    desired = (int(override.get("after_sentence", 0)) - 1) if override is not None else (
        0 if position == "early" else len(variant.sentences) - 2)
    if len(current) == 1 and current[0][1].comment_hook == phrase and current[0][1].comment_hook_mode == position and (
            position == "auto" or current[0][0] == max(0, desired)):
        return variant
    strip_hook(variant, speed)
    if not phrase:
        return variant
    if override is not None:
        if not 0 <= desired < len(variant.sentences):
            raise ValueError("评论话术请放在某一行故事正文之后")
        selected, basis = desired, "user_review"
    elif position == "auto":
        selected, basis = _choose_position(pipeline, variant, phrase)
    else:
        selected, basis = max(0, desired), "manual"
    attach_hook(variant.sentences[selected], phrase, speed)
    variant.sentences[selected].comment_hook_mode = position
    _refresh(variant)
    info = {"phrase": phrase, "after_sentence": selected + 1, "basis": basis,
            "purpose": "comment_engagement_aside", "audio": "whole_script_single_submission"}
    pipeline.state.artifacts.setdefault("comment_hooks", {})[variant.candidate_id + ":" + variant.variant_id] = info
    write_json(pipeline.job_dir / "analysis" / "comment_hooks.json", pipeline.state.artifacts["comment_hooks"])
    pipeline.emitter.log("评论引导话术插在第%d句后（%s）：%s" % (selected + 1, basis, phrase))
    return variant


def _choose_position(pipeline, variant, phrase):
    provider = pipeline.router.get(pipeline.config.param("comment_hook_role", "story_analysis"))
    if provider.is_mock:
        pipeline.emitter.log("话术位置模型未配置，按参考片放在首个危险之后", "warning")
        return 0, "local_fallback"
    # At most three early boundaries; no vision or story regeneration is needed.
    choices = list(range(1, min(3, len(variant.sentences) - 1) + 1)) or [1]
    brief = [{"sentence": i + 1, "text": story_text(s)} for i, s in enumerate(variant.sentences)]
    prompt = ("为惊悚短片选择评论引导话术的插入位置。话术是独立插话，不是剧情、因果或道具。"
              "参考：女孩冲出车道，没带玫瑰花，司机吓得急刹。先有事件，短暂插话，马上接回正文。"
              "不要改写正文或话术，不要解释花的用途。只返回JSON {\"after_sentence\":整数}，可选%s。"
              "话术：%s；正文：%s") % (choices, json.dumps(phrase, ensure_ascii=False), json.dumps(brief, ensure_ascii=False))
    key = hashlib.sha256((str(provider.conf.get("base_url", "")) + provider.model + prompt).encode()).hexdigest()
    path = pipeline.config.workspace_path() / ".cache" / "comment_hooks" / (key + ".json")
    data = read_json(path)
    cached = data is not None
    if data is None:
        previous = provider.conf.get("max_tokens")
        previous_retry = provider.max_retry
        try:
            provider.conf["max_tokens"] = 256
            provider.conf["enable_thinking"] = False
            provider.max_retry = 1
            try:
                data = provider.complete_json(prompt, images=[], json_mode=True, default=None)
            except ProviderTimeoutError:
                pipeline.emitter.log("话术位置接口超时，按参考片放在开场事件后，正文和话术原句均保留", "warning")
                data = {"after_sentence": 1, "basis": "local_fallback_timeout"}
        finally:
            if previous is None:
                provider.conf.pop("max_tokens", None)
            else:
                provider.conf["max_tokens"] = previous
            provider.max_retry = previous_retry
    selected = (data or {}).get("after_sentence") if isinstance(data, dict) else None
    if isinstance(selected, str) and selected.strip().isdigit():
        selected = int(selected.strip())
    if type(selected) is not int or selected not in choices:
        pipeline.emitter.log("模型未返回可用的前段话术位置，按参考片放在开场事件后", "warning")
        write_json(path, {"after_sentence": 1, "basis": "local_fallback_invalid_ai_position"})
        return 0, "local_fallback_invalid_ai_position"
    if not cached:
        write_json(path, data)
    return selected - 1, data.get("basis") or ("cached_ai" if cached else "ai_text_only")
