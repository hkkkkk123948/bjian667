"""Minimal textual rephrasing for repeat-prone synthesis. Never splice narration."""
import copy
import json

from .quality import compact, script_issues
from .comment_hook import story_text, attach_hook, attach_inline_hooks, comment_phrases
from ..utils import write_json


def repair_for_speech(pipeline, candidate, variant, report):
    provider = pipeline.router.get("script_generation")
    if provider.is_mock:
        raise RuntimeError("配音重复且没有可用的文案修复模型")
    issues = report.get("issues", [])
    targets = []
    for issue in issues:
        phrase = compact(issue.get("text") or issue.get("expected") or "")
        if phrase and any(phrase in compact(p) for s in variant.sentences for p in comment_phrases(s)):
            raise RuntimeError("用户评论话术配音未通过核对；保留原句，停止自动改写话术")
        matches = [i for i, s in enumerate(variant.sentences) if phrase and phrase in compact(s.text)]
        if matches:
            targets.append(matches[-1])
    if not targets:
        raise RuntimeError("无法定位出错旁白句，停止自动改稿")
    fresh = copy.deepcopy(variant)
    changes = []
    provider.max_retry = 1
    provider.conf["max_tokens"] = 240
    for i in sorted(set(targets))[:2]:
        sent = fresh.sentences[i]
        visible = sent.source_evidence.quote
        if not visible and pipeline.index:
            visible = next((scene.visual for scene in pipeline.index.scenes
                            if scene.scene_id in sent.source_evidence.scene_ids), "")
        prompt = ("这句中文配音在TTS中会重复尾句。只换自然口语表达，保留完全相同的人物、动作和结果；"
                  "不能增加事实，不能改变镜头，不要逗号堆叠，不用原来的重复短语。长度15至30字。"
                  "输出JSON {\"text\":\"改写后的完整一句\"}。原句：%s；实际可见画面：%s；异常：%s") % (
                      story_text(sent), visible,
                      json.dumps(issues, ensure_ascii=False))
        data = provider.complete_json(prompt, json_mode=True, default=None)
        text = str((data or {}).get("text") or "").strip()
        if not 6 <= len(text) <= 50 or compact(text) == compact(story_text(sent)):
            raise RuntimeError("配音措辞修复未生成有效新句子")
        changes.append({"index": sent.index, "before": sent.text, "after": text})
        if getattr(sent,'comment_hooks',[]):
            from difflib import SequenceMatcher
            old_body = story_text(sent)
            hooks = copy.deepcopy(sent.comment_hooks)
            for hook in hooks:
                offset = hook["offset"]
                for _, a, b, x, y in SequenceMatcher(None, old_body, text, autojunk=False).get_opcodes():
                    if a <= offset <= b:
                        hook["offset"] = x + min(offset-a, y-x)
                        break
            sent.narrative_text = text
            attach_inline_hooks(sent, hooks, float(pipeline.config.tts.get("speed", 1.2)))
        elif sent.comment_hook:
            hook = sent.comment_hook
            sent.narrative_text = text
            attach_hook(sent, hook, float(pipeline.config.tts.get("speed", 1.2)))
        else:
            sent.text = text
        sent.estimated_seconds = len(sent.text) / (6.2 * float(pipeline.config.tts.get("speed", 1.2)))
    if script_issues(fresh.sentences):
        raise RuntimeError("修复后的旁白未通过重复检查")
    fresh.variant_id = variant.variant_id + "_voicefix"
    fresh.full_text = "".join(s.text for s in fresh.sentences)
    fresh.estimated_seconds = sum(s.estimated_seconds for s in fresh.sentences)
    if getattr(fresh,'review_comment_hooks',None) is not None:
        fresh.review_comment_hooks = [{"after_sentence":s.index, "text":h["text"], "offset":h["offset"]}
                                      for s in fresh.sentences for h in s.comment_hooks]
    pipeline._adopt_variant(candidate, fresh)
    pipeline.state.artifacts.pop("reviewed_source_sha256", None)
    pipeline.state.artifacts["voice_text_repair"] = changes
    write_json(pipeline.job_dir / "analysis" / "voice_text_repair.json", changes)
    pipeline.save_state()
    pipeline.emitter.log("已调整重复句措辞，将整篇重新配音: " + fresh.full_text)
    return fresh
