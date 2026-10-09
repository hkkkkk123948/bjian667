import json
import re
import math

from ..log import get_logger
from ..models import Evidence, StoryCandidate, to_dict
from ..providers import LLMRouter
from ..utils import clamp, normalize_space, split_sentences
from ..story_count import automatic_count


class MiningResult(list):
    def __init__(self,items,discovery):
        super().__init__(items)
        self.discovery = discovery

TURN_MARKERS = [
    "突然", "竟然", "没想到", "原来", "居然", "发现", "但是", "却", "结果", "忽然",
    "就在这时", "随后", "反转", "真相", "秘密", "诡异", "不对劲", "阴森", "黑影",
    "尸体", "失踪", "尖叫", "消失", "监控", "录像", "录音", "报警", "血",
]
HIGHER_VALUE_MARKERS = {
    "尸体": 2.2, "失踪": 1.8, "报警": 1.6, "监控": 1.5, "真相": 1.6,
    "反转": 1.7, "秘密": 1.4, "消失": 1.5, "录像": 1.4, "录音": 1.3,
}
HOOK_PREFIXES = [
    "你敢相信，{text}",
    "这一段太不对劲了，{text}",
    "看似平常的画面，{text}",
    "直到他打开那扇门，{text}",
]


def mine_stories(index, params, router: LLMRouter, role="story_analysis"):
    logger = get_logger()
    provider = router.get(role)
    if automatic_count(params):
        report = {'mode':'auto','complete':False,'scan_complete':False,'discovered':0,'selected':0,
                  'manual_review_required':True,'uncertain_ranges':[],'exhaustive_guarantee':False}
        if provider.is_mock:
            report['reason'] = '故事分析模型未配置，无法自动确认完整故事数量；本次不使用规则片段凑数'
            return MiningResult([],report)
        try:
            raw = _llm_mine(index,params,provider,logger)
            candidates = _normalize(raw,params,index)
            report.update(getattr(raw,'discovery',{}))
            report.update(discovered=len(candidates),selected=len(candidates))
            return MiningResult(candidates,report)
        except Exception as exc:
            logger.warning('自动故事分析未完成: %s',exc)
            report['reason'] = '自动故事分析未完成，未伪造完整故事或兜底成片：' + str(exc)[:160]
            return MiningResult([],report)
    raw = []
    if not provider.is_mock:
        try:
            raw = _llm_mine(index, params, provider, logger)
        except Exception as exc:
            logger.warning("LLM 故事挖掘失败，启用本地规则挖掘: %s", exc)
            raw = []
    if not raw:
        raw = _rule_mine(index, params, logger)
    if not raw:
        raw = _whole_clip_fallback(index, params)
    candidates = _normalize(raw, params, index)
    logger.info("故事挖掘完成: %d 个候选", len(candidates))
    return candidates


def _llm_mine(index, params, provider, logger):
    scenes = _condensed_scenes(index)
    meta = to_dict(index.video) if index.video else {}
    min_d = float(params.get("min_duration", 15))
    max_d = float(params.get("max_duration", 35))
    pool = int(params.get("candidate_pool", 12))
    budget = int(params.get("mine_prompt_chars", 24000))
    if automatic_count(params):
        return _all_index_stories(scenes,meta,params,provider,max(2000,budget))
    prompt = """你是短视频剧情挖掘器。从下面的长视频场景索引中找出 %d 个最值得单独剪成 %d~%d 秒短视频的叙事片段。

场景条目中的 "text" 是该时段原片台词/字幕，"visual" 是视觉模型观察该时段画面后的真实描述。两者都是原片真实内容，务必以此为依据。

输出 JSON 对象，格式 {"candidates": [ ... ]}，其中 candidates 为数组，每个元素包含:
- source_start / source_end: 秒级时间
- story_summary: 一句话概括这个故事（必须来自实际画面/台词）
- title: 12字以内标题
- conflict: 冲突描述（来自实际画面/台词）
- turning_point: 转折点描述（来自实际画面/台词）
- payoff: 结果/爆点描述（来自实际画面/台词）
- characters: 人物列表（仅画面/台词中出现的人物）
- tags: 关键词列表
- best_visual_moments: 3个最值得展示的瞬间(秒)
- evidence: [{start,end,quote}]

要求: 候选之间时间段尽量不重复, 每个故事能独立成立。严禁编造画面和台词里没有的情节与人名。

视频元数据: %s
场景索引:
%s""" % (pool, int(min_d), int(max_d), json.dumps(meta, ensure_ascii=False)[:800],
        json.dumps(scenes, ensure_ascii=False)[:budget])
    data = provider.complete_json(prompt, json_mode=True, default=[])
    if isinstance(data, dict):
        if isinstance(data.get("candidates"), list):
            data = data["candidates"]
        elif isinstance(data.get("stories"), list):
            data = data["stories"]
        elif any(k in data for k in ("source_start", "start", "story_summary", "summary")):
            data = [data]
        else:
            data = []
    return data or []


def _all_index_stories(scenes,meta,params,provider,budget):
    """Read the complete index in bounded text blocks; never silently cut its tail."""
    if not scenes:
        return MiningResult([],{'complete':True,'scan_complete':True,'manual_review_required':False,
                                 'reason':'原片索引没有可用事件画面或台词，未发现可独立成片的故事'})
    chunks,part,size = [],[],0
    for scene in scenes:
        text_size = len(json.dumps(scene,ensure_ascii=False))
        if part and size+text_size > budget:
            chunks.append(part)
            part = part[-2:]
            size = sum(len(json.dumps(row,ensure_ascii=False)) for row in part)
        part.append(scene)
        size += text_size
    if part:
        chunks.append(part)
    stories,uncertain = [],[]
    for chunk in chunks:
        prompt = ('从此原片场景索引找出所有完整、独立、不重复的故事，数量由实际内容决定，允许0。'
                  '不受默认候选池限制，不按标题中的数字凑数，不把同一故事的逃跑或转折拆成多篇。'
                  '一个完整故事须有危险、主角应对、升级或发现及结果，具体动作有索引依据。'
                  '源片范围保留完整事件，不按最终短片%d至%d秒强行截断。跨本块边界无法确认完整的故事'
                  '不要猜结局，放入uncertain_ranges。输出JSON {"candidates":[{"source_start":秒,'
                  '"source_end":秒,"title":"标题","story_summary":"危险→应对→变化→结果",'
                  '"complete_story":true,"conflict":"危险","turning_point":"变化","payoff":"结果",'
                  '"characters":[],"evidence":[]}],"scan_complete":true,"uncertain_ranges":[],"reason":"零篇须说明"}。'
                  '索引:\n%s') % (params.get('min_duration',15),params.get('max_duration',35),
                                  json.dumps(chunk,ensure_ascii=False,separators=(',',':')))
        data = provider.complete_json(prompt,json_mode=True,default=None)
        if (not isinstance(data,dict) or not isinstance(data.get('candidates'),list) or
                type(data.get('scan_complete')) is not bool or not isinstance(data.get('uncertain_ranges'),list)):
            raise ValueError('故事分析缺少完整目录与覆盖说明')
        if not data['candidates'] and not str(data.get('reason','')).strip():
            raise ValueError('空故事结果没有说明原因')
        uncertain.extend(data['uncertain_ranges'])
        if not data['scan_complete']:
            uncertain.append({'source_start':chunk[0]['start'],'source_end':chunk[-1]['end'],'reason':'此段未完成分析'})
        for row in data['candidates']:
            if not isinstance(row,dict) or row.get('complete_story') is not True:
                uncertain.append({'reason':'模型返回的候选未确认独立完整'})
                continue
            lo,hi = float(row.get('source_start',-1)),float(row.get('source_end',-1))
            if not math.isfinite(lo+hi) or not 0 <= lo < hi <= float(meta.get('duration',0)) or hi-lo < 3:
                raise ValueError('完整故事范围不合法')
            if any(min(hi,float(old['source_end']))-max(lo,float(old['source_start'])) >
                   .6*min(hi-lo,float(old['source_end'])-float(old['source_start'])) for old in stories):
                continue
            stories.append(row)
    complete = not uncertain
    return MiningResult(stories,{'complete':complete,'scan_complete':True,'manual_review_required':not complete,
        'uncertain_ranges':uncertain,'reason':'已检查全片索引，发现%d个完整故事%s' %
        (len(stories),'，仍有边界需要人工复核' if uncertain else '')})


def _condensed_scenes(index, limit=0):
    """把场景压成索引串：场景越多单条越精简，尽量覆盖全片（不丢后段故事）。"""
    scenes = []
    n = len(index.scenes)
    text_cap = 160 if n <= 120 else (100 if n <= 300 else 64)
    vis_cap = 300 if n <= 120 else (180 if n <= 300 else 110)
    keep_all = n <= 1
    for scene in index.scenes:
        if _is_black_scene(scene) and not keep_all:
            continue
        text = normalize_space(scene.transcript)[:text_cap]
        visual = normalize_space(scene.visual)[:vis_cap]
        ocr = normalize_space(getattr(scene, "ocr", ""))[:120]
        if not text and not visual and not ocr and scene.speech_ratio < 0.05 and not keep_all:
            continue
        scenes.append({
            "id": scene.scene_id, "start": scene.start, "end": scene.end,
            "speech_ratio": round(scene.speech_ratio, 3),
            "text": text, "visual": visual, "ocr": ocr,
        })
        if limit and len(scenes) >= limit:
            break
    return scenes


def _rule_mine(index, params, logger):
    min_dur = float(params.get("min_duration", 15))
    max_dur = float(params.get("max_duration", 35))
    pool = int(params.get("candidate_pool", 12))
    segments = [s for s in index.transcript if s.text.strip()]
    if not segments:
        segments = _visual_pseudo_segments(index)
    scenes = index.scenes
    duration = (index.video.duration if index.video else 0.0) or max((s.end for s in scenes), default=0.0)
    logger.info("规则挖掘窗口: %d 转写段, %d 场景, 总时长 %.1fs", len(segments), len(scenes), duration)

    windows = _find_windows(segments, scenes, duration, min_dur, max_dur)
    picked = _greedy_pick(windows, pool)
    raw = []
    for i, window in enumerate(picked):
        raw.append(_build_candidate(i + 1, window, index))
    return raw


def _visual_pseudo_segments(index):
    """无字幕时，用视觉描述当作可挖文本，保证仍能定位候选片段。"""
    from ..models import TranscriptSegment

    out = []
    for scene in index.scenes:
        if _is_black_scene(scene):
            continue
        text = normalize_space(scene.visual or scene.transcript)
        if text:
            out.append(TranscriptSegment(start=scene.start, end=scene.end, text=text, source="visual"))
    return out


def _mostly_black(scenes, start, end):
    covered = [s for s in scenes if s.end > start and s.start < end]
    if not covered:
        return False
    black = [s for s in covered if _is_black_scene(s)]
    return len(black) / len(covered) > 0.5


def _is_black_scene(scene):
    brightness = getattr(scene, "brightness", -1.0)
    return 0.0 <= brightness < 15.0


def _find_windows(segments, scenes, duration, min_dur, max_dur):
    if not segments or duration <= 0:
        return []
    step = 3.0
    windows = []
    cursor = 0.0
    while cursor + min_dur <= duration:
        end = cursor + max_dur
        text_parts = [s.text for s in segments if s.end > cursor + 1.0 and s.start < end - 1.0]
        text = normalize_space(" ".join(text_parts))
        if len(text) < 16:
            cursor += step
            continue
        if _mostly_black(scenes, cursor, end):
            cursor += step
            continue
        score = _window_score(text, scenes, cursor, end, min_dur, max_dur)
        if score >= 0.34:
            windows.append({
                "start": round(cursor, 3), "end": round(end, 3),
                "text": text[:400], "score": round(score, 3),
            })
        cursor += step
    return windows


def _window_score(text, scenes, start, end, min_dur, max_dur):
    score = 0.0
    marker_hits = 0
    for marker in TURN_MARKERS:
        count = text.count(marker)
        if count:
            score += min(count, 3) * HIGHER_VALUE_MARKERS.get(marker, 0.7)
            marker_hits += 1
    if marker_hits >= 3:
        score += 1.4
    if re.search(r"[?？]", text):
        score += 1.1
    sentence_count = len(split_sentences(text))
    if sentence_count >= 2:
        score += 0.5
    for scene in scenes:
        if scene.end > start and scene.start < end:
            score += scene.speech_ratio * 0.5
            break
    target_len = int((min_dur + max_dur) / 2 * 4.0)
    if target_len > 0:
        score += min(1.0, abs(len(text) - target_len) / target_len) * 0.6
    return score


def _greedy_pick(windows, pool):
    windows = sorted(windows, key=lambda x: x["score"], reverse=True)
    picked = []
    for window in windows:
        if _overlaps(picked, window):
            continue
        picked.append(window)
        if len(picked) >= pool:
            break
    picked.sort(key=lambda x: x["start"])
    return picked


def _overlaps(picked, window):
    for item in picked:
        inter = max(0.0, min(item["end"], window["end"]) - max(item["start"], window["start"]))
        union = max(1e-6, max(item["end"], window["end"]) - min(item["start"], window["start"]))
        if inter / union > 0.42:
            return True
    return False


def _build_candidate(idx, window, index):
    start = window["start"]
    end = window["end"]
    text = window["text"]
    summary = _summarize(text)
    conflict, turning, payoff = _three_beats(text)
    moments = [round(start + 0.5, 3), round((start + end) / 2.0, 3), round(end - 0.5, 3)]
    completeness = clamp(40 + marker_score(text, 0.35) * 100 + min(len(summary), 40) / 40 * 12, 24, 96)
    visual = clamp(44 + marker_score(text, 0.25) * 100 + (12 if _has_visual_marker(text) else 0), 24, 96)
    potential = clamp(42 + completeness * 0.3 + visual * 0.3 + _score_density(text), 24, 97)
    candidate = StoryCandidate(
        candidate_id="C%03d" % idx,
        source_start=start,
        source_end=end,
        story_summary=summary,
        title=summary[:16],
        hook=_make_hook(summary),
        conflict=conflict,
        turning_point=turning,
        payoff=payoff,
        hook_visual={"start": start, "end": round(start + 3.0, 3)},
        best_visual_moments=moments,
        characters=_characters(text),
        evidence=[Evidence(start=start, end=end, scene_ids=[], quote=text[:160])],
        story_completeness=int(completeness),
        visual_quality=int(visual),
        short_video_potential=int(potential),
        tags=[m for m in TURN_MARKERS if m in text][:8],
    )
    candidate.story_score = int(round(candidate.story_completeness * 0.4 + candidate.visual_quality * 0.25 + candidate.short_video_potential * 0.35))
    return candidate


def _summarize(text):
    sentences = split_sentences(text)
    if not sentences:
        return text[:48]
    ranked = []
    for sent in sentences:
        score = marker_score(sent, 1.0) * 10 + len(sent) * 0.02
        if re.search(r"[?？]", sent):
            score += 8
        ranked.append((score, sent))
    ranked.sort(key=lambda x: x[0], reverse=True)
    if ranked[0][0] > 0.5:
        return ranked[0][1][:52]
    return sentences[0][:52]


def marker_score(text, weight=1.0):
    score = 0.0
    for marker in TURN_MARKERS:
        if marker in text:
            score += HIGHER_VALUE_MARKERS.get(marker, 0.7) * weight
    return score


def _whole_clip_fallback(index, params):
    min_dur = float(params.get("min_duration", 15))
    max_dur = float(params.get("max_duration", 35))
    duration = (index.video.duration if index.video else 0.0) or max((s.end for s in index.scenes), default=0.0)
    if duration < min_dur or duration <= 0:
        return []
    end = min(duration, max_dur)
    if end - 0.0 < min_dur:
        return []
    text = normalize_space(" ".join(s.transcript for s in index.scenes))[:300]
    summary = _summarize(text) if text else "整段未解之谜"
    return [{
        "source_start": 0.0,
        "source_end": end,
        "story_summary": summary,
        "title": summary[:12],
        "conflict": "真相被层层扒开",
        "turning_point": "关键线索接连浮现",
        "payoff": "最终真相让人细思极恐",
        "characters": [],
        "tags": [],
        "best_visual_moments": [0.0, round(end / 2.0, 3), round(max(0.1, end - 0.5), 3)],
        "evidence": [{"start": 0.0, "end": end, "quote": text[:160]}],
    }]


def _has_visual_marker(text):
    visual_words = ["尸体", "监控", "录像", "录音", "血", "尖叫", "消失", "门", "房间", "街道", "走廊", "电梯", "楼梯", "床", "镜子"]
    return any(w in text for w in visual_words)


def _score_density(text):
    return min(8.0, marker_score(text, 0.22) * 10)


def _three_beats(text):
    sentences = split_sentences(text)
    if len(sentences) <= 1:
        parts = [text[:36], text[36:72], text[72:108] or text[:36]]
        return parts[0], parts[1], parts[2]
    first = sentences[0][:36]
    middle = sentences[len(sentences) // 2][:36]
    last = sentences[-1][:36]
    return first, middle, last


def _make_hook(summary):
    if not summary:
        return "这一段内容，值得你看到最后。"
    chosen = HOOK_PREFIXES[0]
    for marker, template in [("尸体", HOOK_PREFIXES[3]), ("监控", HOOK_PREFIXES[2]), ("消失", HOOK_PREFIXES[1]), ("突然", HOOK_PREFIXES[1])]:
        if marker in summary:
            chosen = template
            break
    return chosen.replace("{text}", summary[:30])


def _characters(text):
    names = re.findall(r"([\u4e00-\u9fff]{1,4}(?:先生|女士|警官|医生|司机|老人|女孩|男孩|男人|女人|女人|男孩|女孩))", text)
    return names[:6]


def _normalize(raw, params, index):
    max_dur = float(params.get("max_duration", 35))
    min_dur = float(params.get("min_duration", 15))
    duration = (index.video.duration if index.video else 0.0) or max((s.end for s in index.scenes), default=0.0)
    out = []
    for i, item in enumerate(raw):
        cand = _coerce(item, i + 1)
        if automatic_count(params):
            if (not math.isfinite(cand.source_start+cand.source_end) or
                    not 0 <= cand.source_start < cand.source_end <= duration or not cand.story_summary):
                continue
            cand.story_score = int(round(cand.story_completeness*.4+cand.visual_quality*.25+cand.short_video_potential*.35))
            out.append(cand)
            continue
        cand.source_start = round(clamp(cand.source_start, 0.0, max(0.0, duration - min_dur)), 3)
        cand.source_end = round(clamp(cand.source_end, cand.source_start + min_dur, duration or cand.source_end), 3)
        if cand.source_end - cand.source_start < min_dur:
            cand.source_end = round(cand.source_start + min_dur, 3)
        if cand.source_end - cand.source_start > max_dur:
            cand.source_end = round(cand.source_start + max_dur, 3)
        if duration and cand.source_end > duration:
            cand.source_end = round(duration, 3)
        if not cand.story_summary:
            cand.story_summary = "候选故事 %s" % cand.candidate_id
        cand.story_score = int(round(cand.story_completeness * 0.4 + cand.visual_quality * 0.25 + cand.short_video_potential * 0.35))
        out.append(cand)
    out.sort(key=lambda c: c.story_score, reverse=True)
    for i, cand in enumerate(out):
        cand.candidate_id = "C%03d" % (i + 1)
    return out


def _coerce(item, idx):
    if isinstance(item, StoryCandidate):
        return item
    start = float(item.get("source_start") or item.get("start") or 0.0)
    end = float(item.get("source_end") or item.get("end") or start + 30.0)
    if end <= start:
        end = start + 30.0
    summary = normalize_space(str(item.get("story_summary") or item.get("summary") or ""))[:120]
    evidence_raw = item.get("evidence") or []
    evidence = []
    for ev in evidence_raw:
        if isinstance(ev, Evidence):
            evidence.append(ev)
        elif isinstance(ev, dict):
            evidence.append(Evidence(
                start=float(ev.get("start", start)),
                end=float(ev.get("end", end)),
                scene_ids=ev.get("scene_ids", []),
                quote=normalize_space(str(ev.get("quote", "")))[:200],
            ))
    moments = [float(x) for x in (item.get("best_visual_moments") or [start, (start + end) / 2.0, end]) if _num(x)]
    return StoryCandidate(
        candidate_id=item.get("candidate_id") or ("C%03d" % idx),
        source_start=start,
        source_end=end,
        story_summary=summary,
        title=normalize_space(str(item.get("title") or summary[:18])),
        hook=normalize_space(str(item.get("hook") or ""))[:120],
        characters=item.get("characters") or [],
        conflict=normalize_space(str(item.get("conflict") or ""))[:160],
        turning_point=normalize_space(str(item.get("turning_point") or ""))[:160],
        payoff=normalize_space(str(item.get("payoff") or ""))[:160],
        hook_visual=item.get("hook_visual") or {},
        best_visual_moments=moments,
        tags=item.get("tags") or [],
        evidence=evidence,
        story_completeness=int(clamp(item.get("story_completeness", 55), 0, 100)),
        visual_quality=int(clamp(item.get("visual_quality", 55), 0, 100)),
        short_video_potential=int(clamp(item.get("short_video_potential", 55), 0, 100)),
    )


def _num(value):
    try:
        float(value)
        return True
    except (TypeError, ValueError):
        return False
