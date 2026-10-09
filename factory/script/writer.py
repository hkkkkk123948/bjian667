import json
import re
import tempfile
from pathlib import Path

from ..log import get_logger
from ..models import Evidence, SceneSegment, ScriptVariant, ScriptSentence
from ..utils import clamp, estimate_speech_seconds, ffmpeg_bin, normalize_space, run_cmd, tokenize
from ..style.short_story import STORY_RULES, REFERENCE_EXAMPLES

FLAVORS = [
    {"flavor": "suspense", "name": "悬疑型", "tone": "压低信息量，保留悬念，用疑问句结尾"},
    {"flavor": "conflict", "name": "强冲突型", "tone": "直接放大冲突，强调对立与对抗"},
    {"flavor": "twist", "name": "反转型", "tone": "先给预期，再反转揭示真相"},
]


def write_scripts(candidate, style_profile, transcript, params, router, role="script_generation", index=None,
                  language="zh", reference_text=""):
    logger = get_logger()
    provider = router.get(role)
    variants = []
    flavor_conf = params.get("flavors") or FLAVORS
    if len(flavor_conf) < 3:
        flavor_conf = FLAVORS[:3] + flavor_conf
    max_sent = int(params.get("max_sentences", 8))
    min_dur = float(params.get("min_duration", 15))
    max_dur = float(params.get("max_duration", 35))
    mode = str(params.get("content_mode", "creative")).lower()

    llm_variants = {}
    if not provider.is_mock:
        llm_variants = _llm_variants(candidate, style_profile, transcript, flavor_conf, provider, index,
                                     language, reference_text, mode, min_dur, max_dur)

    for i, fl in enumerate(flavor_conf[:3]):
        flavor = fl["flavor"]
        variant_id = "S%d" % (i + 1)
        sentences = []
        data = llm_variants.get(flavor) or llm_variants.get(fl["name"])
        if data:
            sentences = _coerce_sentences(data, candidate, i, max_sent)
        if not sentences:
            if not provider.is_mock:
                # An API failure or missing story cannot become a bland,
                # invented draft with uniformly distributed source evidence.
                continue
            if mode == "creative":
                sentences = _creative_fallback(candidate, style_profile, min_dur, max_dur, flavor, i, language)
            else:
                sentences = _rule_variant(candidate, transcript, flavor, style_profile, min_dur, max_dur, i)
        variant = _assemble(candidate, variant_id, flavor, sentences, min_dur, max_dur)
        shot_ids = [getattr(s, "shot_id", 0) for s in sentences]
        if any(shot_ids):
            variant.scene_matches = [
                {"sentence_index": k + 1, "scene_ids": [sid] if sid else [], "score": 1.0}
                for k, sid in enumerate(shot_ids)]
        variants.append(variant)
    logger.info("候选 %s 生成 %d 个脚本变体 (模式=%s)", candidate.candidate_id, len(variants), mode)
    return variants


def _cjk_ratio(text):
    text = normalize_space(text)
    if not text:
        return 0.0
    cjk = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
    return cjk / len(text)


def _creative_fallback(candidate, style_profile, min_dur, max_dur, flavor, idx, language):
    """LLM 不可用时的中文兜底：只使用候选里的中文信息，绝不拼接原片英文台词。"""
    fields = [candidate.hook, candidate.story_summary, candidate.conflict,
              candidate.turning_point, candidate.payoff]
    zh = [normalize_space(f).strip("。 ") for f in fields if f and _cjk_ratio(f) >= 0.5]
    if zh:
        lines = zh
    else:
        lines = [
            "画面里浮出一段令人不安的故事。",
            "一开始，一切看起来还算平静。",
            "可很快，事情就变得不对劲了。",
            "他察觉到了某种不对劲的迹象。",
            "紧接着，画面出现了意想不到的变化。",
            "气氛骤然紧张起来。",
            "到最后，真相似乎就要浮出水面。",
            "细想之下，才发现一切都细思极恐。",
        ]
    if language == "en":
        lines = ["A strange story unfolds on screen.", "At first everything seemed calm.",
                 "But soon something felt off.", "A sign that should not be there appeared.",
                 "Then the scene changed unexpectedly.", "The tension rose sharply.",
                 "And the truth crept closer.", "Look closer, and it is chilling."]
    span = max(1.0, candidate.source_end - candidate.source_start)
    lo = candidate.source_start
    sentences = []
    count = max(4, min(len(lines), 10))
    lines = lines[:count]
    for i, text in enumerate(lines):
        a = lo + span * (i / max(1, count))
        b = lo + span * ((i + 1) / max(1, count))
        role = "hook" if i == 0 else ("ending" if i == count - 1 else "escalation")
        sentences.append(_make_sentence(i + 1, text, role, round(a, 3), round(b, 3), candidate))
    return _trim_to_duration(sentences, min_dur, max_dur, 1.0)


def _window_visuals(candidate, index, language="zh"):
    """候选时段内原片画面描述（来自视觉分析），供写文案时参考真实画面。"""
    if not index or not getattr(index, "scenes", None):
        return ""
    parts = []
    for scene in index.scenes:
        if scene.end <= candidate.source_start or scene.start >= candidate.source_end:
            continue
        if scene.visual:
            extra = ""
            ocr = (getattr(scene, "ocr", "") or "").strip()
            if ocr:
                extra = " [屏幕文字: %s]" % ocr[:80]
            parts.append("[%s~%s] %s%s" % (round(scene.start, 1), round(scene.end, 1), scene.visual, extra))
    return "\n".join(parts)[:2400]


def _merge_shortest(shots, target):
    while len(shots) > target:
        idx = min(range(len(shots) - 1),
                  key=lambda i: (shots[i].end - shots[i].start) + (shots[i + 1].end - shots[i + 1].start))
        a, b = shots[idx], shots[idx + 1]
        a.end = b.end
        del shots[idx + 1]
    return shots


def _shot_plan(candidate, index, max_shots=9, split_every=4.0):
    """候选窗口内的镜头清单；无场景检测结果时按时间等分降级。"""
    lo, hi = candidate.source_start, candidate.source_end
    shots = []
    if index and getattr(index, "scenes", None):
        for sc in index.scenes:
            if sc.end <= lo or sc.start >= hi:
                continue
            s = SceneSegment(scene_id=sc.scene_id, start=max(sc.start, lo), end=min(sc.end, hi),
                             visual=getattr(sc, "visual", "") or "", ocr=getattr(sc, "ocr", "") or "")
            if s.end - s.start > 0.4:
                shots.append(s)
    if not shots:
        n = max(3, min(round((hi - lo) / split_every), max_shots))
        for i in range(n):
            shots.append(SceneSegment(scene_id=i + 1, start=lo + (hi - lo) * i / n,
                                      end=lo + (hi - lo) * (i + 1) / n))
    if len(shots) > max_shots:
        shots = _merge_shortest(shots, max_shots)
    return shots


def _extract_shot_keyframes(video_path, shots, out_dir, width=360):
    """每个镜头中点抽 1 张关键帧，返回图片路径列表（与 shots 一一对应）。"""
    imgs = []
    for sc in shots:
        t = (sc.start + sc.end) / 2.0
        out = Path(out_dir) / ("shot_%d_%.2f.jpg" % (sc.scene_id, t))
        _, _, code = run_cmd([ffmpeg_bin(), "-hide_banner", "-loglevel", "error", "-y",
                              "-ss", "%.2f" % t, "-i", str(video_path), "-frames:v", "1",
                              "-vf", "scale=%d:-2" % width, str(out)], check=False, timeout=120)
        if code == 0 and out.exists():
            imgs.append(str(out))
    return imgs


def _dedupe_shots_by_visual(shots, imgs, thr=9):
    """按关键帧 dHash 去重相邻的视觉重复镜头，只把「真实不同的画面」交给编剧。"""
    if len(shots) <= 1 or not imgs or len(imgs) != len(shots):
        return shots, imgs
    try:
        from PIL import Image
    except Exception:
        return shots, imgs

    def dh(p):
        img = Image.open(p).convert("L").resize((9, 8))
        px = list(img.getdata())
        h = 0
        for r in range(8):
            row = r * 9
            for c in range(8):
                if px[row + c] > px[row + c + 1]:
                    h |= 1 << (r * 8 + c)
        return h

    hashes = [dh(im) for im in imgs]
    keep = [0]
    for i in range(1, len(hashes)):
        a, b = hashes[keep[-1]], hashes[i]
        if a is None or b is None or bin(a ^ b).count("1") > thr:
            keep.append(i)
    if len(keep) == len(shots):
        return shots, imgs
    return [shots[i] for i in keep], [imgs[i] for i in keep]


def _shot_visual_block(shots):
    lines = []
    for sc in shots:
        vis = normalize_space(sc.visual or "").strip()
        ocr = normalize_space(sc.ocr or "").strip()
        extra = "画面: %s" % vis if vis else "画面: (未标注)"
        if ocr:
            extra += "；屏幕文字: %s" % ocr[:80]
        lines.append("[镜头%d] id=%d, 时间 %.1f~%.1f秒。%s" % (
            sc.scene_id, sc.scene_id, sc.start, sc.end, extra))
    return "\n".join(lines)


def _ref_sentence_template(reference_text):
    """把参考文案切成句子，作为句式/节奏模板。"""
    parts = [normalize_space(p) for p in re.split(r"(?<=[。！？!?])", reference_text) if p and p.strip()]
    return parts[:14]


def _llm_variants(candidate, style_profile, transcript, flavor_conf, provider, index=None, language="zh",
                  reference_text="", mode="creative", min_dur=15.0, max_dur=35.0):
    """一次 LLM 调用生成全部口吻脚本。creative 模式依参考示例写原创故事，commentary 模式依原片解说。

    creative 采用「镜头锚定」改写：先把候选窗口切分为镜头并抽关键帧，写稿时逐镜头对着
    关键帧描述真实可见内容，句子绑定镜头——故事与画面先天地一一对应，无需事后强行匹配。
    """
    window = _transcript_in(candidate, transcript)
    visuals = _window_visuals(candidate, index, language)
    target = int(clamp(candidate.duration, float(min_dur), float(max_dur)))
    flavor_names = json.dumps([f["name"] for f in flavor_conf[:3]], ensure_ascii=False)
    n = len(flavor_conf[:3])
    lang_hint = "用简体中文" if language == "zh" else "用英文"

    shot_imgs = []
    shots = []
    if mode == "creative" and index is not None:
        try:
            video_path = getattr(index.video, "path", "") if index.video else ""
            shots = _shot_plan(candidate, index)
            if video_path and shots:
                tmp = tempfile.mkdtemp(prefix="shot_kf_")
                shot_imgs = _extract_shot_keyframes(video_path, shots, tmp)
        except Exception:
            shot_imgs, shots = [], []

    if mode == "creative" and shot_imgs:
        shots, shot_imgs = _dedupe_shots_by_visual(shots, shot_imgs)

    if mode == "creative" and shot_imgs and len(shots) >= 2:
        return _creative_anchored(candidate, style_profile, flavor_conf, provider, language,
                                  reference_text, min_dur, max_dur, target, shots, shot_imgs, window)

    if mode == "creative":
        ref_block = ("【风格示例（必须严格模仿其题材、叙事人称、句式长短、节奏、语气与结尾方式，输出风格需与示例高度一致）】:\n%s"
                     % reference_text[:2000]) if reference_text else "（未提供示例，使用风格画像）"
        prompt = (
            """你是短视频编剧。请严格模仿下面的「风格示例」，仅依据「原片画面描述」%s，创作 %d 个、%d 秒左右、自成起承转合的完整小故事解说。
可以进行艺术加工与想象，但故事氛围、人物与场景要和画面描述协调一致；不必忠实复述原片台词。输出必须与示例文案的风格高度一致（同样的题材取向、人称、句式长短、节奏快慢、语气与收尾方式）。

要求:
【因果密度】
%s
1. 口吻分别为: %s
2. 结构因果链: 遇险 → 行动理由 → 行动 → 危险升级/新发现 → 解决办法 → 结果；首句直接进入危险，禁止注水外貌描写；每句只推进一件事。
3. 句数 6~9，整体朗读时长约 %d 秒（贴近参考片22–36秒密度），不要留大段空白。
4. 每句给出 evidence_start / evidence_end（秒）对齐画面。
5. 只输出 JSON 对象，格式 {"variants": [ 恰好 %d 个元素 ]}，每个元素 {flavor, sentences:[{text, role, evidence_start, evidence_end}]}。

%s
风格画像: %s
原片画面描述:
%s
原片台词（仅作背景参考）:
%s"""
        ) % (
            lang_hint, n, target,
            STORY_RULES[:220],
            flavor_names,
            target,
            n,
            ref_block,
            json.dumps(_profile_dict(style_profile), ensure_ascii=False),
            visuals or "（无画面描述）",
            json.dumps(window, ensure_ascii=False)[:2000],
        )
    else:
        visuals_block = "原片画面描述（视觉模型观察该时段画面的真实内容）:\n%s" % visuals if visuals else ""
        prompt = """你是短视频解说词写手。请根据候选片段一次生成 %d 个不同口吻的解说脚本，每个脚本约 %d 秒、5~10 句话，必须严格对应原片时间轴。

要求:
1. 口吻分别为: %s
2. 第一句是 1-3 秒的 hook，直接抓人，不铺垫。
3. 每句必须给出 evidence_start / evidence_end（秒）。
4. 解说必须建立在原片的真实画面与台词之上，严禁编造。
5. 只输出 JSON 对象，格式 {"variants": [ 恰好 %d 个元素 ]}，每个元素 {flavor, sentences:[{text, role, evidence_start, evidence_end}]}。

风格画像: %s
候选故事: %s
%s
原片台词窗口:
%s""" % (
            n, target,
            flavor_names,
            n,
            json.dumps(_profile_dict(style_profile), ensure_ascii=False),
            json.dumps(candidate.__dict__, ensure_ascii=False, default=str)[:1500],
            visuals_block,
            json.dumps(window, ensure_ascii=False)[:3000],
        )
    data = provider.complete_json(prompt, json_mode=True, default=[])
    return _parse_variants(data, flavor_conf)


def _creative_anchored(candidate, style_profile, flavor_conf, provider, language, reference_text,
                       min_dur, max_dur, target, shots, shot_imgs, window):
    """creative 模式的核心改版：逐镜头对着关键帧写稿，句子绑定镜头，风格强模仿参考文案。"""
    n = len(flavor_conf[:3])
    flavor_names = json.dumps([f["name"] for f in flavor_conf[:3]], ensure_ascii=False)
    ref_block = ("【参考文案——必须逐句模仿其句式长短、人称、节奏、情绪递进与收尾方式，输出语感应与其高度接近，严禁套话/水词】:\n%s"
                 % reference_text[:2500]) if reference_text else "（未提供参考文案，仅用风格画像）"
    templates = "\n".join("- %s" % s for s in _ref_sentence_template(reference_text)) or "（无）"
    shot_block = _shot_visual_block(shots)
    prompt = """你是短视频惊悚解说编剧。核心铁律是「音画一一对应」：我已给出候选片段内每个镜头的关键帧画像，句子必须绑定镜头（shot_index），只描述该镜头画面上真实可见的元素（人物、动作、环境、光影、可读字幕）。

同时你要像一个真正的惊悚片解说员那样讲故事，而不是罗列画面：
- 允许基于「可见画面」的合理氛围联想与因果推断——比如把楼梯旁的黑影读作“有人慢慢上楼”、把孩子紧盯着空处读作“她在看一个我们看不见的东西”。这种推断是惊悚叙事的一部分，参考文案也是这么写气氛的。
- 严禁写与画面冲突的具体内容：错误的房间/人物、画面里不存在的物件、别的镜头才有的东西；推断只能建立在同镜/紧邻可见的信息上，不得空降设定。
- 必须依据镜头顺序构建完整因果小故事（遇险→行动→升级→解决→结果），有结尾爆点或明确结局；情绪曲线照参考模板，禁止只串惊吓镜头。

风格必须与下面的参考文案高度相似：一样的题材取向、人称、句式长短、节奏快慢、语气和收尾。参考里怎么断句、怎么吊胃口、怎么收尾，你就照那个语感组织句子（用词可换，语感照抄）。

参考文案：%s

参考句句式模板（学节奏，不抄词）:
%s

【镜头清单】（关键帧图片已按顺序附上）:
%s

要求:
1. 生成恰好 %d 个风格变体，口吻分别为: %s。
2. 每个变体 %d 句（等于上面镜头数，每个镜头**一次且仅一次**按顺序使用），整段朗读时长约 %d 秒。
3. 句子顺序与镜头顺序一致，绝不回退、不跳镜头；已按画面去重，全部是真实不同的画面，严禁任何一句描述重复出现过的画面。
4. 首句是 1-3 秒的 hook，结尾落在最后 1-2 个镜头上，要有爆点/反转/悬念，不要平淡收尾。
5. 每句必须给出: text, role, shot_index(对应上面镜头 id), evidence_start/evidence_end（必须落在该镜头的起止时间内）, seconds(该句配音时长的估计)。
6. 只要关键帧里有可读文字（门牌/申请单/标语/字幕），就引用来强化悬念。
7. 只输出 JSON：{"variants":[恰好 %d 个元素]}，每个元素 {flavor, sentences:[{text, role, shot_index, evidence_start, evidence_end, seconds}]}。

风格画像: %s
原片台词（仅当对应镜头画面字幕可读时参考，其余忽略）:
%s""" % (
        ref_block,
        templates,
        shot_block,
        n, flavor_names, len(shots), target, n,
        json.dumps(_profile_dict(style_profile), ensure_ascii=False),
        json.dumps(window[:12], ensure_ascii=False)[:1200],
    )
    data = provider.complete_json(prompt, json_mode=True, default=[], images=shot_imgs)
    parsed = _parse_variants(data, flavor_conf)
    for item in parsed.values():
        item["_shots"] = shots
    return parsed


def _parse_variants(data, flavor_conf):
    """把 LLM 返回的 JSON 规整为 {flavor: item}。"""
    if isinstance(data, dict):
        if isinstance(data.get("variants"), list):
            data = data["variants"]
        elif isinstance(data.get("scripts"), list):
            data = data["scripts"]
        elif "sentences" in data:
            data = [data]
        else:
            data = []
    out = {}
    for item in data or []:
        if not isinstance(item, dict):
            continue
        key = str(item.get("flavor") or "")
        for fl in flavor_conf:
            if key in (fl["flavor"], fl["name"]):
                out.setdefault(fl["flavor"], item)
    return out


def _profile_dict(p):
    return {
        "sentence_length": p.sentence_length,
        "narration": p.narration,
        "pacing": p.pacing,
        "emotion": p.emotion,
        "structure": p.structure,
        "ending": p.ending,
    }


def _transcript_in(candidate, transcript):
    out = []
    for seg in transcript:
        if seg.end > candidate.source_start and seg.start < candidate.source_end:
            out.append({"start": seg.start, "end": seg.end, "text": seg.text})
        if len(out) > 40:
            break
    return out


def _coerce_sentences(data, candidate, idx, max_sent):
    out = []
    raw = data.get("sentences") if isinstance(data, dict) else data
    shots = data.get("_shots") or [] if isinstance(data, dict) else []
    for j, s in enumerate((raw or [])[:max_sent]):
        if isinstance(s, dict):
            text = normalize_space(str(s.get("text") or s.get("sentence") or ""))
            role = str(s.get("role") or "")
            es = s.get("evidence_start")
            ee = s.get("evidence_end")
            shot = None
            if shots:
                si = _num(s.get("shot_index"), None)
                if si is not None:
                    shot = next((sh for sh in shots if sh.scene_id == int(si)), None)
                if shot is not None:
                    es, ee = _clamp_to_shot(es, ee, shot)
            if text:
                sent = _make_sentence(j + 1, text, role, es, ee, candidate)
                if shot is not None:
                    sent.shot_id = shot.scene_id
                secs = _num(s.get("seconds"), None)
                if secs:
                    sent.estimated_seconds = clamp(float(secs), 1.0, 12.0)
                out.append(sent)
        elif isinstance(s, str) and s.strip():
            out.append(_make_sentence(j + 1, s.strip(), "", None, None, candidate))
    return out


def _clamp_to_shot(es, ee, shot):
    """把句子证据时间钳进它绑定的镜头区间内（默认取镜头中后段的画面）。"""
    lo, hi = shot.start, shot.end
    if es is None or ee is None:
        front = lo + (hi - lo) * 0.30
        return float(round(front, 3)), float(round(min(hi, front + (hi - lo) * 0.45), 3))
    es, ee = clamp(float(es), lo, hi), clamp(float(ee), lo, hi)
    if ee <= es:
        ee = min(hi, es + 2.0)
    if ee <= es:
        es = max(lo, ee - 2.0)
    return float(round(es, 3)), float(round(ee, 3))


def _make_sentence(idx, text, role, es, ee, candidate):
    lo = candidate.source_start
    hi = candidate.source_end
    start = _num(es, None)
    end = _num(ee, None)
    if start is None:
        start = (lo + hi) / 2.0
    else:
        start = clamp(start, lo, hi)
    if end is None:
        end = min(hi, start + 2.5)
    else:
        end = clamp(end, lo, hi)
    if end <= start:
        end = min(hi, start + 2.5)
    if end <= start:
        start = max(lo, end - 2.5)
    start = round(start, 3)
    end = round(end, 3)
    if end <= start:
        start, end = round(lo, 3), round(min(hi, lo + 1.0), 3)
    ev = Evidence(start=start, end=end, quote=text[:60])
    return ScriptSentence(
        index=idx, text=text, source_evidence=ev,
        estimated_seconds=estimate_speech_seconds(text), role=role,
    )


def _num(value, default):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


CONFLICT_RE = re.compile(r"但是|却|发现|不对|失踪|报警|有人|跟着|不对劲")
TURNING_RE = re.compile(r"真相|原来|竟然|结果|监控|尸体|失踪|录音|录像|忽然|突然")
PAYOFF_RE = re.compile(r"报警|血|消失|尖叫|反转|真相|最后一秒|原来")


def _rule_variant(candidate, transcript, flavor, style_profile, min_dur, max_dur, idx):
    window = _transcript_in(candidate, transcript)
    lo, hi = candidate.source_start, candidate.source_end
    quotes = []
    for s in window:
        text = normalize_space(s["text"]).strip("。 ")
        if text:
            quotes.append((text, float(s["start"]), float(s["end"])))
    span = max(0.5, hi - lo)
    want = int(clamp(round(span / 5.0), 3, 8))
    picks = _pick_quotes(quotes, want)
    for field in (candidate.conflict, candidate.turning_point, candidate.payoff):
        text = normalize_space(field or "").strip("。 ")
        ev = _match_evidence(text, quotes)
        if not text or not ev:
            continue
        if any(_same_quote(text, p[0]) for p in picks):
            continue
        picks.append((text, ev[0], ev[1]))
    picks.sort(key=lambda p: p[1])
    if len(picks) > want + 2:
        picks = picks[:want + 2]

    hook = normalize_space(candidate.hook or "").strip("。 ") or (quotes[0][0] if quotes else "这一段，你真的敢看到最后吗？")
    ending = _ending_line(candidate, flavor, style_profile)

    items = []
    hook_ev = _match_evidence(hook, quotes)
    if hook_ev is None or (picks and hook_ev[1] > picks[0][1] + 0.5):
        if quotes:
            hook_ev = (quotes[0][1], min(quotes[0][2], quotes[0][1] + 4.0))
        else:
            hook_ev = (lo, min(hi, lo + 3.0))
    items.append(("hook", hook, hook_ev[0], hook_ev[1]))

    has_conflict = False
    has_twist = False
    for i, (text, qs, qe) in enumerate(picks):
        if _same_quote(text, hook):
            continue
        if not has_conflict and CONFLICT_RE.search(text):
            role = "conflict"
            has_conflict = True
        elif not has_twist and flavor in ("suspense", "twist") and TURNING_RE.search(text):
            role = "twist"
            has_twist = True
        elif i == 0:
            role = "context"
        elif i == len(picks) - 1:
            role = "payoff"
        else:
            role = "escalation"
        items.append((role, text, qs, qe))

    end_ev = (quotes[-1][1], quotes[-1][2]) if quotes else (max(lo, hi - 2.5), hi)
    if not _same_quote(ending, items[-1][1]):
        items.append(("ending", ending, end_ev[0], end_ev[1]))

    deduped = []
    seen = set()
    for role, text, qs, qe in items:
        text = normalize_space(text).strip("。 ")
        key = text[:24]
        if not text or key in seen:
            continue
        seen.add(key)
        deduped.append((role, text, qs, qe))

    sentences = [_make_sentence(i + 1, text, role, qs, qe, candidate)
                 for i, (role, text, qs, qe) in enumerate(deduped)]
    return _trim_to_duration(sentences, min_dur, max_dur, 1.0)


def _same_quote(a, b):
    a = normalize_space(a or "")[:24]
    b = normalize_space(b or "")[:24]
    return bool(a) and a == b


def _match_evidence(text, quotes):
    norm = normalize_space(text or "").strip("。 ")
    if not norm or not quotes:
        return None
    tset = set(tokenize(norm))
    best = None
    best_score = 0.0
    for qt, qs, qe in quotes:
        if norm == qt:
            return (qs, qe)
        score = 0.0
        if norm in qt or qt in norm:
            score = 0.85
        else:
            qset = set(tokenize(qt))
            if tset and qset:
                score = len(tset & qset) / len(tset | qset)
        if score > best_score:
            best = (qs, qe)
            best_score = score
    if best and best_score >= 0.45:
        return best
    return None


def _pick_quotes(quotes, want):
    if not quotes:
        return []
    if len(quotes) <= want:
        return list(quotes)
    picked = []
    seen = set()
    for i in range(want):
        idx = round(i * (len(quotes) - 1) / max(1, want - 1))
        while idx in seen and idx + 1 < len(quotes):
            idx += 1
        if idx in seen:
            continue
        seen.add(idx)
        picked.append(quotes[idx])
    return picked


def _trim_to_duration(sentences, min_dur, max_dur, scale):
    total = sum(s.estimated_seconds for s in sentences)
    if total <= max_dur * 1.1 and len(sentences) >= 2:
        while total > max_dur * 1.1 and len(sentences) > 2:
            dropped = sentences.pop()
            total -= dropped.estimated_seconds
    if total <= 0:
        return sentences
    if total < min_dur * 0.9:
        factor = clamp(min_dur * 0.95 / total, 1.0, 1.35)
        for s in sentences:
            s.estimated_seconds = round(s.estimated_seconds * factor, 3)
    return sentences


def _ending_line(candidate, flavor, style_profile):
    if style_profile.ending == "closed_truth":
        return "答案，就在最后那一秒。"
    if flavor == "suspense":
        return "你敢想，事情根本不是你以为的那样。"
    if flavor == "conflict":
        return "这一仗，才刚刚开始。"
    return "真相揭开的那一刻，所有人才发现——细思极恐。"


def _assemble(candidate, variant_id, flavor, sentences, min_dur, max_dur):
    sentences = sentences or []
    full_text = " ".join(s.text for s in sentences)
    est = sum(s.estimated_seconds for s in sentences)
    variant = ScriptVariant(
        variant_id=variant_id,
        candidate_id=candidate.candidate_id,
        flavor=flavor,
        sentences=sentences,
        full_text=full_text,
        estimated_seconds=round(est, 3),
    )
    return variant
