"""解说落地化（grounding）：让每句解说严格贴合它播放的那段画面。

问题根源：creative 编的故事常“画面里没有”，导致音画不匹配（match 分低）。
做法：先按非重叠分区确定每句的播放窗口 → 抽该窗口的画面帧 → 让 LLM 把这句话改写为
“画面里真实发生的事”（保留惊悚/悬疑风格，长度相近）→ 重新配音 → 重建时间轴并重渲染。

只改文本（软件行为），不改任何画面内容。
"""

from pathlib import Path

from ..log import get_logger
from ..utils import clamp, ensure_dir, ffmpeg_bin, normalize_space, run_cmd

ROLE = "story_writer"


def extract_segment_frames(video_path, src_start, dur, out_dir, width=640):
    frames = []
    for rel in (0.35, 0.7):
        t = clamp(src_start + dur * rel, 0.0, 10 ** 9)
        out = Path(out_dir) / ("seg_%.1f_%d.jpg" % (src_start, int(rel * 100)))
        _, _, code = run_cmd([ffmpeg_bin(), "-hide_banner", "-loglevel", "error", "-y",
                              "-ss", "%.2f" % t, "-i", str(video_path), "-frames:v", "1",
                              "-vf", "scale=%d:-2" % width, str(out)], check=False, timeout=120)
        if code == 0 and out.exists():
            frames.append(str(out))
    return frames


def _neighbor_visual(index, start, end):
    try:
        for sc in index.scenes:
            if sc.end > start - 1.0 and sc.start < end + 1.0:
                v = normalize_space(getattr(sc, "visual", "") or "")
                if v:
                    return v[:140]
    except Exception:
        pass
    return ""


def ground_sentence(router, sentence_index, text, prev_text, next_text, frames, neighbor_visual,
                    ref_style, estimated_seconds, seg_dur=None, total_target=None, orig_total=None,
                    extra_hint=""):
    """把单句解说改写为严格贴合画面。返回 (new_text, new_seconds) 或 (None, None)。"""
    logger = get_logger()
    if not router or router.is_mock(ROLE):
        return None, None
    duration_hint = int(estimated_seconds)
    dur_note = (" 这句解说的画面窗口约 %.1f 秒，配音时长请控制在 0.85~1.15 倍该值之间。"
                % seg_dur) if seg_dur else ""
    total_note = ""
    if total_target:
        total_note = (" 整片目标时长约 %.0f 秒（原稿约 %.0f 秒），请全片精简、避免啰嗦，本句字数大致对应 %.1f 秒。"
                      % (total_target, orig_total or total_target, seg_dur or estimated_seconds))
    prompt = (
        "你负责把解说词改写到“与画面完全一致”。下面这句解说会播放于对应的画面片段（帧已给出），"
        "旁白时间约 %d 秒。请把它改写为一句**严格描述画面里真实发生的事**的中文解说：\n"
        "- 只写画面确实能看到的内容（人物动作/场景变化/可见字幕等），画面里没有的（想法、记忆、声音、未来情节）一律不写；\n"
        "- 保留惊悚/悬疑的叙事语气，句子通顺自然；\n"
        "- 长度与原句接近（±30%%）%s%s；\n"
        "- 若原句与画面完全无关，不要硬搬，改成描写画面本身。%s\n"
        "只输出 JSON：{\"sentence\":\"改写后的句子\",\"seconds\":估算的配音秒数}\n\n"
        "上一句（若有）: %s\n本句（原标题）: %s\n下一句（若有）: %s\n"
        "画面剧情提示: %s\n"
        "参考文案风格（仅学语气，不抄内容）:\n%s\n"
    ) % (duration_hint, dur_note, total_note, prev_text or "（无）", text, next_text or "（无）",
         neighbor_visual or "（无）", ref_style, (" " + extra_hint) if extra_hint else "")
    try:
        data = router.get(ROLE).complete_json(prompt, json_mode=True, default=None, images=frames)
    except Exception as exc:
        logger.warning("第%d句落地化失败: %s", sentence_index, exc)
        return None, None
    new_text = ""
    if isinstance(data, dict):
        new_text = str(data.get("sentence") or "").strip()
        try:
            seconds = float(data.get("seconds") or 0.0)
        except (TypeError, ValueError):
            seconds = 0.0
    if len(new_text) < 6:
        return None, None
    if seconds <= 0.0:
        seconds = estimated_seconds * (len(new_text) / max(1.0, len(text)))
    return new_text, seconds


def _ref_style(pipeline):
    refs = pipeline.config.param("references") or []
    for path in refs:
        p = Path(str(path))
        if p.exists():
            try:
                txt = p.read_text(encoding="utf-8", errors="replace")
            except Exception:
                continue
            if txt.strip():
                return txt.strip()[:1200]
    return ""


def _image_dhash(path, size=8):
    """8x8 感知哈希，用于判断两帧是否同一画面。"""
    try:
        from PIL import Image
        img = Image.open(path).convert("L").resize((size + 1, size))
        px = list(img.getdata())
        h = 0
        for r in range(size):
            row = r * (size + 1)
            for c in range(size):
                if px[row + c] > px[row + c + 1]:
                    h |= 1 << (r * size + c)
        return h
    except Exception:
        return None


def _hamming(a, b):
    return bin(a ^ b).count("1")


def _mid_frame_hash(work_dir, seg):
    path = Path(work_dir) / ("seg_%.1f_70.jpg" % seg["src_start"])
    if not path.exists():
        path = Path(work_dir) / ("seg_%.1f_35.jpg" % seg["src_start"])
    return _image_dhash(str(path)) if path.exists() else None


def _dedupe_similar_segments(variant, segs, work_dir, min_keep=3, thr=9):
    """相邻窗口画面几乎相同（汉明≤thr）时合并：删重复句，避免配音‘念重复的画面’。

    撞上源片本来就是长静止镜头的段落，宁可少说、也不硬编不同内容。
    """
    hashes = [_mid_frame_hash(work_dir, s) for s in segs]
    keep, drop = [], 0
    for i, h in enumerate(hashes):
        if (keep and h is not None and hashes[keep[-1]] is not None
                and _hamming(hashes[keep[-1]], h) <= thr and len(segs) - drop > min_keep):
            drop += 1
            continue
        keep.append(i)
    if drop:
        variant.sentences = [variant.sentences[i] for i in keep]
        variant.full_text = "\n".join(s.text for s in variant.sentences)
    return drop


def _ground_pass(pipeline, variant, segs, work_dir, ref_style, router, clip_id,
                 target_total, orig_total, logger, tag=""):
    """针对当前段落窗口逐句落地化（含与上句文字重复的重试）。"""
    originals = [s.text for s in variant.sentences]
    for i, sent in enumerate(variant.sentences):
        seg = segs[i]
        frames = extract_segment_frames(str(pipeline.state.artifacts["video_path"]),
                                        seg["src_start"], seg["dur"], work_dir)
        if not frames:
            logger.warning("clip %s 第%d句抽帧失败，保留原文", clip_id, i + 1)
        prev_text = variant.sentences[i - 1].text if i > 0 else ""
        next_text = variant.sentences[i + 1].text if i + 1 < len(variant.sentences) else ""
        neighbor = _neighbor_visual(pipeline.index, seg["src_start"], seg["src_start"] + seg["dur"])
        new_text, seconds_i = ground_sentence(
            router, i + 1, originals[i], prev_text, next_text, frames or [],
            neighbor, ref_style, sent.estimated_seconds, seg_dur=seg["dur"],
            total_target=target_total, orig_total=orig_total)
        if new_text and i > 0 and new_text == variant.sentences[i - 1].text:
            new_text_2, seconds_i2 = ground_sentence(
                router, i + 1, originals[i], prev_text, next_text, frames or [],
                neighbor, ref_style, sent.estimated_seconds, seg_dur=seg["dur"],
                total_target=target_total, orig_total=orig_total,
                extra_hint="刚才的改写与上一句完全相同。请换一个视角/画面细节/动作重新描述，"
                           "不得与上一句在文字上重复，但仍须严格贴合本句画面。")
            if new_text_2 and new_text_2 != variant.sentences[i - 1].text:
                new_text, seconds_i = new_text_2, seconds_i2
        if new_text:
            sent.text = new_text
            sent.estimated_seconds = max(1.2, seconds_i)
        prefix = "pass%s" % tag if tag else "fallback"
        logger.info("clip %s 第%d句(%s): %s", clip_id, i + 1, prefix, sent.text[:34])
    return variant


def ground_clip(pipeline, clip, max_passes=3):
    """对单条 clip 做落地化重写 → 重新配音 → 重建时间轴 → 重渲染。

    落地化必须**迭代**：改句依据的是“当时的画面窗口”，重配音会改变句子时长、重新分区，
    窗口一旦移位，上一轮改的句子就不再匹配。因此每一轮都用最新窗口重新落地化，
    直到窗口漂移小于阈值。
    """
    from ..render.timeline import build_timeline
    from ..production import rerender_export

    logger = get_logger()
    cand = pipeline._by_id(clip.candidate_id)
    variant = pipeline._pick_variant(cand)
    if not variant or not variant.sentences:
        logger.warning("clip %s 无可用变体/解说", clip.clip_id)
        return clip
    if pipeline.tts is None:
        logger.warning("clip %s 无 TTS 引擎，跳过落地化", clip.clip_id)
        return clip
    clip_dir = pipeline.job_dir / "clips" / clip.clip_id
    work_dir = ensure_dir(clip_dir / "ground")
    ref_style = _ref_style(pipeline)
    router = pipeline.router
    n = len(variant.sentences)
    orig_total = sum(max(1.2, s.estimated_seconds) for s in variant.sentences)
    target_total = float(pipeline.config.param("max_duration", 35) or 35) * 0.98

    cur_segs = (_read_json(clip_dir / "timeline.json").get("segments") or [])
    if len(cur_segs) != n:
        cur_segs = _partition_from_timings(cand, variant, pipeline, clip)
    if len(cur_segs) != n:
        logger.warning("clip %s 分段与解说句数不一致(%d vs %d)，跳过", clip.clip_id,
                       len(cur_segs), n)
        return clip
    for seg in cur_segs:
        seg["src_start"] = float(seg.get("src_start", 0.0))
        seg["dur"] = float(seg.get("dur", 0.0))

    timeline = None
    for _pass in range(max_passes):
        n = len(variant.sentences)
        _ground_pass(pipeline, variant, cur_segs, work_dir, ref_style, router, clip.clip_id,
                     target_total, orig_total, logger, tag=str(_pass + 1))
        result = pipeline.tts.synth(variant.full_text, clip_dir / "voice.mp3")
        _write_json(clip_dir / "tts.json", result)
        timeline = build_timeline(cand, variant, result, pipeline.config.params,
                                  video_duration=pipeline.index.video.duration if pipeline.index else 0.0,
                                  index=pipeline.index, router=pipeline.router)
        new_segs = timeline.get("segments") or []
        if not new_segs or len(new_segs) != n:
            break
        drift = max(abs(float(new_segs[i]["src_start"]) - float(cur_segs[i]["src_start"]))
                    for i in range(n))
        cur_segs = new_segs
        logger.info("clip %s pass%d 窗口最大漂移 %.2fs", clip.clip_id, _pass + 1, drift)
        if drift < 1.2:
            break

    for _dd in range(3):
        before = len(variant.sentences)
        dropped = _dedupe_similar_segments(variant, cur_segs, work_dir)
        if not dropped:
            break
        logger.info("clip %s 合并相似画面段落 %d → %d 句", clip.clip_id, before,
                    len(variant.sentences))
        result = pipeline.tts.synth(variant.full_text, clip_dir / "voice.mp3")
        _write_json(clip_dir / "tts.json", result)
        timeline = build_timeline(cand, variant, result, pipeline.config.params,
                                  video_duration=pipeline.index.video.duration if pipeline.index else 0.0,
                                  index=pipeline.index, router=pipeline.router)
        cur_segs = timeline.get("segments") or []
        if len(cur_segs) != len(variant.sentences):
            break
        _ground_pass(pipeline, variant, cur_segs, work_dir, ref_style, router,
                     clip.clip_id, target_total, orig_total, logger, tag="dedupe%d" % (_dd + 1))
        result = pipeline.tts.synth(variant.full_text, clip_dir / "voice.mp3")
        _write_json(clip_dir / "tts.json", result)
        timeline = build_timeline(cand, variant, result, pipeline.config.params,
                                  video_duration=pipeline.index.video.duration if pipeline.index else 0.0,
                                  index=pipeline.index, router=pipeline.router)
        cur_segs = timeline.get("segments") or []
        if len(cur_segs) != len(variant.sentences):
            break

    _write_json(clip_dir / "timeline.json", timeline)
    clip.script_text = variant.full_text
    clip.scores = clip.scores or {}
    clip.scores["tts_path"] = result["audio_path"]
    clip.scores["tts_duration"] = result.get("duration", 0.0)
    clip.variant_id = variant.variant_id
    pipeline._adopt_variant(cand, variant)
    rerender_export(pipeline, clip, clip_dir, timeline)
    pipeline._save_clips()
    return clip


def _partition_from_timings(cand, variant, pipeline, clip):
    """当前时间轴不可用时，先按原证据重建一次非重叠分区。"""
    from ..render.timeline import build_timeline
    clip_dir = pipeline.job_dir / "clips" / clip.clip_id
    tts_data = _read_json(clip_dir / "tts.json") or {}
    timeline = build_timeline(cand, variant, tts_data, pipeline.config.params,
                              video_duration=pipeline.index.video.duration if pipeline.index else 0.0,
                              index=pipeline.index, router=pipeline.router)
    _write_json(clip_dir / "timeline.json", timeline)
    return timeline.get("segments") or []


def _read_json(path):
    from ..utils import read_json
    return read_json(path) or {}


def _write_json(path, data):
    from ..utils import write_json
    write_json(path, data)
