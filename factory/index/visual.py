"""原片视觉分析：把每个场景的关键帧喂给 video_analysis 模型，得到画面描述。

先让模型真正“看”原片画面，之后 STORY_MINING 与 SCRIPT_GENERATED 才能
基于原片内容写故事和文案，而不是凭空编造。

为控制延迟：多个场景合并为一批请求（一次调用分析若干场景），
并用少量线程并行多个批次。
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from ..log import get_logger
from ..utils import normalize_space


def _scene_keyframes(scene, max_frames):
    frames = list(scene.keyframes or [])[:max_frames]
    return [str(p) for p in frames if Path(p).exists()]


def describe_scene(provider, scene, max_frames=2, language="zh"):
    """对单个场景送关键帧给视觉模型，返回画面描述文本。

    返回 None 表示无关键帧/模型不可用/调用失败，调用方自行降级。
    """
    frames = _scene_keyframes(scene, max_frames)
    if not frames:
        return None
    prompt = _single_prompt(scene, language)
    try:
        text = provider.complete(prompt, images=frames, json_mode=False)
        text = normalize_space(text)
        if not text:
            return None
        return text[:700]
    except Exception as exc:
        get_logger().warning("场景 %s 视觉分析失败: %s", scene.scene_id, exc)
        return None


def describe_scenes_batch(provider, scenes, max_frames=2, language="zh"):
    """一次请求分析多个场景，返回 {scene_id: 描述}。

    scenes 为 (scene, frames) 列表。整批失败返回 {}，调用方逐场景降级。
    """
    if not scenes:
        return {}
    items = []
    images = []
    for scene, frames in scenes:
        items.append((scene, len(images), len(images) + len(frames)))
        images.extend(frames)
    if not images:
        return {}
    prompt = _batch_prompt(items, language)
    try:
        data = provider.complete_json(prompt, images=images, json_mode=True, default=None)
    except Exception as exc:
        get_logger().warning("视觉分析批处理失败(%d 场景): %s", len(scenes), exc)
        return {}
    if not isinstance(data, dict):
        flat = {}
        if isinstance(data, list):
            for entry in data:
                if not isinstance(entry, dict):
                    continue
                for k, v in entry.items():
                    flat.setdefault(k, v)
        else:
            return {}
        data = flat
    out = {}
    for k, v in data.items():
        key = str(k).strip()
        key = key.split("scene")[-1].strip().lstrip("_").strip()
        key = key.split("_")[0].strip()
        if key.lstrip("-").isdigit() and isinstance(v, str) and v.strip():
            out[int(key)] = v.strip()[:700]
    return out


def _single_prompt(scene, language):
    lang_hint = "用简体中文" if language == "zh" else "用英文"
    return (
        "你是视频画面理解专家。下面图片是从原片某一场景抽取的关键帧。"
        "请基于画面本身，概括这一段时间里观众能看到什么。%s。"
        "不得编造画面里不存在的人物、台词或剧情。输出格式："
        "环境/地点；人物与外观；动作或事情经过；画面情绪氛围；可疑或惊悚细节（如有）。"
    ) % lang_hint


def _batch_prompt(items, language):
    lang_hint = "用简体中文" if language == "zh" else "用英文"
    lines = []
    for scene, start_i, end_i in items:
        idx = "%d-%d" % (start_i + 1, end_i) if end_i > start_i + 1 else str(start_i + 1)
        lines.append("图片 %s 属于场景 %d（时间 %.1fs-%.1fs）" % (
            idx, scene.scene_id, scene.start, scene.end))
    return (
        "你是视频画面理解专家。下面按顺序列出多个场景的关键帧图片，%s。"
        "请逐张图片基于画面概括每个场景观众能看到什么。不得编造画面里不存在的人物、台词或剧情。\n"
        "图片归属：\n%s\n"
        "输出 JSON 对象，键为整数场景编号，值为该场景的描述，"
        "格式：环境/地点；人物与外观；动作或事情经过；画面情绪氛围；可疑或惊悚细节（如有）。"
    ) % (lang_hint, "\n".join(lines))


def analyze_scene_visuals(index, provider, max_frames=2, language="zh", limit=0,
                          batch_size=6, workers=3, progress_cb=None):
    """对索引中的所有场景做视觉分析并回填 scene.visual。

    batch_size: 一次模型请求合并的场景数；workers: 并行批次线程数。
    return 成功分析数。progress_cb(pct, detail) 向 UI 回报进度。
    """
    logger = get_logger()
    if provider.is_mock:
        msg = "video_analysis 为 mock，跳过视觉分析"
        logger.info(msg)
        if progress_cb:
            progress_cb(100.0, msg)
        return 0
    if provider.kind in ("openai", "gemini") and not getattr(provider, "api_key", ""):
        msg = "video_analysis 未配置 api_key，跳过视觉分析"
        logger.warning(msg)
        if progress_cb:
            progress_cb(100.0, msg)
        return 0

    todo = [s for s in index.scenes if not s.visual]
    if limit:
        todo = todo[:limit]
    total = len(todo)
    if not total:
        if progress_cb:
            progress_cb(100.0, "原片视觉分析: 无待分析场景")
        return 0

    batches = [todo[i:i + batch_size] for i in range(0, total, batch_size)]
    done = 0
    finished = 0
    lock = _Lock()
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futs = {}
        for batch in batches:
            items = [(s, _scene_keyframes(s, max_frames)) for s in batch]
            items = [(s, fr) for s, fr in items if fr]
            if items:
                futs[pool.submit(_run_batch, provider, items, max_frames, language)] = len(items)
        for fut in as_completed(futs):
            scenes_done, recovered = fut.result()
            with lock:
                done += scenes_done + recovered
                finished += futs[fut]
                pct = (finished / max(1, sum(futs.values()))) * 100.0
            logger.info("视觉分析进度: %d/%d 场景 (失败降级 %d)", done, total, recovered)
            if progress_cb:
                progress_cb(pct, "视觉分析 %d/%d 场景" % (done, total))
    logger.info("原片视觉分析完成: %d 个场景", done)
    if progress_cb:
        progress_cb(100.0, "原片视觉分析完成: %d 个场景" % done)
    return done


def _run_batch(provider, items, max_frames, language):
    """执行一批：先整批请求，结果缺失的场景逐个降级。返回 (成功数, 降级数)。"""
    scenes_done = 0
    result = describe_scenes_batch(provider, items, max_frames=max_frames, language=language)
    for scene, _frames in items:
        if result.get(scene.scene_id):
            scene.visual = result[scene.scene_id]
            scenes_done += 1
    recovered = 0
    for scene, _frames in items:
        if scene.visual:
            continue
        desc = describe_scene(provider, scene, max_frames=max_frames, language=language)
        if desc:
            scene.visual = desc
            scenes_done += 1
    return scenes_done, recovered


class _Lock:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def scene_visuals(window_scenes, start, end):
    """取与 [start,end) 重叠场景的画面描述，拼接成段落文本。"""
    parts = []
    for scene in window_scenes:
        if scene.end <= start or scene.start >= end:
            continue
        if scene.visual:
            parts.append("[%s~%s] %s" % (_fmt(scene.start), _fmt(scene.end), scene.visual))
    return "\n".join(parts)


def _fmt(sec):
    return "%.1fs" % sec
