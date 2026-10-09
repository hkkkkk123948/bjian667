from datetime import datetime
from pathlib import Path

from ..log import get_logger
from ..models import MediaIndex
from ..utils import read_json, write_json
from .audio_index import analyze_audio
from .cache import index_cache_key, load_cached, save_cached
from .keyframes import assign_scene_keyframes, sample_global
from .scenes import detect_scenes
from .transcript import build_transcript

INDEX_FIELDS = ["transcript", "scenes", "audio", "metadata", "keyframes"]


def build_index(meta, video_path, subtitle_path, params, out_dir, progress_cb=None, force=False, asr=None,
                cache_dir=None):
    logger = get_logger()
    out_dir = out_dir if hasattr(out_dir, "mkdir") else Path(out_dir)
    index_path = out_dir / "video_index.json"
    if not force and index_path.exists():
        cached = load_index(index_path)
        if cached and cached.video and cached.video.path == str(video_path):
            logger.info("复用已有媒体索引: %s", index_path)
            return cached

    cache_key = None
    if not force and cache_dir:
        cache_key = index_cache_key(video_path, params)
        cached = load_cached(cache_dir, cache_key)
        if cached:
            index = _from_cache(cached, meta, video_path, out_dir)
            if index:
                logger.info("命中全局索引缓存 %s：跳过场景/转写/抽帧/视觉分析", cache_key[:10])
                save_index(index, index_path)
                return index

    index = MediaIndex(video=meta)
    steps = 4
    step = 0

    if progress_cb:
        progress_cb(step / steps, "场景检测")
    index.scenes = detect_scenes(video_path, threshold=float(params.get("scene_threshold", 0.35)))
    step += 1
    if progress_cb:
        progress_cb(step / steps, "语音转写")

    index.transcript, index.transcript_source = build_transcript(
        subtitle_path, video_path,
        mode=params.get("transcribe", "auto"),
        language=params.get("language", "zh"),
        whisper_model=params.get("whisper_model", "base"),
        duration=meta.duration,
        asr=asr,
        out_quality=index.transcript_quality,
        full=_full_transcribe(params, meta.duration),
        asr_language=(asr or {}).get("language", "auto"),
    )
    step += 1
    if progress_cb:
        progress_cb(step / steps, "音频事件")

    try:
        index.audio_events, audio_stats = analyze_audio(video_path, index.scenes)
        write_json(out_dir / "audio_stats.json", audio_stats)
    except Exception as exc:
        logger.warning("音频事件分析失败: %s", exc)
        index.audio_events = []
    step += 1
    if progress_cb:
        progress_cb(step / steps, "抽帧")

    frames_dir = out_dir / "frames"
    global_frames = sample_global(video_path, frames_dir / "l1",
                                  fps=float(params.get("frame_l1_fps", 0.5)),
                                  max_frames=int(params.get("max_l1_frames", 1800)),
                                  width=480, duration=meta.duration)
    assign_scene_keyframes(index.scenes, global_frames, per_scene=3, video_path=video_path,
                           out_dir=frames_dir, width=640)
    index.frames_dir = str(frames_dir)
    _attach_brightness(index)
    _attach_transcript_to_scenes(index)
    step += 1
    if progress_cb:
        progress_cb(1.0, "索引完成")

    index.built_at = datetime.now().isoformat(timespec="seconds")
    save_index(index, index_path)
    if cache_key:
        try:
            save_cached(cache_dir, cache_key, index)
        except Exception as exc:
            logger.debug("写入索引缓存失败: %s", exc)
    logger.info("媒体索引完成: %d 场景 / %d 转写段", len(index.scenes), len(index.transcript))
    return index


def _full_transcribe(params, duration):
    """决定是否对整片做 Whisper：无字幕时，长视频只对候选片段转写。"""
    raw = params.get("transcribe_full", "auto")
    if isinstance(raw, bool):
        return raw
    value = str(raw).lower()
    if value in ("off", "false", "0", "no", "window", "deferred"):
        return False
    if value in ("on", "true", "1", "yes", "full"):
        return True
    limit = float(params.get("window_transcribe_min", 1200))
    return (duration or 0.0) <= limit


def _from_cache(data, meta, video_path, out_dir):
    from ..models import AudioEvent, MediaIndex, SceneSegment, TranscriptSegment, from_dict

    index = MediaIndex()
    index.video = meta
    index.video.path = str(video_path)
    index.scenes = [from_dict(SceneSegment, s) for s in data.get("scenes") or []]
    index.transcript = [from_dict(TranscriptSegment, s) for s in data.get("transcript") or []]
    index.audio_events = [from_dict(AudioEvent, s) for s in data.get("audio_events") or []]
    index.frames_dir = str(Path(out_dir) / "frames")
    index.built_at = data.get("built_at", "")
    index.transcript_source = data.get("transcript_source", "none")
    index.transcript_quality = data.get("transcript_quality") or {}
    if not index.scenes and not index.transcript:
        return None
    _attach_transcript_to_scenes(index)
    return index


def _attach_transcript_to_scenes(index):
    for scene in index.scenes:
        scene.transcript = index.text_in(scene.start, scene.end)


def _attach_brightness(index):
    """用关键帧估计场景亮度，供挖故事/边界精修排除黑场。"""
    try:
        from PIL import Image, ImageStat
    except Exception:
        return
    for scene in index.scenes:
        values = []
        for path in (scene.keyframes or [])[:3]:
            try:
                with Image.open(path) as img:
                    values.append(ImageStat.Stat(img.convert("L")).mean[0])
            except Exception:
                continue
        scene.brightness = round(sum(values) / len(values), 1) if values else -1.0


def save_index(index, path):
    from ..models import to_dict

    write_json(path, to_dict(index))


def load_index(path):
    from ..models import MediaIndex, SceneSegment, TranscriptSegment, VideoMeta, AudioEvent
    from ..models import from_dict

    data = read_json(path)
    if not data:
        return None
    index = MediaIndex()
    index.video = from_dict(VideoMeta, data.get("video"))
    index.scenes = [from_dict(SceneSegment, s) for s in data.get("scenes") or []]
    index.transcript = [from_dict(TranscriptSegment, s) for s in data.get("transcript") or []]
    index.audio_events = [from_dict(AudioEvent, s) for s in data.get("audio_events") or []]
    index.frames_dir = data.get("frames_dir", "")
    index.built_at = data.get("built_at", "")
    index.transcript_source = data.get("transcript_source", "none")
    return index
