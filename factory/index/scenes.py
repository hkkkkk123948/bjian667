from ..log import get_logger
from ..models import SceneSegment
from ..utils import media_duration, run_cmd


def detect_scenes(video_path, threshold=0.35, min_scene_len=0.6, max_scenes=0):
    logger = get_logger()
    try:
        scenes = _scenedetect(video_path, threshold, min_scene_len)
    except Exception as exc:
        logger.warning("scenedetect 失败(%s)，改用 ffmpeg scdet", exc)
        scenes = []
    if not scenes:
        scenes = _ffmpeg_scdet(video_path, threshold, min_scene_len)
    if not scenes:
        duration = media_duration(video_path)
        scenes = [(0.0, duration)]
    cleaned = []
    for start, end in scenes:
        if end - start < min_scene_len and cleaned:
            cleaned[-1] = (cleaned[-1][0], end)
        else:
            cleaned.append((start, end))
    if max_scenes and len(cleaned) > max_scenes:
        logger.info("场景过多 %d，合并到 %d", len(cleaned), max_scenes)
        cleaned = _merge_to_count(cleaned, max_scenes)
    logger.info("场景检测: %d 个场景", len(cleaned))
    return [SceneSegment(scene_id=i + 1, start=round(s, 3), end=round(e, 3))
            for i, (s, e) in enumerate(cleaned)]


def _scenedetect(video_path, threshold, min_scene_len):
    from scenedetect import ContentDetector, SceneManager, open_video

    video = open_video(str(video_path))
    manager = SceneManager()
    rate = getattr(video, "frame_rate", None) or getattr(video, "framerate", None) or 25.0
    # FFmpeg uses 0..1 whereas PySceneDetect uses 0..255 pixel differences.
    content_threshold = threshold * 100.0 if threshold <= 1.0 else threshold
    manager.add_detector(ContentDetector(threshold=content_threshold, min_scene_len=int(min_scene_len * rate)))
    manager.detect_scenes(video, show_progress=False)
    scene_list = manager.get_scene_list()
    out = []
    for scene in scene_list:
        try:
            start, end = scene.get_seconds()
        except Exception:
            start = scene[0].get_seconds()
            end = scene[1].get_seconds()
        out.append((float(start), float(end)))
    if not out:
        out = [(0.0, float(video.duration))]
    return out


def _ffmpeg_scdet(video_path, threshold, min_scene_len):
    _, err, code = run_cmd([
        "ffmpeg", "-hide_banner", "-i", str(video_path),
        "-vf", "select='gt(scene,%.4f)',metadata=print" % threshold,
        "-an", "-f", "null", "-"
    ], check=False, timeout=1800)
    if code != 0 and not err:
        return []
    times = []
    lines = (err or "").splitlines()
    for i, line in enumerate(lines):
        if "pts_time:" in line:
            try:
                value = line.split("pts_time:")[1].strip().split()[0]
                times.append(float(value))
            except (IndexError, ValueError):
                continue
    duration = media_duration(video_path)
    points = [0.0] + [t for t in times if min_scene_len < t < duration] + [duration]
    merged = []
    for i in range(len(points) - 1):
        if points[i + 1] - points[i] > 0.01:
            merged.append((points[i], points[i + 1]))
    return merged


def _merge_to_count(scenes, count):
    while len(scenes) > count:
        lengths = [(end - start, i) for i, (start, end) in enumerate(scenes)]
        _, idx = min(lengths)
        start, _ = scenes[idx]
        _, end = scenes[idx + 1]
        scenes[idx] = (start, end)
        del scenes[idx + 1]
    return scenes


def _escape(path):
    return path.replace("\\", "/").replace(":", "\\:")
