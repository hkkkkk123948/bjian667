from ..log import get_logger
from ..utils import clamp

BLACK_BRIGHTNESS = 15.0


def refine_boundaries(candidate, index, min_dur, max_dur):
    logger = get_logger()
    start, end = candidate.source_start, candidate.source_end
    start, end = _snap_to_scene_edges(start, end, index, prefer_outside=True)
    start, end = _snap_to_silence_edges(start, end, index)
    start, end = _trim_black_edges(start, end, index)
    start = round(clamp(start, 0.0, max(0.0, (index.video.duration if index.video else 0.0) - min_dur)), 3)
    end = round(clamp(end, start + min_dur, (index.video.duration if index.video else 0.0) or end), 3)
    if end - start > max_dur:
        end = round(start + max_dur, 3)
    if candidate.source_start != start or candidate.source_end != end:
        logger.info("候选 %s 边界微调: %.2f-%.2f → %.2f-%.2f",
                    candidate.candidate_id, candidate.source_start, candidate.source_end, start, end)
    candidate.source_start = start
    candidate.source_end = end
    hook_start = pick_hook_point(candidate, index)
    candidate.hook_visual = {"start": round(hook_start, 3), "end": round(max(hook_start + 3.0, start + 3.0), 3)}
    return candidate


def _trim_black_edges(start, end, index):
    """裁掉窗口首尾的黑场场景，避免成片出现黑屏。"""
    scenes = index.scenes or []
    if not scenes:
        return start, end
    guard = 0
    while guard < 20 and start < end - 1.0:
        scene = index.scene_at(start + 0.05)
        if scene and _is_black(scene) and scene.end <= end + 0.01:
            start = scene.end
            guard += 1
        else:
            break
    guard = 0
    while guard < 20 and end > start + 1.0:
        scene = index.scene_at(end - 0.05)
        if scene and _is_black(scene) and scene.start >= start - 0.01:
            end = scene.start
            guard += 1
        else:
            break
    return start, end


def _is_black(scene):
    brightness = getattr(scene, "brightness", -1.0)
    return 0.0 <= brightness < BLACK_BRIGHTNESS


def _snap_to_scene_edges(start, end, index, prefer_outside=True):
    scenes = index.scenes or []
    if not scenes:
        return start, end
    if start > 0.2:
        nearest = _nearest_scene_boundary(scenes, start, direction="before")
        if nearest is not None and abs(nearest - start) < 1.6:
            start = nearest
    if end < (index.video.duration if index.video else end):
        nearest = _nearest_scene_boundary(scenes, end, direction="after")
        if nearest is not None and abs(nearest - end) < 1.6:
            end = nearest
    return start, end


def _snap_to_silence_edges(start, end, index):
    silences = [ev for ev in index.audio_events if ev.kind == "silence"]
    if not silences:
        return start, end
    best_start = start
    for ev in silences:
        mid = (ev.start + ev.end) / 2.0
        if 0 < mid < start and (start - ev.end) < 2.2 and abs(mid - start) < 3.0:
            if ev.end < start:
                best_start = ev.end
    best_end = end
    for ev in silences:
        mid = (ev.start + ev.end) / 2.0
        if end < mid and ev.start > end and (ev.start - end) < 2.2:
            best_end = ev.start
    return best_start, best_end


def _nearest_scene_boundary(scenes, t, direction):
    best = None
    for scene in scenes:
        point = scene.start if direction == "before" else scene.end
        if point <= t + 0.05 and (best is None or abs(t - point) < abs(t - best)):
            best = point
        elif direction == "after" and point >= t - 0.05 and (best is None or abs(point - t) < abs(best - t)):
            best = point
    return best


def pick_hook_point(candidate, index):
    scene = index.scene_at(candidate.source_start + 0.5)
    if scene and scene.transition_count > 0:
        return scene.start + 0.05
    for ev in index.audio_events:
        if ev.kind == "loud" and ev.start >= candidate.source_start and ev.start <= candidate.source_start + 3.0:
            return ev.start
    return candidate.source_start
