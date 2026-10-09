import json
import re

from ..log import get_logger
from ..models import SceneSegment
from ..utils import ensure_dir, ffmpeg_bin, run_cmd

SHOWINFO_RE = re.compile(r"pts_time:\s*(-?\d+(?:\.\d+)?)")


def extract_frames(video_path, out_dir, fps, start=None, end=None, width=480, max_frames=0, prefix="frame"):
    out_dir = ensure_dir(out_dir)
    existing = sorted(out_dir.glob(prefix + "_*.jpg"))
    if existing:
        return _reload_existing(out_dir, prefix, fps, start)
    duration = (end - start) if (start is not None and end is not None) else None
    if fps <= 0:
        fps = 1.0
    if max_frames and duration and duration * fps > max_frames:
        fps = max(0.05, max_frames / max(1.0, duration))
    args = [ffmpeg_bin(), "-hide_banner", "-loglevel", "info"]
    if start:
        args += ["-ss", "%.3f" % start]
    args += ["-i", str(video_path)]
    if duration:
        args += ["-t", "%.3f" % duration]
    pattern = str(out_dir / (prefix + "_%06d.jpg"))
    args += ["-vf", "fps=%.5f,scale=%d:-2,showinfo" % (fps, width), "-q:v", "4", "-y", pattern]
    timeout = max(180, int((duration or 600) * 4 + 120))
    _, err, code = run_cmd(args, check=False, timeout=timeout)
    if code != 0:
        raise RuntimeError("抽帧失败: %s" % (err or "")[-800:])
    times = [float(m) for m in SHOWINFO_RE.findall(err or "")]
    files = sorted(out_dir.glob(prefix + "_*.jpg"))
    base = float(start or 0.0)
    results = []
    for i, path in enumerate(files):
        t_abs = base + times[i] if i < len(times) else base + i / fps
        results.append((round(t_abs, 3), path))
    _write_times(out_dir, prefix, [t for t, _ in results])
    return results


def _write_times(out_dir, prefix, times):
    try:
        (out_dir / (prefix + "_times.json")).write_text(json.dumps(times), encoding="utf-8")
    except OSError:
        pass


def _reload_existing(out_dir, prefix, fps, start):
    files = sorted(out_dir.glob(prefix + "_*.jpg"))
    sidecar = out_dir / (prefix + "_times.json")
    if sidecar.exists():
        try:
            times = json.loads(sidecar.read_text(encoding="utf-8"))
            if len(times) == len(files):
                return [(round(float(t), 3), p) for t, p in zip(times, files)]
        except (OSError, ValueError):
            pass
    base = float(start or 0.0)
    return [(round(base + i / max(0.01, fps), 3), p) for i, p in enumerate(files)]


def sample_global(video_path, out_dir, fps=0.5, max_frames=1800, width=480, duration=0.0):
    logger = get_logger()
    frames = extract_frames(video_path, out_dir, fps=fps, width=width, max_frames=max_frames, prefix="l1")
    logger.info("L1 全局抽帧: %d 张", len(frames))
    return frames


def sample_window(video_path, out_dir, start, end, fps=4.0, width=640, tag="l2", max_frames=600):
    ensure_dir(out_dir)
    frames = extract_frames(video_path, out_dir, fps=fps, start=start, end=end, width=width,
                            max_frames=max_frames, prefix=tag)
    return frames


def sample_dense(video_path, out_dir, start, end, fps=8.0, width=720, tag="l3", max_frames=400):
    return sample_window(video_path, out_dir, start, end, fps=fps, width=width, tag=tag, max_frames=max_frames)


def assign_scene_keyframes(scenes, global_frames, per_scene=3, video_path=None, out_dir=None, width=640):
    if isinstance(scenes, list) and scenes and not isinstance(scenes[0], SceneSegment):
        return scenes
    times = [(t, p) for t, p in global_frames]
    for scene in scenes:
        picks = _pick_times(scene.start, scene.end, per_scene)
        chosen = []
        for want in picks:
            best = None
            for t, p in times:
                if scene.start <= t <= scene.end:
                    dist = abs(t - want)
                    if best is None or dist < best[0]:
                        best = (dist, t, p)
            if best and best[2] not in [c[1] for c in chosen]:
                chosen.append((best[1], best[2]))
        if not chosen and video_path and out_dir:
            try:
                frames = extract_frames(video_path, out_dir / ("scene_%04d" % scene.scene_id),
                                        fps=max(0.2, 1.0 / max(1.0, scene.end - scene.start)),
                                        start=scene.start, end=scene.end, width=width, prefix="s")
                chosen = frames[:per_scene]
            except Exception:
                chosen = []
        scene.keyframes = [str(p) for _, p in chosen]
    return scenes


def _pick_times(start, end, count):
    span = max(0.0, end - start)
    if count <= 1:
        return [start + span * 0.5]
    fracs = [0.18, 0.5, 0.82] if count == 3 else [i / (count + 1) for i in range(1, count + 1)]
    return [start + span * f for f in fracs[:count]]
