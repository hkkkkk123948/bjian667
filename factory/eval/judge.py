"""第三方独立评审（软件内功能）。

把成片与参考片的画面抽帧 + 旁白音频 + 旁白 SRT 一起发给 T8 中转的 gemini-3.1-flash，
由它作为独立评委逐维度打分；各维度再按权重加权得到总分，最后判断我方是否优于参考。

评审前先过“机器审计”，任何硬性问题（画面重复、画幅与源片不符）直接判定不通过：
- 抽帧按“每句旁白一张图”对齐，重复画面一眼可见；
- 对抽帧做感知哈希(dHash)查重，重复达到阈值即失败；
- 成片画幅必须与源片一致。
"""

from pathlib import Path

from ..log import get_logger
from ..utils import ensure_dir, ffmpeg_bin, media_duration, run_cmd

DEFAULT_WEIGHTS = {
    "match": 0.30,
    "visual": 0.20,
    "story": 0.20,
    "copy": 0.15,
    "pacing": 0.10,
    "originality": 0.05,
}

RUBRIC = """你是严格的动画惊悚短片评委。两条匿名作品A、B来自不同制作方，身份未知。
A对应音频1和前{n_ours}张图；B对应音频2和后{n_ref}张图。图按各自时间排列。
请实际听两段音频，再观察各自画面。不要因为某条有额外文字说明而加分，也不要因为缺少字幕而扣分（交付要求无字幕）。参考素材本来可能有字幕。
同样尺度分别给A/B打0至100分：
match音画内容与时序是否匹配；visual构图清晰度；story具体事件、因果、结尾和人物一致性；copy直白口语、信息密度；pacing剪辑与配音节奏、停顿、情绪和声场；originality改编是否有自己的叙事。
80分是合格成熟短片；90以上须给出明显优秀的证据。动作或物件说错、衣服流水账、反复说同一件事、结尾空泛都要扣分；静止动画本身不能当成重复剪辑。音乐低于旁白是正常的，无音乐不自动判差。
必须给出各自具体优缺点和秒级问题点。如果无法听到音频，audio_access必须为false，配音/声音评价写无法判断，严禁根据文案猜测听感。
输出JSON {{"A":{{"match":0,"visual":0,"story":0,"copy":0,"pacing":0,"originality":0}},"B":{{"match":0,"visual":0,"story":0,"copy":0,"pacing":0,"originality":0}},"audio_access":true,"A_notes":["秒级优缺点"],"B_notes":["秒级优缺点"],"suggestions":["可执行改进"]}}。
"""


def _weighted(scores, weights):
    total = 0.0
    for key, w in weights.items():
        total += float(scores.get(key, 0.0) or 0.0) * float(w)
    return round(total, 1)


def timeline_to_srt(timeline, path=None):
    timings = (timeline or {}).get("sentence_timings") or []
    lines = []
    for i, item in enumerate(timings, 1):
        text = str(item.get("text", "")).strip()
        if not text:
            continue
        lines += [str(i), "%s --> %s" % (_srt_time(item.get("start", 0.0)), _srt_time(item.get("end", 0.0))), text, ""]
    content = "\n".join(lines)
    if path:
        ensure_dir(Path(path).parent)
        Path(path).write_text(content, encoding="utf-8")
    return content


def _srt_time(seconds):
    seconds = max(0.0, float(seconds or 0.0))
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return ("%02d:%02d:%06.3f" % (h, m, s)).replace(".", ",")


def _frame_at(video_path, out_path, t, width=720):
    _, _, code = run_cmd([ffmpeg_bin(), "-hide_banner", "-loglevel", "error", "-y",
                          "-threads","1","-ss", "%.2f" % max(0.0, float(t)), "-i", str(video_path), "-frames:v", "1",
                          "-filter_threads","1","-threads","1","-vf", "scale=%d:-2" % width, str(out_path)], check=False, timeout=120)
    if code == 0 and Path(out_path).exists():
        return str(out_path)
    return None


def extract_frames(video_path, out_dir, count=6, width=720, times=None):
    out_dir = ensure_dir(out_dir)
    try:
        duration = media_duration(video_path)
    except Exception:
        duration = 0.0
    frames = []
    for old in out_dir.glob("frame_*.jpg"):
        try:
            old.unlink()
        except OSError:
            pass
    if times:
        times = [float(t) for t in times if float(t) >= 0.0]
        if duration > 0.0:
            times = [t for t in times if t < duration]
    for i in range(max(1, count)):
        srt = out_dir / ("frame_%02d.jpg" % i)
        if times and i < len(times):
            t = times[i]
        else:
            t = duration * (i + 0.5) / max(1, count)
        one = _frame_at(video_path, srt, t, width)
        if one:
            frames.append(one)
    return frames


def extract_audio(video_path, out_wav):
    out_wav = Path(out_wav)
    ensure_dir(out_wav.parent)
    _, _, code = run_cmd([ffmpeg_bin(), "-hide_banner", "-loglevel", "error", "-y", "-i", str(video_path),
                          "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(out_wav)],
                         check=False, timeout=300)
    return str(out_wav) if code == 0 and out_wav.exists() else None


def _dhash(path, size=8):
    try:
        from PIL import Image
        im = Image.open(path).convert("L").resize((size + 1, size))
        px = list(im.getdata())
    except Exception:
        return None
    bits = 0
    for y in range(size):
        row = y * (size + 1)
        for x in range(size):
            bits = (bits << 1) | (1 if px[row + x] > px[row + x + 1] else 0)
    return bits


def audit_frames(frame_paths):
    """对抽帧做感知哈希查重：两个不同时刻的帧哈希越接近，说明同一段画面被重复剪辑。"""
    issues = []
    hashes = []
    for p in frame_paths:
        h = _dhash(p)
        hashes.append(h if h is not None else None)
    for i in range(len(hashes)):
        if hashes[i] is None:
            continue
        for j in range(i + 1, len(hashes)):
            if hashes[j] is None:
                continue
            if bin(hashes[i] ^ hashes[j]).count("1") <= 3:
                issues.append("相似帧提示：抽帧 %d 与 %d 高度相似（静止动画可能正常，需结合源片区间）" % (i + 1, j + 1))
    return issues


def probe_size(video_path):
    try:
        from ..utils import media_size
        return media_size(video_path)
    except Exception:
        return 0, 0


def check_aspect(our_video, expect_size):
    if not expect_size or expect_size[0] <= 0 or expect_size[1] <= 0:
        return []
    w, h = probe_size(our_video)
    if w <= 0 or h <= 0:
        return []
    e_w, e_h = int(expect_size[0]), int(expect_size[1])
    our_ratio = w / h
    exp_ratio = e_w / e_h
    if abs(our_ratio - exp_ratio) / exp_ratio > 0.05:
        return ["画幅错误：成片 %dx%d，源片 %dx%d（比例必须跟随源片）" % (w, h, e_w, e_h)]
    return []


def judge_video(provider, our_video, our_srt, ref_video, ref_text, workdir,
                our_audio=None, weights=None, frames_per_video=12, timeline=None,
                expect_size=None):
    import hashlib
    from ..utils import write_json
    from ..providers.base import extract_json

    logger = get_logger()
    weights = weights or DEFAULT_WEIGHTS
    workdir = ensure_dir(Path(workdir))
    timings = (timeline or {}).get("sentence_timings") or []
    times = [float(t["start"]) + (float(t["end"]) - float(t["start"])) * f
             for t in timings for f in (.25, .75)]
    ours = extract_frames(our_video, workdir / "ours_frames", len(times) if times else frames_per_video, times=times)
    ref = extract_frames(ref_video, workdir / "ref_frames", frames_per_video)
    if not ours or not ref:
        raise RuntimeError("对比评审必须成功读取双方成片")
    # Review the exported sound, not an unmixed or stale TTS recording.
    ours_wav = extract_audio(our_video, workdir / "our_audio.wav")
    ref_wav = extract_audio(ref_video, workdir / "ref_audio.wav")
    if not ours_wav or not ref_wav:
        raise RuntimeError("对比评审必须成功读取双方音轨")
    digest = hashlib.sha256(Path(our_video).read_bytes()).hexdigest()
    swapped = int(digest[:2], 16) % 2 == 1
    frames_a, frames_b = (ref, ours) if swapped else (ours, ref)
    audio_a, audio_b = (ref_wav, ours_wav) if swapped else (ours_wav, ref_wav)
    # Only actual exported audio/images are supplied, equally for both entries.
    prompt = RUBRIC.format(n_ours=len(frames_a), n_ref=len(frames_b))
    text = provider.complete(prompt, images=frames_a + frames_b, audios=[audio_a, audio_b], json_mode=True)
    data = extract_json(text) or {}
    for label in ("A", "B"):
        if not isinstance(data.get(label), dict):
            raise ValueError("评委未返回双方有效分数")
        for dim in weights:
            score = data[label].get(dim)
            if isinstance(score, bool) or not isinstance(score, (int, float)) or not 0 <= score <= 100:
                raise ValueError("评委维度分数无效: %s.%s" % (label, dim))
    ours_label = "B" if swapped else "A"
    ref_label = "A" if swapped else "B"
    data["ours"], data["reference"] = data[ours_label], data[ref_label]
    issues = check_aspect(our_video, expect_size)
    provenance = audit_timeline(timeline or {})
    issues.extend(provenance["issues"])
    result = {"raw": text, "data": data, "weights": weights,
              "ours_label": ours_label, "reference_label": ref_label,
              "source_sha256": digest, "audio_supplied": True,
              "audio_access_claimed": data.get("audio_access") is True,
              "review_issues": issues, "review_failed": bool(issues),
              "similar_frame_warnings": audit_frames(ours), "provenance": provenance,
              "ours_overall": _weighted(data["ours"], weights),
              "reference_overall": _weighted(data["reference"], weights),
              "ours_notes": data.get(ours_label + "_notes", []),
              "reference_notes": data.get(ref_label + "_notes", [])}
    result["ours_better"] = (not issues and result["audio_access_claimed"]
                             and result["ours_overall"] > result["reference_overall"])
    write_json(workdir / "comparison.json", result)
    logger.info("匿名对比 %.1f / %.1f, 音频可访问声明=%s", result["ours_overall"], result["reference_overall"], result["audio_access_claimed"])
    return result


def audit_timeline(timeline):
    issues, intervals = [], []
    timings = timeline.get("sentence_timings") or []
    segments = timeline.get("segments") or []
    cursor = 0.0
    if segments and all("sentence_index" in s for s in segments):
        for tm in timings:
            pieces = [s for s in segments if s["sentence_index"] == tm["index"]]
            expected = float(tm["end"]) - float(tm["start"])
            if not pieces or abs(sum(float(s["dur"]) for s in pieces) - expected) > .06:
                issues.append("第%d句镜头未完整覆盖旁白" % tm["index"])
    elif len(timings) != len(segments):
        issues.append("旁白与剪辑段数量不一致")
    for i, seg in enumerate(segments):
        start = float(seg["src_start"])
        end = start + float(seg.get("src_duration", seg["dur"]))
        if abs(float(seg.get("out_start", cursor)) - cursor) > .06:
            issues.append("第%d段输出时间轴出现空隙或重叠" % (i + 1))
        if start < float(seg.get("evidence_start", start)) - .02 or end > float(seg.get("evidence_end", end)) + .02:
            issues.append("第%d段播放超出画面证据" % (i + 1))
        if any(min(b, end) - max(a, start) > .08 for a, b in intervals):
            issues.append("第%d段重复使用源片画面" % (i + 1))
        intervals.append((start, end))
        cursor += float(seg["dur"])
    return {"issues": issues, "output_duration": round(cursor, 6), "segment_count": len(segments)}
