import json
import re
from pathlib import Path

from ..log import get_logger
from ..models import TranscriptSegment
from ..utils import normalize_space, split_sentences

TIME_TS = re.compile(r"(\d+):(\d{2}):(\d{2})[.,](\d{3})")
TIME_SHORT = re.compile(r"(\d{1,2}):(\d{2})[.,](\d{3})")


def _ts_to_seconds(h, m, s, ms):
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000.0


def parse_json3(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    segments = []
    for event in data.get("events") or []:
        start = float(event.get("tStartMs") or 0) / 1000.0
        dur = float(event.get("dDurationMs") or 0) / 1000.0
        text = "".join(seg.get("utf8", "") for seg in event.get("segs") or [])
        text = normalize_space(text)
        if not text:
            continue
        segments.append(TranscriptSegment(start=round(start, 3), end=round(start + dur, 3), text=text, source="youtube"))
    return segments


def parse_vtt(path):
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    segments = []
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if "-->" in line:
            parts = re.split(r"\s*-->\s*", line)
            start = _parse_clock(parts[0])
            end = _parse_clock(parts[1].split()[0]) if len(parts) > 1 else start
            i += 1
            block = []
            while i < len(lines) and lines[i].strip():
                block.append(lines[i].strip())
                i += 1
            body = normalize_space(" ".join(block))
            body = re.sub(r"<[^>]+>", "", body)
            if body:
                segments.append(TranscriptSegment(start=round(start, 3), end=round(end, 3), text=body, source="youtube"))
            continue
        i += 1
    return segments


def _parse_clock(value):
    value = value.strip().replace(",", ".")
    if not value:
        return 0.0
    if ":" not in value:
        return float(value)
    parts = value.split(":")
    seconds = 0.0
    for part in parts:
        seconds = seconds * 60 + float(part)
    return seconds


def split_long_segments(segments, max_chars=64, max_span=8.0):
    out = []
    for seg in segments:
        text = seg.text
        span = max(0.5, seg.end - seg.start)
        if len(text) <= max_chars and span <= max_span:
            out.append(seg)
            continue
        pieces = _split_sentences(text)
        total_chars = sum(len(p) for p in pieces) or 1
        cursor = seg.start
        for piece in pieces:
            share = len(piece) / total_chars
            dur = span * share
            piece_end = min(seg.end, cursor + dur)
            out.append(TranscriptSegment(start=round(cursor, 3), end=round(piece_end, 3),
                                         text=piece, source=seg.source, speaker=seg.speaker))
            cursor = piece_end
        if cursor < seg.end - 0.05 and out:
            out[-1].end = round(seg.end, 3)
    return out


def _split_sentences(text):
    return split_sentences(text)


def from_subtitle_file(path):
    if not path:
        return []
    p = Path(path)
    if not p.exists():
        return []
    suffix = p.suffix.lower()
    try:
        if suffix == ".json3":
            segments = parse_json3(p)
        elif suffix in {".vtt", ".webvtt"}:
            segments = parse_vtt(p)
        elif suffix == ".srt":
            segments = _parse_srt(p)
        elif suffix in {".ass", ".ssa"}:
            segments = _parse_ass(p)
        else:
            segments = parse_vtt(p)
    except Exception as exc:
        get_logger().warning("字幕解析失败 %s: %s", p.name, exc)
        return []
    return split_long_segments([s for s in segments if s.text.strip()])


def _parse_srt(path):
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    segments = []
    blocks = re.split(r"\n\s*\n", text)
    for block in blocks:
        lines = [ln.strip() for ln in block.splitlines() if ln.strip()]
        if not lines:
            continue
        idx = 0
        if lines[0].isdigit():
            idx = 1
        if idx >= len(lines) or "-->" not in lines[idx]:
            continue
        parts = re.split(r"\s*-->\s*", lines[idx])
        start = _parse_clock(parts[0])
        end = _parse_clock(parts[1].split()[0])
        body = normalize_space(" ".join(lines[idx + 1:]))
        if body:
            segments.append(TranscriptSegment(start=round(start, 3), end=round(end, 3), text=body, source="youtube"))
    return segments


def _parse_ass(path):
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    segments = []
    for line in text.splitlines():
        if not line.startswith("Dialogue:"):
            continue
        fields = line[len("Dialogue:"):].split(",", 9)
        if len(fields) < 10:
            continue
        start = _parse_clock(fields[1].strip())
        end = _parse_clock(fields[2].strip())
        body = re.sub(r"\{[^}]*\}", "", fields[9])
        body = normalize_space(body.replace("\\N", " ").replace("\\n", " "))
        if body and end > start:
            segments.append(TranscriptSegment(start=round(start, 3), end=round(end, 3), text=body, source="youtube"))
    return segments


def transcribe_whisper(video_path, model_name="base", language="zh", max_duration=0, start=0.0):
    import tempfile
    import uuid

    logger = get_logger()
    lang = None if str(language).lower() in ("auto", "", "none") else language
    logger.info("本地语音识别 (%s, 语言=%s)...", model_name, lang or "auto")
    tmp_wav = Path(tempfile.gettempdir()) / ("ytsf_whisper_%s_%s.wav" % (Path(video_path).stem, uuid.uuid4().hex[:8]))
    cmd = ["ffmpeg", "-y"]
    if start:
        cmd += ["-ss", "%.3f" % start]
    cmd += ["-i", str(video_path), "-ac", "1", "-ar", "16000", "-vn"]
    if max_duration:
        cmd += ["-t", str(int(max_duration))]
    cmd += [str(tmp_wav)]
    from ..utils import run_cmd

    _, err, code = run_cmd(cmd, check=False, timeout=1800)
    if code != 0:
        logger.warning("提取音频失败: %s", (err or "")[-400:])
        return []
    try:
        result = _openai_whisper_call(tmp_wav, model_name, lang, offset=start)
        if result is None:
            result = _pywhispercpp_call(tmp_wav, model_name, lang, offset=start)
        if result:
            return split_long_segments(result)
        return []
    except Exception as exc:
        logger.warning("whisper 转写失败: %s", exc)
        return []
    finally:
        try:
            tmp_wav.unlink()
        except OSError:
            pass


def _local_whisper_models_dir():
    candidates = []
    cache = Path.home() / ".cache" / "whisper"
    if cache.is_dir():
        candidates.append(cache)
    here = Path(__file__).resolve().parent.parent.parent / "whisper-models"
    if here.is_dir():
        candidates.append(here)
    return candidates


def _openai_whisper_call(wav_path, model_name, language, offset=0.0):
    try:
        import whisper
    except Exception as exc:
        get_logger().debug("openai-whisper 不可用: %s", exc)
        return None
    checkpoint = None
    for d in _local_whisper_models_dir():
        candidate = d / ("%s.pt" % model_name)
        if candidate.exists():
            checkpoint = str(candidate)
            break
    if not checkpoint and model_name:
        user_home = Path.home() / ".cache" / "whisper" / ("%s.pt" % model_name)
        if user_home.exists():
            checkpoint = str(user_home)
    model = whisper.load_model(checkpoint)
    result = model.transcribe(str(wav_path), language=language, fp16=False, verbose=False,
                              no_speech_threshold=0.6)
    segments = []
    for seg in result.get("segments", []):
        text = normalize_space(seg.get("text", ""))
        if not text:
            continue
        segments.append(TranscriptSegment(start=round(float(offset) + float(seg["start"]), 3),
                                          end=round(float(offset) + float(seg["end"]), 3),
                                          text=text, source="whisper"))
    return segments or None


def _pywhispercpp_call(wav_path, model_name, language, offset=0.0):
    try:
        from pywhispercpp.model import Model
    except Exception as exc:
        get_logger().debug("pywhispercpp 不可用: %s", exc)
        return None
    logger = get_logger()
    try:
        model = Model(model_name, print_progress=False, print_realtime=False)
        kwargs = {} if language is None else {"language": language}
        result = model.transcribe(str(wav_path), **kwargs)
    except Exception as exc:
        logger.warning("pywhispercpp 转写失败: %s", exc)
        return None
    segments = []
    for seg in result:
        text = normalize_space(getattr(seg, "text", ""))
        if not text:
            continue
        segments.append(TranscriptSegment(start=round(float(offset) + float(seg.t0) / 100.0, 3),
                                          end=round(float(offset) + float(seg.t1) / 100.0, 3),
                                          text=text, source="whisper"))
    return segments or None


def transcribe_api(video_path, base_url="", api_key="", model="whisper-1", language="zh",
                   max_duration=0.0, chunk_seconds=600):
    import tempfile
    import time

    import requests

    logger = get_logger()
    if not api_key:
        logger.warning("未配置语音转写 API Key，跳过 API 转写")
        return []
    base = (base_url or "").rstrip("/")
    url = base + "/audio/transcriptions"
    logger.info("API 语音转写 (%s)...", model)
    tmp_dir = Path(tempfile.gettempdir())
    total = float(max_duration) if max_duration else 0.0
    if total <= 0:
        try:
            from ..utils import media_duration
            total = media_duration(video_path)
        except Exception as exc:
            logger.warning("探测视频时长失败: %s", exc)
    segments = []
    start = 0.0
    while total <= 0 or start < total:
        seg = chunk_seconds if total <= 0 else min(chunk_seconds, total - start)
        if seg <= 0:
            break
        out_mp3 = tmp_dir / ("ytsf_asr_%d_%ds.mp3" % (int(time.time() * 1000), int(start)))
        cmd = ["ffmpeg", "-y", "-ss", str(start), "-t", str(seg), "-i", str(video_path),
               "-ac", "1", "-ar", "16000", "-vn", "-c:a", "libmp3lame", "-b:a", "96k", str(out_mp3)]
        from ..utils import run_cmd

        _, err, code = run_cmd(cmd, check=False, timeout=1800)
        if code != 0 or not out_mp3.exists() or out_mp3.stat().st_size == 0:
            logger.warning("音频切段失败(%.0fs): %s", start, (err or "")[-300:])
            break
        try:
            with open(out_mp3, "rb") as fh:
                resp = requests.post(
                    url,
                    headers={"Authorization": "Bearer %s" % api_key},
                    data={"model": model, "response_format": "verbose_json",
                          **({"language": language} if language else {})},
                    files={"file": ("chunk_%d.mp3" % int(start), fh, "audio/mpeg")},
                    timeout=600,
                )
            if resp.status_code != 200:
                logger.warning("API 转写 HTTP %s: %s", resp.status_code, resp.text[:300])
            else:
                data = resp.json()
                for seg_info in (data.get("segments") or []):
                    text = normalize_space(seg_info.get("text") or "")
                    if not text:
                        continue
                    segments.append(TranscriptSegment(
                        start=round(start + float(seg_info.get("start", 0)), 3),
                        end=round(start + float(seg_info.get("end", 0)), 3),
                        text=text, source="api"))
        except Exception as exc:
            logger.warning("API 转写请求失败: %s", exc)
        finally:
            try:
                out_mp3.unlink()
            except OSError:
                pass
        start += seg
    return split_long_segments(segments)


def build_transcript(subtitle_path, video_path, mode="auto", language="zh", whisper_model="base",
                     duration=0.0, asr=None, out_quality=None, full=True, asr_language=""):
    logger = get_logger()
    segments = from_subtitle_file(subtitle_path)
    if segments:
        logger.info("字幕来源: %s (%d 段)", Path(subtitle_path).name if subtitle_path else "?", len(segments))
        return _finalize(segments, "youtube", duration, out_quality)
    if mode == "off":
        logger.info("未启用转写，字幕为空")
        return [], "none"
    if not full:
        logger.info("长视频且无字幕：延迟转写，仅对候选片段做 Whisper")
        return [], "deferred"
    if video_path and mode == "api":
        asr = dict(asr or {})
        segments = transcribe_api(video_path, max_duration=min(duration, 5400) if duration else 0,
                                  **{k: v for k, v in asr.items()
                                     if k in ("base_url", "api_key", "model", "language")})
        if segments:
            return _finalize(segments, "api", duration, out_quality)
        logger.warning("API 转写无结果，字幕为空")
        return [], "none"
    if mode in ("auto", "force") and video_path:
        segments = transcribe_whisper(video_path, model_name=whisper_model,
                                      language=(asr_language or language),
                                      max_duration=min(duration, 5400) if duration else 0)
        if segments:
            return _finalize(segments, "whisper", duration, out_quality)
    return [], "none"


def transcribe_windows(video_path, windows, model_name="base", language="zh", pad=3.0):
    """只对给定时间窗做 Whisper 转写（长视频无字幕时用），合并去重。"""
    out = []
    for start, end in windows:
        s = max(0.0, float(start) - float(pad))
        span = float(end) - float(start) + 2.0 * float(pad)
        if span <= 0.5:
            continue
        out.extend(transcribe_whisper(video_path, model_name=model_name, language=language,
                                      max_duration=span, start=s))
    return split_long_segments(merge_adjacent_duplicates([s for s in out if s.text.strip()]))


def _finalize(segments, source, duration, out_quality=None):
    valid = [s for s in segments if s.end > s.start and s.text.strip()]
    merged = merge_adjacent_duplicates(valid)
    quality = quality_check_subtitles(merged, duration)
    if out_quality is not None:
        out_quality.update(quality)
    return merged, source


def merge_adjacent_duplicates(segments):
    if not segments:
        return []
    out = []
    prev = segments[0]
    for seg in segments[1:]:
        same = normalize_space(seg.text) == normalize_space(prev.text)
        tail_dup = seg.start <= prev.end + 0.5 and text_ends_with(normalize_space(seg.text),
                                                                  normalize_space(prev.text))
        if same or tail_dup:
            prev.end = max(prev.end, seg.end)
            continue
        out.append(prev)
        prev = seg
    out.append(prev)
    return out


def text_ends_with(cur, prev):
    if not cur or not prev:
        return False
    if len(cur) < 12 or len(prev) < len(cur):
        return False
    head = prev[len(cur):]
    return head and normalize_space(head) and any(
        len(w) >= 2 for w in re.findall(r"[\w\u4e00-\u9fff]+", head))


def quality_check_subtitles(segments, duration=0.0):
    total_span = 0.0
    prev_end = 0.0
    gaps = 0
    for s in sorted(segments, key=lambda x: x.start):
        span = max(0.0, min(s.end, duration or s.end) - max(s.start, prev_end))
        total_span += span
        if duration and s.start > prev_end + 8.0:
            gaps += 1
        prev_end = max(prev_end, s.end)
    n = len(segments)
    coverage = (total_span / duration) if duration and duration > 0 else 1.0
    density = n / (duration / 60.0) if duration and duration > 0 else 0.0
    bad_times = sum(1 for s in segments if s.end <= s.start or s.end - s.start > 60.0)
    issues = []
    if duration and duration > 0:
        if coverage < 0.25:
            issues.append("转写覆盖率过低 %.0f%%" % (coverage * 100))
        if n < 5:
            issues.append("字幕段过少 %d" % n)
    if bad_times:
        issues.append("时间轴异常 %d 段" % bad_times)
    return {
        "coverage": round(coverage, 3),
        "density_per_min": round(density, 2),
        "segments": n,
        "gap_count": gaps,
        "bad_timelines": bad_times,
        "issues": issues,
        "ok": not issues,
    }
