import math
import re
from pathlib import Path

from ..log import get_logger
from ..utils import ffmpeg_bin, ffprobe_bin, run_cmd


def final_qc(clip, script_text, params, min_duration=8.0, max_duration=60.0, source_size=None,
             tts_duration=None, sentences=None):
    """Validate that export produced a readable video, without adding a quality review."""
    logger = get_logger()
    path = Path(clip.path)
    report = {"status": "pass", "issues": [], "basis": "export_file_validity"}
    if not path.is_file():
        return {"status": "fail", "issues": ["输出文件不存在"]}
    try:
        meta = _probe(path)
        report.update(meta, size_bytes=path.stat().st_size)
        if not meta["has_video"] or meta.get("probe_success") is False:
            report["status"] = "fail"
            report["issues"].append("导出文件没有可读的视频流")
    except Exception as exc:
        report["status"] = "fail"
        report["issues"].append("导出文件无法读取: %s" % str(exc)[:120])
    logger.info("导出文件验证 %s: %s issues=%s", path.name, report["status"], report["issues"])
    return report


def semantic_final_check(clip, script_text, window_text, router, params=None, visuals=""):
    logger = get_logger()
    if not router or router.is_mock("qc") or not script_text:
        return None
    try:
        provider = router.get("qc")
        provider.max_retry = 1
        provider.conf["max_tokens"] = 1400
        script = script_text[:1200]
        from ..utils import read_json
        from ..eval.judge import extract_frames
        timeline = read_json(Path(clip.path).parent / "timeline.json") or {}
        cache_key = _semantic_cache_key(clip.path, script_text, timeline, provider, params or {})
        cache_path = Path(clip.path).parent / "semantic_review.json"
        cached = read_json(cache_path) or {}
        if (cache_key and cached.get("signature") == cache_key
                and cached.get("review", {}).get("status") in ('pass', 'fail')
                and cached.get("review", {}).get("basis") == "approved_script_visual_alignment"):
            return dict(cached["review"], cache_hit=True)
        def keep_review(review):
            if cache_key and review.get('status') in ('pass','fail'):
                from ..utils import write_json
                review['model'] = provider.model
                write_json(cache_path,{'signature':cache_key,'review':review})
            return review
        timings = timeline.get("sentence_timings") or []
        times = [float(t["start"]) + (float(t["end"])-float(t["start"]))*f for t in timings for f in (.25,.75)]
        frames = extract_frames(clip.path, Path(clip.path).parent / "qc_frames", len(times) or 8, width=384, times=times)
        if len(frames) != (len(times) or 8):
            return {"status":"review","reason":"质检抽帧不完整（%d/%d），已跳过画面匹配评分；请检查视频轨和抽帧。" %
                    (len(frames),len(times) or 8),"score":None,"mismatches":[],"basis":"incomplete_visual_evidence"}
        import json
        mapped_script = json.dumps([{"sentence_index": i + 1, "text": t.get("narrative_text") or t["text"],
                                    "start": t["start"], "end": t["end"],
                                    "image_numbers": [2*i+1, 2*i+2]}
                                   for i, t in enumerate(timings)], ensure_ascii=False) if timings else script
        prompt = (
            "你是成片画面匹配质检员。文案已经确认，是镜头匹配目标；只检查实际成片镜头与该句解说是否匹配。"
            "不要重新审稿，不评价文案好坏、故事复杂度、因果是否完整、转折是否合理或剧情过渡是否缺失。"
            "不要求补写任何文案，也不改写用户确认的叙述。原片字幕和作者画面描述不是实际成片的证据。"
            "输入已排除独立评论插话，不按插话字面寻找动作或道具。只依据所附实际成片图片。"
            "按JSON里的sentence_index、start/end和image_numbers核对，不自行重新拆句。"
            "同一句和前后相邻镜头可共同概括一个行动，不要求人物始终同框或完整拍出中间走路等过渡动作。"
            "心理、情绪、动机、合理概括及邻接镜头可推断的人物关系不应被当作画面错误。"
            "例如求助镜头后接警方接待可以表现协助报警，不要求拨号特写；只有明确说拨号码等操作才核对具体操作画面。"
            "抽帧没捕获动作过程不等于动作不存在；看不清、只缺少动作瞬间或无法确定时写uncertain_sentences，不写确定错误。"
            "mismatches仅列有可见正面证据的明确配对错误：此处匹配了不同人物、相反动作、不同道具/场景，"
            "或正确事件镜头出现在另一句的时间段。每条都写图中实际可见内容(evidence)，并说明与目标句的具体差异(reason)。"
            "不得仅凭标题、文案或想象认定画面出现手机、刀等物品；也不得仅因两张抽帧没看见某物就断言它不存在。"
            "分数范围0至100，只评价画面与解说的贴合度，不评价故事结构。没有明确错误时mismatches为空数组，"
            "没有不确定项时uncertain_sentences为空数组。只返回JSON："
            '{"score":95,"mismatches":[{"sentence_index":1,"start":0,"end":3,'
            '"reason":"目标句与此处镜头如何错位","evidence":"对应图实际可见的具体内容"}],'
            '"uncertain_sentences":[{"sentence_index":1,"reason":"需要核查的画面证据"}],"reason":"匹配情况简述"}。'
            "\n已确认解说与成片图片索引：%s" % mapped_script
        )
        data = provider.complete_json(prompt, images=frames, json_mode=True, default=None)
        check = _alignment_review(data, timings, float((params or {}).get("qc_semantic_min", 80)))
        if check["status"] == "fail":
            logger.warning("画面匹配质检未通过: %s", check["reason"])
        return keep_review(check)
    except Exception as exc:
        logger.info("画面匹配质检不可用: %s", exc)
        return {"status": "review", "score": None, "mismatches": [],
                "basis": "approved_script_visual_alignment", "reason": "画面匹配质检不可用: %s" % str(exc)[:180]}


def _alignment_review(data, timings, minimum):
    """Only concrete shot errors can request rematching; uncertain evidence stays reviewable."""
    check = {"status": "review", "score": None, "mismatches": [],
             "basis": "approved_script_visual_alignment", "reason": "画面匹配质检未返回有效结果"}
    if not isinstance(data, dict) or not isinstance(data.get("mismatches"), list):
        return check
    score = data.get("score")
    if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score) or not 0 <= score <= 100:
        return check
    uncertain = data.get("uncertain_sentences", [])
    if not isinstance(uncertain, list):
        return check
    errors = []
    for mismatch in data["mismatches"]:
        if not isinstance(mismatch, dict):
            return check
        index = mismatch.get("sentence_index")
        if isinstance(index, bool) or not isinstance(index, int) or index < 1 or timings and index > len(timings):
            return check
        reason = mismatch.get("reason")
        evidence = mismatch.get("evidence")
        if not isinstance(reason, str) or not reason.strip() or not isinstance(evidence, str) or not evidence.strip():
            check["reason"] = "画面匹配问题缺少明确原因或可见证据，需核查"
            return check
        timing = timings[index - 1] if timings else mismatch
        start, end = timing.get("start"), timing.get("end")
        if (isinstance(start, bool) or isinstance(end, bool) or not isinstance(start, (int, float))
                or not isinstance(end, (int, float)) or not math.isfinite(start) or not math.isfinite(end)
                or start < 0 or end <= start):
            return check
        errors.append({"sentence_index": index, "start": round(start, 3), "end": round(end, 3),
                       "reason": reason.strip(), "evidence": evidence.strip()})
    check.update(score=round(score, 1), mismatches=errors,
                 reason=str(data.get("reason") or "").strip(), uncertain_sentences=uncertain)
    if errors:
        check["status"] = "fail"
        check["reason"] = "；".join("第%d句（%.2f–%.2f秒）：%s" %
                                  (error["sentence_index"], error["start"], error["end"], error["reason"]) for error in errors)
    elif uncertain:
        check["reason"] = "部分镜头证据尚不明确，需核查画面；不据此改写已确认文案。" + check["reason"]
    elif score < minimum:
        check["reason"] = "画面匹配评分偏低，但未定位到明确镜头错误，需核查。" + check["reason"]
    else:
        check["status"] = "pass"
    return check


def _semantic_cache_key(path, text, timeline, provider, params):
    import hashlib
    import json
    from ..versioning import PROMPT_VERSIONS
    path = Path(path)
    if not path.is_file():
        return None
    data = {"media_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "approved_script_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "timeline": timeline, "model": provider.model, "base_url": provider.conf.get("base_url", ""),
            "mode": params.get("content_mode", "creative"), "threshold": params.get("qc_semantic_min", 80),
            "policy": PROMPT_VERSIONS["qc_semantic"]}
    return hashlib.sha256(json.dumps(data, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def _probe(path):
    out, _, code = run_cmd([
        ffprobe_bin(), "-v", "error", "-show_entries", "stream=codec_type,width,height",
        "-show_entries", "format=duration", "-of", "json", str(path)
    ], check=False, timeout=60)
    data = {}
    try:
        import json
        data = json.loads(out or "{}")
    except Exception:
        data = {}
    has_video = False
    has_audio = False
    width = height = 0
    for stream in data.get("streams") or []:
        if stream.get("codec_type") == "video":
            has_video = True
            width = stream.get("width", 0)
            height = stream.get("height", 0)
        elif stream.get("codec_type") == "audio":
            has_audio = True
    duration = 0.0
    try:
        duration = float((data.get("format") or {}).get("duration") or 0.0)
    except (TypeError, ValueError):
        duration = 0.0
    return {"has_video": has_video, "has_audio": has_audio, "width": width, "height": height,
            "duration": duration, "probe_success": code == 0}


def _loudness(path):
    _, err, _ = run_cmd([ffmpeg_bin(), "-hide_banner", "-i", str(path), "-af", "volumedetect", "-f", "null", "-"],
                        check=False, timeout=300)
    match = re.search(r"mean_volume:\s*(-?\d+(?:\.\d+)?)", err or "")
    if match:
        return float(match.group(1))
    return None


def _sample_brightness(path, duration, count=3):
    import tempfile

    results = []
    for i in range(count):
        t = duration * (i + 1) / (count + 1)
        tmp = Path(tempfile.gettempdir()) / ("ytsf_qc_%s_%d.jpg" % (path.stem, i))
        _, err, code = run_cmd([
            ffmpeg_bin(), "-hide_banner", "-loglevel", "error", "-ss", "%.2f" % t,
            "-i", str(path), "-frames:v", "1", "-y", str(tmp)
        ], check=False, timeout=60)
        if code != 0 or not tmp.exists():
            continue
        try:
            from PIL import Image
            with Image.open(tmp) as img:
                gray = img.convert("L").resize((32, 32))
                pixels = list(gray.getdata())
                results.append(sum(pixels) / len(pixels))
        finally:
            try:
                tmp.unlink()
            except OSError:
                pass
    return results
