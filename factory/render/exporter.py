import sys
from pathlib import Path

from ..log import get_logger
from ..utils import ensure_dir, ffmpeg_bin, run_cmd
from .audio import audio_graph
from .clean import build_video_graph
from .reframing import video_filter


def export_clip(video_path, voice_path, subtitle_path, timeline, params, out_path, workdir=None, focus=None,
                focus_expr=None, black_box=None, src_size=None, regions=None, region_method=None):
    logger = get_logger()
    video_path = Path(video_path).resolve()
    out_path = Path(out_path)
    ensure_dir(out_path.parent)

    has_source_audio = _has_audio(video_path)
    has_voice = bool(voice_path and Path(voice_path).exists())
    speed = float(timeline.get("speed_factor", 1.0))
    start = float(timeline.get("source_start", 0.0))
    duration = float(timeline.get("source_duration", 30.0))

    base_filter = video_filter(params, focus=focus, focus_expr=focus_expr)
    tail = ""
    if subtitle_path and Path(subtitle_path).exists():
        tail = "ass='%s'" % _escape_filter_path(Path(subtitle_path).resolve())

    segments = timeline.get("segments") or []
    if (_use_evidence(params) and len(segments) >= 1):
        return _export_evidence(video_path, voice_path, out_path, params, segments,
                                base_filter, focus, black_box, speed, logger, timeline, tail, src_size, regions, region_method)

    vout_graph = build_video_graph(params, base_filter, black_box=black_box, tail=tail, src_size=src_size,
                                   regions=regions, region_method=region_method)

    agraph = audio_graph(has_source_audio, has_voice, params, speed=speed)

    args = [ffmpeg_bin(), "-hide_banner", "-loglevel", "error", "-y"]
    args += ["-ss", "%.3f" % start, "-t", "%.3f" % duration, "-i", str(video_path)]
    if has_voice:
        args += ["-i", str(Path(voice_path).resolve())]
    parts = [vout_graph]
    if agraph:
        parts.append(agraph)
    args += ["-filter_complex", ";".join(parts)]
    args += ["-map", "[vout]"]
    if agraph:
        args += ["-map", "[aout]"]
    else:
        args += ["-an"]
    args += _encoder_args(out_path, params)
    logger.info("渲染: %s (源 %.1f-%.1f)", out_path.name, start, start + duration)
    _, err, code = run_cmd(args, check=False, timeout=1800)
    if code != 0:
        raise RuntimeError("渲染失败: %s" % (err or "")[-1500:])
    if not out_path.exists():
        raise RuntimeError("渲染未产出文件: %s" % out_path)
    return out_path


def _use_evidence(params):
    return str(params.get("evidence_cut", "on")).lower() not in ("off", "0", "false")


def _export_evidence(video_path, voice_path, out_path, params, segments, base_filter,
                     focus, black_box, speed, logger, timeline=None, tail="", src_size=None,
                     regions=None, region_method=None):
    """Render each beat from its actual evidence, preserving measured output times."""
    import tempfile
    from ..utils import write_json
    from .sound import make_soundbed
    from .transitions import transition_plan, pair_xfade_graph, pair_concat_graph
    from .pacing import render_runs
    from .resources import render_budget

    has_voice = bool(voice_path and Path(voice_path).is_file())
    if not has_voice:
        raise RuntimeError("证据剪辑需要真实旁白音轨")
    params,budget = render_budget(params)
    write_json(out_path.parent/'render_resources.json',budget)
    total = sum(float(s["dur"]) for s in segments)
    trimmed = _prepare_context_sources(video_path, segments)
    if trimmed:
        write_json(out_path.parent/'timeline.json', timeline)
    from .motion import warp_available
    from .pacing import boundary_is_jump
    runs = render_runs(segments)
    roles = [_segment_role(s, timeline) for s in runs]
    force_hard = [i for i in range(len(runs) - 1) if boundary_is_jump(runs[i], runs[i + 1])]
    plan = transition_plan([s["dur"] for s in runs],
                           enabled=str(params.get("short_transitions", "on")) == "on",
                           seconds=params.get("transition_duration", .125),
                           roles=roles, force_hard=force_hard)
    use_warp = warp_available()
    write_json(out_path.parent/'edit_decisions.json', {
        'policy':'continuous_context_v4','motion':'cubic_warp' if use_warp else 'scale_crop',
        'hard_cuts':force_hard,'logical_segments':len(segments),'rendered_shots':len(runs),
        'context_joins':len(segments)-len(runs),
        'shots':[{'start':s.get('out_start',0),'duration':s['dur'],
                  'parts':len(s['parts']),'kind':s['visual_kind'],
                  'motion':[p.get('motion') or 'none' for p in s.get('parts') or [s]]} for s in runs]})
    write_json(out_path.parent / "transitions.json", plan)
    # Sequential render bounds decoder/filter memory for long shot lists.
    with tempfile.TemporaryDirectory(prefix="evidence_", dir=out_path.parent) as tmp:
        tmp = Path(tmp)
        rendered = []
        for i, seg in enumerate(runs):
            handle = plan["handles"][i]
            src_dur = float(seg.get("src_duration", seg["dur"]))
            if src_dur <= 0 or handle["nominal_frames"] <= 0:
                raise ValueError("无效剪辑时长")
            target = tmp / ("shot_%03d.mp4" % i)
            _render_shot(video_path, seg, handle, params, base_filter, black_box, tail,
                         src_size, regions, region_method, target, use_warp)
            _check_video_frames(target, handle["nominal_frames"] + handle["before"] + handle["after"])
            rendered.append(target)
        manifest = tmp / "shots.txt"
        manifest.write_text("\n".join("file '%s'" % p.as_posix().replace("'", "'\\''") for p in rendered), encoding="utf-8")
        picture = manifest
        picture_args = ["-f", "concat", "-safe", "0", "-i", str(manifest)]
        if plan["transitions"]:
            # A long xfade chain retains many decoded HD frames. Keep only two
            # inputs alive and preserve each intermediate losslessly in 4:4:4.
            picture = rendered[0]
            nominal_so_far = plan["handles"][0]["nominal_frames"]
            for i, tr in enumerate(plan["boundaries"], 1):
                nominal_so_far += plan["handles"][i]["nominal_frames"]
                count = nominal_so_far + plan["handles"][i]["after"]
                last = i == len(rendered) - 1
                target = tmp / ("dissolve_%03d%s" % (i, ".mp4" if last else ".mkv"))
                blend = [ffmpeg_bin(), "-v", "error", "-y", "-filter_complex_threads", "1",
                         "-threads", "1", "-i", str(picture), "-threads", "1", "-i", str(rendered[i]),
                         "-filter_complex", pair_xfade_graph(tr, count) if tr['duration_frames'] else pair_concat_graph(count), "-map", "[vout]", "-an",
                         "-frames:v", str(count)]
                if last:
                    blend += _encoder_args(target, dict(params, render_crf=16))
                else:
                    blend += ["-c:v", "libx264", "-preset", "ultrafast", "-tune", "zerolatency",
                              "-crf", "0", "-threads", "2", "-pix_fmt", "yuv444p", str(target)]
                run_cmd(blend, timeout=1800)
                _check_video_frames(target,count)
                picture = target
            picture_args = ["-i", str(picture)]
        enhance = str(params.get("picture_enhance", "on")) == "on"
        assembled = tmp / "assembled.mp4" if enhance else out_path.resolve()
        args = [ffmpeg_bin(), "-v", "error", "-y", *picture_args,
                "-i", str(Path(voice_path).resolve())]
        voice_filters = "atempo=%.6f," % speed if abs(speed - 1) > .001 else ""
        voice_filters += "loudnorm=I=-16:TP=-2:LRA=7,aresample=24000,volume=%.3f" % float(params.get("voice_volume", 1))
        if str(params.get("sound_design", "off")).lower() == "on":
            meta = make_soundbed(dict(timeline or {}, source_duration=total), out_path.parent / "soundbed.wav", params.get("sound_bed_db", -29))
            write_json(out_path.parent / "sound_design.json", meta)
            args += ["-i", meta["path"]]
            graph = "[1:a]%s[voice];[voice][2:a]amix=inputs=2:duration=first:normalize=0,alimiter=limit=0.79:level=false[aout]" % voice_filters
        else:
            graph = "[1:a]%s,alimiter=limit=0.79:level=false[aout]" % voice_filters
        args += ["-filter_complex", graph, "-map", "0:v", "-map", "[aout]", "-c:v", "copy",
                 "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-t", str(total),
                 "-movflags", "+faststart", str(assembled)]
        logger.info("渲染证据剪辑: %d段 %.2fs, 旁白使用真实句时长", len(segments), total)
        run_cmd(args, timeout=1800)
        if enhance:
            from .finishing import finish_picture
            finish_picture(assembled, out_path.resolve(), params)
        _check_video_frames(out_path,plan['output_frames'])
    return out_path


def _prepare_context_sources(video_path, segments):
    """Trim a whole context before remapping its body and aside frame together."""
    from .pacing import render_runs
    changed = False
    for i, seg in enumerate(segments):
        if seg.get('visual_kind') != 'comment_still':
            continue
        def same_context(other):
            return (other.get('sentence_index') == seg.get('sentence_index') and
                    (other.get('evidence_start'),other.get('evidence_end')) ==
                    (seg.get('evidence_start'),seg.get('evidence_end')))
        previous = segments[i-1] if i and same_context(segments[i-1]) else None
        following = segments[i+1] if i+1 < len(segments) and same_context(segments[i+1]) else None
        if following and following.get('visual_kind') != 'comment_still':
            anchor = float(following['src_start'])
        elif previous and previous.get('visual_kind') != 'comment_still':
            anchor = float(previous['src_start']) + float(previous['src_duration'])
        else:
            continue
        anchor = min(anchor, float(seg.get('evidence_end',anchor+1/24))-1/24)
        if abs(anchor-float(seg['src_start'])) > 1e-7:
            seg['src_start'] = anchor
            changed = True
    for run in render_runs(segments):
        old_start, old_duration = run['src_start'],run['src_duration']
        if old_duration <= .2 or run['visual_kind'] == 'comment_still':
            continue
        new_start,new_duration = _trim_black_edges(video_path,old_start,old_duration)
        if new_start == old_start and abs(new_duration-old_duration) <= .001:
            continue
        ratio = new_duration / old_duration
        for part in run['parts']:
            original_start, original_duration = part['src_start'],part['src_duration']
            mapped = new_start + (original_start-old_start)*ratio
            if part.get('visual_kind') == 'comment_still':
                mapped = min(mapped,new_start+new_duration-1/24)
            else:
                part['src_duration'] = original_duration*ratio
            part['src_start'] = mapped
            part['source_black_trim'] = {'original_start':original_start,'original_duration':original_duration,
                                         'context_start':new_start,'context_duration':new_duration}
        changed = True
    return changed


def _shot_moves(seg):
    if seg.get("visual_kind") == "context_run":
        return any(p.get("motion") not in (None, "", "none") for p in seg.get("parts") or [])
    return seg.get("motion") not in (None, "", "none")


def _timing_chain(seg, nominal, base_filter, motion):
    """Build the nominal frames. Handles are cloned after motion, not inside it."""
    from .pacing import fill_plan, time_resample
    out_dur = nominal / 24.0
    if seg.get("visual_kind") == "comment_still":
        return ("loop=loop=-1:size=1:start=0,setpts=N/(24*TB),%s,fps=24,%s,"
                "trim=end_frame=%d,setpts=PTS-STARTPTS") % (base_filter, motion, nominal)
    src_dur = max(1 / 24, float(seg.get("src_duration", seg["dur"])))
    preset = min(float(seg.get("freeze_duration") or 0), max(0.0, out_dur - .2))
    preset_start = preset if seg.get("freeze_position") == "start" else 0.0
    body = max(1 / 24, out_dur - preset)
    fill = fill_plan(src_dur, body)
    freeze_end = (preset - preset_start) + fill["freeze"]
    resample = time_resample(fill["interpolate"], nominal)
    return ("trim=duration=%.9f,setpts=%.9f*(PTS-STARTPTS),%s,%s,"
            "tpad=start_mode=clone:start_duration=%.9f:stop_mode=clone:stop_duration=%.9f,"
            "trim=end_frame=%d,setpts=PTS-STARTPTS,%s") % (
                fill["src_used"], fill["active"] / max(fill["src_used"], 1 / 24),
                base_filter, resample, preset_start, freeze_end + 0.25, nominal, motion)


def _with_handles(chain, handle, include):
    nominal = handle["nominal_frames"]
    if not include:
        return chain
    render_frames = nominal + handle["before"] + handle["after"]
    return ("%s,tpad=start_mode=clone:start_duration=%.9f:stop_mode=clone:stop_duration=%.9f,"
            "trim=end_frame=%d,setpts=PTS-STARTPTS") % (
                chain, handle["before"] / 24, handle["after"] / 24 + 0.2, render_frames)


def _render_shot(video_path, seg, handle, params, base_filter, black_box, tail,
                 src_size, regions, region_method, target, use_warp):
    from .motion import hold_spans, motion_filter
    nominal = handle["nominal_frames"]
    width = int(params.get("target_width", 1920)) // 2 * 2
    height = int(params.get("target_height", 1080)) // 2 * 2
    warp = bool(use_warp and _shot_moves(seg))
    spans = []
    kind = seg.get("motion") or "none"
    if seg.get("visual_kind") == "context_run":
        prefix, context_filter, spans = _context_run_graph(
            seg, handle, params, base_filter, with_motion=not warp, include_handles=not warp)
        graph = prefix + ";" + build_video_graph(
            params, context_filter, black_box=black_box, tail=tail,
            src_size=src_size, regions=regions, region_method=region_method).replace("[0:v]", "[context]")
    else:
        if seg.get("visual_kind") == "comment_still":
            preset = min(float(seg.get("freeze_duration") or 0), max(0.0, nominal / 24 - .2))
            preset_start = preset if seg.get("freeze_position") == "start" else 0.0
            spans = hold_spans(nominal, kind, freeze_frames=round(preset_start * 24))
            kind = "none"
        motion = "null"
        if not warp:
            motion = motion_filter(kind, width, height, nominal / 24.0, nominal,
                                   oversample=params.get("motion_oversample", 8),
                                   local_spans=spans or None)
        chain = _with_handles(_timing_chain(seg, nominal, base_filter, motion), handle, include=not warp)
        graph = build_video_graph(params, chain, black_box=black_box, tail=tail,
                                  src_size=src_size, regions=regions, region_method=region_method)
    src_dur = float(seg.get("src_duration", seg["dur"]))
    decode = [ffmpeg_bin(), "-v", "error", "-filter_complex_threads", "1", "-ss", str(seg["src_start"]),
              "-t", str(max(src_dur, 1 / 24)), "-threads", "1", "-i", str(video_path),
              "-filter_complex", graph, "-map", "[vout]", "-an"]
    if warp:
        _encode_warped(decode, width, height, nominal, handle, kind, spans, seg, target, params)
        return
    render_frames = nominal + handle["before"] + handle["after"]
    decode = [ffmpeg_bin(), "-v", "error", "-y", *decode[3:], "-frames:v", str(render_frames)]
    decode += _encoder_args(target, dict(params, render_crf=16))
    run_cmd(decode, timeout=300)


def _encode_warped(decode, width, height, nominal, handle, kind, spans, seg, target, params):
    import subprocess
    import threading
    import numpy as np
    from .motion import frame_motion, warp_frame
    frame_bytes = width * height * 3
    total = nominal + handle["before"] + handle["after"]
    duration = nominal / 24.0
    decode_proc = subprocess.Popen(decode + ["-frames:v", str(nominal), "-f", "rawvideo", "-pix_fmt", "bgr24", "pipe:1"],
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    encode_cmd = [ffmpeg_bin(), "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
                  "-s", "%dx%d" % (width, height), "-r", "24", "-i", "pipe:0",
                  "-an", "-frames:v", str(total)]
    encode_cmd += _encoder_args(target, dict(params, render_crf=16))
    encode_proc = subprocess.Popen(encode_cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    errors = {}

    def drain(proc, key):
        errors[key] = proc.stderr.read()

    threading.Thread(target=drain, args=(decode_proc, "decode"), daemon=True).start()
    threading.Thread(target=drain, args=(encode_proc, "encode"), daemon=True).start()

    def read_frame():
        buf = bytearray()
        while len(buf) < frame_bytes:
            chunk = decode_proc.stdout.read(frame_bytes - len(buf))
            if not chunk:
                break
            buf += chunk
        return buf if len(buf) == frame_bytes else None

    def move(raw, index):
        frame = np.frombuffer(raw, np.uint8).reshape(height, width, 3)
        zoom, fraction = frame_motion(kind, index, nominal, duration, spans or None)
        if abs(zoom - 1) < 1e-5 and abs(fraction - 0.5) < 1e-5:
            return bytes(raw)
        return np.ascontiguousarray(warp_frame(frame, zoom, fraction)).tobytes()

    failed = None
    try:
        first = read_frame()
        if first is None:
            raise RuntimeError("镜头没有输出画面")
        payload = move(first, 0)
        for _ in range(handle["before"] + 1):
            encode_proc.stdin.write(payload)
        last = payload
        for index in range(1, nominal):
            raw = read_frame()
            if raw is None:
                raise RuntimeError("镜头帧不足：需要%d帧" % nominal)
            last = move(raw, index)
            encode_proc.stdin.write(last)
        for _ in range(handle["after"]):
            encode_proc.stdin.write(last)
    except Exception as exc:
        failed = exc
    finally:
        try:
            encode_proc.stdin.close()
        except Exception:
            pass
        try:
            decode_proc.stdout.read()
        except Exception:
            pass
        decode_proc.wait()
        encode_proc.wait()
    if failed:
        raise failed
    if decode_proc.returncode or encode_proc.returncode:
        detail = (errors.get("decode") or b"").decode("utf-8", "replace")[-400:]
        detail += (errors.get("encode") or b"").decode("utf-8", "replace")[-400:]
        raise RuntimeError("镜头平滑采样失败: %s" % detail.strip())


def _context_run_graph(run, handle, params, base_filter, with_motion=True, include_handles=True):
    """Assemble continuous source/holds, then one local motion clock.

    A short prelude stays inside this shot. The aside eases away from the
    current frame and back, so the join is the same picture.
    """
    from .motion import motion_filter
    from .pacing import fill_plan, time_resample
    parts, frames = run["parts"], handle["nominal_frames"]
    total = sum(float(p["dur"]) for p in parts)
    graph = ["[0:v]setpts=PTS-STARTPTS,split=%d%s" % (
        len(parts), "".join("[ctxin%d]" % i for i in range(len(parts))))]
    spans, outputs, elapsed, previous = [], [], 0.0, 0
    for i, part in enumerate(parts):
        elapsed += float(part["dur"])
        end = round(elapsed / total * frames)
        count = end - previous
        if count <= 0:
            graph.append("[ctxin%d]nullsink" % i)
            continue
        offset = max(0.0, float(part["src_start"]) - run["src_start"])
        source_duration = float(part["src_duration"])
        if part.get("visual_kind") == "comment_still":
            clip = ("trim=start=%.9f:duration=%.9f,setpts=PTS-STARTPTS,"
                    "loop=loop=-1:size=1:start=0,setpts=N/(24*TB)") % (offset, max(source_duration, 1 / 24))
            start_frame = previous + round(float(part.get("freeze_duration") or 0) * 24)
            spans.append({"start_frame": min(end - 1, start_frame), "end_frame": end,
                          "motion": part.get("motion") or "none"})
            stop = 0.2
        else:
            fill = fill_plan(source_duration, count / 24.0)
            clip = ("trim=start=%.9f:end=%.9f,setpts=%.9f*(PTS-STARTPTS),%s" % (
                offset, offset + fill["src_used"],
                fill["active"] / max(fill["src_used"], 1 / 24),
                time_resample(fill["interpolate"], count)))
            stop = fill["freeze"] + 0.2
        graph.append("[ctxin%d]%s,tpad=stop_mode=clone:stop_duration=%.9f,trim=end_frame=%d,setpts=N/(24*TB)[ctxpart%d]" % (
            i, clip, stop, count, i))
        outputs.append("[ctxpart%d]" % i)
        previous = end
    if not outputs:
        raise RuntimeError("连续镜头没有可输出的画面")
    graph.append("%sconcat=n=%d:v=1:a=0[context]" % ("".join(outputs), len(outputs)))
    motion = "null"
    if with_motion:
        motion = motion_filter("none", int(params.get("target_width", 1920)), int(params.get("target_height", 1080)),
                               frames / 24.0, frames, oversample=params.get("motion_oversample", 8),
                               local_spans=spans)
    before = handle["before"] / 24 if include_handles else 0
    after = handle["after"] / 24 + 0.2 if include_handles else 0.2
    trim_frames = frames + (handle["before"] + handle["after"] if include_handles else 0)
    filters = ("%s,fps=24,%s,tpad=start_mode=clone:start_duration=%.9f:"
               "stop_mode=clone:stop_duration=%.9f,trim=end_frame=%d,setpts=PTS-STARTPTS") % (
                   base_filter, motion, before, after, trim_frames)
    return ";".join(graph), filters, spans


def _check_video_frames(path,expected):
    import json
    out,_,_ = run_cmd(['ffprobe','-v','error','-select_streams','v:0','-count_packets',
                       '-show_entries','stream=nb_read_packets','-of','json',str(path)],timeout=120)
    streams=json.loads(out).get('streams',[])
    count=int(streams[0].get('nb_read_packets',0)) if streams else 0
    if abs(count-int(expected)) > 1:
        raise RuntimeError('视频合成帧数不足：%s，应有%d帧，实际%d帧；已停止导出，避免生成后半段无画面的文件' %
                           (Path(path).name,expected,count))



def _segment_role(seg, timeline=None):
    role = str(seg.get("role") or "")
    if role:
        return role
    idx = seg.get("sentence_index")
    timings = (timeline or {}).get("sentence_timings") or []
    if isinstance(idx, int):
        for item in timings:
            if item.get("sentence_index") == idx:
                return str(item.get("role") or "")
        if 0 <= idx < len(timings):
            return str(timings[idx].get("role") or "")
    return ""


def _trim_black_edges(source,start,duration):
    import re
    _,log,code=run_cmd([ffmpeg_bin(),'-hide_banner','-nostats','-threads','1','-ss',str(start),'-t',str(duration),
                       '-i',str(source),'-vf','setpts=PTS-STARTPTS,blackdetect=d=0.08:pix_th=0.08:pic_th=0.98',
                       '-an','-f','null','-'],check=False,timeout=120)
    if code:
        raise RuntimeError('原片黑场检查失败，已停止该镜头导出')
    intervals=[(float(a),float(b)) for a,b in re.findall(r'black_start:([\d.]+) black_end:([\d.]+)',log)]
    lo,hi=0.0,duration
    for a,b in intervals:
        if a<=.05:
            lo=max(lo,b+1/24)
        if b>=duration-.08:
            hi=min(hi,a-1/24)
    if hi-lo<.15:
        raise RuntimeError('所选原片镜头为黑幕，缺少有效画面，需要重新选择镜头')
    return start+lo,hi-lo


def _has_audio(path):
    from ..ingest.probe import has_audio_stream
    try:
        return has_audio_stream(path)
    except Exception:
        return False


def _has_nvenc():
    import subprocess

    try:
        out = subprocess.run([ffmpeg_bin(), "-hide_banner", "-encoders"], capture_output=True,
                             text=True, timeout=20).stdout
        return "h264_nvenc" in out
    except Exception:
        return False


_NVENC_CACHE = {}


def _encoder_args(out_path, params):
    base = ["-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "192k", "-ar", "44100",
            "-movflags", "+faststart",
            str(out_path.resolve())]
    prefer_nvenc = str(params.get("prefer_nvenc", "auto")).lower() != "off"
    if prefer_nvenc not in _NVENC_CACHE:
        _NVENC_CACHE[prefer_nvenc] = prefer_nvenc and _has_nvenc()
    if _NVENC_CACHE[prefer_nvenc]:
        return ["-c:v", "h264_nvenc", "-preset", "p5", "-cq", str(params.get("nvenc_cq", 18)), "-b:v", "0"] + base
    return ["-c:v", "libx264", "-preset", "veryfast", "-tune", "zerolatency",
            "-crf", str(params.get("render_crf", 17)), "-threads", "1",
            "-x264-params", "rc-lookahead=0:sync-lookahead=0:ref=2:bframes=0"] + base


def _escape_filter_path(path):
    text = str(path)
    if sys.platform == "win32":
        text = text.replace("\\", "/")
        if len(text) > 1 and text[1] == ":":
            text = text[0] + "\\:" + text[2:]
    return text.replace("'", "\\'")
