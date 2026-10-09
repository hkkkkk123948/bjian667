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
    runs = render_runs(segments)
    roles = [_segment_role(s, timeline) for s in runs]
    plan = transition_plan([s["dur"] for s in runs],
                           enabled=str(params.get("short_transitions", "on")) == "on",
                           seconds=params.get("transition_duration", .125),
                           roles=roles)
    write_json(out_path.parent/'edit_decisions.json', {
        'policy':'continuous_context_v3','logical_segments':len(segments),'rendered_shots':len(runs),
        'context_joins':len(segments)-len(runs),
        'shots':[{'start':s.get('out_start',0),'duration':s['dur'],
                  'parts':len(s['parts']),'kind':s['visual_kind']} for s in runs]})
    write_json(out_path.parent / "transitions.json", plan)
    # Sequential render bounds decoder/filter memory for long shot lists.
    with tempfile.TemporaryDirectory(prefix="evidence_", dir=out_path.parent) as tmp:
        tmp = Path(tmp)
        rendered = []
        for i, seg in enumerate(runs):
            src_dur = float(seg.get("src_duration", seg["dur"]))
            handle = plan["handles"][i]
            frame_count = handle["nominal_frames"]
            render_frames = frame_count + handle["before"] + handle["after"]
            out_dur = frame_count / 24.0
            if src_dur <= 0 or out_dur <= 0:
                raise ValueError("无效剪辑时长")
            from .motion import motion_filter
            freeze = min(float(seg.get("freeze_duration", 0)), max(0, out_dur - .2))
            active = out_dur - freeze
            motion = motion_filter(seg.get("motion", "none"), int(params.get("target_width", 1920)),
                                   int(params.get("target_height", 1080)), active, round(active * 24),
                                   oversample=params.get("motion_oversample", 3))
            freeze_start = freeze if seg.get("freeze_position") == "start" else 0
            freeze_end = freeze - freeze_start
            filters = "setpts=%.9f*(PTS-STARTPTS),%s,fps=24,%s,tpad=start_mode=clone:start_duration=%.9f:stop_mode=clone:stop_duration=%.9f,trim=end_frame=%d,setpts=PTS-STARTPTS" % (
                active / src_dur, base_filter, motion, handle["before"] / 24 + freeze_start,
                freeze_end + handle["after"] / 24 + .15, render_frames)
            if seg.get("visual_kind") == "comment_still":
                filters = ("loop=loop=-1:size=1:start=0,setpts=N/(24*TB),%s,fps=24,%s,"
                           "tpad=start_mode=clone:start_duration=%.9f,trim=end_frame=%d,setpts=PTS-STARTPTS") % (
                               base_filter,motion,handle['before']/24+freeze_start,render_frames)
            graph = build_video_graph(params, filters, black_box=black_box, tail=tail,
                                      src_size=src_size, regions=regions, region_method=region_method)
            if seg.get('visual_kind') == 'context_run':
                prefix, context_filter = _context_run_graph(seg, handle, params, base_filter)
                graph = prefix + ';' + build_video_graph(
                    params, context_filter, black_box=black_box, tail=tail,
                    src_size=src_size, regions=regions, region_method=region_method).replace('[0:v]', '[context]')
            target = tmp / ("shot_%03d.mp4" % i)
            args = [ffmpeg_bin(), "-v", "error", "-y", "-filter_complex_threads", "1", "-ss", str(seg["src_start"]),
                    "-t", str(src_dur), "-threads", "1", "-i", str(video_path), "-filter_complex", graph,
                    "-map", "[vout]", "-an", "-frames:v", str(render_frames)]
            args += _encoder_args(target, dict(params, render_crf=16))
            run_cmd(args, timeout=300)
            _check_video_frames(target,render_frames)
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


def _context_run_graph(run, handle, params, base_filter):
    """Assemble continuous source/holds before applying one local motion clock.

    A three-frame prelude is part of this shot, never a separate dissolve.
    Local motion starts and ends at the original scale, avoiding zoom snaps.
    """
    from .motion import motion_filter
    parts, frames = run['parts'], handle['nominal_frames']
    total = sum(float(p['dur']) for p in parts)
    graph = ['[0:v]setpts=PTS-STARTPTS,split=%d%s' % (
        len(parts), ''.join('[ctxin%d]' % i for i in range(len(parts))))]
    spans, outputs, elapsed, previous = [], [], 0., 0
    for i, part in enumerate(parts):
        elapsed += float(part['dur'])
        end = round(elapsed / total * frames)
        count = end - previous
        if count <= 0:
            graph.append('[ctxin%d]nullsink' % i)
            continue
        offset = max(0., float(part['src_start']) - run['src_start'])
        source_duration = float(part['src_duration'])
        if part.get('visual_kind') == 'comment_still':
            clip = 'trim=start=%.9f:duration=%.9f,setpts=PTS-STARTPTS,loop=loop=-1:size=1:start=0,setpts=N/(24*TB)' % (offset,source_duration)
            start_frame = previous + round(float(part.get('freeze_duration',0)) * 24)
            spans.append({'start_frame':min(end-1,start_frame),'end_frame':end,
                          'motion':part.get('motion','none')})
        else:
            clip = 'trim=start=%.9f:duration=%.9f,setpts=%.9f*(PTS-STARTPTS),fps=24' % (
                offset,source_duration,(count/24)/source_duration)
        graph.append('[ctxin%d]%s,tpad=stop_mode=clone:stop_duration=0.15,trim=end_frame=%d,setpts=N/(24*TB)[ctxpart%d]' % (i,clip,count,i))
        outputs.append('[ctxpart%d]' % i)
        previous = end
    graph.append('%sconcat=n=%d:v=1:a=0[context]' % (''.join(outputs),len(outputs)))
    motion = motion_filter('none',int(params.get('target_width',1920)),int(params.get('target_height',1080)),
                           frames/24,frames,oversample=params.get('motion_oversample',3),local_spans=spans)
    filters = '%s,fps=24,%s,tpad=start_mode=clone:start_duration=%.9f:stop_mode=clone:stop_duration=%.9f,trim=end_frame=%d,setpts=PTS-STARTPTS' % (
        base_filter,motion,handle['before']/24,handle['after']/24+.15,
        frames+handle['before']+handle['after'])
    return ';'.join(graph), filters


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
                       '-i',str(source),'-vf','setpts=PTS-STARTPTS,blackdetect=d=0.18:pix_th=0.08:pic_th=0.98',
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
