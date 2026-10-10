from ..log import get_logger
from .motion import choose_motion


def shorten_variant_to_fit(variant, limit):
    """Compatibility entry: length targets no longer remove selected narration."""
    if variant and getattr(variant, 'estimated_seconds', 0) > limit:
        get_logger().info('文案预计超过目标时长，仅提示；保留全部已选文字')
    return variant


def build_timeline(candidate, variant, tts_result, params, video_duration=0.0, index=None, router=None):
    logger = get_logger()
    max_dur = float(params.get("max_duration", 35))
    min_dur = float(params.get("min_duration", 15))
    tts_duration = float(tts_result.get("duration") or 0.0)
    source_len = candidate.source_end - candidate.source_start
    narration = tts_duration

    # 解说句 ↔ 场景 语义匹配（参考“解说猫”），驱动画面剪辑，提升音画匹配
    if (index and router and variant is not None and not getattr(variant, "scene_matches", None)
            and str(params.get("scene_align", "on")).lower() not in ("off", "0", "false")):
        try:
            from ..matching.scene_align import align_sentences_to_scenes
            variant.scene_matches = align_sentences_to_scenes(
                candidate, variant.sentences, index, router,
                role=params.get("scene_align_role", "match_judge"),
                language=params.get("language", "zh")) or []
        except Exception as exc:
            logger.warning("句子-场景匹配异常: %s", exc)

    # Intact narration must match the current script, including comment asides.
    source_text = tts_result.get("text")
    if source_text is not None:
        current = "".join(s.text for s in (variant.sentences or []))
        if source_text != current:
            raise ValueError("旁白文案已变更，须整段重新配音后再剪辑")

    # The audio already has its native synthesis speed. Requested movie
    # length is a planning hint, never a reason to speed up or cut narration.
    speed = 1.0
    clip_duration = narration if narration > 0 else max(0.0, source_len)
    duration_note = ""
    if narration > max_dur:
        duration_note = "整段配音%.2f秒，超过目标%.2f秒；完整保留配音" % (narration, max_dur)
    elif narration < min_dur:
        duration_note = "整段配音%.2f秒，低于目标%.2f秒；按实际配音长度制作" % (narration, min_dur)
    if duration_note:
        logger.info(duration_note)

    start = candidate.source_start
    if video_duration:
        if start + clip_duration > video_duration:
            start = max(0.0, video_duration - clip_duration)

    measured = tts_result.get('sentence_timings') or []
    timing_metadata_mismatch = bool(measured and (len(measured) != len(variant.sentences)
                                    or any(t.get('text') != s.text for t,s in zip(measured,variant.sentences))))
    timings = _sentence_timings(variant, tts_result, narration, speed)
    for sent, timing in zip(variant.sentences, timings):
        timing["narrative_text"] = getattr(sent, 'narrative_text', '') or sent.text
        timing["comment_hook"] = getattr(sent, 'comment_hook', '')
    from .comment_visuals import prepare_comment_picture, apply_comment_picture
    picture_timings, comments, comment_mode, preface = prepare_comment_picture(variant,timings,tts_result,params)
    picture_params = dict(params,max_freeze_seconds=0) if comments else params
    segments = _evidence_segments(candidate, variant, picture_timings, index, picture_params)
    segments = apply_comment_picture(segments,comments,comment_mode,preface)
    segments, editing_diagnostics = _smooth_short_contexts(segments, params)
    if segments:
        clip_duration = sum(s["dur"] for s in segments)
    timeline = {
        "source_start": round(start, 3),
        "source_duration": round(clip_duration, 3),
        "narration_duration": round(narration, 3),
        "tts_duration": round(tts_duration, 3),
        "voice_path": tts_result.get("audio_path", ""),
        "timing_source": "estimated_text_metadata_mismatch" if timing_metadata_mismatch
                         else tts_result.get("timing_source", "estimated"),
        "alignment_report": dict(tts_result.get('alignment_report') or {}, confidence='low',
                                 reason='sentence_metadata_mismatch') if timing_metadata_mismatch
                            else tts_result.get('alignment_report') or {},
        "speed_factor": round(speed, 3),
        "voice_playback_speed": tts_result.get("playback_speed", speed),
        "duration_guidance": {"gate_enabled": False, "target_minimum": min_dur,
                              "target_maximum": max_dur, "actual_narration": narration, "note": duration_note},
        "editing_diagnostics": {"gate_enabled": False, "short_context_adjustments": editing_diagnostics},
        "sentence_timings": timings,
        "segments": segments,
        "comment_picture_plan": {"mode":comment_mode,"comments":comments,
                                 "literal_matching":False,"context_matching":True,
                                 "position_policy":"exact_comment_intervals","local_visual_treatment":True},
        "tail_padding": round(max(0.0, clip_duration - narration), 3),
    }
    return timeline


def _smooth_short_contexts(segments, params):
    """Give a short action nearby picture time, or continue adjacent context.

    Sentence/audio clocks and comment positions stay intact. Only picture cut
    positions can move a little. When the whole narration has no room for an
    independent shot, continuity is preferable to a two-frame flash.
    """
    from .pacing import render_runs
    minimum = max(.35, float(params.get('min_required_shot_seconds', .35)))
    preferred = max(.85, minimum, float(params.get('min_shot_seconds', .85)))
    diagnostics = [{'sentence_index':s.get('sentence_index'), 'kind':'continuous_primary_picture',
                    'omitted_windows':s['continuity_omitted_windows']}
                   for s in segments if s.get('continuity_omitted_windows')]
    rows = list(segments)

    def sensitive(run):
        return any(p.get('comment_spans') or p.get('visual_kind') == 'comment_still' for p in run['parts'])

    def resize(part, duration, anchor='center'):
        old_duration = float(part['dur'])
        rate = float(part['src_duration']) / old_duration
        lo, hi = float(part['evidence_start']), float(part['evidence_end'])
        length = min(hi-lo, duration*rate)
        start = float(part['src_start'])
        if anchor == 'end':
            start += float(part['src_duration'])-length
        elif anchor == 'center':
            start += (float(part['src_duration'])-length)/2
        part.update(dur=duration, src_start=max(lo, min(hi-length, start)), src_duration=length,
                    playback_speed=length/duration, freeze_duration=0)

    def reset_clock():
        clock = 0.0
        for row in rows:
            row['out_start'] = clock
            clock += float(row['dur'])

    original_runs = render_runs(rows)
    for original in original_runs:
        if original['dur'] >= preferred or len(original_runs) < 2:
            continue
        current = render_runs(rows)
        position = next((i for i, run in enumerate(current)
                         if any(run['parts'][0] is p for p in original['parts'])), None)
        if position is None:
            continue
        run = current[position]
        before = float(run['dur'])
        neighbours = [(position-1, 'start'), (position+1, 'end')]
        if len(run['parts']) == 1 and not sensitive(run):
            need = preferred-before
            for donor_index, anchor in neighbours:
                if not 0 <= donor_index < len(current):
                    continue
                donor = current[donor_index]
                if len(donor['parts']) != 1 or sensitive(donor):
                    continue
                # At most 0.50 s per boundary, with the neighbouring picture
                # retaining readable dwell time. Source playback stays bounded
                # by the selected evidence, without moving the audio clock.
                amount = min(need, .50, max(0.0, float(donor['parts'][0]['dur'])-preferred))
                if amount > 1e-8:
                    resize(donor['parts'][0], float(donor['parts'][0]['dur'])-amount, anchor)
                    resize(run['parts'][0], float(run['parts'][0]['dur'])+amount)
                    need -= amount
            run['dur'] = sum(float(p['dur']) for p in run['parts'])
            if run['dur'] > before+1e-8:
                diagnostics.append({'sentence_index':run.get('sentence_index'), 'kind':'extended_picture_dwell',
                                    'before':before, 'after':run['dur']})
        if run['dur'] < minimum-1e-8:
            # Borrow no time from comment spans. Instead use the same context
            # frame across the short interval, letting the renderer join it to
            # the neighbouring shot. Its original narration is left intact.
            target = current[position-1] if position else current[position+1]
            neighbour = target['parts'][-1] if position else target['parts'][0]
            anchor = float(neighbour['src_start'])
            if position and neighbour.get('visual_kind') != 'comment_still':
                anchor += float(neighbour['src_duration'])
            anchor = max(float(neighbour['evidence_start']), min(anchor,float(neighbour['evidence_end'])-1/24))
            for part in run['parts']:
                original_index = part.get('sentence_index')
                part.update(src_start=anchor, src_duration=1/24, visual_kind='comment_still',
                            sentence_index=neighbour.get('sentence_index'),
                            evidence_start=neighbour['evidence_start'], evidence_end=neighbour['evidence_end'],
                            continuity_for_sentence_index=original_index, motion=part.get('motion','none')
                            if part.get('comment_spans') else 'none')
            diagnostics.append({'sentence_index':run.get('sentence_index'), 'kind':'adjacent_context_continuation',
                                'duration':run['dur'], 'context_sentence_index':neighbour.get('sentence_index')})
        reset_clock()
    return rows, diagnostics


def _evidence_segments(candidate, variant, timings, index=None, params=None):
    """Support multiple non-reused shots per sentence while leaving its voice intact."""
    params = params or {}
    segments, used = [], []
    matches = {m.get("sentence_index"): m.get("source_windows", []) for m in getattr(variant, "scene_matches", []) if isinstance(m, dict)}
    for sent, tm in zip(variant.sentences, timings):
        duration = float(tm["end"]) - float(tm["start"])
        ev = sent.source_evidence
        if duration <= 0 or not ev or ev.end <= ev.start:
            raise ValueError("每句必须有有效配音时长和画面证据")
        windows = matches.get(sent.index) or [{"start": ev.start, "end": ev.end, "weight": 1.0}]
        from .pacing import readable_windows
        windows = readable_windows(windows, max(candidate.source_start, ev.start),
                                   min(candidate.source_end, ev.end), duration, params)
        weights = [max(.01, float(w.get("weight", 1))) for w in windows]
        total_weight = sum(weights)
        cursor = float(tm["start"])
        for part, (window, weight) in enumerate(zip(windows, weights)):
            out_end = float(tm["end"]) if part == len(windows) - 1 else cursor + duration * weight / total_weight
            out_duration = out_end - cursor
            lo = max(candidate.source_start, ev.start, float(window["start"]))
            hi = min(candidate.source_end, ev.end, float(window["end"]))
            if hi <= lo:
                raise ValueError("证据不在源片窗口内")
            motion_enabled = str(params.get("visual_motion", "on")) == "on"
            # Freezes are an explicit aside treatment. Normal story motion
            # should not stop automatically at every twist/payoff sentence.
            freeze = 0.0
            active = out_duration - freeze
            source_duration = min(active, hi - lo)
            start = lo + (hi - lo - source_duration) / 2
            shared_source = any(min(b, start + source_duration) - max(a, start) > .08 for a, b in used)
            used.append((start, start + source_duration))
            segments.append({"src_start": round(start, 6), "src_duration": round(source_duration, 6),
                             "dur": out_duration, "out_start": cursor, "sentence_index": sent.index,
                             "shot_part": part + 1, "playback_speed": round(source_duration / active, 6),
                             "freeze_duration": freeze,
                             "shared_source_interval": shared_source,
                             "continuity_omitted_windows": window.get('continuity_omitted_windows', []),
                             "role": getattr(sent, "role", "") or "",
                             "motion": choose_motion(sent.role, sent.index + part, out_duration) if motion_enabled else "none",
                             "context_id": "%d:%d" % (sent.index, part + 1),
                             "evidence_start": lo, "evidence_end": hi})
            cursor = out_end
    return segments


def _sentence_timings(variant, tts_result, narration, speed):
    sentences = variant.sentences
    if not sentences:
        return []
    measured = tts_result.get("sentence_timings") or []
    if measured:
        if len(measured) != len(sentences) or any(t.get("text") != sent.text for t, sent in zip(measured, sentences)):
            get_logger().warning('时间元数据与当前文字分句不同，保留原音频，按现有文字估计时间')
            return _align_proportional(sentences, narration, speed)
        out = []
        for sent, item in zip(sentences, measured):
            tm = dict(item, start=float(item["start"]) / speed, end=float(item["end"]) / speed)
            sent.start_offset = tm["start"]
            sent.actual_seconds = tm["end"] - tm["start"]
            out.append(tm)
        return out
    word_timings = tts_result.get("word_timings") or []
    if word_timings and len(word_timings) >= max(1, len(sentences)):
        return _align_by_words(sentences, word_timings, narration, speed)
    return _align_proportional(sentences, narration, speed)


def _align_proportional(sentences, narration, speed):
    weights = [max(0.3, s.estimated_seconds / max(0.5, speed)) for s in sentences]
    total = sum(weights) or 1.0
    cursor = 0.0
    out = []
    for sent, w in zip(sentences, weights):
        dur = narration * (w / total)
        out.append({
            "index": sent.index,
            "text": sent.text,
            "role": sent.role,
            "start": round(cursor, 3),
            "end": round(cursor + dur, 3),
        })
        cursor += dur
    if out:
        out[-1]["end"] = round(narration, 3)
    for sent, item in zip(sentences, out):
        sent.start_offset = item["start"]
        sent.actual_seconds = item["end"] - item["start"]
    return out


def _align_by_words(sentences, word_timings, narration, speed):
    concat = []
    cursor_char = 0
    for w in word_timings:
        text = str(w.get("text", ""))
        compact = _spoken_chars(text)
        if not compact:
            continue
        concat.append({
            "char_start": cursor_char,
            "char_end": cursor_char + len(compact),
            "start": float(w.get("start", 0.0)) / max(0.5, speed),
            "end": float(w.get("end", 0.0)) / max(0.5, speed),
            "text": compact,
        })
        cursor_char += len(compact)
    if not concat:
        return _align_proportional(sentences, narration, speed)

    out = []
    char_cursor = 0
    for sent in sentences:
        chars = len(_spoken_chars(sent.text))
        seg_start, seg_end = char_cursor, char_cursor + chars
        char_cursor = seg_end
        hits = [c for c in concat if c["char_end"] > seg_start and c["char_start"] < seg_end]
        if hits:
            start = hits[0]["start"]
            end = hits[-1]["end"]
        else:
            last = out[-1] if out else None
            start = last["end"] if last else 0.0
            end = start + max(0.4, sent.estimated_seconds)
        start = max(start, out[-1]["end"] if out else 0.0)
        if end <= start:
            return _align_proportional(sentences, narration, speed)
        out.append({
            "index": sent.index,
            "text": sent.text,
            "role": sent.role,
            "start": round(start, 3),
            "end": round(end, 3),
        })
    if out and out[-1]["end"] < narration:
        out[-1]["end"] = round(narration, 3)
    for sent, item in zip(sentences, out):
        sent.start_offset = item["start"]
        sent.actual_seconds = item["end"] - item["start"]
    return out


def _spoken_chars(text):
    import re
    return re.sub(r"[^\w\u4e00-\u9fff]", "", text or "")
