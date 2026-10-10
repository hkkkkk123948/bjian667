"""Keep real shots readable and render contextual asides as one continuous shot."""
import math


def readable_windows(windows, lo, hi, duration, params=None):
    """Allocate readable shot lengths without dropping required story evidence.

    Fact-bound windows are preferred. When the narration has no room for every
    independent cut, the primary picture continues and omitted inserts become
    non-blocking editing diagnostics instead of forcing another script/TTS.
    """
    params = params or {}
    # Floor above the saved default so a 0.4s flash cannot survive as its own cut.
    minimum = max(.85, float(params.get('min_shot_seconds', .85)))
    rows = []
    for window in windows:
        start, end = max(lo, float(window['start'])), min(hi, float(window['end']))
        if end - start < 1 / 24:
            continue
        row = dict(window, start=start, end=end, weight=max(.001, float(window.get('weight', 1))))
        if rows and 0 <= start - rows[-1]['end'] <= .16:
            rows[-1]['end'] = end
            rows[-1]['weight'] += row['weight']
        else:
            rows.append(row)
    if not rows:
        raise ValueError('所选镜头没有可播放的画面证据')
    capacity = max(1, math.floor((duration + 1e-8) / minimum))
    while len(rows) > 1:
        total = sum(row['weight'] for row in rows)
        optional = [i for i,row in enumerate(rows) if row.get('optional') is True]
        if not optional:
            break
        weakest = min(optional, key=lambda i: rows[i]['weight'])
        if len(rows) <= capacity and min(duration * r['weight'] / total for r in rows) >= minimum:
            break
        rows.pop(weakest)
    hard_minimum = max(.25, float(params.get('min_required_shot_seconds', .35)))
    if len(rows) > 1 and duration + 1e-8 < len(rows) * hard_minimum:
        capacity = max(1, math.floor((duration+1e-8)/hard_minimum))
        ranked = sorted(range(len(rows)), key=lambda i: (rows[i]['weight'], rows[i]['end']-rows[i]['start']), reverse=True)
        retained = set(ranked[:capacity])
        omitted = [{'start':r['start'],'end':r['end'],'weight':r['weight']}
                   for i,r in enumerate(rows) if i not in retained]
        rows = [r for i,r in enumerate(rows) if i in retained]
        rows[0]['continuity_omitted_windows'] = omitted
    # Reserve minimum dwell time for every necessary action, then distribute
    # the remainder by the original relevance weights (water filling).
    floor = min(minimum, duration / len(rows))
    pending, remaining = list(range(len(rows))), duration
    allocations = [0.] * len(rows)
    while pending:
        total = sum(rows[i]['weight'] for i in pending)
        short = [i for i in pending if remaining * rows[i]['weight'] / total < floor - 1e-8]
        if not short:
            for i in pending:
                allocations[i] = remaining * rows[i]['weight'] / total
            break
        for i in short:
            allocations[i] = floor
            remaining -= floor
            pending.remove(i)
    for row, allocation in zip(rows,allocations):
        row['weight'] = allocation / duration
    return rows


def _same_context(left, right):
    if left.get('sentence_index') != right.get('sentence_index'):
        return False
    if (left.get('evidence_start'), left.get('evidence_end')) != (right.get('evidence_start'), right.get('evidence_end')):
        return False
    left_end = float(left['src_start']) if left.get('visual_kind') == 'comment_still' else float(left['src_start']) + float(left['src_duration'])
    return abs(left_end - float(right['src_start'])) <= .085


def render_runs(segments):
    """Group source-continuous body/aside/body without introducing picture cuts.

    Logical segments stay intact for narration/provenance checks. The renderer
    concatenates source pieces first and applies local motion once to the run.
    """
    runs = []
    for seg in segments:
        if runs and _same_context(runs[-1]['parts'][-1], seg):
            runs[-1]['parts'].append(seg)
        else:
            runs.append({'parts': [seg]})
    for run in runs:
        parts = run['parts']
        run.update(parts[0])
        run['parts'] = parts
        run['dur'] = sum(float(p['dur']) for p in parts)
        run['src_start'] = min(float(p['src_start']) for p in parts)
        run['src_duration'] = max(float(p['src_start']) + float(p['src_duration']) for p in parts) - run['src_start']
        run['visual_kind'] = 'context_run' if len(parts) > 1 else parts[0].get('visual_kind', 'source')
    return runs


def fill_plan(source_duration, output_duration, fps=24):
    """Cover `output_duration` without a stuttering slideshow.

    A mild shortfall is blended slow motion. A severe shortfall plays at a
    readable slow speed and holds the last frame for the rest. The hold is
    what a freeze is for; stretching a few source frames across many seconds
    is not.
    """
    fps = float(fps or 24)
    tick = 1.0 / fps
    output = max(tick, float(output_duration))
    source = max(tick, float(source_duration or 0))
    if source + 1e-6 >= output:
        # The selected action is at least as long as the slot. Play all of it,
        # speeding up only as much as the slot requires.
        return {"speed": source / output, "active": output, "freeze": 0.0, "src_used": source,
                "interpolate": False, "freeze_position": "end"}
    natural = source / output
    if natural >= 0.42:
        return {"speed": natural, "active": output, "freeze": 0.0, "src_used": source,
                "interpolate": natural < 0.90, "freeze_position": "end"}
    hold_cap = 0.70
    speed = min(0.50, max(0.34, source / max(tick, output - hold_cap)))
    active = min(output, source / speed)
    freeze = output - active
    if freeze < tick:
        return {"speed": natural, "active": output, "freeze": 0.0, "src_used": source,
                "interpolate": True, "freeze_position": "end"}
    return {"speed": source / active, "active": active, "freeze": freeze, "src_used": source,
            "interpolate": True, "freeze_position": "end"}


def time_resample(interpolate, frames):
    """Blend only when stretched frames would otherwise repeat."""
    if interpolate and int(frames) >= 8:
        return "minterpolate=fps=24:mi_mode=blend"
    return "fps=24"


def boundary_is_jump(left, right):
    """A dissolve across a near-repeat or a tiny time gap ghosts the cut."""
    if not left or not right:
        return False
    if left.get("visual_kind") == "comment_still" or right.get("visual_kind") == "comment_still":
        return False
    if left.get("shot_id") and left.get("shot_id") == right.get("shot_id"):
        return True
    try:
        left_start, left_end = float(left["src_start"]), float(left["src_start"]) + float(left["src_duration"])
        right_start, right_end = float(right["src_start"]), float(right["src_start"]) + float(right["src_duration"])
    except (KeyError, TypeError, ValueError):
        return False
    gap = right_start - left_end
    left_key = (left.get("evidence_start"), left.get("evidence_end"))
    right_key = (right.get("evidence_start"), right.get("evidence_end"))
    same = None not in left_key and left_key == right_key
    overlap = min(left_end, right_end) - max(left_start, right_start)
    if overlap > 0.2:
        return True
    return bool(same and 0.04 < abs(gap) < 0.45)
