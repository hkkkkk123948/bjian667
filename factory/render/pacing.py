"""Keep real shots readable and render contextual asides as one continuous shot."""
import math


def readable_windows(windows, lo, hi, duration, params=None):
    """Allocate readable shot lengths without dropping required story evidence.

    Fact-bound windows are preferred. When the narration has no room for every
    independent cut, the primary picture continues and omitted inserts become
    non-blocking editing diagnostics instead of forcing another script/TTS.
    """
    params = params or {}
    minimum = max(.4, float(params.get('min_shot_seconds', .7)))
    rows = []
    for window in windows:
        start, end = max(lo, float(window['start'])), min(hi, float(window['end']))
        if end - start < 1 / 24:
            continue
        row = dict(window, start=start, end=end, weight=max(.001, float(window.get('weight', 1))))
        if rows and 0 <= start - rows[-1]['end'] <= .08:
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
