"""Choose real shot IDs once, then bind spoken action anchors in code.

The model never controls source or audio seconds. Comments are excluded from
semantic shot choice and inserted locally after the measured clock is bound.
"""
import hashlib
import json
import math
from pathlib import Path

from ..script.comment_hook import story_text
from ..script.quality import compact
from ..utils import read_json, write_json

POLICY = "real_action_frames_and_request_controls_v3"


class InitialPlanError(ValueError):
    pass


def _hash(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(',', ':'), default=str).encode()).hexdigest()


def normalize_catalog(catalog, candidate):
    if not isinstance(catalog, list) or not catalog:
        raise InitialPlanError('没有真实切镜头目录，不能用旧文案的镜头标签假定匹配正确')
    result, seen = [], set()
    for item in catalog:
        sid = str(item.get('shot_id', ''))
        lo, hi = float(item['start']), float(item['end'])
        if (not sid or sid in seen or not math.isfinite(lo + hi) or lo >= hi
                or lo < candidate.source_start - .05 or hi > candidate.source_end + .05):
            raise InitialPlanError('镜头目录含重复ID、越界或无效真实切点')
        visible = str(item.get('visible') or '').strip()
        if not visible:
            raise InitialPlanError('真实镜头缺少独立观察描述，无法首次选镜头')
        frames = []
        for frame in item.get('frames') or []:
            fid, time = str(frame.get('frame_id') or ''), float(frame['time'])
            if not fid or not math.isfinite(time) or not lo <= time < hi:
                raise InitialPlanError('镜头帧编号或代码绑定帧时间无效')
            frames.append({'frame_id': fid, 'time': time, 'position': frame.get('position', '')})
        frame_lookup = {frame['frame_id']: frame for frame in frames}
        if not frames or len(frame_lookup) != len(frames):
            raise InitialPlanError('镜头未保存独立真实帧，不能用镜头中心假定动作时点')
        observations = []
        for observation in item.get('observations') or []:
            if observation.get('frame_id') not in frame_lookup:
                raise InitialPlanError('镜头观察引用了不存在的真实帧')
            observations.append({'frame_id': observation['frame_id'], 'visible': str(observation.get('visible') or '')})
        actions = []
        for action in item.get('actions') or []:
            if not isinstance(action, dict):
                raise InitialPlanError('动作缺少代码绑定的frame_ids，不可仅凭整体镜头标签截片')
            refs = action.get('frame_ids') or []
            if not refs or any(fid not in frame_lookup for fid in refs):
                raise InitialPlanError('动作引用了不存在的真实帧')
            actions.append({'text': str(action.get('text') or ''), 'frame_ids': refs,
                            'anchor_times': [frame_lookup[fid]['time'] for fid in refs]})
        seen.add(sid)
        result.append({'shot_id': sid, 'start': lo, 'end': hi, 'visible': visible,
                       'characters': item.get('characters') or [], 'actions': actions,
                       'frames': frames, 'observations': observations})
    return result


def _overlap_length(left, right):
    """Longest shared slice. A contained phrase counts as the shorter length."""
    if not left or not right:
        return 0
    if left in right or right in left:
        return min(len(left), len(right))
    shorter, longer = (left, right) if len(left) <= len(right) else (right, left)
    for size in range(len(shorter), 0, -1):
        for start in range(0, len(shorter) - size + 1):
            if shorter[start:start + size] in longer:
                return size
    return 0


def _frames_for_phrase(shot, phrase, supported):
    """Bind the action whose text overlaps the spoken phrase the most."""
    best_ids, best_score = [], 0
    for action in shot.get('actions') or []:
        score = _overlap_length(compact(str(action.get('text') or '')), phrase)
        if score <= best_score:
            continue
        ids = [fid for fid in (action.get('frame_ids') or []) if fid in supported]
        if ids:
            best_ids, best_score = ids, score
    if best_ids:
        return list(dict.fromkeys(best_ids))
    for observation in shot.get('observations') or []:
        if phrase and phrase in compact(str(observation.get('visible') or '')):
            fid = observation.get('frame_id')
            if fid in supported:
                return [fid]
    frames = shot.get('frames') or []
    if frames:
        mid = frames[len(frames) // 2]['frame_id']
        if mid in supported:
            return [mid]
    return []


def _merge_repeated_shots(beats):
    """Adjacent beats of one shot are one hold, not a jump back to itself."""
    merged = []
    for beat in beats:
        if merged and merged[-1]['shot_id'] == beat['shot_id']:
            frames = merged[-1]['frame_ids']
            for fid in beat['frame_ids']:
                if fid not in frames:
                    frames.append(fid)
            continue
        merged.append(beat)
    return merged


def normalize_choices(data, variant, catalog):
    lookup_shots = {item['shot_id']: item for item in catalog}
    ids = set(lookup_shots)
    if not isinstance(data, dict) or not isinstance(data.get('sentences'), list):
        raise InitialPlanError('首次匹配未返回逐句镜头方案；不沿用旧稿绑定假装匹配完成')
    if data.get('unmatched'):
        notes = [str(item.get('reason') or item) if isinstance(item, dict) else str(item)
                 for item in data['unmatched']]
        raise InitialPlanError('可用镜头不足：' + '；'.join(notes))
    lookup = {sentence.index: sentence for sentence in variant.sentences}
    output = {}
    for item in data['sentences']:
        index = item.get('sentence_index')
        if index not in lookup or index in output or not isinstance(item.get('beats'), list) or not item['beats']:
            raise InitialPlanError('首次镜头方案遗漏、重复或引用了不存在的文案句子')
        text = compact(story_text(lookup[index]))
        beats, previous = [], -1
        for beat in item['beats']:
            sid = str(beat.get('shot_id') or '')
            phrase = compact(str(beat.get('target_text') or ''))
            if sid not in ids or not phrase:
                raise InitialPlanError('动作锚点必须引用真实shot_id和原文短语')
            supported = {frame['frame_id'] for frame in lookup_shots[sid]['frames']}
            frame_ids = beat.get('frame_ids')
            if not isinstance(frame_ids, list) or not frame_ids:
                # Models often return only shot_id+phrase; bind to catalog action frames.
                frame_ids = _frames_for_phrase(lookup_shots[sid], phrase, supported)
            if (not isinstance(frame_ids, list) or not frame_ids
                    or any(not isinstance(fid, str) or fid not in supported for fid in frame_ids)
                    or len(set(frame_ids)) != len(frame_ids)):
                raise InitialPlanError('动作锚点必须引用该镜头真实frame_ids，不能用整体镜头中心代替')
            positions, cursor = [], 0
            while True:
                at = text.find(phrase, cursor)
                if at < 0:
                    break
                positions.append(at)
                cursor = at + 1
            occurrence = beat.get('occurrence', 1)
            if (type(occurrence) is not int or not 1 <= occurrence <= len(positions)
                    or positions[occurrence - 1] <= previous):
                raise InitialPlanError('动作锚点不在正文、重复不明确或顺序不正确')
            offset = positions[occurrence - 1]
            beats.append({'shot_id': sid, 'target_text': beat['target_text'], 'body_offset': offset,
                          'frame_ids': frame_ids})
            previous = offset
        beats = _merge_repeated_shots(beats)
        if len(beats) > 3:
            raise InitialPlanError('一句分配过多镜头会产生碎切，请减少动作分段')
        output[index] = beats
    if set(output) != set(lookup):
        raise InitialPlanError('可用镜头不足：部分已确认正文尚未分配真实镜头')
    return [{'sentence_index': index, 'beats': output[index]} for index in lookup]


def choose_shots(pipeline, candidate, variant, catalog):
    """At most one bounded text request per unchanged body/catalogue/provider."""
    role = pipeline.config.param('initial_match_role', 'match_judge')
    provider = pipeline.router.get(role)
    body = [{'sentence_index': sentence.index, 'text': story_text(sentence)} for sentence in variant.sentences]
    identity = {'model': provider.model, 'base_url': provider.conf.get('base_url', '')}
    signature = _hash({'policy': POLICY, 'body': body, 'catalog': catalog, 'provider': identity})
    path = Path(pipeline.job_dir) / 'analysis' / 'initial_match' / (candidate.candidate_id + '_' + signature + '.json')
    cached = read_json(path) or {}
    if cached.get('signature') == signature:
        if cached.get('error'):
            raise InitialPlanError(cached['error'])
        return normalize_choices(cached.get('response'), variant, catalog), signature, True
    if provider.is_mock:
        raise InitialPlanError('首次镜头匹配模型未配置，不能按旧稿镜头或平均时长代替')
    concise = [{key: item[key] for key in ('shot_id', 'characters', 'actions')}
               | {'visible': item['visible'][:180]} for item in catalog]
    prompt = (
        '这是首次制作的镜头分配，不是文案审核，不评价或修改文案。以下正文已选定，评论插话已排除。'
        '从独立观察的真实镜头目录中，为每句及句内主要动作选择shot_id。允许同一故事镜头按旁白重排，'
        '不要按原片编号顺序硬排，不沿用旧文案证据。提到狗/刀/门/人物动作时，镜头须在该短语出现时切到对应对象和动作。'
        '每句通常1镜头，含不同关键动作时2镜头，最多3个。连续同场景尽量承接，避免几帧碎切。'
        '不要把同一机位拆成相邻跳切，不要切回刚离开的同一角度。'
        '不要选用黑场、空镜或只有字幕的镜头。'
        '评论插话不是镜头，停在当前这句的画面上，不得占用下一句的画面。'
        'target_text必须逐字取自该句正文的动作或对象短语，按其在句中出现顺序列出；若相同短语重复，occurrence从1开始。'
        '第一镜头同时承接句子开头语境。镜头不够则unmatched说明缺什么，不编道具、不修改文字。'
        '只返JSON {"sentences":[{"sentence_index":1,"beats":[{"target_text":"正文原短语","shot_id":"真实ID","occurrence":1,"frame_ids":["可选真实帧ID"]}]}],'
        '"unmatched":[]}。不要返回时间、分数、改稿或长解释。软件自行定位文字并绑定真实源片与配音时间。\n正文：'
        + json.dumps(body, ensure_ascii=False, separators=(',', ':')) + '\n镜头目录：'
        + json.dumps(concise, ensure_ascii=False, separators=(',', ':')))
    previous_retry, previous_tokens = provider.max_retry, provider.conf.get('max_tokens')
    try:
        provider.max_retry = 1
        provider.conf['max_tokens'] = min(3200, max(900, len(body) * 180))
        provider.conf['enable_thinking'] = False
        response = provider.complete_json(prompt, json_mode=True, default=None)
        choices = normalize_choices(response, variant, catalog)
        write_json(path, {'signature': signature, 'response': response, 'policy': POLICY})
    except Exception as exc:
        error = str(exc)
        write_json(path, {'signature': signature, 'error': error, 'policy': POLICY})
        raise InitialPlanError('首次镜头分配未完成：' + error) from exc
    finally:
        provider.max_retry = previous_retry
        if previous_tokens is None:
            provider.conf.pop('max_tokens', None)
        else:
            provider.conf['max_tokens'] = previous_tokens
    return choices, signature, False


def _clock(variant, result, timings):
    words = result.get('word_timings') or []
    characters = []
    for word in words:
        text = compact(word.get('text', ''))
        if text:
            start, end = float(word['start']), float(word['end'])
            for index, char in enumerate(text):
                characters.append((char, start + (end - start) * index / len(text)))
    full = ''.join(compact(sentence.text) for sentence in variant.sentences)
    exact = ''.join(char for char, _ in characters) == full
    points, base = {}, 0
    for sentence, timing in zip(variant.sentences, timings):
        spoken = compact(sentence.text)
        excluded = set()
        hooks = getattr(sentence, 'comment_hooks', []) or []
        for hook in hooks:
            a = int(hook['spoken_start'])
            excluded.update(range(a, a + len(compact(hook['text']))))
        if not hooks and getattr(sentence, 'comment_hook', ''):
            phrase = compact(sentence.comment_hook)
            a = spoken.rfind(phrase)
            if a >= 0:
                excluded.update(range(a, a + len(phrase)))
        indices = [index for index in range(len(spoken)) if index not in excluded]
        if ''.join(spoken[index] for index in indices) != compact(story_text(sentence)):
            raise InitialPlanError('正文与评论位置记录不一致，不能猜动作发生时点')
        points[sentence.index] = [characters[base + index][1] if exact else
            timing['start'] + (timing['end'] - timing['start']) * index / max(1, len(spoken))
            for index in indices]
        base += len(spoken)
    return points, result.get('timing_source', 'estimated') if exact else 'estimated_within_sentence'


def build_initial_timeline(pipeline, candidate, variant, result, catalog=None):
    """Bind the first shot plan to the measured intact narration, no old bindings."""
    if catalog is None:
        from ..mining.shot_evidence import get_shot_catalog
        catalog = get_shot_catalog(pipeline, candidate)
    catalog = normalize_catalog(catalog, candidate)
    choices, signature, hit = choose_shots(pipeline, candidate, variant, catalog)
    from ..render.timeline import _sentence_timings
    from ..render.comment_visuals import prepare_comment_picture, apply_comment_picture
    from ..render.motion import choose_motion
    duration = float(result['duration'])
    timings = _sentence_timings(variant, result, duration, 1.0)
    for sentence, timing in zip(variant.sentences, timings):
        timing['narrative_text'] = story_text(sentence)
        timing['comment_hook'] = getattr(sentence, 'comment_hook', '')
    picture_times, comments, mode, preface = prepare_comment_picture(variant, timings, result, pipeline.config.params)
    points, clock_source = _clock(variant, result, timings)

    def body_clock(time):
        return time - sum(max(0., min(time, comment['end']) - comment['start'])
                          for comment in comments if time > comment['start'])

    spans = []
    minimum = float(pipeline.config.param('min_required_shot_seconds', .35))
    for plan, timing in zip(choices, picture_times):
        cuts = [timing['start']]
        for beat in plan['beats'][1:]:
            cuts.append(body_clock(points[plan['sentence_index']][beat['body_offset']]))
        cuts.append(timing['end'])
        if cuts[-1] - cuts[0] < minimum * len(plan['beats']):
            raise InitialPlanError('第%d句动作过密，无法在实测配音中分配可读镜头；不插入几帧碎切' % plan['sentence_index'])
        # On a long line, keep each action readable. A cut that already sits
        # well clear of its neighbours is left on the spoken word.
        span = cuts[-1] - cuts[0]
        floor = .55 if span >= .55 * len(plan['beats']) else minimum
        for index in range(1, len(cuts) - 1):
            cuts[index] = max(cuts[index - 1] + floor,
                              min(cuts[index], cuts[-1] - floor * (len(cuts) - 1 - index)))
        for beat, start, end in zip(plan['beats'], cuts, cuts[1:]):
            spans.append({**beat, 'sentence_index': plan['sentence_index'], 'out_start': start, 'dur': end - start})
    totals = {}
    for span in spans:
        totals[span['shot_id']] = totals.get(span['shot_id'], 0.) + span['dur']
    lookup, used, segments = {item['shot_id']: item for item in catalog}, {}, []
    beginnings = {}
    for sid, needed in totals.items():
        shot = lookup[sid]
        available = shot['end'] - shot['start']
        if available < .1 or needed > available * 5:
            raise InitialPlanError('真实镜头%s可用%.2f秒，却需承接%.2f秒旁白；可用动作镜头不够，不能把几帧硬拉长' %
                                   (sid, available, needed))
        # A whole shot observation proves its sampled middle most strongly.
        # Use the centre excerpt rather than blindly starting at its opening.
        beginnings[sid] = shot['start'] + max(0., available - min(available, needed)) / 2
    for span in spans:
        shot = lookup[span['shot_id']]
        length = min(1., (shot['end'] - shot['start']) / totals[span['shot_id']]) * span['dur']
        start = beginnings[span['shot_id']] + used.get(span['shot_id'], 0.)
        used[span['shot_id']] = used.get(span['shot_id'], 0.) + length
        role = ''
        if 0 <= int(span['sentence_index']) < len(variant.sentences):
            role = getattr(variant.sentences[int(span['sentence_index'])], 'role', '') or ''
        segments.append({**span, 'src_start': start, 'src_duration': length,
                         'evidence_start': shot['start'], 'evidence_end': shot['end'],
                         'playback_speed': length / span['dur'], 'freeze_duration': 0.,
                         'role': role,
                         'motion': choose_motion(role, len(segments) + 1, span['dur']),
                         'shot_part': len(segments) + 1, 'context_id': span['shot_id']})
    segments = apply_comment_picture(segments, comments, mode, preface)
    bindings = [{'sentence_index': choice['sentence_index'], 'initial_match_signature': signature,
                 'source_windows': [{'shot_id': beat['shot_id'], 'start': lookup[beat['shot_id']]['start'],
                                     'end': lookup[beat['shot_id']]['end'], 'target_text': beat['target_text']}
                                    for beat in choice['beats']]} for choice in choices]
    variant.scene_matches = bindings
    binding_signature = _hash({'semantic_signature': signature, 'clock': result.get('word_timings'),
                               'sentence_timings': timings, 'comments': comments, 'duration': duration})
    return {'source_start': candidate.source_start, 'source_duration': candidate.source_end - candidate.source_start,
            'output_duration': duration, 'narration_duration': duration,
            'tts_duration': duration, 'voice_path': result['audio_path'], 'speed_factor': 1.,
            'voice_playback_speed': result.get('playback_speed', 1.), 'timing_source': result.get('timing_source', 'estimated'),
            'alignment_report': result.get('alignment_report') or {}, 'sentence_timings': timings,
            'segments': segments, 'tail_padding': 0.,
            'initial_match': {'policy': POLICY, 'signature': signature, 'binding_signature': binding_signature,
                              'cache_hit': hit, 'clock_source': clock_source, 'choices': choices,
                              'source_times_basis': 'real_cut_catalogue', 'model_time_values_accepted': False},
            'comment_picture_plan': {'mode': mode, 'comments': comments, 'literal_matching': False,
                                     'context_matching': True, 'position_policy': 'exact_comment_intervals'}}
