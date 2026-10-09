"""Repair concrete picture mismatches without rewriting a syllable of narration.

Only existing, independently observed facts or already indexed keyframes are
eligible. The model chooses IDs; Python supplies every source time and checks
that the new picture does not borrow an unaffected sentence's footage.
"""
import copy
import hashlib
import json
import math
from pathlib import Path

from ..models import Evidence
from ..script.comment_hook import story_text
from ..utils import ensure_dir, read_json, write_json, tokenize

POLICY = 'approved_body_visual_rematch_v1'


def missing_evidence(variant):
    """Targets to match after editing; absence of pictures never rejects text."""
    flagged = {match.get('sentence_index') for match in getattr(variant, 'scene_matches', None) or []
               if isinstance(match, dict) and match.get('needs_rematch')}
    out, used = [], []
    registered = _effective_windows(variant, 0, float('inf'))
    for sentence in variant.sentences:
        ev = sentence.source_evidence
        try:
            valid = ev is not None and math.isfinite(ev.start+ev.end) and ev.start >= 0 and ev.end-ev.start >= .15
        except (TypeError, ValueError):
            valid = False
        windows = registered.get(sentence.index, [])
        duplicate = any(_overlaps(window, old) for window in windows for old in used)
        if not valid or sentence.index in flagged or duplicate:
            out.append({'sentence_index': sentence.index,
                        'reason': ('这句与其他正文重复绑定同一源片区间，需要重新分配可用镜头' if duplicate else
                                   '人工修改后的这一句尚未绑定可用镜头，需要在当前故事中重新匹配'),
                        'evidence': ('当前登记源片窗口与前句重复；正文已确认，不得修改正文' if duplicate else
                                     '保存的镜头证据为空或已标记needs_rematch；正文已确认，不得修改正文')})
        else:
            used.extend(windows)
    return out


def rematch_missing(p, cand, variant, clip):
    """Call immediately before build_timeline, using the untouched whole voice."""
    targets = missing_evidence(variant)
    result = rematch(p, cand, variant, clip, targets)
    result['missing_sentence_indices'] = [item['sentence_index'] for item in missing_evidence(variant)]
    return result


def _window(item, lo, hi):
    if not isinstance(item, dict):
        return None
    try:
        start, end = float(item['start']), float(item['end'])
    except (KeyError, TypeError, ValueError):
        return None
    if not math.isfinite(start+end) or not lo <= start < end <= hi:
        return None
    return {'start': start, 'end': end}


def _overlaps(first, second):
    return min(first['end'], second['end']) - max(first['start'], second['start']) > .08


def _fact_catalogue(p, candidate):
    try:
        number = int(candidate.candidate_id[1:])
    except (TypeError, ValueError):
        return []
    folder = Path(p.job_dir) / 'analysis' / 'director' / ('story_%02d' % number)
    saved = read_json(folder / 'visual_facts.json') or {}
    lo, hi = candidate.source_start, candidate.source_end
    catalog = []
    rows = saved.get('facts', []) if isinstance(saved, dict) else []
    for item in rows if isinstance(rows, list) else []:
        window = _window(item, lo, hi)
        if not window or not isinstance(item.get('id'), str) or not str(item.get('visible', '')).strip():
            continue
        catalog.append({'id': item['id'], **window, 'visible': str(item['visible'])[:180], 'basis': 'independent_cached_fact'})
    if catalog:
        return _unique_facts(catalog)
    # Legacy indexes can supply real pictures, but unproved writer labels alone
    # are not turned into an authoritative alternative shot catalogue.
    index = getattr(p, 'index', None) or p._load_index()
    for scene in getattr(index, 'scenes', []) if index else []:
        window = _window({'start': scene.start, 'end': scene.end}, lo, hi)
        frames = _existing_frames(getattr(scene, 'keyframes', []))
        if window and frames:
            catalog.append({'id': 'S%d' % scene.scene_id, **window, 'visible': str(scene.visual or '已索引原片镜头')[:180],
                            'basis': 'indexed_actual_keyframe', 'images': frames, 'scene_id': scene.scene_id})
    return _unique_facts(catalog)


def _unique_facts(catalog):
    out, used = [], set()
    for fact in sorted(catalog, key=lambda item: (item['start'], item['id'])):
        if fact['id'] not in used:
            out.append(fact)
            used.add(fact['id'])
    return out


def _existing_frames(frames):
    out = []
    for item in frames if isinstance(frames, list) else []:
        value = item.get('path') if isinstance(item, dict) else item
        if isinstance(value, str) and Path(value).is_file():
            out.append(value)
    return out


def _effective_windows(variant, lo, hi):
    mapped = {}
    for match in getattr(variant, 'scene_matches', None) or []:
        if not isinstance(match, dict) or type(match.get('sentence_index')) is not int:
            continue
        rows = match.get('source_windows') or []
        windows = [_window(window, lo, hi) for window in rows if isinstance(window, dict)] if isinstance(rows, list) else []
        valid = [window for window in windows if window]
        if valid:
            mapped[match['sentence_index']] = valid
    for sentence in variant.sentences:
        ev = sentence.source_evidence
        if sentence.index not in mapped and ev:
            window = _window({'start': ev.start, 'end': ev.end}, lo, hi)
            if window:
                mapped[sentence.index] = [window]
    return mapped


def _occupied_windows(clip, variant, targets, current, lo, hi):
    timeline = read_json(Path(clip.path).parent / 'timeline.json') if getattr(clip, 'path', '') else None
    timings = timeline.get('sentence_timings', []) if isinstance(timeline, dict) else []
    valid_body = (isinstance(timings, list) and len(timings) == len(variant.sentences) and all(
        isinstance(timing, dict) and timing.get('text') == sent.text for timing, sent in zip(timings, variant.sentences)))
    actual = []
    covered = set()
    if valid_body:
        rows = timeline.get('segments') or []
        for segment in rows if isinstance(rows, list) else []:
            number = segment.get('sentence_index') if isinstance(segment, dict) else None
            if type(number) is not int or number in targets:
                continue
            try:
                start = float(segment['src_start'])
                end = start + float(segment['src_duration'])
            except (KeyError, ValueError, TypeError):
                continue
            window = _window({'start': start, 'end': end}, lo, hi)
            if window:
                actual.append(window)
                covered.add(number)
    # Old/missing timelines fall back to the conservative registered windows.
    actual.extend(window for number, windows in current.items() if number not in targets and number not in covered for window in windows)
    return actual


def _free_facts(catalog, occupied, minimum=.35):
    out = []
    for fact in catalog:
        intervals = [(fact['start'], fact['end'])]
        for used in occupied:
            remaining = []
            for start, end in intervals:
                if used['end'] <= start or used['start'] >= end:
                    remaining.append((start, end))
                else:
                    if used['start'] > start:
                        remaining.append((start, min(end, used['start'])))
                    if used['end'] < end:
                        remaining.append((max(start, used['end']), end))
            intervals = remaining
        for number, (start, end) in enumerate(intervals, 1):
            if end-start < minimum:
                continue
            full = abs(start-fact['start']) < .001 and abs(end-fact['end']) < .001
            out.append(dict(fact, id=fact['id'] if full else '%s_free%d' % (fact['id'], number),
                            original_fact_id=fact['id'], start=start, end=end))
    return out


def _result(status, reason, **extra):
    return {'status': status, 'changed': status == 'changed', 'reason': reason, 'policy': POLICY, **extra}


def _identity(candidate, variant, mismatches, windows, catalog, provider):
    payload = {'policy': POLICY, 'candidate': candidate.candidate_id, 'variant': variant.variant_id,
               'source_span': [candidate.source_start, candidate.source_end], 'full_text': variant.full_text,
               'sentences': [{'index': sent.index, 'text': sent.text, 'narrative_text': getattr(sent, 'narrative_text', ''),
                              'comment_hooks': getattr(sent, 'comment_hooks', []),
                              'evidence': [sent.source_evidence.start, sent.source_evidence.end] if sent.source_evidence else None}
                             for sent in variant.sentences],
               'mismatches': mismatches, 'current_windows': windows, 'facts': catalog,
               'model': provider.model, 'base_url': provider.conf.get('base_url', '')}
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def _validate_response(response, targets, catalog, current, occupied):
    if not isinstance(response, dict) or not isinstance(response.get('replacements'), list) or not isinstance(response.get('unavailable'), list):
        raise ValueError('重新选镜头未返回合法替换表，保留现有画面')
    by_id = {fact['id']: fact for fact in catalog}
    changed, absent, seen = [], [], set()
    used = list(occupied)
    for row in response['unavailable']:
        if not isinstance(row, dict) or type(row.get('sentence_index')) is not int or row['sentence_index'] not in targets or row['sentence_index'] in seen:
            raise ValueError('未找到镜头项需对应一个尚未列出的指定句')
        if not str(row.get('reason', '')).strip():
            raise ValueError('未找到镜头项需说明原因')
        seen.add(row['sentence_index'])
        used.extend(current.get(row['sentence_index'], []))
        absent.append({'sentence_index': row['sentence_index'], 'reason': str(row['reason'])[:180]})
    for row in response['replacements']:
        if not isinstance(row, dict) or set(row) - {'sentence_index', 'fact_ids', 'reason'}:
            raise ValueError('替换表只能选择已有fact_ids，不能返回新时间或修改文案')
        number, ids = row.get('sentence_index'), row.get('fact_ids')
        if type(number) is not int or number not in targets or number in seen:
            raise ValueError('替换表修改了未指定句子或重复列出目标句')
        if (not isinstance(ids, list) or not 1 <= len(ids) <= 3 or any(not isinstance(fid, str) for fid in ids)
                or len(set(ids)) != len(ids) or any(fid not in by_id for fid in ids)):
            raise ValueError('替换表包含未知、重复或过多的事实编号')
        chosen = sorted([by_id[fid] for fid in ids], key=lambda fact: fact['start'])
        windows = [{'start': fact['start'], 'end': fact['end']} for fact in chosen]
        for window in windows:
            if any(_overlaps(window, old) for old in used):
                raise ValueError('新镜头与其他句子或同句另一镜头重复，保留原匹配')
            used.append(window)
        seen.add(number)
        same = len(windows) == len(current.get(number, [])) and all(
            abs(new['start']-old['start']) < .001 and abs(new['end']-old['end']) < .001
            for new, old in zip(windows, current.get(number, [])))
        if not same:
            changed.append({'sentence_index': number, 'facts': chosen, 'windows': windows,
                            'reason': str(row.get('reason', ''))[:180]})
    if seen != set(targets):
        raise ValueError('重新选镜头遗漏了指定错误句，未修改现有匹配')
    return changed, absent


def rematch(p, cand, variant, clip, mismatches):
    """One bounded matching request. Never synthesize or edit approved narration."""
    by_sentence = {sentence.index: sentence for sentence in variant.sentences}
    targets, problems = [], []
    for item in mismatches if isinstance(mismatches, list) else []:
        number = item.get('sentence_index') if isinstance(item, dict) else None
        if type(number) is int and number in by_sentence and number not in targets:
            targets.append(number)
            problems.append({'sentence_index': number, 'reason': str(item.get('reason', ''))[:200],
                             'evidence': str(item.get('evidence', ''))[:200]})
    if not targets:
        return _result('unchanged', '没有明确定位到句子的画面错误，保留已确认文案和现有配音')
    catalog = _fact_catalogue(p, cand)
    if not catalog:
        return _result('unavailable', '缺少已缓存的独立镜头事实或实际索引关键帧，不能虚构替代镜头')
    current = _effective_windows(variant, cand.source_start, cand.source_end)
    occupied = _occupied_windows(clip, variant, targets, current, cand.source_start, cand.source_end)
    available = _free_facts(catalog, occupied)
    if not available:
        return _result('unavailable', '当前故事没有未被其他句子使用的替代镜头，保留正文与声音')
    catalogue_total = len(available)
    limit = max(24, min(64, int(p.config.param('visual_rematch_catalogue_limit', 48))))
    if len(available) > limit:
        words = set(tokenize(' '.join(story_text(by_sentence[number]) for number in targets)))
        def relevance(fact):
            tokens = set(tokenize(fact['visible']))
            return len(tokens & words) / max(1, len(tokens | words))
        available = sorted(available, key=lambda fact: (-relevance(fact), fact['start']))[:limit]
        available.sort(key=lambda fact: fact['start'])
    role = str(p.config.param('visual_rematch_role', 'video_analysis'))
    if not p.router or p.router.is_mock(role):
        return _result('unavailable', '画面重新匹配模型未配置，保留现有匹配')
    provider = p.router.get(role)
    signature = _identity(cand, variant, problems, current, available, provider)
    base = Path(clip.path).parent if getattr(clip, 'path', '') else Path(p.job_dir) / 'analysis' / 'visual_rematch'
    cache = ensure_dir(base / 'rematch_cache') / (signature + '.json')
    saved = read_json(cache)
    cached = isinstance(saved, dict) and 'response' in saved
    response = saved.get('response') if cached else None
    if not cached:
        facts = [{key: fact[key] for key in ('id', 'start', 'end', 'visible')} for fact in available]
        images = []
        for fact in available:
            for frame in fact.get('images', [])[:1]:
                if len(images) >= 8:
                    break
                images.append(frame)
                facts[available.index(fact)]['image_number'] = len(images)
        prompt = ('这是画面重新匹配，不是文案审核。正文已确认，绝对不能改词、漏句、改评论话术或声音。'
                  '只针对给定错误句，从当前故事已经观察过的镜头事实选更贴合的fact_ids。'
                  '评论已从匹配目标排除，不找评论字面道具。允许心理/动机/合理概括，用相邻真实动作承接；'
                  '选同一主角的正确事件，避免借用已由其他句占用的源区间。可选目录已排除被占用镜头。'
                  '一个句子可选1至3个按时间顺序的事实，优先一个清楚连续镜头，不为了变化插几帧。'
                  '只选提供的编号，禁止返回source_start/end、source_windows、text或新事实；源时间由软件绑定。'
                  '如果原片没有合适替代，在unavailable解释，不建议重写旁白或编造镜头。'
                  '每个错误句必须且只能出现在一个replacements或unavailable项。'
                  '只返回JSON {"replacements":[{"sentence_index":1,"fact_ids":["F01"],"reason":"镜头对应目标动作"}],'
                  '"unavailable":[{"sentence_index":2,"reason":"没有可用镜头"}]}。目标：' +
                  json.dumps([{'sentence_index': number, 'text': story_text(by_sentence[number]),
                               'current_windows': current.get(number, []), **next(problem for problem in problems if problem['sentence_index'] == number)}
                              for number in targets], ensure_ascii=False, separators=(',', ':')) +
                  '\n可选镜头事实：' + json.dumps(facts, ensure_ascii=False, separators=(',', ':')))
        old_retry, old_tokens = provider.max_retry, provider.conf.get('max_tokens')
        provider.max_retry, provider.conf['max_tokens'] = 1, min(1600, 300+180*len(targets))
        try:
            response = provider.complete_json(prompt, images=images, json_mode=True, default=None)
            _validate_response(response, targets, available, current, occupied)
            write_json(cache, {'policy': POLICY, 'response': response})
        except Exception as exc:
            return _result('unavailable', '画面重新匹配未完成：' + str(exc)[:200])
        finally:
            provider.max_retry = old_retry
            if old_tokens is None:
                provider.conf.pop('max_tokens', None)
            else:
                provider.conf['max_tokens'] = old_tokens
    try:
        changed, absent = _validate_response(response, targets, available, current, occupied)
    except ValueError as exc:
        return _result('unavailable', str(exc), cache_hit=cached)
    if not changed:
        return _result('unchanged', '没有选到不同且可用的替代镜头，停止重复重匹配', unavailable=absent, cache_hit=cached,
                       catalogue_total=catalogue_total, catalogue_provided=len(available))
    replacements = {}
    audit = []
    for replacement in changed:
        number, windows = replacement['sentence_index'], replacement['windows']
        source_windows = [dict(window, weight=1/len(windows)) for window in windows]
        replacements[number] = {'sentence_index': number, 'source_windows': source_windows,
                                'rematch_fact_ids': [fact['id'] for fact in replacement['facts']],
                                'rematch_policy': POLICY}
        audit.append({'sentence_index': number, 'previous_windows': current.get(number, []),
                      'source_windows': source_windows, 'reason': replacement['reason']})
    new_matches = [copy.deepcopy(match) for match in getattr(variant, 'scene_matches', None) or []
                   if not isinstance(match, dict) or match.get('sentence_index') not in replacements]
    new_matches.extend(replacements.values())
    old_matches = variant.scene_matches
    old_evidence = {replacement['sentence_index']: by_sentence[replacement['sentence_index']].source_evidence
                    for replacement in changed}
    try:
        for replacement in changed:
            sentence = by_sentence[replacement['sentence_index']]
            windows = replacement['windows']
            sentence.source_evidence = Evidence(start=min(window['start'] for window in windows),
                                               end=max(window['end'] for window in windows),
                                               scene_ids=[fact['scene_id'] for fact in replacement['facts'] if 'scene_id' in fact])
        variant.scene_matches = new_matches
        p._save_candidates()
    except Exception as exc:
        variant.scene_matches = old_matches
        for number, evidence in old_evidence.items():
            by_sentence[number].source_evidence = evidence
        return _result('unavailable', '替代镜头记录未能保存，已保留原匹配：' + str(exc)[:160], cache_hit=cached)
    result = _result('changed', '只重新匹配%d句画面，已确认正文、评论和整段配音保持原样' % len(changed),
                     changed_sentence_indices=[row['sentence_index'] for row in changed], replacements=audit,
                     unavailable=absent, cache_hit=cached, catalogue_total=catalogue_total, catalogue_provided=len(available))
    write_json(base / 'visual_rematch_result.json', result)
    return result
