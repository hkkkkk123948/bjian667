"""Real source cuts and individually identified frames, never model-made times.

The visual reader observes a small batch of six separate images. Shot IDs,
frame IDs and source bounds are created locally, and the reader can only refer
to those IDs. This is observation for authoring/matching, not a script critic.
"""
import copy
import hashlib
import json
import math
import re
from pathlib import Path

from ..utils import ensure_dir, read_json, write_json, run_cmd, ffmpeg_bin, media_duration
from ..providers.base import ProviderEmptyResponseError, ProviderError

POLICY = 'actual_cut_individual_frames_v1'


def _source_signature(video):
    path = Path(video)
    stat = path.stat()
    digest = hashlib.sha256(('%d:%d' % (stat.st_size, stat.st_mtime_ns)).encode())
    with path.open('rb') as stream:
        for offset in (0, max(0, stat.st_size//2-65536), max(0, stat.st_size-131072)):
            stream.seek(offset)
            digest.update(stream.read(131072))
    return digest.hexdigest()


def _detect_cuts(video, lo, hi, threshold=.22):
    """FFmpeg timestamps are relative to the accurately decoded seek window."""
    _, err, _ = run_cmd([ffmpeg_bin(), '-hide_banner', '-loglevel', 'info', '-threads', '1',
        '-ss', str(lo), '-i', str(video), '-t', str(hi-lo), '-an', '-sn', '-filter_threads', '1',
        '-vf', "scale=320:-2,select='gt(scene,%.4f)',showinfo" % threshold, '-f', 'null', '-'],
        timeout=max(180, int((hi-lo)*3)))
    times = [lo + float(value) for value in re.findall(r'pts_time:\s*([\d.]+)', err)]
    return sorted({round(time, 6) for time in times if lo+.15 < time < hi-.15})


def build_local_index(video, lo, hi, folder, check_cancel=lambda: None, width=384, max_span=8.):
    root = ensure_dir(Path(folder) / 'shot_evidence')
    duration = media_duration(video)
    lo, hi = max(0., float(lo)), min(duration, float(hi))
    if not math.isfinite(lo+hi) or hi-lo < .15:
        raise ValueError('实际分镜源范围无效')
    source = _source_signature(video)
    settings = {'policy': POLICY, 'source_signature': source, 'source_start': lo, 'source_end': hi,
                'width': width, 'max_span': max_span}
    saved = read_json(root / 'index.json')
    if isinstance(saved, dict) and saved.get('settings') == settings and all(
            Path(frame['path']).is_file() for shot in saved.get('shots', []) for frame in shot['frames']):
        return saved
    check_cancel()
    cuts = _detect_cuts(video, lo, hi)
    points = [lo] + cuts + [hi]
    shots = []
    for parent, (start, end) in enumerate(zip(points, points[1:]), 1):
        # A long continuous camera shot can contain several event phases. Local
        # subwindows retain their parent cut ID; they are not new story chapters.
        parts = max(1, math.ceil((end-start)/max_span))
        for part in range(parts):
            a, b = start+(end-start)*part/parts, start+(end-start)*(part+1)/parts
            if b-a < .15:
                continue
            number = len(shots)+1
            sid = 'SH%04d' % number
            frames = []
            for suffix, position, fraction in [('A', 'first', .08), ('B', 'middle', .5), ('C', 'last', .92)]:
                check_cancel()
                time = a+(b-a)*fraction
                path = root / 'frames' / (sid+'_'+suffix+'.jpg')
                ensure_dir(path.parent)
                run_cmd([ffmpeg_bin(), '-v', 'error', '-y', '-threads', '1', '-ss', '%.6f' % time,
                         '-i', str(video), '-frames:v', '1', '-vf', 'scale=%d:-2' % width,
                         '-filter_threads', '1', str(path)], timeout=90)
                if not path.is_file():
                    raise RuntimeError('实际分镜独立抽帧失败：' + sid)
                frames.append({'frame_id': sid+'_'+suffix, 'position': position, 'time': round(time, 6), 'path': str(path.resolve())})
            shots.append({'shot_id': sid, 'start': round(a, 6), 'end': round(b, 6),
                          'parent_cut_id': 'CUT%04d' % parent, 'continuous_part': part+1,
                          'frames': frames})
    result = {'settings': settings, 'cuts': cuts, 'shots': shots, 'cut_method': 'local_ffmpeg_scene_score',
              'time_policy': 'all_bounds_and_frame_times_assigned_by_code'}
    write_json(root / 'index.json', result)
    return result


def _normalize_observations(raw, shots, registry):
    if not isinstance(raw, dict) or not isinstance(raw.get('shots'), list) or len(raw['shots']) != len(shots):
        raise ValueError('实际分镜观察需逐个返回本批shot_id，不允许缺图或猜时间')
    catalog = {shot['shot_id']: shot for shot in shots}
    people = copy.deepcopy(registry)
    for person in raw.get('characters', []):
        if not isinstance(person, dict) or not re.fullmatch(r'P\d{2,3}', str(person.get('id', ''))):
            raise ValueError('人物需稳定Pxx编号，不能用衣服颜色代替身份')
        identity = person['id']
        label, features = str(person.get('label', '')).strip(), str(person.get('features', '')).strip()
        if not label or not features:
            raise ValueError('人物编号缺少稳定称呼或实际外貌依据')
        if identity not in people:
            people[identity] = {'id': identity, 'label': label[:40], 'features': features[:180]}
        # Later clothes or camera angle do not rename the same person's label.
    seen, observed = set(), []
    for item in raw['shots']:
        if not isinstance(item, dict) or item.get('shot_id') not in catalog or item['shot_id'] in seen:
            raise ValueError('分镜观察包含未知或重复shot_id')
        if any(key in item for key in ('start', 'end', 'source_start', 'source_end', 'source_windows')):
            raise ValueError('视觉模型不能创建源片秒数，必须只引用代码shot/frame编号')
        sid = item['shot_id']
        seen.add(sid)
        shot = catalog[sid]
        frame_catalog = {frame['frame_id']: frame for frame in shot['frames']}
        visible = str(item.get('visible', '')).strip()
        ids = item.get('characters', [])
        if not isinstance(ids, list):
            ids = []
        if not visible or any(not isinstance(pid, str) or pid not in people for pid in ids):
            raise ValueError('分镜观察缺少实际内容或引用了未声明人物')
        observations = item.get('observations')
        frames_seen, description = set(), []
        # Prefer explicit per-frame notes; if the model only gave a shot summary,
        # replicate that visible text onto each coded frame so production can continue.
        usable = []
        if isinstance(observations, list):
            for observation in observations:
                if not isinstance(observation, dict):
                    continue
                fid = observation.get('frame_id')
                note = str(observation.get('visible', '')).strip()
                if fid in frame_catalog and fid not in frames_seen and note:
                    usable.append((fid, note))
                    frames_seen.add(fid)
        if len(usable) != len(frame_catalog):
            if not visible:
                raise ValueError('每个分镜需分别描述首、中、末真实帧，不能用总体印象代替动作位置')
            description = [{'frame_id': frame['frame_id'], 'position': frame['position'],
                            'time': frame['time'], 'visible': visible[:220]}
                           for frame in shot['frames']]
        else:
            for fid, note in usable:
                frame = frame_catalog[fid]
                description.append({'frame_id': fid, 'position': frame['position'], 'time': frame['time'],
                                    'visible': note[:220]})
        actions = []
        for action in item.get('actions', []) or []:
            refs = action.get('frame_ids') if isinstance(action, dict) else None
            if (not isinstance(refs, list) or not refs or any(not isinstance(fid, str) or fid not in frame_catalog for fid in refs)
                    or not str(action.get('text', '')).strip()):
                continue
            actions.append({'text': str(action['text'])[:180], 'frame_ids': refs,
                            'anchor_times': [frame_catalog[fid]['time'] for fid in refs]})
        if not actions:
            mid = shot['frames'][len(shot['frames']) // 2]['frame_id']
            actions = [{'text': visible[:180], 'frame_ids': [mid],
                        'anchor_times': [frame_catalog[mid]['time']]}]
        observed.append(dict(shot, visible=visible[:300], characters=ids, actions=actions,
                             observations=description, uncertain=list(item.get('uncertain') or []),
                             source_windows=[{'start': shot['start'], 'end': shot['end'], 'weight': 1}],
                             evidence_policy=POLICY))
    return observed, people


def build_and_observe(p, video, lo, hi, folder, role=None):
    """Create new evidence in an isolated/new job; never modify a user script."""
    check_cancel = getattr(p, '_check_cancel', lambda: None)
    local = build_local_index(video, lo, hi, folder, check_cancel,
                              width=int(p.config.param('shot_evidence_width', 384)))
    root = ensure_dir(Path(folder) / 'shot_evidence')
    role = role or p.config.param('shot_evidence_role', 'video_analysis')
    provider = p.router.get(role)
    if provider.is_mock:
        raise ValueError('实际分镜需要配置可读取独立图片的视觉模型')
    from .source_context import source_context
    context = source_context(p, lo, hi, max_chars=6000)
    context_signature = hashlib.sha256(json.dumps(context, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
    identity = {'model': provider.model, 'base_url': provider.conf.get('base_url', ''), 'policy': POLICY,
                'source_context_signature': context_signature}
    saved = read_json(root / 'catalog.json') or {}
    if saved.get('settings') == local['settings'] and saved.get('reader') == identity:
        return saved['shots']
    registry, observed, stats = {}, [], {'calls': 0, 'cache_hits': 0, 'total_tokens': 0}
    prior_progress = read_json(root / 'progress.json') or {}
    if prior_progress.get('settings') == local['settings'] and prior_progress.get('reader') == identity:
        stats.update(prior_progress.get('usage') or {})
    for offset in range(0, len(local['shots']), 2):
        check_cancel()
        batch = local['shots'][offset:offset+2]
        images = [frame['path'] for shot in batch for frame in shot['frames']]
        mapping = [{'image_number': i+1, 'shot_id': shot['shot_id'], 'frame_id': frame['frame_id'], 'position': frame['position']}
                   for i, (shot, frame) in enumerate((shot, frame) for shot in batch for frame in shot['frames'])]
        prompt = ('只观察所附独立真实动画帧，不编故事、不审文案。图片顺序和shot/frame编号由软件提供，不猜源片秒数。'
                  '每个shot分别看首、中、末帧，写谁在哪里做什么；动作只能由真正看见的帧支撑，记录动作对应frame_ids。'
                  '同伴与陌生怪人必须区分，穿相似衣服不等于同一个人；不把帽子、警车、狗、警方或追逐移到另一镜头。'
                  '人物用稳定Pxx ID，已有ID不改称呼，新人加入characters并写脸形/发型等可见依据；不确定身份写uncertain。'
                  '称呼优先男孩、女孩、同伴、陌生人/怪人、帮忙男人、警察等；只有动作与相邻帧明确支持才确认关系。'
                  '这些是称呼示例，成年主角称男子/男人，不要一律写男孩；原声的回忆年龄可帮助区分当前讲述者与回忆中的小孩。'
                  '衣服颜色和帽子不是剧情，昏暗帧看不清就不写颜色，不依据旁白或字幕猜。狗与部分人的头必须按可见形态区分。'
                  '总体visible最多70字概括真实动作变化，observations每图最多45字描述当前姿态，features最多60字；没看到完整动作只写姿态，不补脑掀被、挥刀或抓捕。'
                  '仅返回JSON {"characters":[{"id":"P01","label":"同伴","features":"可见稳定特征"}],'
                  '"shots":[{"shot_id":"SH0001","visible":"真实事件", "characters":["P01"],'
                  '"observations":[{"frame_id":"SH0001_A","visible":"首帧实际内容"}],'
                  '"actions":[{"text":"具体可见动作","frame_ids":["SH0001_B","SH0001_C"]}],"uncertain":[]}]}。'
                  'shots必须列本批每个shot，observations必须列其全部真实帧。禁止返回任何start/end/source_windows。'
                  '\n本批原片原声背景（仅助身份/关系/动作目的理解，不是可见动作证据，不得抄成事实；只在图片确实看见时记录动作）：' +
                  json.dumps(source_context(p,batch[0]['start'],batch[-1]['end'],max_chars=600),ensure_ascii=False,separators=(',', ':')) +
                  '\n已有稳定人物：' + json.dumps(registry, ensure_ascii=False, separators=(',', ':')) +
                  '\n真实图片映射：' + json.dumps(mapping, ensure_ascii=False, separators=(',', ':')))
        signature = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()+prompt.encode())
        for image in images:
            signature.update(Path(image).read_bytes())
        cache = p.config.workspace_path() / '.cache' / POLICY / (signature.hexdigest()+'.json')
        raw = read_json(cache)
        if raw is not None:
            try:
                _normalize_observations(raw, batch, registry)
            except ValueError:
                # A stale/corrupt observation is not a permanent successful
                # cache entry; re-read only this batch.
                raw = None
                cache.unlink(missing_ok=True)
        if raw is None:
            p.emitter.log('读取真实分镜%d–%d/%d：%d张独立首中末帧，源时间由软件绑定' %
                          (offset+1, offset+len(batch), len(local['shots']), len(images)))
            try:
                raw = _read_batch(p, provider, prompt, images, batch, registry, stats,
                                  batch_key=signature.hexdigest())
                write_json(cache, raw)
            except ProviderError as exc:
                write_json(root / 'progress.json', {'settings': local['settings'], 'reader': identity,
                    'shots_observed': len(observed), 'shots_total': len(local['shots']),
                    'characters': registry, 'usage': stats, 'complete': False,
                    'pending_shot_ids': [shot['shot_id'] for shot in batch],
                    'pending_error': {'type': type(exc).__name__, 'model': provider.model,
                                      'diagnostics': getattr(exc, 'diagnostics', {})}})
                raise
        else:
            stats['cache_hits'] += 1
        rows, registry = _normalize_observations(raw, batch, registry)
        observed.extend(rows)
        write_json(root / 'progress.json', {'settings': local['settings'], 'reader': identity,
                   'shots_observed': len(observed), 'shots_total': len(local['shots']), 'characters': registry,
                   'usage': stats, 'complete': len(observed) == len(local['shots'])})
    result = {'policy': POLICY, 'settings': local['settings'], 'reader': identity, 'characters': registry,
              'shots': observed, 'usage': stats, 'time_policy': 'code_bound_shots_and_actual_frame_ids'}
    write_json(root / 'catalog.json', result)
    return observed


def _read_batch(p, provider, prompt, images, batch, registry, stats, batch_key=""):
    """One normal request, at most one changed request for proven budget exhaustion."""
    previous = {key: provider.conf[key] for key in ('max_tokens', 'reasoning_effort') if key in provider.conf}
    old_retry = provider.max_retry
    recovered_batches = stats.setdefault('budget_recovery_batches', [])
    known_recovery = bool(batch_key and batch_key in recovered_batches)
    configured_budget = int(provider.conf.get('max_tokens') or 1900)
    provider.conf['max_tokens'], provider.max_retry = (
        3800 if known_recovery else min(3800, max(1900, configured_budget))), 1
    if known_recovery and provider.conf.get('reasoning_effort') in ('medium', 'high', 'xhigh', 'max'):
        provider.conf['reasoning_effort'] = 'low'

    def request():
        try:
            return provider.complete_json(prompt, images=images, json_mode=True, default=None)
        finally:
            usage = getattr(provider, 'last_call_usage', None)
            stats['calls'] += int(getattr(provider, 'last_call_requests', 1))
            stats['total_tokens'] += int((usage if usage is not None else getattr(provider, 'last_usage', {})).get('total_tokens', 0))

    try:
        try:
            raw = request()
        except ProviderEmptyResponseError as exc:
            diagnostics = stats.setdefault('empty_responses', [])
            diagnostics.append(dict(exc.diagnostics))
            del diagnostics[:-4]
            if not exc.budget_exhausted or known_recovery:
                raise
            getattr(p, '_check_cancel', lambda: None)()
            # The server reported length exhaustion with reasoning tokens and
            # no final output. Give final JSON room once; do not replay the
            # original request or substitute an unconfigured model.
            provider.conf['max_tokens'] = 3800
            if batch_key:
                recovered_batches.append(batch_key)
            if provider.conf.get('reasoning_effort') in ('medium', 'high', 'xhigh', 'max'):
                provider.conf['reasoning_effort'] = 'low'
            stats['budget_recoveries'] = stats.get('budget_recoveries', 0) + 1
            p.emitter.log('视觉接口推理耗尽输出预算、未返回最终内容；仅尝试一次调整预算后继续，已完成批次保留', 'warning')
            try:
                raw = request()
            except ProviderEmptyResponseError as retry_exc:
                diagnostics.append(dict(retry_exc.diagnostics))
                del diagnostics[:-4]
                raise
        try:
            _normalize_observations(raw, batch, registry)
        except ValueError as exc:
            raise ProviderError('实际分镜接口返回无效观察：%s；已保留完成批次，可继续未完成部分' % exc) from exc
        return raw
    finally:
        provider.max_retry = old_retry
        for key in ('max_tokens', 'reasoning_effort'):
            if key in previous:
                provider.conf[key] = previous[key]
            else:
                provider.conf.pop(key, None)


def get_shot_catalog(p, candidate):
    """Pure read: old guessed visual_facts are never promoted to real shots."""
    try:
        number = int(candidate.candidate_id[1:])
    except (TypeError, ValueError):
        return []
    path = Path(p.job_dir) / 'analysis' / 'director' / ('story_%02d' % number) / 'shot_evidence' / 'catalog.json'
    catalog = read_json(path) or {}
    settings = catalog.get('settings', {})
    if (catalog.get('policy') != POLICY or settings.get('source_start') != candidate.source_start or
            settings.get('source_end') != candidate.source_end):
        return []
    video = p.state.artifacts.get('video_path') if getattr(p, 'state', None) else None
    if video and Path(video).is_file() and settings.get('source_signature') != _source_signature(video):
        return []
    from .source_context import source_context
    context_signature = hashlib.sha256(json.dumps(source_context(p,candidate.source_start,candidate.source_end,max_chars=6000),
                                                  ensure_ascii=False,sort_keys=True).encode()).hexdigest()
    if catalog.get('reader', {}).get('source_context_signature') != context_signature:
        return []
    return catalog.get('shots', []) if isinstance(catalog.get('shots'), list) else []


def authoring_facts(shots):
    return [{'id': shot['shot_id'], 'start': shot['start'], 'end': shot['end'], 'visible': shot['visible'],
             'characters': shot['characters'], 'actions': shot['actions'], 'observations': shot['observations']}
            for shot in shots]
