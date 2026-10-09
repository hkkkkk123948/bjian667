"""Observe actual chapter cards; filenames and title numbers are never counts."""
import json
from pathlib import Path

import numpy as np
from PIL import Image

from ..index.keyframes import extract_frames
from ..utils import ensure_dir, read_json, write_json
from .source_context import source_digest

POLICY = 'visible_chapter_card_frame_ids_v1'


def _looks_like_card(path):
    with Image.open(path) as image:
        rgb = np.asarray(image.convert('RGB'))
    grey = rgb.mean(axis=2)
    if (grey < 30).mean() < .83:
        return False
    white = (grey > 115) & ((rgb.max(axis=2)-rgb.min(axis=2)) < 45)
    rows, columns = np.nonzero(white)
    if len(rows) < rgb.shape[0]*rgb.shape[1]*.004:
        return False
    height, width = rgb.shape[:2]
    return (columns.max()-columns.min() > width*.43 and rows.max()-rows.min() < height*.28 and
            height*.15 < rows.mean() < height*.85)


def normalize_cards(raw, frames, duration):
    if not isinstance(raw, dict) or not isinstance(raw.get('chapter_cards'), list) or not isinstance(raw.get('ignored'), list):
        raise ValueError('章节卡识别需返回实际frame_id和明确排除项，不能从文件名猜故事数')
    by_id = {frame['frame_id']: frame for frame in frames}
    seen, cards = set(), []
    for item in raw['chapter_cards']:
        ids = item.get('frame_ids') if isinstance(item, dict) else None
        if (not isinstance(ids, list) or not ids or any(not isinstance(fid, str) for fid in ids) or len(set(ids)) != len(ids)
                or any(fid not in by_id or fid in seen for fid in ids) or not str(item.get('title', '')).strip()):
            raise ValueError('章节标题只能引用提供的真实帧，不能猜时间或重复计数')
        times = [by_id[fid]['time'] for fid in ids]
        cards.append({'title': str(item['title'])[:160], 'frame_ids': ids, 'frame_times': times,
                      'card_start': max(0., min(times)-1), 'card_end': min(duration, max(times)+1),
                      'basis': 'visible_title_card', 'boundary_precision': 'sampled_approximately_one_second'})
        seen.update(ids)
    for item in raw['ignored']:
        fid = item.get('frame_id') if isinstance(item, dict) else None
        if fid not in by_id or fid in seen or not str(item.get('reason', '')).strip():
            raise ValueError('非章节图需对应真实帧及原因')
        seen.add(fid)
    if seen != set(by_id):
        raise ValueError('章节识别遗漏真实候选帧，不能静默省略标题卡')
    cards.sort(key=lambda item: item['card_start'])
    for index, card in enumerate(cards):
        card.update(chapter_id='CH%02d' % (index+1), source_start=card['card_end'],
                    source_end=cards[index+1]['card_start'] if index+1 < len(cards) else duration)
    return cards


def cache_verified_cards(p, cards):
    """Only caller-observed actual cards, with explicit real-frame provenance."""
    data = {'policy': POLICY, 'source_sha256': source_digest(p), 'cards': cards, 'basis': 'human_observed_actual_frames'}
    path = p.config.workspace_path() / '.cache' / 'chapter_cards' / (data['source_sha256']+'.json')
    write_json(path, data)
    return str(path.resolve())


def load_chapter_hints(p, video, duration, folder, scout=None, call=None):
    digest = source_digest(p)
    shared = read_json(p.config.workspace_path() / '.cache' / 'chapter_cards' / (digest+'.json')) or {}
    if shared.get('policy') == POLICY and shared.get('source_sha256') == digest:
        return shared['cards']
    root = ensure_dir(Path(folder) / 'chapter_hints')
    saved = read_json(root / 'cards.json') or {}
    if saved.get('policy') == POLICY and saved.get('source_sha256') == digest:
        return saved['cards']
    if not scout or not call or str(p.config.param('source_chapter_scan', 'on')).lower() == 'off':
        return []
    try:
        frames = extract_frames(video, root / 'frames', fps=.5, width=480, prefix='chapter')
        likely = [{'frame_id': 'CARD_%05d' % index, 'time': time, 'path': str(path)}
                  for index, (time, path) in enumerate(frames) if _looks_like_card(path)]
    except Exception as exc:
        p.emitter.log('本地章节卡采样不可用，按人物事件链归并：'+str(exc)[:140], 'warning')
        return []
    # One representative per continuous card, then at most eight independent
    # images per request. No 20-cell sheet whose times can be misassociated.
    groups = []
    for frame in likely:
        if groups and frame['time']-groups[-1][-1]['time'] <= 2.1:
            groups[-1].append(frame)
        else:
            groups.append([frame])
    representatives = [group[len(group)//2] for group in groups]
    cards = []
    for start in range(0, len(representatives), 8):
        batch = representatives[start:start+8]
        prompt = ('只判断所附独立原片帧是否是明确的新故事章节标题卡，不编故事。'
                  '必须看到实际标题文字才列chapter_cards；地图、剧情中看电视/手机文字、广告和片尾不是新故事。'
                  '不参考视频文件名或总标题里的数字。只引用frame_id，不猜开始/结束秒数。'
                  '每个提供帧都必须列为chapter_cards中的frame_ids或ignored中的frame_id。'
                  '返回JSON {"chapter_cards":[{"frame_ids":["CARD_00001"],"title":"实际标题文字"}],'
                  '"ignored":[{"frame_id":"CARD_00002","reason":"不是新故事标题"}]}。帧映射：'+
                  json.dumps([{'frame_id': frame['frame_id'], 'image_number': index+1} for index, frame in enumerate(batch)], separators=(',', ':')))
        raw = call(p, 'chapter_cards', prompt, [frame['path'] for frame in batch], scout)
        observed = normalize_cards(raw, batch, duration)
        for card in observed:
            group = next(group for group in groups if any(frame['frame_id'] in card['frame_ids'] for frame in group))
            card['card_start'] = max(0., group[0]['time']-1)
            card['card_end'] = min(duration, group[-1]['time']+1)
        cards.extend(observed)
    cards.sort(key=lambda item: item['card_start'])
    for index, card in enumerate(cards):
        card.update(chapter_id='CH%02d' % (index+1), source_start=card['card_end'],
                    source_end=cards[index+1]['card_start'] if index+1 < len(cards) else duration)
    data = {'policy': POLICY, 'source_sha256': digest, 'cards': cards, 'basis': 'individual_actual_card_frames'}
    write_json(root / 'cards.json', data)
    return cards
