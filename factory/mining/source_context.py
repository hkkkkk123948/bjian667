"""Reuse original narration for chapter/identity context, never picture evidence."""
import hashlib
import math
import sys
from pathlib import Path

from ..utils import read_json, write_json, run_cmd, ffmpeg_bin, ensure_dir, media_duration

POLICY = 'original_audio_story_context_v1'


def source_digest(p):
    path = Path(p.state.artifacts.get('video_path', ''))
    if not path.is_file():
        return ''
    stat = path.stat()
    signature = [str(path.resolve()), stat.st_size, stat.st_mtime_ns]
    stored = p.state.artifacts.get('source_context_fingerprint') or {}
    if stored.get('stat') == signature:
        return stored['sha256']
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for data in iter(lambda: stream.read(1024*1024), b''):
            digest.update(data)
    value = digest.hexdigest()
    p.state.artifacts['source_context_fingerprint'] = {'stat': signature, 'sha256': value}
    return value


def _segments(raw):
    values = raw.get('segments', []) if isinstance(raw, dict) else raw if isinstance(raw, list) else []
    out = []
    for item in values if isinstance(values, list) else []:
        if not isinstance(item, dict):
            continue
        try:
            start, end = float(item['start']), float(item['end'])
        except (KeyError, TypeError, ValueError):
            continue
        text = str(item.get('text', '')).strip()
        if text and math.isfinite(start+end) and 0 <= start < end:
            out.append({'start': start, 'end': end, 'text': text})
    return sorted(out, key=lambda item: item['start'])


def cache_source_context(p, segments, language='en', origin='local_asr'):
    """Caller transcribes once locally; this function performs no model call."""
    data = {'policy': POLICY, 'source_sha256': source_digest(p), 'language': language, 'origin': origin,
            'basis': 'original_narration_context_not_visual_evidence', 'segments': _segments(segments)}
    path = Path(p.job_dir) / 'analysis' / 'source_context' / 'transcript.json'
    write_json(path, data)
    if data['source_sha256']:
        write_json(p.config.workspace_path() / '.cache' / 'source_context' / (data['source_sha256']+'.json'), data)
    p.state.artifacts['source_transcript_path'] = str(path.resolve())
    p.save_state()
    return str(path.resolve())


def source_context(p, lo, hi, max_chars=3000):
    """Pure cached read, bounded to the current source window."""
    digest = source_digest(p)
    explicit = p.state.artifacts.get('source_transcript_path') or p.config.param('source_transcript_path', '')
    choices = [Path(explicit)] if explicit else []
    choices.append(Path(p.job_dir) / 'analysis' / 'source_context' / 'transcript.json')
    if digest:
        choices.append(p.config.workspace_path() / '.cache' / 'source_context' / (digest+'.json'))
    values, origin = [], 'not_available'
    for path in choices:
        raw = read_json(path)
        if isinstance(raw, dict) and raw.get('source_sha256') and raw['source_sha256'] != digest:
            continue
        values = _segments(raw)
        if values:
            origin = raw.get('origin', 'cached_source_transcript') if isinstance(raw, dict) else 'cached_source_transcript'
            break
    if not values:
        index = getattr(p, 'index', None)
        values = _segments([{'start': seg.start, 'end': seg.end, 'text': seg.text} for seg in getattr(index, 'transcript', [])]) if index else []
        origin = 'existing_source_index' if values else origin
    in_window = [item for item in values if item['end'] > lo and item['start'] < hi]
    total = sum(len(item['text']) for item in in_window)
    selected = []
    if total <= max_chars:
        selected = in_window
    else:
        # Keep both the setup and the real ending. A prefix-only token cap
        # previously removed the arrest/rescue, corrupting the story's cause.
        chosen = {}
        for indices, budget in ((range(len(in_window)), int(max_chars*.3)),
                                (range(len(in_window)-1, -1, -1), int(max_chars*.3))):
            consumed = 0
            for index in indices:
                if index in chosen:
                    continue
                if consumed >= budget:
                    break
                item = in_window[index]
                text = item['text'][:max(1, budget-consumed)]
                chosen[index] = dict(item, text=text)
                consumed += len(text)
        remaining = max_chars-sum(len(item['text']) for item in chosen.values())
        middle = [index for index in range(len(in_window)) if index not in chosen]
        bins = min(8, len(middle))
        for index in [middle[round(i*(len(middle)-1)/max(1,bins-1))] for i in range(bins)]:
            if remaining <= 0:
                break
            text = in_window[index]['text'][:remaining]
            chosen[index] = dict(in_window[index], text=text)
            remaining -= len(text)
        selected = [chosen[index] for index in sorted(chosen)]
    return {'basis': 'original_narration_context_not_visual_evidence', 'origin': origin,
            'segments': selected, 'available': bool(selected), 'truncated': total > max_chars,
            'sampling': 'opening_middle_ending' if total > max_chars else 'complete_window',
            'omitted_middle': total > max_chars}


def ensure_source_context(p, video=None, duration=None):
    """Automatic source subtitles/cache first, then one local CPU read per SHA.

    No remote ASR, checkpoint download, narration editing or GPU reservation.
    The worker exits before vision/rendering, releasing all its model memory.
    """
    video = Path(video or p.state.artifacts.get('video_path', ''))
    if not video.is_file():
        return {'available': False, 'reason': '原片文件不可用'}
    duration = float(duration or media_duration(video))
    available = source_context(p, 0, duration, max_chars=1)
    if available['available']:
        p.emitter.log('复用同源原声/字幕上下文，用于章节、身份和动机理解；不当镜头证据')
        return available
    # Use the index if the normal indexing pipeline already obtained subtitles.
    if not getattr(p, 'index', None) and hasattr(p, '_load_index'):
        p._load_index()
        available = source_context(p, 0, duration, max_chars=1)
        if available['available']:
            return available
    from ..index.transcript import from_subtitle_file
    subtitle = p.state.artifacts.get('subtitle_path') or (p.state.artifacts.get('meta') or {}).get('subtitle_path') or ''
    choices = [Path(subtitle)] if subtitle else []
    choices.extend(video.with_suffix(extension) for extension in ('.srt', '.vtt', '.json3', '.ass'))
    for path in choices:
        if not path.is_file():
            continue
        rows = from_subtitle_file(path)
        if rows:
            cache_source_context(p, [{'start': row.start, 'end': row.end, 'text': row.text} for row in rows],
                                 language='source_subtitles', origin='original_subtitles')
            return source_context(p, 0, duration, max_chars=1)
    if str(p.config.param('source_context_auto', 'on')).lower() in ('off', 'false', '0'):
        return {'available': False, 'reason': '原声上下文自动读取已关闭'}
    requested = str(p.config.param('source_context_model', 'tiny'))
    names = list(dict.fromkeys([requested, 'tiny.en', 'tiny', 'base.en', 'base', 'small.en', 'small']))
    roots = [Path.home()/'.cache'/'whisper', Path(__file__).resolve().parents[2]/'whisper-models']
    checkpoint = next((root/(name+'.pt') for name in names for root in roots if
                       name in ('tiny', 'tiny.en', 'base', 'base.en', 'small', 'small.en') and (root/(name+'.pt')).is_file()), None)
    digest = source_digest(p)
    marker = p.config.workspace_path()/'.cache'/'source_context'/(digest+'_attempt.json')
    identity = {'policy': POLICY, 'checkpoint': str(checkpoint) if checkpoint else '',
                'checkpoint_mtime': checkpoint.stat().st_mtime_ns if checkpoint else None,
                'language': p.config.param('source_context_language', 'auto')}
    previous = read_json(marker) or {}
    if previous.get('identity') == identity and previous.get('attempted'):
        return {'available': False, 'reason': previous.get('reason', '同源本地原声已尝试，未重复读取')}
    if checkpoint is None:
        reason = '没有已下载的本地小型Whisper模型，原声上下文不可用；不自动下载，仍依据真实镜头'
        write_json(marker, {'identity': identity, 'attempted': True, 'reason': reason})
        p.emitter.log(reason, 'warning')
        return {'available': False, 'reason': reason}
    check_cancel = getattr(p, '_check_cancel', lambda: None)
    check_cancel()
    root = ensure_dir(Path(p.job_dir)/'analysis'/'source_context')
    audio, output = root/'original_16k.wav', root/'local_asr_raw.json'
    p.emitter.log('读取原片原声上下文：本地%s、CPU两线程、同源只读取一次；不调用收费接口' % checkpoint.stem)
    try:
        run_cmd([ffmpeg_bin(), '-v', 'error', '-y', '-threads', '2', '-i', str(video), '-vn', '-ac', '1', '-ar', '16000', str(audio)],
                timeout=max(180, int(duration*2)))
        check_cancel()
        run_cmd([sys.executable, '-m', 'factory.mining.source_context_worker', '--checkpoint', str(checkpoint),
                 '--audio', str(audio), '--output', str(output), '--language', str(identity['language'])],
                timeout=max(240, int(duration*8)))
        check_cancel()
        raw = read_json(output) or {}
        rows = _segments(raw)
        if not rows:
            raise ValueError('原声未识别到可靠讲述，可能只有音乐或音效')
        cache_source_context(p, rows, language=raw.get('language', 'auto'), origin='local_whisper_'+checkpoint.stem)
        write_json(marker, {'identity': identity, 'attempted': True, 'available': True})
        p.emitter.log('原声上下文已保存%d段，同源新任务可复用；本地模型进程已退出释放内存' % len(rows))
        return source_context(p, 0, duration, max_chars=1)
    except Exception as exc:
        check_cancel()
        reason = '原声上下文暂不可用：'+str(exc)[:180]+'；继续以真实镜头为依据，不伪造上下文'
        write_json(marker, {'identity': identity, 'attempted': True, 'reason': reason})
        p.emitter.log(reason, 'warning')
        return {'available': False, 'reason': reason}
    finally:
        audio.unlink(missing_ok=True)
