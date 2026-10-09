"""Cover audience asides with story context, never literal prop searches."""
import copy
from ..script.quality import compact


def comment_timings(variant, timings, tts_result):
    words = tts_result.get("word_timings") or []
    scale = float(tts_result.get("timestamp_duration_scale", 1))
    full = "".join(compact(s.text) for s in variant.sentences)
    exact = bool(words) and "".join(compact(w.get("text", "")) for w in words) == full
    indexed, cursor = [], 0
    for word in words:
        length = len(compact(word.get("text", "")))
        if length:
            indexed.append((cursor,cursor+length,float(word["start"])*scale,float(word["end"])*scale))
            cursor += length

    def point(offset, end=False):
        for a,b,start,stop in indexed:
            if a <= offset < b or (end and offset == b):
                return start + (stop-start)*(offset-a)/(b-a)
        return None

    result, base = [], 0
    for sent, tm in zip(variant.sentences,timings):
        body = getattr(sent,'narrative_text','') or sent.text
        hooks = getattr(sent,'comment_hooks',[]) or ([{"text":sent.comment_hook,"offset":len(body.rstrip("。！？!?")),
                                        "spoken_start":len(compact(body)),"spoken_end":len(compact(sent.text))}]
                                      if getattr(sent,'comment_hook','') else [])
        size = max(1,len(compact(sent.text)))
        search = 0
        for hook in hooks:
            a = hook.get("spoken_start")
            if a is None:
                a = compact(sent.text).find(compact(hook["text"]),search)
            a = max(0,a)
            b = a+len(compact(hook["text"]))
            search = b
            start,stop = (point(base+a),point(base+b,True)) if exact else (None,None)
            if start is None or stop is None:
                start = tm["start"]+(tm["end"]-tm["start"])*a/size
                stop = tm["start"]+(tm["end"]-tm["start"])*b/size
            clock_source = tts_result.get('timing_source','provider_word_timestamps')
            result.append({"sentence_index":sent.index,"text":hook["text"],"narrative_offset":hook["offset"],
                           "start":max(tm["start"],start),"end":min(tm["end"],stop),
                           "timing_source":clock_source if exact else "estimated_within_sentence",
                           "confidence":tts_result.get('alignment_report',{}).get('confidence','high' if exact else 'low')})
        base += size
    return result


def prepare_comment_picture(variant,timings,tts_result,params):
    comments = comment_timings(variant,timings,tts_result)
    mode = getattr(variant,'comment_visual_mode','') or params.get("comment_visual_mode","auto")
    adjusted = copy.deepcopy(timings)
    clock = 0.0
    for tm in adjusted:
        aside = sum(c['end']-c['start'] for c in comments if c['sentence_index']==tm['index'])
        duration = tm['end']-tm['start']-aside
        if duration <= 0:
            raise ValueError('评论话术之外需要保留有效的正文配音时长')
        tm['start'],tm['end'] = clock,clock+duration
        clock += duration
    return adjusted,comments,mode,None


def apply_comment_picture(segments,comments,mode,preface):
    if not comments:
        return segments
    comments = sorted(comments,key=lambda c:c['start'])
    out, actual, removed = [], 0.0, 0.0
    interrupted = {c['sentence_index'] for c in comments}

    def body(start,end,shift):
        lo,hi = start-shift,end-shift
        for source in segments:
            a,b = max(lo,source['out_start']),min(hi,source['out_start']+source['dur'])
            if b-a <= 1e-8:
                continue
            ratio = source['src_duration']/source['dur']
            fragment = dict(source,src_start=source['src_start']+(a-source['out_start'])*ratio,
                            src_duration=(b-a)*ratio,dur=b-a,out_start=a+shift,freeze_duration=0)
            if source['sentence_index'] in interrupted:
                fragment['motion']='none'
            out.append(fragment)

    for number,comment in enumerate(comments):
        start,end = comment['start'],comment['end']
        body(actual,start,removed)
        point = start-removed
        options = [s for s in segments if s['sentence_index']==comment['sentence_index']]
        source = next((s for s in options if s['out_start']-1e-8 <= point < s['out_start']+s['dur']),options[-1])
        fraction = max(0,min(1,(point-source['out_start'])/source['dur']))
        anchor = source['src_start']+source['src_duration']*fraction
        anchor = min(anchor,source['evidence_end']-1/24)
        still = dict(source,src_start=anchor,src_duration=1/24,dur=end-start,out_start=start,
                     playback_speed=1,freeze_duration=0,shot_part=0,visual_kind='comment_still',
                     comment_spans=[comment],comment_visual_mode=mode,
                     comment_context='话术所在位置的前后正文镜头',motion='push_in')
        if mode=='follow':
            still['motion']='none'
        elif mode=='motion':
            still['motion']='pan_left' if number%2 else 'pan_right'
        elif mode=='freeze':
            still['freeze_duration']=min(.35,max(0,still['dur']-.2))
            still['freeze_position']='start'
        out.append(still)
        removed += end-start
        actual=end
    total = sum(s['dur'] for s in segments)+removed
    body(actual,total,removed)
    return out
