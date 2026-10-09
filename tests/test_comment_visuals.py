from types import SimpleNamespace

import numpy as np
import pytest

from factory.models import Evidence
from factory.script.comment_hook import story_text
from factory.script.quality import compact
from factory.script.review_edit import edit_variant
from factory.render.timeline import build_timeline
from factory.eval.judge import audit_timeline
from test_review_edit import fixture


def timeline_fixture(tmp_path,position=1,mode='auto'):
    p,c,v=fixture(tmp_path)
    rows=[story_text(s) for s in v.sentences]
    rows[position]=rows[position][:3]+'（评论插话）'+rows[position][3:]
    v,_=edit_variant(p,v,'\n'.join(rows))
    v.comment_visual_mode=mode
    times=[{'index':s.index,'text':s.text,'start':i*4,'end':(i+1)*4} for i,s in enumerate(v.sentences)]
    words=[]
    for i,s in enumerate(v.sentences):
        text=compact(s.text)
        words.extend({'text':ch,'start':i*4+j*4/len(text),'end':i*4+(j+1)*4/len(text)} for j,ch in enumerate(text))
    tts={'duration':16,'audio_path':'voice.wav','sentence_timings':times,'word_timings':words,'playback_speed':1.2}
    return c,v,tts,build_timeline(c,v,tts,{})


@pytest.mark.parametrize('position',[0,1,3])
@pytest.mark.parametrize('mode',['auto','follow','motion','freeze'])
def test_treatment_stays_exactly_at_each_aside_not_whole_sentence(tmp_path,position,mode):
    c,v,tts,timeline=timeline_fixture(tmp_path,position,mode)
    plan=timeline['comment_picture_plan']
    assert plan['literal_matching'] is False and plan['context_matching'] is True
    event=plan['comments'][0]
    effects=[s for s in timeline['segments'] if s.get('visual_kind')=='comment_still']
    assert len(effects)==1
    effect=effects[0]
    assert effect['out_start']==pytest.approx(event['start'])
    assert effect['dur']==pytest.approx(event['end']-event['start'])
    assert effect['dur']<4
    assert position*4<=effect['out_start']<(position+1)*4
    assert sum(s['dur'] for s in timeline['segments'])==pytest.approx(16)
    assert audit_timeline(timeline)['issues']==[]
    assert timeline['voice_path']==tts['audio_path'] and timeline['speed_factor']==1


def test_legacy_live_objects_without_new_optional_attributes_work(tmp_path):
    c,v,tts,timeline=timeline_fixture(tmp_path)
    sentences=[SimpleNamespace(index=i+1,text='原有正文。',narrative_text='',comment_hook='',role='context',
                 estimated_seconds=4,source_evidence=Evidence(start=i*10,end=i*10+5)) for i in range(4)]
    legacy=SimpleNamespace(sentences=sentences,scene_matches=[])
    result=build_timeline(c,legacy,{'duration':16,'audio_path':'old.wav'}, {})
    assert len(result['segments'])==4
    assert result['comment_picture_plan']['comments']==[]


def test_actual_export_keeps_complete_audio_duration_with_middle_aside(tmp_path):
    from factory.utils import ffmpeg_bin,run_cmd,media_duration
    from factory.providers.tts.base import write_wav
    from factory.render.exporter import export_clip
    c,v,tts,timeline=timeline_fixture(tmp_path,position=1,mode='motion')
    source=tmp_path/'source.mp4'
    run_cmd([ffmpeg_bin(),'-v','error','-y','-f','lavfi','-i','testsrc2=size=320x180:rate=24:duration=4',
             '-an','-c:v','libx264','-preset','ultrafast',str(source)])
    for i,s in enumerate(v.sentences):
        s.source_evidence=Evidence(start=i,end=i+.6)
    c.source_end=4
    voice=tmp_path/'voice.wav'
    write_wav(voice,.1*np.sin(np.arange(16*24000)*.05),24000)
    tts['audio_path']=str(voice)
    params={'target_width':320,'target_height':180,'prefer_nvenc':'off','picture_enhance':'off',
            'sound_design':'off','motion_oversample':2,'short_transitions':'on'}
    timeline=build_timeline(c,v,tts,params)
    out=tmp_path/'export.mp4'
    export_clip(source,voice,None,timeline,params,out,src_size=(320,180))
    assert abs(media_duration(out)-16)<.1
    assert audit_timeline(timeline)['issues']==[]


def test_one_frame_dissolve_preserves_video_track_not_just_audio_duration(tmp_path):
    import json
    from factory.utils import ffmpeg_bin,run_cmd
    from factory.providers.tts.base import write_wav
    from factory.render.exporter import export_clip
    c,v,tts,timeline=timeline_fixture(tmp_path,position=0,mode='auto')
    tiny,aside=.1149612995,1.2262538615
    for i,s in enumerate(timeline['segments']):
        if i==0:
            s.update(dur=tiny,out_start=0,src_start=0,src_duration=tiny)
        elif i==1:
            s.update(dur=aside,out_start=tiny,src_start=tiny,src_duration=1/24)
        elif i==2:
            s.update(dur=4-tiny-aside,out_start=tiny+aside,src_start=tiny,src_duration=.6-tiny)
        else:
            s.update(src_start=i-2,src_duration=.6,evidence_start=i-2,evidence_end=i-2+.6)
        if i<3:
            s.update(evidence_start=0,evidence_end=.6)
    source=tmp_path/'source.mp4'
    run_cmd([ffmpeg_bin(),'-v','error','-y','-f','lavfi','-i','testsrc2=size=320x180:rate=24:duration=4',
             '-an','-c:v','libx264','-preset','ultrafast',str(source)])
    voice=tmp_path/'voice.wav'
    write_wav(voice,.1*np.sin(np.arange(16*24000)*.05),24000)
    out=tmp_path/'final.mp4'
    export_clip(source,voice,None,timeline,{'target_width':320,'target_height':180,'prefer_nvenc':'off',
                'picture_enhance':'off','sound_design':'off','motion_oversample':2,'short_transitions':'on'},out,src_size=(320,180))
    data,_,_=run_cmd(['ffprobe','-v','error','-select_streams','v:0','-show_entries','stream=duration,nb_frames','-of','json',str(out)])
    stream=json.loads(data)['streams'][0]
    assert abs(float(stream['duration'])-16)<.1
    assert int(stream['nb_frames'])==384
