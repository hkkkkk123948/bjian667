import hashlib

import numpy as np
import pytest

from factory.models import Evidence, ScriptSentence, ScriptVariant, StoryCandidate
from factory.providers.tts.base import write_wav
from factory.render.pacing import render_runs
from factory.render.timeline import build_timeline


def fixture(lengths):
    texts = ['女孩发现门外有人。', '快跑！', '她随后跑下楼梯。', '同伴最后将她安全接走。']
    variant = ScriptVariant(sentences=[ScriptSentence(index=i+1,text=text,estimated_seconds=length,
                            source_evidence=Evidence(start=i*10,end=i*10+5))
                            for i,(text,length) in enumerate(zip(texts,lengths))])
    clock, times = 0., []
    for sent,length in zip(variant.sentences,lengths):
        times.append({'index':sent.index,'text':sent.text,'start':clock,'end':clock+length})
        clock += length
    return StoryCandidate(source_start=0,source_end=40), variant, {
        'duration':clock,'audio_path':'voice.wav','sentence_timings':times,'timing_source':'local_asr_words'}


def test_legacy_length_helper_cannot_drop_selected_copy():
    from factory.render.timeline import shorten_variant_to_fit
    _,variant,_ = fixture([20,20,20,20])
    variant.estimated_seconds = 80
    original = [s.text for s in variant.sentences]
    assert shorten_variant_to_fit(variant,15) is variant
    assert [s.text for s in variant.sentences] == original


@pytest.mark.parametrize('native_speed',[None,1.2])
def test_overlong_audio_remains_complete_at_original_audio_clock(native_speed):
    candidate,variant,tts = fixture([20,20,20,20])
    if native_speed is not None:
        tts['playback_speed'] = native_speed
    timeline = build_timeline(candidate,variant,tts,{'min_duration':15,'max_duration':35},video_duration=40)
    assert timeline['narration_duration'] == 80
    assert timeline['source_duration'] == 80
    assert timeline['speed_factor'] == 1
    assert timeline['sentence_timings'] == [dict(t,narrative_text=s.text,comment_hook='')
                                         for t,s in zip(tts['sentence_timings'],variant.sentences)]
    assert timeline['duration_guidance']['gate_enabled'] is False
    assert sum(s['dur'] for s in timeline['segments']) == 80
    assert all(s['src_start']+s['src_duration'] <= s['evidence_end'] for s in timeline['segments'])


def test_short_action_gets_neighbour_picture_time_without_moving_narration():
    candidate,variant,tts = fixture([3,.125,3,3])
    timeline = build_timeline(candidate,variant,tts,{'max_duration':35})
    runs = render_runs(timeline['segments'])
    assert all(r['dur'] >= .35 for r in runs)
    assert sum(r['dur'] for r in runs) == pytest.approx(tts['duration'])
    assert timeline['sentence_timings'][1]['start'] == 3
    assert timeline['sentence_timings'][1]['end'] == 3.125
    action = next(r for r in runs if r['sentence_index']==2)
    assert action['out_start'] <= 3 < 3.125 <= action['out_start']+action['dur']
    assert timeline['editing_diagnostics']['short_context_adjustments'][0]['kind'] == 'extended_picture_dwell'


def test_no_room_for_independent_cuts_continues_a_picture_without_voice_gate():
    candidate,variant,tts = fixture([.1,.1,.1,.1])
    timeline = build_timeline(candidate,variant,tts,{'min_duration':35,'max_duration':50})
    runs = render_runs(timeline['segments'])
    assert len(runs) == 1
    assert runs[0]['dur'] == pytest.approx(.4)
    assert timeline['sentence_timings'][-1]['end'] == .4
    assert timeline['speed_factor'] == 1
    assert any(d['kind']=='adjacent_context_continuation'
               for d in timeline['editing_diagnostics']['short_context_adjustments'])


def test_actual_export_preserves_full_voice_and_removes_inserted_flash(tmp_path):
    from factory.render.exporter import export_clip
    from factory.utils import ffmpeg_bin,media_duration,read_json,run_cmd
    source = tmp_path/'source.mp4'
    run_cmd([ffmpeg_bin(),'-v','error','-y','-f','lavfi','-i','testsrc2=size=128x72:rate=24',
             '-t','4','-an','-c:v','libx264','-threads','1',str(source)])
    candidate,variant,tts = fixture([1,.125,1,1])
    for i,sent in enumerate(variant.sentences):
        sent.source_evidence = Evidence(start=i,end=i+1)
    candidate.source_end = 4
    voice = tmp_path/'voice.wav'
    write_wav(voice,.1*np.sin(np.arange(round(tts['duration']*24000))*.1),24000)
    before = hashlib.sha256(voice.read_bytes()).hexdigest()
    tts['audio_path'] = str(voice)
    timeline = build_timeline(candidate,variant,tts,{'max_duration':2},video_duration=4)
    target = tmp_path/'out.mp4'
    export_clip(source,voice,None,timeline,{'target_width':128,'target_height':72,'motion_oversample':1,
                'picture_enhance':'off','sound_design':'off','short_transitions':'on','prefer_nvenc':'off'},target,src_size=(128,72))
    assert hashlib.sha256(voice.read_bytes()).hexdigest() == before
    assert media_duration(target) == pytest.approx(tts['duration'],abs=.06)
    assert all(s['duration'] >= .35 for s in read_json(tmp_path/'edit_decisions.json')['shots'])
