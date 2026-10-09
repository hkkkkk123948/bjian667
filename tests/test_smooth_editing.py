import json
import subprocess

import numpy as np
import pytest

from factory.render.motion import choose_motion, motion_sample
from factory.render.pacing import readable_windows, render_runs
from factory.render.exporter import export_clip
from factory.providers.tts.base import write_wav
from factory.utils import ffmpeg_bin, run_cmd


def test_optional_sliver_windows_are_removed_without_inventing_evidence():
    windows = [{'start':0,'end':2,'weight':1}, {'start':4,'end':4.1,'weight':.01,'optional':True},
               {'start':6,'end':9,'weight':1}]
    selected = readable_windows(windows,0,9,3)
    assert [(w['start'],w['end']) for w in selected] == [(0,2),(6,9)]
    assert all(3*w['weight']/sum(r['weight'] for r in selected) >= .7 for w in selected)
    short = readable_windows(windows,0,9,.3)
    assert len(short) == 1
    assert short[0]['continuity_omitted_windows']
    assert short[0]['weight'] == 1


def test_low_weight_required_action_is_preserved_and_gets_readable_time():
    windows = [{'start':0,'end':2,'weight':1}, {'start':4,'end':5,'weight':.01},
               {'start':6,'end':9,'weight':1}]
    selected = readable_windows(windows,0,9,3)
    assert len(selected) == 3
    assert min(3*row['weight'] for row in selected) >= .7
    pair = readable_windows([{'start':0,'end':2},{'start':5,'end':7}],0,7,1.2)
    assert len(pair) == 2 and all(1.2*row['weight'] == pytest.approx(.6) for row in pair)


@pytest.mark.parametrize('duration',[.12,.5,1.,3.,8.])
@pytest.mark.parametrize('returning',[False,True])
def test_zoom_velocity_is_bounded_and_endpoints_do_not_snap(duration,returning):
    times = np.linspace(0,duration,1001)
    scales = np.array([motion_sample('push_in',t,duration,returning)[0] for t in times])
    assert scales[0] == 1
    assert max(abs(np.diff(scales)) / (duration/1000)) < .012
    assert abs(scales[1]-scales[0]) / (duration/1000) < .0002
    if returning:
        assert scales[-1] == 1
    assert choose_motion('twist',1,.12) == 'none'


def test_three_frame_prelude_and_comment_render_as_one_shot():
    parts = [dict(src_start=0,src_duration=.125,dur=.125),
             dict(src_start=.125,src_duration=1/24,dur=1,visual_kind='comment_still'),
             dict(src_start=.125,src_duration=1.875,dur=2.875)]
    for p in parts:
        p.update(sentence_index=1,evidence_start=0,evidence_end=2)
    runs = render_runs(parts)
    assert len(runs) == 1 and runs[0]['dur'] == 4
    assert len(runs[0]['parts']) == 3
    # Different story evidence remains a real scene change.
    other = dict(parts[0],sentence_index=2,src_start=5,evidence_start=5,evidence_end=6)
    assert len(render_runs(parts+[other])) == 2


def test_comment_at_sentence_start_anchors_selected_frame_not_evidence_start(tmp_path):
    from test_review_edit import fixture
    from factory.script.comment_hook import story_text
    from factory.script.review_edit import edit_variant
    from factory.render.timeline import build_timeline
    p,c,v = fixture(tmp_path)
    rows = [story_text(s) for s in v.sentences]
    rows[0] = '（评论插话）' + rows[0]
    v,_ = edit_variant(p,v,'\n'.join(rows))
    timeline = build_timeline(c,v,{'duration':16,'audio_path':'voice.wav','playback_speed':1.2},{})
    aside,body = timeline['segments'][:2]
    assert aside['visual_kind'] == 'comment_still'
    assert aside['src_start'] == pytest.approx(body['src_start'])
    assert aside['src_start'] > aside['evidence_start']
    assert len(render_runs(timeline['segments'])) == len(v.sentences)


def test_black_edge_trim_remaps_comment_and_body_as_one_context(monkeypatch):
    from factory.render.exporter import _prepare_context_sources
    parts = [dict(src_start=0,src_duration=1/24,dur=1,visual_kind='comment_still'),
             dict(src_start=0,src_duration=2,dur=3)]
    for p in parts:
        p.update(sentence_index=1,evidence_start=0,evidence_end=2)
    monkeypatch.setattr('factory.render.exporter._trim_black_edges',lambda *a:(.8,1.2))
    assert _prepare_context_sources('source.mp4',parts)
    assert parts[0]['src_start'] == parts[1]['src_start'] == .8
    assert parts[1]['src_duration'] == pytest.approx(1.2)
    assert len(render_runs(parts)) == 1
    assert sum(p['dur'] for p in parts) == 4


def test_actual_local_aside_returns_to_same_frame_without_zoom_jump(tmp_path):
    cv2 = pytest.importorskip('cv2')
    grid = np.zeros((180,320),np.uint8)
    cv2.circle(grid,(160,90),38,220,-1)
    cv2.rectangle(grid,(50,30),(80,145),120,-1)
    image = tmp_path/'grid.png'
    cv2.imwrite(str(image),grid)
    source = tmp_path/'source.mp4'
    run_cmd([ffmpeg_bin(),'-v','error','-y','-loop','1','-i',str(image),'-t','3',
             '-r','24','-an','-c:v','libx264','-threads','1',str(source)])
    voice = tmp_path/'voice.wav'
    write_wav(voice,.1*np.sin(np.arange(4*24000)*.05),24000)
    segments = [dict(src_start=0,src_duration=.125,dur=.125,out_start=0,motion='none'),
                dict(src_start=.125,src_duration=1/24,dur=1,out_start=.125,
                     visual_kind='comment_still',motion='push_in'),
                dict(src_start=.125,src_duration=1.875,dur=2.875,out_start=1.125,motion='none')]
    for s in segments:
        s.update(sentence_index=1,evidence_start=0,evidence_end=2)
    target = tmp_path/'smooth.mp4'
    export_clip(source,voice,None,{'segments':segments,'narration_duration':4,'speed_factor':1},
                {'target_width':320,'target_height':180,'prefer_nvenc':'off','picture_enhance':'off',
                 'sound_design':'off','motion_oversample':3},target)
    raw = subprocess.check_output([ffmpeg_bin(),'-v','error','-threads','1','-i',str(target),
                                    '-f','rawvideo','-pix_fmt','gray','-'])
    frames = np.frombuffer(raw,np.uint8).reshape(-1,180,320)
    assert len(frames) == 96
    # At either aside boundary the scale is exactly the original scale.
    assert np.mean(abs(frames[2].astype(float)-frames[3])) < .7
    assert np.mean(abs(frames[26].astype(float)-frames[27])) < .7
    decisions = json.loads((tmp_path/'edit_decisions.json').read_text(encoding='utf-8'))
    assert decisions['rendered_shots'] == 1 and decisions['context_joins'] == 2


def test_mixed_zero_length_cut_and_dissolve_keep_every_video_frame(tmp_path):
    source = tmp_path/'source.mp4'
    run_cmd([ffmpeg_bin(),'-v','error','-y','-f','lavfi','-i',
             'testsrc2=size=320x180:rate=24:duration=5','-an','-c:v','libx264','-threads','1',str(source)])
    durations = [1/24,1.5,1.5]
    total = sum(durations)
    voice = tmp_path/'voice.wav'
    write_wav(voice,.1*np.sin(np.arange(round(total*24000))*.05),24000)
    segments, clock = [], 0.
    for i,dur in enumerate(durations):
        segments.append(dict(src_start=i*1.6,src_duration=dur,dur=dur,out_start=clock,motion='none'))
        clock += dur
    target = tmp_path/'mixed.mp4'
    export_clip(source,voice,None,{'segments':segments,'narration_duration':total,'speed_factor':1},
                {'target_width':320,'target_height':180,'prefer_nvenc':'off','picture_enhance':'off',
                 'sound_design':'off','short_transitions':'on'},target)
    output,_,_ = run_cmd(['ffprobe','-v','error','-select_streams','v:0','-show_entries',
                          'stream=nb_frames','-of','json',str(target)])
    assert int(json.loads(output)['streams'][0]['nb_frames']) == 73
    plan = json.loads((tmp_path/'transitions.json').read_text(encoding='utf-8'))
    assert [b['type'] for b in plan['boundaries']] == ['cut','fade']
