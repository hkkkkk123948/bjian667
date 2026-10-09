import hashlib

import numpy as np
import pytest

from factory.pipeline import ScriptReviewRequired
from factory.production import do_tts, upgrade_narration_alignment
from factory.providers.tts.base import write_wav
from factory.script.comment_hook import story_text
from factory.script.quality import compact
from factory.utils import read_json
from test_review_edit import fixture


def test_selected_overlong_copy_is_kept_and_duration_is_only_a_warning(tmp_path,monkeypatch):
    p,c,v = fixture(tmp_path)
    p.config.params.update(review_mode='semi',min_duration=1,max_duration=2)
    original = [s.text for s in v.sentences]
    result = {'duration': 12, 'audio_path': str(tmp_path/'existing.wav'), 'provider': 'test_fixture',
              'content_check': {'status': 'pass'}, 'audio_edits': []}
    monkeypatch.setattr('factory.providers.tts.continuous.synth_continuous', lambda *a: result)
    monkeypatch.setattr('factory.script.repair.compress_variant',
                        lambda *a: pytest.fail('Approved text must not be silently compressed'))
    do_tts(p,[c])
    assert [s.text for s in v.sentences] == original
    assert c.status == 'approved'
    assert p.clips[0].scores['duration_target_warning']['actual_seconds'] == 12


def test_actual_short_narration_is_saved_before_video_work(tmp_path,monkeypatch):
    p,c,v = fixture(tmp_path)
    p.config.params.update(review_mode='semi',min_duration=35,max_duration=50)
    original = [story_text(s) for s in v.sentences]
    result = {'duration':12,'audio_path':str(tmp_path/'existing.wav'),'provider':'test_fixture',
              'content_check':{'status':'pass'},'audio_edits':[]}
    calls = []
    monkeypatch.setattr('factory.providers.tts.continuous.synth_continuous',
                        lambda *a: calls.append(a) or result)
    do_tts(p,[c])
    assert len(calls) == 1
    assert read_json(p.job_dir/'clips'/'clip_001'/'tts.json')['duration'] == 12
    assert [story_text(s) for s in v.sentences] == original
    assert c.status == 'approved'
    assert not p.completed('RENDERING')


def test_production_preflight_returns_review_instead_of_failed(tmp_path,monkeypatch):
    p,c,v = fixture(tmp_path)
    monkeypatch.setattr(p,'_do_tts',lambda queue: (_ for _ in ()).throw(ScriptReviewRequired('请改文案')))
    result = p.produce()
    assert result.status == 'CANDIDATE_FILTERED'
    assert result.error == ''
    assert result.artifacts['script_review_pending'] is True
    assert not p.completed('TTS_GENERATED')


def test_old_provider_clock_upgrade_preserves_audio_without_asr_or_tts(tmp_path,monkeypatch):
    p,c,v = fixture(tmp_path)
    voice = tmp_path/'voice.wav'
    write_wav(voice,.1*np.sin(np.arange(24000)*.05),24000)
    before = hashlib.sha256(voice.read_bytes()).hexdigest()
    spoken = ''.join(s.text for s in v.sentences)
    text = compact(spoken)
    words = [{'text':ch,'start':i/len(text),'end':(i+1)/len(text)} for i,ch in enumerate(text)]
    metadata = tmp_path/'tts.json'
    monkeypatch.setattr('factory.eval.narration_audit.audit_narration',lambda *a,**k: pytest.fail('Reliable provider clock needs no ASR'))
    upgraded = upgrade_narration_alignment(p,v,{'audio_path':str(voice),'text':spoken,
                  'duration':1,'timing_source':'provider_word_timestamps','word_timings':words,
                  'content_check':{'status':'pass'}},metadata)
    assert upgraded['timing_source'] == 'provider_word_timestamps'
    assert upgraded['timestamp_duration_scale'] == 1
    assert hashlib.sha256(voice.read_bytes()).hexdigest() == before
    assert upgraded['sentence_timings'][-1]['end'] == 1


@pytest.mark.parametrize('unavailable', [False, True])
def test_local_asr_difference_or_unavailable_model_only_changes_timing_confidence(tmp_path, monkeypatch, unavailable):
    p, c, v = fixture(tmp_path)
    voice = tmp_path/'voice.wav'
    write_wav(voice, .1*np.sin(np.arange(24000)*.05), 24000)
    before = hashlib.sha256(voice.read_bytes()).hexdigest()

    def audit(*args, **kwargs):
        if unavailable:
            raise RuntimeError('fixture ASR unavailable')
        return {'status': 'fail', 'issues': [{'kind': 'missing_words'}], 'asr_word_timings': []}

    monkeypatch.setattr('factory.eval.narration_audit.audit_narration', audit)
    upgraded = upgrade_narration_alignment(p, v, {'audio_path': str(voice), 'text': ''.join(s.text for s in v.sentences),
        'duration': 1, 'provider': 'bailian-cosyvoice', 'timing_source': 'estimated', 'word_timings': []}, tmp_path/'tts.json')
    assert upgraded['timing_source'] == 'estimated'
    assert upgraded['alignment_report']['confidence'] == 'low'
    assert hashlib.sha256(voice.read_bytes()).hexdigest() == before
    assert c.status == 'approved'
