import copy
from types import SimpleNamespace

import pytest

from factory.config import Config
from factory.matching.initial_plan import InitialPlanError, build_initial_timeline
from factory.models import Evidence, ScriptSentence, ScriptVariant, StoryCandidate
from factory.script.comment_hook import attach_inline_hooks
from factory.script.quality import compact


def fixture(tmp_path):
    # Old script evidence deliberately points at the wrong source location.
    sentences = [ScriptSentence(index=1, text='男孩冲到门口，两条恶犬扑倒怪人。', estimated_seconds=6,
                                source_evidence=Evidence(start=0, end=2)),
                 ScriptSentence(index=2, text='怪人举起尖刀，男孩赶紧跑走。', estimated_seconds=6,
                                source_evidence=Evidence(start=2, end=4))]
    variant = ScriptVariant(variant_id='D1', candidate_id='C001', sentences=sentences,
                            full_text=''.join(s.text for s in sentences), estimated_seconds=12)
    candidate = StoryCandidate(candidate_id='C001', source_start=0, source_end=90, scripts=[variant],
                               selected_script='D1', status='approved')
    def framed(shot_id, start, end, visible, action=None):
        mid = (start + end) / 2
        fid = shot_id + "_mid"
        act = action or visible
        return {"shot_id": shot_id, "start": start, "end": end, "visible": visible,
                "frames": [{"frame_id": fid, "time": mid, "position": "middle"}],
                "observations": [{"frame_id": fid, "visible": visible}],
                "actions": [{"text": act, "frame_ids": [fid]}], "characters": []}
    shots = [framed("DOG", 60, 66, "两条恶犬扑向怪人并把他压倒", "恶犬扑倒怪人"),
             framed("DOOR", 25, 31, "男孩冲向门口"),
             framed("KNIFE", 10, 16, "怪人举起尖刀"),
             framed("RUN", 40, 46, "男孩跑走")]
    answer = {'sentences': [
        {'sentence_index': 1, 'beats': [{'shot_id': 'DOOR', 'target_text': '男孩冲到门口'},
                                       {'shot_id': 'DOG', 'target_text': '两条恶犬'}]},
        {'sentence_index': 2, 'beats': [{'shot_id': 'KNIFE', 'target_text': '怪人举起尖刀'},
                                       {'shot_id': 'RUN', 'target_text': '男孩赶紧跑走'}]}], 'unmatched': []}
    calls = []
    provider = SimpleNamespace(model='fixture', conf={'max_tokens': 9000}, max_retry=3, is_mock=False)

    def complete(prompt, **kwargs):
        assert '不评价或修改文案' in prompt and '不要返回时间' in prompt
        assert '忘记带玫瑰花' not in prompt
        assert provider.conf['max_tokens'] <= 2200 and provider.max_retry == 1
        calls.append(prompt)
        return copy.deepcopy(answer)

    provider.complete_json = complete
    p = SimpleNamespace(job_dir=tmp_path, config=Config({'params': {'comment_hook_text': ''}}),
                         router=SimpleNamespace(get=lambda role: provider))
    return p, candidate, variant, shots, answer, calls, provider


def speech(variant):
    words, timings, clock = [], [], 0.
    for sentence in variant.sentences:
        text = compact(sentence.text)
        start = clock
        for char in text:
            words.append({'text': char, 'start': clock, 'end': clock + .4})
            clock += .4
        timings.append({'index': sentence.index, 'text': sentence.text, 'start': start, 'end': clock})
    return {'audio_path': 'intact.wav', 'duration': clock, 'playback_speed': 1.2,
            'word_timings': words, 'sentence_timings': timings, 'timing_source': 'provider_word_timestamps',
            'alignment_report': {'confidence': 'high'}}


def test_real_action_anchor_uses_audio_word_clock_and_can_reorder_source_shots(tmp_path):
    p, candidate, variant, shots, answer, calls, provider = fixture(tmp_path)
    original = copy.deepcopy(variant.sentences)
    result = speech(variant)
    timeline = build_initial_timeline(p, candidate, variant, result, shots)
    segments = timeline['segments']
    assert [seg['shot_id'] for seg in segments] == ['DOOR', 'DOG', 'KNIFE', 'RUN']
    dog_word = next(word['start'] for word in result['word_timings'] if word['text'] == '两')
    assert segments[1]['out_start'] == pytest.approx(dog_word)
    assert segments[2]['src_start'] < segments[1]['src_start']
    assert all(seg['evidence_start'] <= seg['src_start'] < seg['src_start'] + seg['src_duration'] <= seg['evidence_end']
               for seg in segments)
    assert sum(seg['dur'] for seg in segments) == pytest.approx(result['duration'])
    assert [s.text for s in variant.sentences] == [s.text for s in original] and len(calls) == 1
    assert all(s.source_evidence == o.source_evidence for s, o in zip(variant.sentences, original))
    assert provider.max_retry == 3 and provider.conf['max_tokens'] == 9000


def test_comment_only_edit_reuses_semantic_shots_and_rebinds_actual_audio_clock(tmp_path):
    p, candidate, variant, shots, answer, calls, provider = fixture(tmp_path)
    first = build_initial_timeline(p, candidate, variant, speech(variant), shots)
    attach_inline_hooks(variant.sentences[0], [{'text': '忘记带玫瑰花了', 'offset': 7}])
    variant.full_text = ''.join(s.text for s in variant.sentences)
    second = build_initial_timeline(p, candidate, variant, speech(variant), shots)
    assert len(calls) == 1 and second['initial_match']['cache_hit']
    assert first['initial_match']['signature'] == second['initial_match']['signature']
    assert first['initial_match']['binding_signature'] != second['initial_match']['binding_signature']
    assert any(seg.get('visual_kind') == 'comment_still' for seg in second['segments'])
    dog = next(seg for seg in second['segments'] if seg['shot_id'] == 'DOG' and seg.get('visual_kind') != 'comment_still')
    assert dog['out_start'] > first['segments'][1]['out_start']


def test_same_line_count_body_edit_invalidates_old_shot_selection(tmp_path):
    p, candidate, variant, shots, answer, calls, provider = fixture(tmp_path)
    first = build_initial_timeline(p, candidate, variant, speech(variant), shots)
    variant.sentences[0].text = '男孩冲到门口，怪人举起尖刀。'
    variant.full_text = ''.join(s.text for s in variant.sentences)
    answer['sentences'][0]['beats'][1] = {'target_text': '怪人举起尖刀', 'shot_id': 'KNIFE'}
    second = build_initial_timeline(p, candidate, variant, speech(variant), shots)
    assert len(calls) == 2 and len(variant.sentences) == 2
    assert first['initial_match']['signature'] != second['initial_match']['signature']
    assert second['segments'][1]['shot_id'] == 'KNIFE'


@pytest.mark.parametrize('failure', ['unknown_shot', 'missing_sentence', 'fabricated_phrase', 'insufficient'])
def test_invalid_selection_does_not_fall_back_to_stale_evidence_or_recharge(tmp_path, failure):
    p, candidate, variant, shots, answer, calls, provider = fixture(tmp_path)
    if failure == 'unknown_shot':
        answer['sentences'][0]['beats'][0]['shot_id'] = 'FAKE'
    elif failure == 'missing_sentence':
        answer['sentences'].pop()
    elif failure == 'fabricated_phrase':
        answer['sentences'][0]['beats'][0]['target_text'] = '根本没有这个动作'
    else:
        answer['unmatched'] = [{'sentence_index': 1, 'reason': '没有狗扑倒怪人的画面'}]
    with pytest.raises(InitialPlanError):
        build_initial_timeline(p, candidate, variant, speech(variant), shots)
    with pytest.raises(InitialPlanError):
        build_initial_timeline(p, candidate, variant, speech(variant), shots)
    assert len(calls) == 1
    assert variant.sentences[0].source_evidence.start == 0


def test_model_time_values_cannot_change_real_source_or_audio_times(tmp_path):
    p, candidate, variant, shots, answer, calls, provider = fixture(tmp_path)
    answer['sentences'][0]['beats'][1].update(source_start=800, source_end=900, audio_start=0)
    result = speech(variant)
    timeline = build_initial_timeline(p, candidate, variant, result, shots)
    dog = timeline['segments'][1]
    assert 60 <= dog['src_start'] < 66 and dog['out_start'] > 0
    assert timeline['initial_match']['model_time_values_accepted'] is False


def test_catalogue_change_or_audio_timing_change_has_distinct_cache_effects(tmp_path):
    p, candidate, variant, shots, answer, calls, provider = fixture(tmp_path)
    result = speech(variant)
    build_initial_timeline(p, candidate, variant, result, shots)
    slower = copy.deepcopy(result)
    for word in slower['word_timings']:
        word['start'] *= 1.1
        word['end'] *= 1.1
    for timing in slower['sentence_timings']:
        timing['start'] *= 1.1
        timing['end'] *= 1.1
    slower['duration'] *= 1.1
    build_initial_timeline(p, candidate, variant, slower, shots)
    assert len(calls) == 1
    shots[0]['visible'] += '，镜头中明确看见男孩站在旁边'
    build_initial_timeline(p, candidate, variant, slower, shots)
    assert len(calls) == 2
