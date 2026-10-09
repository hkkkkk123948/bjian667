import copy
import hashlib
from types import SimpleNamespace

import pytest

from factory.config import Config
from factory.models import ClipResult, Evidence, ScriptSentence, ScriptVariant, StoryCandidate, to_dict
from factory.pipeline import ProductionPipeline
from factory.utils import write_json
from factory.matching.visual_rematch import rematch, _validate_response


def setup(tmp_path, response):
    source = tmp_path / 'source.mp4'
    source.write_bytes(b'source')
    p = ProductionPipeline(Config({'workspace': str(tmp_path / 'ws'), 'params': {'max_clips': 1}}), str(source))
    first = ScriptSentence(index=1, text='怪人站在窗外（忘记带玫瑰花了）。', narrative_text='怪人站在窗外。',
                           comment_hooks=[{'text': '忘记带玫瑰花了', 'offset': 7}], source_evidence=Evidence(start=0, end=4))
    second = ScriptSentence(index=2, text='女孩跑到门口。', source_evidence=Evidence(start=4, end=8))
    variant = ScriptVariant(variant_id='D1', candidate_id='C001', sentences=[first, second],
                            full_text=first.text+second.text, review_comment_hooks=[{'text': '忘记带玫瑰花了', 'after_sentence': 1}],
                            scene_matches=[{'sentence_index': 2, 'source_windows': [{'start': 4, 'end': 8, 'weight': 1}]}])
    candidate = StoryCandidate(candidate_id='C001', source_start=0, source_end=24, scripts=[variant], selected_script='D1', status='approved')
    p.candidates = [candidate]
    p._save_candidates()
    folder = p.job_dir / 'analysis' / 'director' / 'story_01'
    write_json(folder / 'visual_facts.json', {'facts': [{'id': 'F%02d' % (i+1), 'start': i*4, 'end': i*4+4,
               'visible': ['女孩睡觉', '女孩跑到门口', '怪人站在窗外', '男人牵恶犬', '恶犬扑倒怪人', '女孩安全离开'][i]} for i in range(6)]})
    output = tmp_path / 'clip' / 'final.mp4'
    output.parent.mkdir()
    output.write_bytes(b'old movie')
    voice = output.parent / 'voice.wav'
    voice.write_bytes(b'whole original narration, never edit')
    clip = ClipResult(clip_id='K1', candidate_id='C001', variant_id='D1', path=str(output))
    clip.tts_path = str(voice)
    provider = SimpleNamespace(model='fixture', conf={'max_tokens': 2200, 'base_url': 'fixture'}, max_retry=3)
    calls = []
    def complete(prompt, **kwargs):
        calls.append({'prompt': prompt, 'kwargs': kwargs, 'limit': provider.conf['max_tokens']})
        return response
    provider.complete_json = complete
    p.router = SimpleNamespace(get=lambda role: provider, is_mock=lambda role: False)
    mismatch = [{'sentence_index': 1, 'reason': '本句窗外怪人却匹配了睡觉女孩', 'evidence': '此处成片看到女孩睡觉'}]
    return p, candidate, variant, clip, provider, calls, mismatch


def proposal(number=1, ids=None):
    return {'replacements': [{'sentence_index': number, 'fact_ids': ids or ['F03'], 'reason': '窗外怪人镜头'}], 'unavailable': []}


def test_rematch_changes_only_target_picture_and_preserves_all_approved_audio_and_comments(tmp_path):
    p, cand, variant, clip, provider, calls, mismatches = setup(tmp_path, proposal())
    old = copy.deepcopy(variant)
    voice_hash = hashlib.sha256(open(clip.tts_path, 'rb').read()).hexdigest()
    result = rematch(p, cand, variant, clip, mismatches)
    assert result['status'] == 'changed' and result['changed_sentence_indices'] == [1]
    assert variant.full_text == old.full_text
    assert [sent.text for sent in variant.sentences] == [sent.text for sent in old.sentences]
    assert variant.review_comment_hooks == old.review_comment_hooks
    assert variant.sentences[0].comment_hooks == old.sentences[0].comment_hooks
    assert to_dict(variant.sentences[1]) == to_dict(old.sentences[1])
    assert (variant.sentences[0].source_evidence.start, variant.sentences[0].source_evidence.end) == (8, 12)
    assert cand.status == 'approved'
    assert hashlib.sha256(open(clip.tts_path, 'rb').read()).hexdigest() == voice_hash
    assert len(calls) == 1 and provider.conf['max_tokens'] == 2200 and provider.max_retry == 3
    assert '忘记带玫瑰花了' not in calls[0]['prompt']


def test_same_request_replays_valid_rematch_cache_without_second_cloud_call(tmp_path):
    p, cand, variant, clip, provider, calls, mismatches = setup(tmp_path, proposal())
    original = copy.deepcopy(variant)
    assert rematch(p, cand, variant, clip, mismatches)['changed']
    cand.scripts[0] = original
    result = rematch(p, cand, original, clip, mismatches)
    assert result['cache_hit'] and result['changed'] and len(calls) == 1


@pytest.mark.parametrize('response', [proposal(ids=['unknown']), proposal(number=2), proposal(ids=['F02']),
    {'replacements': [dict(proposal()['replacements'][0], text='改写旁白')], 'unavailable': []},
    {'replacements': [dict(proposal()['replacements'][0], source_start=100)], 'unavailable': []},
    {'replacements': [], 'unavailable': []}])
def test_invalid_or_occupied_replacements_are_atomic_and_do_not_change_variant(tmp_path, response):
    p, cand, variant, clip, provider, calls, mismatches = setup(tmp_path, response)
    before = to_dict(variant)
    result = rematch(p, cand, variant, clip, mismatches)
    assert result['status'] == 'unavailable'
    assert to_dict(variant) == before and len(calls) == 1


def test_same_fact_selection_stops_with_cached_unchanged_result(tmp_path):
    p, cand, variant, clip, provider, calls, mismatches = setup(tmp_path, proposal(ids=['F01']))
    assert rematch(p, cand, variant, clip, mismatches)['status'] == 'unchanged'
    assert rematch(p, cand, variant, clip, mismatches)['cache_hit']
    assert len(calls) == 1


def test_fact_unused_remainder_is_available_without_borrowing_actual_played_frames(tmp_path):
    p, cand, variant, clip, provider, calls, mismatches = setup(tmp_path, proposal(ids=['F02_free1']))
    write_json(__import__('pathlib').Path(clip.path).parent / 'timeline.json', {
        'sentence_timings': [{'text': sent.text, 'start': i, 'end': i+1} for i, sent in enumerate(variant.sentences)],
        'segments': [{'sentence_index': 1, 'src_start': 1, 'src_duration': 1},
                     {'sentence_index': 2, 'src_start': 6, 'src_duration': 1}]})
    result = rematch(p, cand, variant, clip, mismatches)
    assert result['changed']
    window = next(match['source_windows'][0] for match in variant.scene_matches if match['sentence_index'] == 1)
    assert (window['start'], window['end']) == (4, 6)
    assert variant.sentences[1].source_evidence.start == 4 and variant.sentences[1].source_evidence.end == 8


def test_unavailable_target_reserves_its_current_footage_against_another_replacement():
    response = {'replacements': [{'sentence_index': 1, 'fact_ids': ['F2']}],
                'unavailable': [{'sentence_index': 2, 'reason': '保留原画面'}]}
    with pytest.raises(ValueError, match='重复'):
        _validate_response(response, [1, 2], [{'id': 'F2', 'start': 4, 'end': 8}],
                           {1: [{'start': 0, 'end': 4}], 2: [{'start': 4, 'end': 8}]}, [])


def test_missing_cached_visual_evidence_does_not_pay_for_a_fabricated_alternative(tmp_path):
    p, cand, variant, clip, provider, calls, mismatches = setup(tmp_path, proposal())
    (p.job_dir / 'analysis' / 'director' / 'story_01' / 'visual_facts.json').unlink()
    assert rematch(p, cand, variant, clip, mismatches)['status'] == 'unavailable'
    assert calls == []
