import copy

import pytest

from factory.models import ClipResult
from factory.pipeline import PipelineError
from factory.script.review_edit import review_text
from test_review_edit import fixture


def old_finished_task(tmp_path, monkeypatch):
    p, candidate, variant = fixture(tmp_path)
    p.config.params['review_mode'] = 'semi'
    p.state.status = 'COMPLETED'
    p.state.completed_stages = ['CANDIDATE_FILTERED', 'TTS_GENERATED', 'MATCHING', 'TIMELINE_ALIGNED',
                                'RENDERING', 'FINAL_MATCH_CHECK', 'QUALITY_CHECK', 'DRAFT_EXPORT', 'COMPLETED']
    p.clips = [ClipResult(candidate_id=candidate.candidate_id, status='completed',
                          qc={'status': 'pass'}, gate2={'status': 'pass'})]
    monkeypatch.setattr(p, '_load_index', lambda: None)
    monkeypatch.setattr(p, '_load_style', lambda: None)
    return p, candidate, variant


def test_non_director_rewrite_invalidates_old_assets_only_after_success_and_requires_confirmation(tmp_path, monkeypatch):
    p, candidate, variant = old_finished_task(tmp_path, monkeypatch)
    rewritten = copy.deepcopy(variant)
    rewritten.variant_id = 'S4'
    rewritten.sentences[1].text = '女孩换了条路寻找可以求助的人。'
    rewritten.full_text = ''.join(sent.text for sent in rewritten.sentences)
    monkeypatch.setattr('factory.pipeline.write_scripts', lambda *a, **k: [rewritten])
    p._regenerate_candidate(candidate)
    assert candidate.status == 'review_required' and candidate.selected_script == 'S4'
    assert p.state.completed_stages == ['CANDIDATE_FILTERED']
    assert p.state.status == 'CANDIDATE_FILTERED' and p.state.artifacts['script_review_pending']
    assert p.clips[0].status == 'outdated_script' and not p.clips[0].qc and not p.clips[0].gate2
    assert p.production_queue() == []
    assert not p.state.artifacts.get('director_local_gate')


@pytest.mark.parametrize('failure', ['empty_result', 'write_exception'])
def test_failed_non_director_rewrite_keeps_approved_script_and_finished_task(tmp_path, monkeypatch, failure):
    p, candidate, variant = old_finished_task(tmp_path, monkeypatch)
    original_candidate, original_state, original_clips = copy.deepcopy(candidate), copy.deepcopy(p.state), copy.deepcopy(p.clips)
    rewritten = copy.deepcopy(variant)
    rewritten.full_text = '新稿尚未检查。'
    def write(*args, **kwargs):
        if failure == 'write_exception':
            raise RuntimeError('fixture failure')
        return [] if failure == 'empty_result' else [rewritten]
    monkeypatch.setattr('factory.pipeline.write_scripts', write)
    with pytest.raises((RuntimeError, PipelineError)):
        p._regenerate_candidate(candidate)
    assert candidate == original_candidate and p.state == original_state and p.clips == original_clips


@pytest.mark.parametrize('director', [False, True])
def test_user_edit_does_not_forge_a_director_provenance_marker(tmp_path, director):
    p, candidate, variant = fixture(tmp_path)
    if director:
        p.state.artifacts['director_local_gate'] = True
    changed = review_text(variant).replace('她赶紧躲到垃圾箱后面', '女孩转身寻找旁边的藏身处')
    p.apply_review(candidate.candidate_id, 'edit', new_text=changed)
    assert bool(p.state.artifacts.get('director_local_gate')) is director
