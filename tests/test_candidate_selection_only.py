from types import SimpleNamespace

import pytest

from factory.config import Config
from factory.models import ScriptSentence, ScriptVariant, StoryCandidate
from factory.pipeline import ProductionPipeline


def candidate(identity, score, status='generated'):
    variant = ScriptVariant(variant_id='draft', full_text='用户可确认的正文。',
                            sentences=[ScriptSentence(index=1, text='用户可确认的正文。')])
    return StoryCandidate(candidate_id=identity, story_score=score, status=status, scripts=[variant])


def test_legacy_selection_does_not_review_write_or_match_before_tts(tmp_path):
    p = ProductionPipeline(Config({'workspace': str(tmp_path), 'params': {
        'max_clips': 0, 'review_mode': 'semi'}}), 'fixture.mp4')
    p.candidates = [candidate('C001', 80), candidate('C002', 60), candidate('C003', 100, 'rejected_by_user')]
    p.router = SimpleNamespace(get=lambda *a: pytest.fail('Selection cannot call a model'))
    p._do_match()
    assert [c.status for c in p.candidates] == ['review_required', 'review_required', 'rejected_by_user']
    assert p.production_queue() == []
    assert all(c.match_report.get('gate1', {}).get('basis') == 'selection_only' for c in p.candidates[:2])


def test_explicit_auto_limit_selects_without_fabricated_match_score(tmp_path):
    p = ProductionPipeline(Config({'workspace': str(tmp_path), 'params': {
        'max_clips': 1, 'review_mode': 'auto'}}), 'fixture.mp4')
    p.candidates = [candidate('C001', 60), candidate('C002', 90)]
    p.router = SimpleNamespace(get=lambda *a: pytest.fail('Selection cannot call a model'))
    p._do_match()
    assert [c.candidate_id for c in p.production_queue()] == ['C002']
    assert p.candidates[0].status == 'not_selected'
    assert p.candidates[1].final_match == 0
