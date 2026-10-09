import copy

import pytest

from factory.models import ClipResult, Evidence, to_dict, variant_from_dict
from factory.script.review_edit import edit_variant, review_text
from factory.matching.visual_rematch import missing_evidence, rematch_missing
from test_review_edit import fixture
from test_visual_rematch import setup, proposal


@pytest.mark.parametrize('text', [
    '女孩出门。\n女孩出门。\n女孩出门。',
    ('女孩在门外等了很久，她一直想知道外面究竟发生了什么，所以又走回房间继续等待。'*4) + '\n她出门。\n结束。',
    '开始。\n结束。',
])
def test_user_can_save_and_approve_short_repeated_or_long_opening_script(tmp_path, text):
    p, candidate, variant = fixture(tmp_path)
    p.apply_review(candidate.candidate_id, 'edit', new_text=text)
    saved = p._pick_variant(candidate)
    assert candidate.status == 'review_required'
    assert [sent.text for sent in saved.sentences] == text.splitlines()
    p.apply_review(candidate.candidate_id, 'approve')
    assert candidate.status == 'approved'
    assert p.production_queue() == [candidate]


def test_edit_with_missing_old_evidence_keeps_all_words_and_defers_picture_matching(tmp_path):
    p, candidate, variant = fixture(tmp_path)
    variant.sentences[0].source_evidence = None
    text = '我写的开始。\n中间发生变化。\n我写的结果。'
    p.apply_review(candidate.candidate_id, 'approve', new_text=text)
    saved = p._pick_variant(candidate)
    assert candidate.status == 'approved' and saved.full_text == ''.join(text.splitlines())
    assert all(sent.source_evidence is None for sent in saved.sentences)
    assert len(missing_evidence(saved)) == 3
    restored = variant_from_dict(to_dict(saved))
    assert all(sent.source_evidence is None for sent in restored.sentences)
    assert all(match['needs_rematch'] for match in restored.scene_matches)


def test_tiny_split_piece_is_saved_as_unmatched_instead_of_rejecting_document(tmp_path):
    p, candidate, variant = fixture(tmp_path)
    variant.sentences[0].source_evidence = Evidence(start=0, end=.2)
    text = review_text(variant).replace('女孩发现门外一直有人', '她。\n女孩发现门外一直有人，又继续在外面等待了很长一段时间。')
    fresh, changed = edit_variant(p, variant, text)
    assert changed and len(fresh.sentences) == 5
    assert fresh.sentences[0].source_evidence is None
    assert fresh.sentences[0].text.startswith('她')
    assert any(match.get('needs_rematch') for match in fresh.scene_matches)
    assert fresh.sentences[-1].source_evidence == variant.sentences[-1].source_evidence


def test_matching_only_metadata_change_does_not_invalidate_whole_narration(tmp_path):
    p, candidate, variant = fixture(tmp_path)
    normalized, _ = edit_variant(p, variant, review_text(variant))
    normalized.sentences[0].source_evidence = None
    candidate.scripts = [normalized]
    p._save_candidates()
    p.state.completed_stages = ['CANDIDATE_FILTERED', 'TTS_GENERATED', 'MATCHING', 'RENDERING']
    p.clips = [ClipResult(candidate_id=candidate.candidate_id, variant_id=normalized.variant_id, status='completed')]
    p.apply_review(candidate.candidate_id, 'edit', new_text=review_text(normalized))
    assert p.completed('TTS_GENERATED')
    saved = p._pick_variant(candidate)
    assert saved.full_text == normalized.full_text and missing_evidence(saved)
    p.apply_review(candidate.candidate_id, 'edit', new_text=review_text(saved).replace('女孩发现', '女孩忽然发现'))
    assert not p.completed('TTS_GENERATED')


def test_missing_evidence_can_be_bound_before_timeline_without_touching_voice(tmp_path):
    p, candidate, variant, clip, provider, calls, mismatches = setup(tmp_path, proposal())
    variant.sentences[0].source_evidence = None
    variant.scene_matches.append({'sentence_index': 1, 'source_windows': [], 'needs_rematch': True})
    before_text = variant.full_text
    before_second = copy.deepcopy(variant.sentences[1])
    result = rematch_missing(p, candidate, variant, clip)
    assert result['changed'] and result['missing_sentence_indices'] == []
    assert variant.full_text == before_text and variant.sentences[1] == before_second
    assert (variant.sentences[0].source_evidence.start, variant.sentences[0].source_evidence.end) == (8, 12)
    assert len(calls) == 1 and candidate.status == 'approved'


@pytest.mark.parametrize('text', ['', ' \n ', '（只有评论）', '正文（未闭合'])
def test_empty_or_invalid_comment_marker_still_cannot_be_saved(tmp_path, text):
    p, candidate, variant = fixture(tmp_path)
    with pytest.raises(ValueError):
        edit_variant(p, variant, text)
