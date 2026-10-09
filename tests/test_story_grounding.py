import copy
from types import SimpleNamespace

import pytest

from factory.config import Config
from factory.providers.base import ProviderTimeoutError
from factory.script.grounding import (
    apply_safe_rewrites, bind_facts, check_prompt, normalize_facts, normalize_report,
)
from factory.utils import read_json


def observations():
    return [{"id": "F%d" % i, "start": i * 10, "end": i * 10 + 5,
             "visible": text} for i, text in enumerate((
                 "女孩独自走到林地，弯腰拾石", "女孩发现锅内有奇怪的东西", "女人站在门外怒视女孩",
                 "女孩从侧门离开房子", "警察听女孩陈述", "女孩坐在警车旁，屋门被警察守住"))]


def draft():
    return {"title": "林间屋", "story_logic": "遇险后离开并求助", "sentences": [
        {"text": fact["visible"] + "。", "fact_ids": [fact["id"]], "role": "hook" if i == 0 else "payoff"}
        for i, fact in enumerate(observations())]}


def review(status="supported", count=6):
    return {"beats": [{"index": i + 1, "status": status, "reason": ""} for i in range(count)],
            "causal_complete": True, "missing_links": []}


def test_writer_cannot_forge_its_own_visual_or_source_times():
    data = draft()
    data['sentences'][0].update(evidence_start=900, evidence_end=999, visual='女孩被人绑架',
                                source_windows=[{'start': 900, 'end': 999}])
    bound = bind_facts(data, observations())
    first = bound['sentences'][0]
    assert (first['evidence_start'], first['evidence_end']) == (0, 5)
    assert first['visual'] == observations()[0]['visible']
    assert 'source_windows' not in first
    assert data['sentences'][0]['evidence_start'] == 900


@pytest.mark.parametrize("ids", [[], ['missing'], ['F0', 'F0'], ['F0', 'F1', 'F2', 'F3'], 'F0'])
def test_unknown_duplicate_or_unbounded_fact_bindings_are_invalid(ids):
    data = draft()
    data['sentences'][0]['fact_ids'] = ids
    bound = bind_facts(data, observations())
    assert bound['sentences'][0]['evidence_start'] == -1


def test_multiple_facts_generate_only_verified_source_windows():
    data = draft()
    data['sentences'][0]['fact_ids'] = ['F1', 'F0']
    bound = bind_facts(data, observations())
    assert bound['sentences'][0]['source_windows'] == [
        {'start': 0, 'end': 5, 'weight': .5}, {'start': 10, 'end': 15, 'weight': .5}]


@pytest.mark.parametrize('mutation', ['too_few', 'duplicate', 'nan', 'out_of_bounds', 'overlap', 'no_visual'])
def test_bad_visual_observations_cannot_enter_the_fact_cache(mutation):
    facts = observations()
    if mutation == 'too_few':
        facts.pop()
    elif mutation == 'duplicate':
        facts[1]['id'] = facts[0]['id']
    elif mutation == 'nan':
        facts[0]['start'] = float('nan')
    elif mutation == 'out_of_bounds':
        facts[0]['start'] = -1
    elif mutation == 'overlap':
        facts[0]['end'] = 13
    else:
        facts[0]['visible'] = ''
    with pytest.raises(ValueError, match='分镜事实表'):
        normalize_facts({'facts': facts}, 0, 100)


@pytest.mark.parametrize('bad', [None, [], 'story', {'sentences': 'bad'}, {'sentences': [None]}, {'sentences': [{'text': 5}]}])
def test_malformed_writer_responses_return_reviewable_errors_instead_of_attribute_crashes(bad):
    from factory.mining.visual_story import _validate, _bind_known_shots
    assert _validate(bind_facts(bad, observations()), 0, 100)
    assert _validate(_bind_known_shots(bad, []), 0, 100)


def test_checker_uses_independent_observations_and_excludes_all_comment_asides():
    data = bind_facts(draft(), observations())
    data['sentences'][0]['text'] = '女孩独自拾石（忘记带玫瑰花了），然后发现异常(大家说怪不怪)。'
    prompt = check_prompt(data, observations())
    assert '忘记带玫瑰花' not in prompt and '大家说怪不怪' not in prompt
    assert '女孩独自走到林地，弯腰拾石' in prompt


@pytest.mark.parametrize('status', ['supported', 'inference'])
def test_reasonable_motivation_is_allowed_without_claiming_a_numeric_score(status):
    result = normalize_report(review(status), 6)
    assert result['status'] == 'pass' and result['causal_complete']
    assert result['model_score'] is None
    assert result['evidence_limit']


@pytest.mark.parametrize('fault', ['missing_beat', 'duplicate', 'no_chain', 'wrong_status', 'unclear', 'unsupported', 'gap'])
def test_incomplete_uncertain_or_contradictory_reviews_cannot_silently_pass(fault):
    answer = review()
    if fault == 'missing_beat':
        answer['beats'].pop()
    elif fault == 'duplicate':
        answer['beats'][-1]['index'] = 1
    elif fault == 'no_chain':
        answer.pop('causal_complete')
    elif fault == 'wrong_status':
        answer['beats'][0]['status'] = 'probably'
    elif fault == 'gap':
        answer['missing_links'] = [{'index': 5, 'reason': '女孩还困在屋内却突然坐上警车'}]
    else:
        answer['beats'][0].update(status=fault, reason='拾石画面不能证明被人带走')
    result = normalize_report(answer, 6)
    assert result['status'] == 'review_required' and result['issues']


def test_draft_repair_preserves_every_source_binding_and_original():
    original = bind_facts(draft(), observations())
    original['sentences'][0]['text'] = '女孩被带到林地里。'
    snapshot = copy.deepcopy(original)
    changed = apply_safe_rewrites(original, {'safe_rewrites': [{'index': 1, 'text': '女孩独自来到林地拾石。'}]})
    assert original == snapshot
    assert changed['sentences'][0]['text'] == '女孩独自来到林地拾石。'
    assert {k: v for k, v in changed['sentences'][0].items() if k != 'text'} == {
        k: v for k, v in original['sentences'][0].items() if k != 'text'}
    assert apply_safe_rewrites(original, {'safe_rewrites': [{'index': 1, 'text': '女孩出门（评论）。'}]}) is None


def small_pipeline(tmp_path):
    return SimpleNamespace(config=Config({'workspace': str(tmp_path / 'ws')}),
                           router=SimpleNamespace(get=lambda role: SimpleNamespace(is_mock=False)),
                           _check_cancel=lambda: None)


def test_only_one_revision_is_adopted_and_only_after_independent_recheck(tmp_path, monkeypatch):
    import factory.mining.visual_story as director
    calls = []
    answers = [review(), review()]
    answers[0]['beats'][0].update(status='unsupported', reason='没有同行者')
    answers[0]['safe_rewrites'] = [{'index': 1, 'text': '女孩独自来到林地拾石。'}]
    def call(p, stage, prompt, images, provider):
        calls.append((stage, images))
        return answers.pop(0)
    monkeypatch.setattr(director, '_cached_call', call)
    monkeypatch.setattr(director, '_validate', lambda *a, **k: [])
    before = bind_facts(draft(), observations())
    before['sentences'][0]['text'] = '女孩被人带到林地。'
    data, report, timeout = director._check_new_draft(small_pipeline(tmp_path), before, observations(), tmp_path, 0, 100, (0, 1000))
    assert data['sentences'][0]['text'] == '女孩独自来到林地拾石。'
    assert report['status'] == 'pass' and report['draft_revised']
    assert calls == [('grounding', []), ('grounding', [])] and not timeout


def test_unverified_repair_is_not_applied(tmp_path, monkeypatch):
    import factory.mining.visual_story as director
    answer = review('unsupported')
    answer['safe_rewrites'] = [{'index': 1, 'text': '女孩独自走过林地。'}]
    monkeypatch.setattr(director, '_cached_call', lambda *a, **k: answer)
    monkeypatch.setattr(director, '_validate', lambda *a, **k: [])
    before = bind_facts(draft(), observations())
    data, report, timeout = director._check_new_draft(small_pipeline(tmp_path), before, observations(), tmp_path, 0, 100, (0, 1000))
    assert data == before and report['status'] == 'review_required' and not timeout


def test_timeout_preserves_draft_and_records_honest_manual_review(tmp_path, monkeypatch):
    import factory.mining.visual_story as director
    calls = []
    def timeout(*a, **k):
        calls.append(1)
        raise ProviderTimeoutError('fixture timeout')
    monkeypatch.setattr(director, '_cached_call', timeout)
    before = bind_facts(draft(), observations())
    data, report, failed = director._check_new_draft(small_pipeline(tmp_path), before, observations(), tmp_path, 0, 100, (0, 1000))
    assert data == before and failed and calls == [1]
    assert report['status'] == 'review_required' and '超时' in report['issues'][0]['reason']
    assert read_json(tmp_path / 'story_grounding.json') == report


def test_cached_call_restores_provider_limits_and_reuses_same_facts(tmp_path):
    import factory.mining.visual_story as director
    calls = []
    class Provider:
        model = 'fixture'
        conf = {'max_tokens': 700, 'base_url': 'fixture'}
        max_retry = 3
        last_usage = {'total_tokens': 5}
        def complete_json(self, prompt, **kwargs):
            calls.append((self.conf['max_tokens'], self.max_retry, kwargs['images']))
            return {'facts': observations()}
    p = small_pipeline(tmp_path)
    p.state = SimpleNamespace(artifacts={})
    p.save_state = lambda: None
    image = tmp_path / 'frame.jpg'
    image.write_bytes(b'fixture')
    provider = Provider()
    first = director._cached_call(p, 'facts', 'fixture prompt', [str(image)], provider)
    second = director._cached_call(p, 'facts', 'fixture prompt', [str(image)], provider)
    assert first == second and calls == [(2000, 1, [str(image)])]
    assert provider.conf['max_tokens'] == 700 and provider.max_retry == 3
    assert p.state.artifacts['token_saving']['cache_hits'] == 1


def director_pipeline(tmp_path):
    from factory.pipeline import ProductionPipeline
    from factory.models import Evidence, MediaIndex, SceneSegment, ScriptSentence, ScriptVariant, StoryCandidate, VideoMeta, to_dict
    from factory.utils import write_json
    p = ProductionPipeline(Config({'workspace': str(tmp_path / 'ws'), 'params': {
        'comment_hook_text': '', 'review_mode': 'semi', 'min_duration': 5, 'max_duration': 50}}), 'fixture')
    p.state.artifacts['director_local_gate'] = True
    p.state.completed_stages = ['SCRIPT_GENERATED', 'CANDIDATE_FILTERED', 'TTS_GENERATED', 'MATCHING', 'RENDERING']
    facts = observations()
    sentences = [ScriptSentence(index=i + 1, text=fact['visible'] + '。', role='hook' if i == 0 else 'payoff',
                                source_evidence=Evidence(start=fact['start'], end=fact['end'], scene_ids=[i + 1]))
                 for i, fact in enumerate(facts)]
    variant = ScriptVariant(variant_id='D1', candidate_id='C001', sentences=sentences,
                            full_text=''.join(sentence.text for sentence in sentences),
                            review_comment_hooks=[{'after_sentence': 3, 'offset': 3, 'text': '忘记带玫瑰花了'}])
    from factory.script.comment_hook import apply_comment_hook
    apply_comment_hook(p, variant)
    candidate = StoryCandidate(candidate_id='C001', title='原文案', source_start=0, source_end=65,
                               status='approved', scripts=[variant], selected_script='D1')
    p.candidates = [candidate]
    p.index = MediaIndex(video=VideoMeta(), scenes=[SceneSegment(scene_id=i + 1, start=fact['start'], end=fact['end'],
                                                               visual=fact['visible']) for i, fact in enumerate(facts)])
    write_json(p.job_dir / 'index' / 'video_index.json', to_dict(p.index))
    write_json(p.job_dir / 'analysis' / 'director' / 'story_01' / 'visual_facts.json', {'facts': facts})
    from shot_catalog_fixture import write_trusted_catalog
    write_trusted_catalog(p,candidate,facts)
    p._save_candidates()
    return p, candidate


def test_explicit_director_rewrite_is_one_new_draft_with_comments_and_requires_approval(tmp_path, monkeypatch):
    import factory.mining.visual_story as director
    from factory.models import ClipResult
    p, candidate = director_pipeline(tmp_path)
    original = copy.deepcopy(candidate.scripts[0])
    calls = []
    def call(pipeline, stage, prompt, images, provider):
        calls.append(stage)
        assert not images
        return draft() if stage == 'script' else review()
    monkeypatch.setattr(director, '_cached_call', call)
    monkeypatch.setattr(p.router, 'get', lambda role: SimpleNamespace(is_mock=False))
    p.clips = [ClipResult(clip_id='old', candidate_id='C001')]
    p._regenerate_candidate(candidate)
    assert calls == ['script']
    assert candidate.status == 'review_required' and candidate.selected_script == 'D2'
    assert candidate.scripts[0] == original
    assert len(candidate.scripts) == 2
    assert candidate.scripts[-1].full_text.count('忘记带玫瑰花了') == 1
    assert candidate.scripts[-1].review_comment_hooks[0]['after_sentence'] == 3
    assert not p.completed('TTS_GENERATED') and p.state.artifacts['script_review_pending']
    assert p.clips[0].status == 'outdated_script'
    assert p.production_queue() == []


def test_bad_rewrite_preserves_approved_original_and_voice_stages(tmp_path, monkeypatch):
    import factory.mining.visual_story as director
    p, candidate = director_pipeline(tmp_path)
    original, completed = copy.deepcopy(candidate), list(p.state.completed_stages)
    monkeypatch.setattr(director, '_cached_call', lambda *a, **k: {'sentences': []})
    with pytest.raises(ValueError, match='原文案保留'):
        p._regenerate_candidate(candidate)
    assert candidate == original and p.state.completed_stages == completed


def test_edited_story_preflight_never_changes_text_and_reports_specific_contradiction(tmp_path, monkeypatch):
    import factory.mining.visual_story as director
    p, candidate = director_pipeline(tmp_path)
    candidate.scripts[0].sentences[0].text = '女孩被人绑住并带进林地。'
    snapshot = copy.deepcopy(candidate)
    answer = review()
    answer['beats'][0].update(status='unsupported', reason='画面只有女孩独自拾石，没有绑架或同行者')
    monkeypatch.setattr(director, '_cached_call', lambda *a, **k: answer)
    monkeypatch.setattr(p.router, 'get', lambda role: SimpleNamespace(is_mock=False, model='fixture', conf={}))
    result = director.review_candidate_story(p, candidate)
    assert result['status'] == 'fail' and result['verified_text_hash'] and result['signature']
    assert '拾石' in result['reason'] and candidate == snapshot


def test_legacy_approved_story_without_observations_is_not_reanalyzed_or_overwritten(tmp_path, monkeypatch):
    import factory.mining.visual_story as director
    p, candidate = director_pipeline(tmp_path)
    snapshot = copy.deepcopy(candidate)
    monkeypatch.setattr(director, '_candidate_facts', lambda pipeline, cand, extract=False: ([], tmp_path))
    monkeypatch.setattr(director, '_cached_call', lambda *a, **k: pytest.fail('旧批准稿不得被重新分析'))
    result = director.review_candidate_story(p, candidate)
    assert result['status'] == 'not_available' and candidate == snapshot


def test_legacy_integer_scene_ids_and_malformed_window_entries_use_evidence_bounds(tmp_path, monkeypatch):
    import factory.mining.visual_story as director
    p, candidate = director_pipeline(tmp_path)
    candidate.scripts[0].scene_matches = [1, 2, {'sentence_index': 3, 'source_windows': [5, {'start': 'bad'}]}]
    captured = []
    def check(*args, **kwargs):
        captured.append(args[2])
        return review()
    monkeypatch.setattr(director, '_cached_call', check)
    monkeypatch.setattr(p.router, 'get', lambda role: SimpleNamespace(is_mock=False, model='fixture', conf={}))
    result = director.review_candidate_story(p, candidate)
    assert result['status'] == 'pass'
    assert all(fact['visible'] in captured[0] for fact in observations())


def test_empty_candidate_is_unavailable_without_any_model_request(tmp_path, monkeypatch):
    import factory.mining.visual_story as director
    p, candidate = director_pipeline(tmp_path)
    candidate.scripts = []
    monkeypatch.setattr(director, '_cached_call', lambda *a, **k: pytest.fail('empty draft must not be billed'))
    assert director.review_candidate_story(p, candidate)['status'] == 'not_available'


def test_new_draft_check_reused_before_tts_without_second_model_call(tmp_path, monkeypatch):
    import factory.mining.visual_story as director
    p, candidate = director_pipeline(tmp_path)
    reviewer = SimpleNamespace(is_mock=False, model='fixture', conf={'base_url': 'fixture'})
    monkeypatch.setattr(p.router, 'get', lambda role: reviewer)
    calls = []
    monkeypatch.setattr(director, '_cached_call', lambda *a, **k: calls.append(a[1]) or review())
    data, checked, _ = director._check_new_draft(p, bind_facts(draft(), observations()), observations(),
                                               tmp_path, 0, 65, (20, 500))
    candidate.match_report = {'story_grounding': checked}
    result = director.review_candidate_story(p, candidate)
    assert calls == ['grounding'] and result['cache_hit']
    assert result['signature'] == checked['signature'] and checked['facts_signature']
    assert result['status'] == 'pass'


@pytest.mark.parametrize('change', ['body', 'facts', 'binding', 'provider'])
def test_grounding_reuse_invalidates_when_text_facts_evidence_or_model_changes(tmp_path, monkeypatch, change):
    import factory.mining.visual_story as director
    p, candidate = director_pipeline(tmp_path)
    reviewer = SimpleNamespace(is_mock=False, model='fixture', conf={'base_url': 'fixture'})
    monkeypatch.setattr(p.router, 'get', lambda role: reviewer)
    calls = []
    monkeypatch.setattr(director, '_cached_call', lambda *a, **k: calls.append(a[1]) or review())
    _, checked, _ = director._check_new_draft(p, bind_facts(draft(), observations()), observations(), tmp_path, 0, 65, (20, 500))
    candidate.match_report = {'story_grounding': checked}
    if change == 'body':
        candidate.scripts[0].sentences[0].text = '她独自走入另一条小路。'
    elif change == 'facts':
        changed = observations()
        changed[0]['visible'] = '女孩独自站在窗前'
        monkeypatch.setattr(director, '_candidate_facts', lambda *a, **k: (changed, tmp_path))
    elif change == 'binding':
        candidate.scripts[0].sentences[0].source_evidence.start = 10
        candidate.scripts[0].sentences[0].source_evidence.end = 15
    else:
        reviewer.model = 'different-model'
    result = director.review_candidate_story(p, candidate)
    assert calls == ['grounding', 'grounding'] and result['signature'] != checked['signature']


def test_fact_binding_order_is_canonical_for_cache_keys():
    import factory.mining.visual_story as director
    before = draft()
    before['sentences'][0]['fact_ids'] = ['F1', 'F0']
    after = bind_facts(before, observations())
    assert after['sentences'][0]['fact_ids'] == ['F0', 'F1']
    reviewer = SimpleNamespace(model='fixture', conf={})
    assert director._grounding_identity(before, observations(), reviewer) == director._grounding_identity(after, observations(), reviewer)
