from types import SimpleNamespace

import pytest

from factory.config import Config
from factory.pipeline import ProductionPipeline
from factory.providers.base import ProviderTimeoutError
from factory.mining.story_discovery import (
    discovery_windows, normalize_window, normalize_merge, preserve_catalogue,
)


def episode(start, end, title='一', opening=True, outcome=True):
    return {'title': title, 'source_start': start, 'source_end': end,
            'story_summary': '遇到陌生人→逃离→获得帮助', 'protagonist': '红衣女孩',
            'opening_seen': opening, 'outcome_seen': outcome}


def result(*parts, excluded=None, uncertain=None):
    return {'episodes': list(parts), 'excluded': excluded or [], 'uncertain': uncertain or []}


def window_result(lo, hi, *parts):
    excluded, cursor = [], lo
    for part in sorted(parts, key=lambda item: item['source_start']):
        if part['source_start'] > cursor:
            excluded.append({'source_start': cursor, 'source_end': part['source_start'], 'reason': '测试片头或无故事转场'})
        cursor = max(cursor, part['source_end'])
    if cursor < hi:
        excluded.append({'source_start': cursor, 'source_end': hi, 'reason': '测试片尾'})
    return result(*parts, excluded=excluded)


def test_windows_cover_short_and_long_source_without_count_or_tail_cutoff():
    assert discovery_windows(20) == [(0, 20)]
    spans = discovery_windows(7200)
    assert spans[0][0] == 0 and spans[-1][1] == 7200
    assert len(spans) > 20
    assert all(left[1] > right[0] for left, right in zip(spans, spans[1:]))
    assert all(end-start <= 240 for start, end in spans)


@pytest.mark.parametrize('bad', [None, {}, {'episodes': []}, result(),
    result(episode(0, 2)), result(episode(-1, 20)), result(episode(0, float('inf'))),
    result(dict(episode(0, 20), opening_seen='true'))])
def test_malformed_discovery_cannot_become_zero_valid_stories(bad):
    with pytest.raises(ValueError):
        normalize_window(bad, 0, 240, 'W000')


def test_cross_window_story_is_merged_without_inventing_endpoints():
    left = normalize_window(result(episode(180, 240, opening=True, outcome=False)), 0, 240, 'W000')['episodes']
    right = normalize_window(result(episode(195, 330, opening=False, outcome=True)), 195, 360, 'W001')['episodes']
    merged = {'stories': [{'title': '整篇', 'story_summary': '完整因果',
                           'observation_ids': ['W000E01', 'W001E01']}], 'unresolved': []}
    stories, uncertain = normalize_merge(merged, left+right)
    assert uncertain == [] and len(stories) == 1
    assert (stories[0]['source_start'], stories[0]['source_end']) == (180, 330)
    with pytest.raises(ValueError, match='遗漏'):
        normalize_merge({'stories': [], 'unresolved': []}, left+right)
    left[0]['opening_seen'] = False
    with pytest.raises(ValueError, match='起因和结果'):
        normalize_merge(merged, left+right)


def test_catalogue_append_preserves_user_story_number_and_original_span():
    previous = {'candidates': [episode(150, 260, title='原已编辑故事')]}
    fresh = [episode(10, 100, title='新发现前章节'), episode(145, 265, title='原故事')]
    kept = preserve_catalogue(previous, fresh)
    assert [part['title'] for part in kept] == ['原已编辑故事', '新发现前章节']
    assert kept[0]['source_start'] == 150


def test_omitted_tail_is_uncertain_instead_of_complete_window():
    normalized = normalize_window(result(episode(0, 30)), 0, 240, 'W000')
    assert normalized['uncertain'] == [{'source_start': 30, 'source_end': 240,
                                       'reason': '这段时间未列故事或排除依据，不能静默忽略，需局部复核'}]


@pytest.mark.parametrize('refs', [['W000E01', 'W000E01'], [{}], [[]]])
def test_merge_rejects_duplicate_or_nonstring_unresolved_ids(refs):
    parts = normalize_window(window_result(0, 20, episode(0, 20)), 0, 20, 'W000')['episodes']
    with pytest.raises(ValueError):
        normalize_merge({'stories': [], 'unresolved': [{'observation_ids': refs, 'reason': '不确定'}]}, parts)


def test_old_broad_story_cannot_silently_swallow_two_new_independent_chapters():
    conflicts = []
    previous = {'candidates': [episode(0, 300, title='原已编辑长选题')]}
    fresh = [episode(10, 100, title='独立第一章'), episode(180, 280, title='独立第二章')]
    kept = preserve_catalogue(previous, fresh, conflicts)
    assert kept == previous['candidates']
    assert len(conflicts) == 1 and len(conflicts[0]['options']) == 2
    assert '不能静默' in conflicts[0]['reason']


def fixture_pipeline(tmp_path, monkeypatch, limit=0, duration=360):
    import factory.mining.visual_story as director
    source = tmp_path / 'source.mp4'
    source.write_bytes(b'fixture video')
    p = ProductionPipeline(Config({'workspace': str(tmp_path / 'ws'), 'params': {
        'max_clips': limit, 'review_mode': 'semi', 'comment_hook_text': ''}}), str(source))
    p.state.completed_stages = ['URL_ANALYZING']
    p.state.artifacts.update(meta={}, video_path=str(source))
    provider = SimpleNamespace(is_mock=False, model='fixture')
    p.router = SimpleNamespace(get=lambda role: provider, is_mock=lambda role: False)
    monkeypatch.setattr(director, 'media_duration', lambda path: duration)
    monkeypatch.setattr(director, 'storyboard', lambda *args, **kw: ['fixture.jpg'])
    monkeypatch.setattr(director, '_validate', lambda *args, **kw: [])
    from shot_catalog_fixture import inject_observer
    inject_observer(monkeypatch, director)
    return p, director


def test_auto_generates_actual_independent_count_and_never_uses_one_story_shortcut(tmp_path, monkeypatch):
    import factory.story_library as library
    p, director = fixture_pipeline(tmp_path, monkeypatch)
    monkeypatch.setattr(library, 'find_reviewed_story', lambda *a: pytest.fail('自动模式不可只复用一条'))
    monkeypatch.setattr(library, 'find_story_outline', lambda *a: pytest.fail('自动模式不可省略全片发现'))
    stages = []
    fact_call = [0]

    def call(pipeline, stage, prompt, images, provider):
        stages.append(stage)
        if stage == 'discover_window':
            if stages.count(stage) == 1:
                return window_result(0, 240, episode(10, 100, '第一篇'), episode(180, 240, '跨块篇', outcome=False))
            return window_result(195, 360, episode(195, 330, '跨块篇', opening=False))
        if stage == 'discover_global':
            assert not images
            return {'stories': [{'title':'第一篇','story_summary':'第一篇完整因果','continuity_reason':'独立主角事件',
                                 'observation_ids':['W000E01']},
                                {'title': '跨块完整篇', 'story_summary': '完整因果','continuity_reason':'同一主角跨块连续事件',
                                 'observation_ids': ['W000E02', 'W001E01']}], 'unresolved': []}
        if stage == 'facts':
            fact_call[0] += 1
            lo = 10 if fact_call[0] == 1 else 180
            return {'facts': [{'id': 'F%d' % i, 'start': lo+i*5, 'end': lo+i*5+4,
                              'visible': '事件%d' % i} for i in range(6)]}
        if stage == 'script':
            assert not images
            return {'title': '故事', 'story_logic': '遇险到脱险', 'sentences': [
                {'text': '事件%d。' % i, 'role': 'hook' if i == 0 else 'payoff',
                 'fact_ids': ['F%d' % i]} for i in range(6)]}
        if stage == 'grounding':
            return {'beats': [{'index': i+1, 'status': 'supported'} for i in range(6)],
                    'causal_complete': True, 'missing_links': []}
        pytest.fail(stage)

    monkeypatch.setattr(director, '_cached_call', call)
    director.analyze(p)
    assert len(p.candidates) == 2
    assert [(candidate.source_start, candidate.source_end) for candidate in p.candidates] == [(10, 100), (180, 330)]
    assert stages.count('script') == 2 and stages.count('facts') == 2
    assert p.state.artifacts['story_discovery']['complete']
    assert p.state.artifacts['story_discovery']['coverage_seconds'] == 360
    assert not p.completed('TTS_GENERATED')
    stages.clear()
    director.analyze(p)
    assert stages == []


def test_auto_timeout_saves_first_window_and_resumes_without_repeated_calls(tmp_path, monkeypatch):
    p, director = fixture_pipeline(tmp_path, monkeypatch)
    calls, fail = [], [True]

    def call(pipeline, stage, prompt, images, provider):
        calls.append(stage)
        if stage != 'discover_window':
            pytest.fail('零合格故事不允许编剧或配音')
        if '覆盖195.000' in prompt and fail[0]:
            raise ProviderTimeoutError('测试超时')
        lo, hi = (195, 360) if '覆盖195.000' in prompt else (0, 240)
        return result(excluded=[{'source_start': lo, 'source_end': hi, 'reason': '只有静图，没有完整动作故事'}])

    monkeypatch.setattr(director, '_cached_call', call)
    director.analyze(p)
    assert p.state.artifacts['discovery_resume_required']
    assert p.state.artifacts['story_discovery']['windows_done'] == 1
    assert not p.completed('CANDIDATE_FILTERED')
    assert p.state.artifacts['pending_stories'] == []
    fail[0] = False
    director.analyze(p)
    assert calls == ['discover_window'] * 3
    report = p.state.artifacts['story_discovery']
    assert report['complete'] and report['discovered'] == 0 and report['coverage_seconds'] == 360
    assert not p.state.artifacts['discovery_resume_required'] and not p.completed('TTS_GENERATED')


def test_uncertain_source_gets_one_cached_refinement_then_waits_for_person(tmp_path, monkeypatch):
    p, director = fixture_pipeline(tmp_path, monkeypatch, duration=20)
    calls = []

    def call(pipeline, stage, prompt, images, provider):
        calls.append(stage)
        return result(uncertain=[{'source_start': 0, 'source_end': 20, 'reason': '黑暗中无法确认事件与结果'}])

    monkeypatch.setattr(director, '_cached_call', call)
    director.analyze(p)
    assert calls == ['discover_window', 'discover_window']
    report = p.state.artifacts['story_discovery']
    assert not report['complete'] and report['manual_review_required']
    assert report['scan_complete'] and not p.state.artifacts['discovery_resume_required']
    assert not p.completed('CANDIDATE_FILTERED') and not p.completed('TTS_GENERATED')
    director.analyze(p)
    assert calls == ['discover_window', 'discover_window']


def test_manual_limit_allows_zero_with_reason_and_does_not_generate_fake_fallback(tmp_path, monkeypatch):
    p, director = fixture_pipeline(tmp_path, monkeypatch, limit=1)
    calls = []

    def call(pipeline, stage, prompt, images, provider):
        calls.append(stage)
        assert '最多选择1' in prompt
        return {'candidates': [], 'reason': '片头和静态图，没有可见结果'}

    monkeypatch.setattr(director, '_cached_call', call)
    director.analyze(p)
    assert calls == ['select'] and p.candidates == []
    assert p.state.artifacts['story_discovery']['selected'] == 0
    assert not p.completed('TTS_GENERATED')


def test_switch_one_to_auto_appends_new_chapter_without_overwriting_user_edits(tmp_path, monkeypatch):
    from factory.script.review_edit import review_text
    p, director = fixture_pipeline(tmp_path, monkeypatch, limit=1)
    stages = []

    def call(pipeline, stage, prompt, images, provider):
        stages.append(stage)
        if stage == 'select':
            return {'candidates': [episode(180, 330, '原先选中的第二章')]}
        if stage == 'discover_window':
            if '覆盖0.000' in prompt:
                return window_result(0, 240, episode(10, 100, '新发现第一章'), episode(180, 240, '原第二章', outcome=False))
            return window_result(195, 360, episode(195, 330, '原第二章', opening=False))
        if stage == 'discover_global':
            return {'stories': [{'title':'新第一章','story_summary':'第一章完整因果','continuity_reason':'不同主角独立事件',
                                 'observation_ids':['W000E01']},
                                {'title': '原第二章', 'story_summary': '完整因果','continuity_reason':'同一主角跨块连续事件',
                                 'observation_ids': ['W000E02', 'W001E01']}], 'unresolved': []}
        if stage == 'facts':
            lo = 180 if p.config.param('max_clips') == 1 else 10
            return {'facts': [{'id': 'F%d' % i, 'start': lo+i*5, 'end': lo+i*5+4,
                              'visible': '事件%d' % i} for i in range(6)]}
        if stage == 'script':
            return {'title': '草稿', 'story_logic': '完整因果', 'sentences': [
                {'text': '事件%d。' % i, 'role': 'hook' if i == 0 else 'payoff',
                 'fact_ids': ['F%d' % i]} for i in range(6)]}
        return {'beats': [{'index': i+1, 'status': 'supported'} for i in range(6)],
                'causal_complete': True, 'missing_links': []}

    monkeypatch.setattr(director, '_cached_call', call)
    director.analyze(p)
    p.apply_review('C001', 'edit', new_text=review_text(p.candidates[0].scripts[0]).replace('事件0', '我的手动修改'))
    p.config.params['max_clips'] = 0
    director.analyze(p)
    assert [(candidate.candidate_id, candidate.source_start) for candidate in p.candidates] == [('C001', 180), ('C002', 10)]
    assert '我的手动修改' in p._pick_variant(p.candidates[0]).full_text
    assert stages.count('script') == 2 and stages.count('facts') == 2
    assert p.state.artifacts['story_discovery']['mode'] == 'auto'
    assert len({scene.scene_id for scene in p.index.scenes}) == 12
    stages.clear()
    p.config.params['max_clips'] = 1
    director.analyze(p)
    assert stages == [] and len(p.candidates) == 2
    assert p.state.artifacts['story_discovery']['selected'] == 1


def test_small_complete_chapter_is_allowed_inside_long_compilation():
    from factory.mining.visual_story import _valid_story_span
    assert _valid_story_span(600, 620, 7200)


def test_auto_rejects_unusable_observations_without_blocking_other_story(tmp_path, monkeypatch):
    p, director = fixture_pipeline(tmp_path, monkeypatch, duration=200)
    stages = []
    def call(pipeline, stage, prompt, images, provider):
        stages.append(stage)
        if stage == 'discover_window':
            return window_result(0, 200, episode(10, 70, '看似第一篇'), episode(100, 190, '有效第二篇'))
        if stage == 'discover_global':
            return {'stories':[{'title':'看似第一篇','story_summary':'第一主线','continuity_reason':'独立人物事件',
                                'observation_ids':['W000E01']},
                               {'title':'第二篇','story_summary':'第二主线','continuity_reason':'独立人物事件',
                                'observation_ids':['W000E02']}],'unresolved':[]}
        if stage == 'facts':
            if '范围10.000至70.000' in prompt:
                return {'facts': []}  # no usable observed shot IDs, rather than a literary minimum beat count
            return {'facts': [{'id': 'F%d' % i, 'start': 100+i*5, 'end': 104+i*5,
                              'visible': '事件%d' % i} for i in range(6)]}
        if stage == 'script':
            return {'title': '有效故事', 'story_logic': '完整因果', 'sentences': [
                {'text': '事件%d。' % i, 'role': 'hook' if i == 0 else 'payoff',
                 'fact_ids': ['F%d' % i]} for i in range(6)]}
        return {'beats': [{'index': i+1, 'status': 'supported'} for i in range(6)],
                'causal_complete': True, 'missing_links': []}
    monkeypatch.setattr(director, '_cached_call', call)
    director.analyze(p)
    assert len(p.candidates) == 1 and p.candidates[0].candidate_id == 'C002'
    assert stages.count('script') == 2
    assert p.state.artifacts['story_discovery']['rejected_count'] == 1
    assert '时间不合法' in p.state.artifacts['rejected_discovered_stories'][0]['reason']
    assert not p.completed('TTS_GENERATED')


def test_large_manual_limit_does_not_leave_unbounded_select_token_cap(tmp_path, monkeypatch):
    p, director = fixture_pipeline(tmp_path, monkeypatch, limit=9999)
    used = []
    provider = SimpleNamespace(model='fixture', conf={'max_tokens': 2000}, max_retry=3,
                               last_usage={'total_tokens': 1})
    def complete(prompt, **kw):
        used.append(provider.conf['max_tokens'])
        return {'candidates': []}
    provider.complete_json = complete
    director._cached_call(p, 'select', 'fixture', [], provider)
    assert used == [6000] and provider.conf['max_tokens'] == 2000 and provider.max_retry == 3
    assert p.state.artifacts['token_saving']['calls'] == 1
