from types import SimpleNamespace

import pytest

from factory.config import Config
from factory.pipeline import ProductionPipeline


def test_new_draft_is_not_criticized_expanded_or_rewritten_for_editorial_rules(tmp_path, monkeypatch):
    import factory.mining.visual_story as director
    source = tmp_path / 'source.mp4'
    source.write_bytes(b'source')
    p = ProductionPipeline(Config({'workspace': str(tmp_path / 'ws'), 'params': {
        'max_clips': 1, 'review_mode': 'semi', 'comment_hook_text': '', 'min_duration': 35, 'max_duration': 50}}), str(source))
    p.state.completed_stages = ['URL_ANALYZING']
    p.state.artifacts.update(meta={}, video_path=str(source))
    provider = SimpleNamespace(is_mock=False)
    p.router = SimpleNamespace(get=lambda role: provider, is_mock=lambda role: False)
    monkeypatch.setattr(director, 'media_duration', lambda path: 120)
    monkeypatch.setattr(director, 'storyboard', lambda *args, **kw: ['fixture.jpg'])
    from shot_catalog_fixture import inject_observer
    inject_observer(monkeypatch,director)
    monkeypatch.setattr(director, '_check_new_draft', lambda *args, **kw: pytest.fail('生成不得自动审稿'))
    monkeypatch.setattr(director, '_repair_text', lambda *args, **kw: pytest.fail('生成不得自动改词'))
    monkeypatch.setattr(director, '_validate', lambda *args, **kw: pytest.fail('生成不得使用措辞/字数审稿规则'))
    calls = []
    long = '女孩想了很久，她不知道接下来该怎么做，但还是决定继续向前走去。' * 2
    texts = [long, long, '结束。']  # old first-length, duplicates, beat-count and word-budget rules would reject
    def call(pipeline, stage, prompt, images, model):
        calls.append(stage)
        if stage == 'select':
            return {'candidates': [{'title': '故事', 'source_start': 0, 'source_end': 100, 'story_summary': '可见故事'}]}
        if stage == 'facts':
            return {'facts': [{'id': 'F%d' % i, 'start': i*10, 'end': i*10+5, 'visible': '可见事件%d' % i} for i in range(6)]}
        if stage == 'script':
            return {'title': '用户待审故事', 'story_logic': '事件摘要', 'sentences': [
                {'text': text, 'role': 'hook' if i == 0 else 'payoff', 'fact_ids': ['F%d' % i]}
                for i, text in enumerate(texts)]}
        pytest.fail('不得发起额外文案审稿请求：' + stage)
    monkeypatch.setattr(director, '_cached_call', call)
    p.run()
    assert calls == ['select', 'facts', 'script']
    assert [sentence.text for sentence in p.candidates[0].scripts[0].sentences] == texts
    assert p.candidates[0].status == 'review_required'
    assert not p.completed('TTS_GENERATED')


def test_technical_validation_still_rejects_unknown_fact_binding_and_illegal_media_time():
    from factory.mining.visual_story import _validate_technical
    data = {'title': '故事', 'story_logic': '摘要', 'sentences': [
        {'text': '短句。', 'role': 'hook', 'visual': '真实镜头', 'evidence_start': -1, 'evidence_end': 20}]}
    assert any('时间' in issue for issue in _validate_technical(data, 0, 100))
    data['sentences'][0]['evidence_start'] = float('nan')
    assert _validate_technical(data, 0, 100)
    data['sentences'][0].update(evidence_start=0, evidence_end=1)
    assert _validate_technical(data, 0, 100) == []
