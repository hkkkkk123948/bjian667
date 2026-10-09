from types import SimpleNamespace

import pytest
import requests

from factory.config import Config
from factory.pipeline import ProductionPipeline
from factory.providers.base import ProviderTimeoutError
from factory.providers.openai_compat import OpenAICompatProvider
from factory.utils import read_json


def test_timeout_is_not_silently_retried():
    class TimeoutProvider(OpenAICompatProvider):
        calls = 0

        def _request(self, *a, **k):
            self.calls += 1
            raise requests.ReadTimeout("fixture timeout")

    provider = TimeoutProvider({"type":"openai", "model":"fixture", "timeout":120, "max_retry":3})
    with pytest.raises(ProviderTimeoutError, match="已保存"):
        provider.complete_json("fixture")
    assert provider.calls == 1


def test_completed_story_and_user_edits_survive_later_timeout(tmp_path, monkeypatch):
    import factory.mining.visual_story as director
    from factory.script.review_edit import review_text
    source = tmp_path / "source.mp4"
    source.write_bytes(b"fixture")
    p = ProductionPipeline(Config({"workspace":str(tmp_path / "ws"), "params":{
        "max_clips":2, "review_mode":"semi", "comment_hook_text":""}}), str(source))
    p.state.completed_stages = ["URL_ANALYZING"]
    p.state.artifacts.update(meta={}, video_path=str(source))
    provider = SimpleNamespace(is_mock=False)
    p.router = SimpleNamespace(get=lambda role:provider, is_mock=lambda role:False)
    monkeypatch.setattr(director, "media_duration", lambda p:300)
    monkeypatch.setattr(director, "storyboard", lambda *a,**k:["fixture.jpg"])
    monkeypatch.setattr(director, "_validate", lambda *a,**k:[])
    from shot_catalog_fixture import inject_observer
    inject_observer(monkeypatch,director)
    calls = []
    fail = [True]

    def draft(number):
        lo = 0 if number == 1 else 150
        return {"title":"故事%d" % number, "story_logic":"遇险到求助脱险", "sentences":[
            {"text": "镜头事件%d。" % i, "role":"hook" if i == 0 else "payoff",
             "visual":"事实%d" % i, "fact_ids":["F%d" % i],
             "evidence_start":lo+i*10, "evidence_end":lo+i*10+5} for i in range(6)]}

    def call(pipeline, stage, prompt, images, model):
        calls.append(stage)
        if stage == "select":
            return {"candidates":[{"title":"一", "source_start":0, "source_end":110, "story_summary":"第一段"},
                                  {"title":"二", "source_start":150, "source_end":260, "story_summary":"第二段"}]}
        if stage == "facts":
            lo = 0 if calls.count("facts") == 1 else 150
            return {"facts":[{"id":"F%d" % i, "start":lo+i*10, "end":lo+i*10+5,
                              "visible":"事实%d" % i} for i in range(6)]}
        if stage == "grounding":
            return {"beats":[{"index":i+1,"status":"supported"} for i in range(6)],
                    "causal_complete":True,"missing_links":[]}
        if calls.count("script") == 1:
            return draft(1)
        if fail[0]:
            raise ProviderTimeoutError("fixture timeout")
        return draft(2)

    monkeypatch.setattr(director, "_cached_call", call)
    state = p.run()
    assert state.status == "CANDIDATE_FILTERED"
    assert not p.completed("CANDIDATE_FILTERED")
    assert len(read_json(p.job_dir / "analysis/story_candidates.json")) == 1
    assert state.artifacts["pending_stories"][0]["number"] == 2
    first = p.candidates[0]
    p.apply_review(first.candidate_id, "edit", new_text=review_text(first.scripts[0]).replace("镜头事件0", "修改后的开场"))
    fail[0] = False
    p.run()
    assert len(p.candidates) == 2 and p.completed("CANDIDATE_FILTERED")
    assert p.state.artifacts["pending_stories"] == []
    assert "修改后的开场" in p._pick_variant(p.candidates[0]).full_text
    assert calls.count("select") == 1 and calls.count("script") == 3
    assert not p.completed("TTS_GENERATED")


def test_new_story_budget_tracks_requested_duration_and_native_voice_speed():
    from factory.mining.visual_story import _word_budget
    short=SimpleNamespace(config=Config({'params':{'min_duration':16,'max_duration':30},'tts':{'speed':1.2}}))
    long=SimpleNamespace(config=Config({'params':{'min_duration':35,'max_duration':50},'tts':{'speed':1.2}}))
    low_short,high_short=_word_budget(short)
    low_long,high_long=_word_budget(long)
    assert low_long>220 and high_long>high_short and low_long>low_short


def test_shorter_source_story_is_not_silently_extended_into_the_next_chapter(tmp_path, monkeypatch):
    import factory.mining.visual_story as director
    source = tmp_path / 'source.mp4'
    source.write_bytes(b'fixture')
    p = ProductionPipeline(Config({'workspace':str(tmp_path / 'ws'), 'params':{
        'max_clips':1,'review_mode':'semi','comment_hook_text':''}}), str(source))
    p.state.completed_stages = ['URL_ANALYZING']
    p.state.artifacts.update(meta={},video_path=str(source))
    provider = SimpleNamespace(is_mock=False)
    p.router = SimpleNamespace(get=lambda role:provider,is_mock=lambda role:False)
    monkeypatch.setattr(director,'media_duration',lambda path:300)
    monkeypatch.setattr(director,'storyboard',lambda *a,**k:['fixture.jpg'])
    monkeypatch.setattr(director,'_validate',lambda *a,**k:[])
    from shot_catalog_fixture import inject_observer
    inject_observer(monkeypatch,director)
    calls = []
    def call(pipeline,stage,prompt,images,model):
        calls.append(stage)
        if stage == 'select':
            return {'candidates':[{'title':'短篇','source_start':10,'source_end':75,'story_summary':'先遇险后离开'}]}
        if stage == 'facts':
            return {'facts':[{'id':'F%d' % i,'start':10+i*10,'end':15+i*10,'visible':'画面事件%d' % i} for i in range(6)]}
        if stage == 'grounding':
            return {'beats':[{'index':i+1,'status':'supported'} for i in range(6)],'causal_complete':True,'missing_links':[]}
        assert not images  # expensive writer receives cached facts, never the vision sheets again
        return {'title':'完整短篇','story_logic':'遇险→应对→离开','sentences':[
            {'text':'画面事件%d。' % i,'role':'hook' if i==0 else 'payoff','fact_ids':['F%d' % i]} for i in range(6)]}
    monkeypatch.setattr(director,'_cached_call',call)
    p.run()
    assert p.candidates[0].source_end == 75
    assert calls == ['select','facts','script']
    assert not p.completed('TTS_GENERATED')
