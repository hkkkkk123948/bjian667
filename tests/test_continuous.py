import hashlib
from pathlib import Path

import numpy as np

from factory.config import Config
from factory.models import ScriptSentence, ScriptVariant
from factory.providers.tts.base import write_wav
from factory.providers.tts.continuous import synth_continuous, sentence_boundaries


def test_one_complete_submission_native_speed_and_unchanged_bytes(tmp_path):
    class Recorder:
        config = Config({"workspace": str(tmp_path / "ws"), "tts": {"speed": 1.2}})
        calls = []

        def synth(self, text, out_path, speed=None):
            self.calls.append((text, speed))
            write_wav(out_path, .2 * np.sin(np.arange(24000) * .1), 24000)
            return {"audio_path": str(out_path), "duration": 1, "provider": "test_fixture"}

    texts = ["女孩发现门外一直有人。", "她不敢开门只能往后退。", "窗外突然多出了一条梯子。", "警车赶来后她才终于获救。"]
    variant = ScriptVariant(sentences=[ScriptSentence(index=i+1, text=t, estimated_seconds=1) for i, t in enumerate(texts)])
    tts = Recorder()
    result = synth_continuous(tts, variant, tmp_path / "clip1")
    assert tts.calls == [("".join(texts), 1.2)]
    assert result["audio_edits"] == []
    assert result["voice_sha256"] == result["original_audio_sha256"]
    assert hashlib.sha256(Path(result["audio_path"]).read_bytes()).hexdigest() == result["voice_sha256"]
    synth_continuous(tts, variant, tmp_path / "clip2")
    assert len(tts.calls) == 1
    tts.config.data["tts"]["speed"] = 1.1
    synth_continuous(tts, variant, tmp_path / "clip3")
    assert len(tts.calls) == 2


def test_word_boundaries_cover_the_full_audio_including_pauses():
    sents = [ScriptSentence(index=1, text="女孩退后。"), ScriptSentence(index=2, text="男人跑了。")]
    words = [{"text": c, "start": t, "end": t+.1} for c, t in zip("女孩退后男人跑了", [.1,.2,.3,.4,.9,1,1.1,1.2])]
    times = sentence_boundaries(sents, words, 1.6)
    assert times[0]["start"] == 0
    assert times[0]["end"] == times[1]["start"] == .9
    assert times[-1]["end"] == 1.6


def test_only_final_provider_timestamp_revision_is_used():
    from factory.providers.tts.bailian import _cosy_word_timings
    provisional = {"type": "sentence-synthesis", "sentence": {"index": 0, "words": [{"text":"女","begin_time":100,"end_time":300}]}}
    final = {"type":"sentence-end", "sentence":{"index":0,"words":[{"text":"女","begin_index":0,"begin_time":120,"end_time":240}]}}
    assert _cosy_word_timings([provisional, final]) == [{"text":"女","start":.12,"end":.24}]


def test_reviewed_source_reuse_requires_matching_bytes(tmp_path, monkeypatch):
    import factory.story_library as library
    from factory.utils import write_json
    source = tmp_path / "a.mp4"
    source.write_bytes(b"original")
    db = tmp_path / "library.json"
    write_json(db, {"stories":[{"source_size":8,"source_sha256":library.source_digest(source),"script":{"title":"fixture"}}]})
    monkeypatch.setattr(library, "LIBRARY", db)
    assert library.find_reviewed_story(source, {})
    source.write_bytes(b"modified")
    assert library.find_reviewed_story(source, {}) is None


def test_high_score_cannot_override_unsupported_visual_claims(tmp_path, monkeypatch):
    from factory.render.qc import semantic_final_check
    from factory.models import ClipResult

    class Provider:
        conf = {}

        def complete_json(self, *args, **kwargs):
            return {"score": 99, "mismatches": [{"sentence_index": 1, "start": 0, "end": 3,
                    "reason": "解说是收到短信，此处镜头却对应另一段厨房切菜事件",
                    "evidence": "对应镜头为男人在厨房切菜，另一句才有女孩查看手机镜头"}]}

    class Router:
        def is_mock(self, role):
            return False

        def get(self, role):
            return Provider()

    monkeypatch.setattr("factory.eval.judge.extract_frames", lambda *a, **k: ["fixture.jpg"]*8)
    clip = ClipResult(path=str(tmp_path / "clip.mp4"))
    result = semantic_final_check(clip, "女孩收到了短信。", "", Router(), params={"content_mode":"creative"})
    assert result["status"] == "fail"


def test_text_only_repair_keeps_all_visual_evidence(tmp_path, monkeypatch):
    import factory.mining.visual_story as director
    data = {"sentences":[{"text":"女孩开门。","visual":"女孩站在门口","evidence_start":10,"evidence_end":14}]}

    def call(pipeline, stage, prompt, images, provider):
        assert images == []
        assert stage == "text_repair"
        return {"texts":["女孩慢慢拉开房门，紧张地看着昏暗的走廊。"]}

    monkeypatch.setattr(director, "_cached_call", call)
    repaired = director._repair_text(None, data, None)
    assert repaired["sentences"][0]["evidence_start"] == 10
    assert repaired["sentences"][0]["evidence_end"] == 14
    assert data["sentences"][0]["text"] == "女孩开门。"


def test_legacy_speech_verdict_reuses_intact_cache_without_gate_or_new_charge(tmp_path, monkeypatch):
    import factory.providers.tts.continuous as continuous
    from factory.utils import write_json

    class Recorder:
        config = Config({"workspace": str(tmp_path / "ws"), "tts": {"speed": 1.2}})
        calls = 0

        def synth(self, text, out_path, **kwargs):
            self.calls += 1
            write_wav(out_path, .1 * np.sin(np.arange(24000) * .1), 24000)
            return {"audio_path": str(out_path), "provider": "test_fixture", "duration": 1}

    variant = ScriptVariant(sentences=[ScriptSentence(index=i+1, text=t, estimated_seconds=1) for i, t in enumerate([
        "女孩发现门外一直有人。", "她赶紧后退并锁好了门。", "窗外忽然出现一个人影。", "警察最后把她安全救走。" ])])
    tts = Recorder()
    first = continuous.synth_continuous(tts, variant, tmp_path / "clip")
    metadata = next((tmp_path / "ws/.cache/tts_continuous").glob("*.json"))
    import json
    data = json.loads(metadata.read_text(encoding="utf-8"))
    data["content_check"] = {"status": "fail", "issues": [{"kind":"repeated_phrase", "text":"警察最后把她安全救走"}]}
    write_json(metadata, data)
    data["content_failure_count"] = 2
    write_json(metadata, data)
    second = continuous.synth_continuous(tts, variant, tmp_path / "clip2")
    assert tts.calls == 1
    assert second['voice_sha256'] == first['voice_sha256']
    assert second['audio_edits'] == []
    assert second['synthesis_identity'] == first['synthesis_identity']


def test_voice_repair_only_changes_faulty_sentence_and_preserves_shots(tmp_path):
    from factory.script.voice_repair import repair_for_speech
    from factory.models import Evidence, StoryCandidate, MediaIndex
    from types import SimpleNamespace

    class Provider:
        is_mock = False
        conf = {}

        def complete_json(self, prompt, **kwargs):
            assert "警察最后赶到调查" in prompt
            return {"text": "警察向女孩询问刚才发生的事情。"}

    texts = ["女孩发现陌生人站在门口。", "她马上后退躲到箱子旁边。", "门缝中突然伸进一只手。", "警察最后赶到调查。"]
    variant = ScriptVariant(variant_id="v1", sentences=[ScriptSentence(index=i+1, text=t, estimated_seconds=2,
                          source_evidence=Evidence(start=i*10, end=i*10+4, scene_ids=[i+1], quote="警察和女孩交谈"))
                          for i, t in enumerate(texts)], scene_matches=[{"sentence_index":2,"source_windows":[{"start":10,"end":12}]}])
    provider = Provider()
    pipeline = SimpleNamespace(router=SimpleNamespace(get=lambda role: provider), config=Config(), index=MediaIndex(),
                               state=SimpleNamespace(artifacts={"reviewed_source_sha256":"old"}), job_dir=tmp_path,
                               _adopt_variant=lambda c,v: None, save_state=lambda:None, emitter=SimpleNamespace(log=lambda text:None))
    fixed = repair_for_speech(pipeline, StoryCandidate(), variant,
                             {"issues":[{"kind":"repeated_phrase","text":"警察最后赶到调查"}]})
    assert [s.text for s in fixed.sentences[:3]] == texts[:3]
    assert variant.sentences[-1].text == texts[-1]
    assert fixed.sentences[-1].source_evidence == variant.sentences[-1].source_evidence
    assert fixed.scene_matches == variant.scene_matches
    assert "reviewed_source_sha256" not in pipeline.state.artifacts
