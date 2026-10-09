import hashlib

import numpy as np

from factory.config import Config
from factory.models import ScriptSentence, ScriptVariant
from factory.providers.tts.base import write_wav
from factory.providers.tts.continuous import (
    _legacy_cache_key,
    synth_continuous,
    synthesis_identity,
)
from factory.script.quality import compact
from factory.utils import read_json, write_json


TEXTS = ["女孩发现门外一直有人。", "她赶紧后退并锁好了门。", "窗外忽然出现一个人影。", "警察最后把她安全救走。"]
TEXT = "".join(TEXTS)


def variant():
    return ScriptVariant(sentences=[ScriptSentence(index=i+1, text=t, estimated_seconds=1)
                                    for i, t in enumerate(TEXTS)])


class Recorder:
    def __init__(self, tmp_path):
        self.config = Config({"workspace": str(tmp_path / "ws"), "tts": {
            "type": "bailian", "model": "cosyvoice-v2", "voice": "N1", "speed": 1.2,
            "voices": [{"name": "N1", "voice_id": "cosyvoice-v2-clone-a", "model": "cosyvoice-v2"}]},
            "params": {"narration_audit": "off"}})
        self.calls = []

    def synth(self, text, out_path, speed=None, **kwargs):
        identity = synthesis_identity(self.config.tts, speed)
        self.calls.append(identity)
        write_wav(out_path, .1 * np.sin(np.arange(24000) * .1), 24000)
        return {"audio_path": str(out_path), "voice": identity["voice"], "provider": "bailian-cosyvoice",
                "playback_speed": speed, "seed": self.config.tts.get("seed", 0)}


def legacy_audio(tmp_path, recorder, local_job=False, text=TEXT):
    key = _legacy_cache_key(text, recorder.config.tts, 1.2)
    directory = tmp_path / "job" if local_job else tmp_path / "ws/.cache/tts_continuous"
    directory.mkdir(parents=True, exist_ok=True)
    audio = directory / ("voice.wav" if local_job else key+".wav")
    write_wav(audio, .1 * np.sin(np.arange(24000) * .1), 24000)
    digest = hashlib.sha256(audio.read_bytes()).hexdigest()
    result = {"audio_path": str(audio), "provider": "bailian-cosyvoice", "voice": "cosyvoice-v2-clone-a",
              "playback_speed": 1.2, "seed": 0,
              "content_check": {"status": "pass", "audio_sha256": digest}}
    if local_job:
        result.update(text=text, voice_sha256=digest, original_audio_sha256=digest)
    metadata = directory / ("tts.json" if local_job else key+".json")
    write_json(metadata, result)
    return audio, metadata, result


def test_registry_unrelated_entries_and_alias_rename_do_not_charge_again(tmp_path):
    recorder = Recorder(tmp_path)
    first = synth_continuous(recorder, variant(), tmp_path / "one")
    recorder.config.tts["voices"].append({"name": "other", "voice_id": "cosyvoice-v2-other", "model": "cosyvoice-v2"})
    second = synth_continuous(recorder, variant(), tmp_path / "two")
    recorder.config.tts["voices"][0]["name"] = "renamed"
    recorder.config.tts["voice"] = "renamed"
    third = synth_continuous(recorder, variant(), tmp_path / "three")
    assert len(recorder.calls) == 1
    assert first["voice_sha256"] == second["voice_sha256"] == third["voice_sha256"]
    assert "voices" not in first["synthesis_identity"]


def test_actual_cosyvoice_model_and_selected_voice_change_miss_the_cache(tmp_path):
    recorder = Recorder(tmp_path)
    synth_continuous(recorder, variant(), tmp_path / "one")
    recorder.config.tts["cosyvoice_model"] = "cosyvoice-v3-flash"
    model_changed = synth_continuous(recorder, variant(), tmp_path / "two")
    assert model_changed["synthesis_identity"]["model"] == "cosyvoice-v3-flash"
    recorder.config.tts["voices"][0]["voice_id"] = "cosyvoice-v3-clone-b"
    synth_continuous(recorder, variant(), tmp_path / "three")
    assert len(recorder.calls) == 3


def test_only_actual_cosyvoice_parameters_affect_identity(tmp_path):
    recorder = Recorder(tmp_path)
    before = synthesis_identity(recorder.config.tts)
    recorder.config.tts.update(model="unused-default", rate="ignored", pitch="ignored", volume="ignored",
                               base_url="https://ignored-http-endpoint.example", api_key="do-not-cache-credentials")
    assert synthesis_identity(recorder.config.tts) == before
    recorder.config.tts["seed"] = 12
    assert synthesis_identity(recorder.config.tts) != before


def test_http_cloned_voice_model_and_effective_rate_affect_cache(tmp_path):
    recorder = Recorder(tmp_path)
    recorder.config.tts.update(voice="qwen-tts-vc-example", voices=[], model="qwen3-tts-flash", voice_model="clone-model-a", rate=1.2)
    first = synth_continuous(recorder, variant(), tmp_path / "one")
    recorder.config.tts["voice_model"] = "clone-model-b"
    second = synth_continuous(recorder, variant(), tmp_path / "two")
    recorder.config.tts["rate"] = 1.4
    third = synth_continuous(recorder, variant(), tmp_path / "three")
    assert first["synthesis_identity"]["model"] == "clone-model-a"
    assert second["synthesis_identity"]["model"] == "clone-model-b"
    assert third["synthesis_identity"]["parameters"]["rate"] == 1.4
    assert len(recorder.calls) == 3


def test_verified_legacy_cache_migrates_without_new_tts_or_asr(tmp_path, monkeypatch):
    from factory.eval import narration_audit
    recorder = Recorder(tmp_path)
    audio, metadata, result = legacy_audio(tmp_path, recorder)
    audit = {"status": "pass", "audio_sha256": hashlib.sha256(audio.read_bytes()).hexdigest(),
             "expected_text": compact(TEXT), "asr_transcript": TEXT, "recognized_text": compact(TEXT),
             "asr_cache_policy": narration_audit.ASR_CACHE_POLICY,
             "asr_word_timings": [{"text": c, "start": i*.015, "end": (i+1)*.015}
                                  for i, c in enumerate(compact(TEXT))]}
    write_json(metadata.with_name(metadata.stem+".audit.json"), audit)
    recorder.config.params["narration_audit"] = "local"
    monkeypatch.setattr(narration_audit, "local_transcribe", lambda *args: (_ for _ in ()).throw(AssertionError("repeated ASR")))
    output = synth_continuous(recorder, variant(), tmp_path / "job")
    assert recorder.calls == []
    assert output["voice_sha256"] == result["content_check"]["audio_sha256"]
    assert output["content_check"]["asr_word_timings"] == audit["asr_word_timings"]
    new_metadata = [p for p in metadata.parent.glob("*.json") if p != metadata and not p.name.endswith(".audit.json")]
    assert read_json(new_metadata[0])["synthesis_identity"] == synthesis_identity(recorder.config.tts)


def test_job_audio_survives_registry_change_and_new_cache_policy(tmp_path):
    recorder = Recorder(tmp_path)
    audio, _, before = legacy_audio(tmp_path, recorder, local_job=True)
    recorder.config.tts["voices"].append({"name": "unrelated", "voice_id": "cosyvoice-v2-other"})
    output = synth_continuous(recorder, variant(), audio.parent)
    assert recorder.calls == []
    assert output["voice_sha256"] == before["voice_sha256"]
    assert hashlib.sha256(audio.read_bytes()).hexdigest() == before["voice_sha256"]


def test_existing_new_identity_cannot_be_reinterpreted_as_a_legacy_voice(tmp_path):
    recorder = Recorder(tmp_path)
    _, metadata, result = legacy_audio(tmp_path, recorder, local_job=True)
    wrong_conf = dict(recorder.config.tts, cosyvoice_model="cosyvoice-v3-flash")
    result["synthesis_identity"] = synthesis_identity(wrong_conf)
    write_json(metadata, result)
    synth_continuous(recorder, variant(), metadata.parent)
    assert len(recorder.calls) == 1


def test_job_audio_requires_full_text_speed_voice_and_matching_hash(tmp_path):
    changes = [{"text": TEXT.replace("。", "，")}, {"playback_speed": 1.1},
               {"voice": "cosyvoice-v2-different"}, {"voice_sha256": "wrong"}, {"model": "cosyvoice-v3"}]
    for index, change in enumerate(changes):
        folder = tmp_path / str(index)
        recorder = Recorder(folder)
        _, metadata, result = legacy_audio(folder, recorder, local_job=True)
        result.update(change)
        write_json(metadata, result)
        synth_continuous(recorder, variant(), metadata.parent)
        assert len(recorder.calls) == 1


def test_missing_legacy_model_evidence_does_not_guess_a_http_voice(tmp_path):
    recorder = Recorder(tmp_path)
    _, metadata, result = legacy_audio(tmp_path, recorder, local_job=True)
    recorder.config.tts.update(voice="Cherry", voices=[], model="qwen3-tts-flash")
    result["provider"], result["voice"] = "bailian", "Cherry"
    write_json(metadata, result)
    synth_continuous(recorder, variant(), metadata.parent)
    assert len(recorder.calls) == 1
