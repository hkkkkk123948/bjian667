import hashlib

import numpy as np
import pytest

from factory.eval.narration_audit import ASR_CACHE_POLICY, audit_narration
from factory.eval.speech_alignment import align_recognized_words, select_word_alignment
from factory.models import ScriptSentence
from factory.providers.tts.continuous import align_continuous_result, sentence_boundaries
from factory.utils import write_json


def timed(text, start=0.2, step=.15):
    return [{"text": character, "start": start+index*step, "end": start+(index+1)*step}
            for index, character in enumerate(text)]


def test_estimated_provider_words_never_masquerade_as_measured_alignment():
    words = timed("女孩退后男人跑了")
    aligned, report = select_word_alignment("女孩退后。男人跑了。", 2, words, "estimated")
    assert aligned == []
    assert report["source"] == "estimated"
    assert report["confidence"] == "low"
    local = timed("女孩退后男人跑了", step=.2)
    aligned, report = select_word_alignment("女孩退后。男人跑了。", 2, words, "estimated", local)
    assert aligned[4]["start"] == 1
    assert report["source"] == "local_asr_words"


def test_short_homophone_uses_actual_word_times_without_changing_spoken_text():
    actual = timed("她们赶紧往后想跑")
    aligned, report = align_recognized_words("她们赶紧往后巷跑。", actual, 2)
    assert "".join(w["text"] for w in aligned) == "她们赶紧往后巷跑"
    assert aligned[6]["start"] == actual[6]["start"]
    assert aligned[6]["granularity"] == "asr_substitution"
    assert report["confidence"] == "medium"
    assert report["source"] == "local_asr_interpolated"
    assert report["uncertain_spans"][0]["recognized"] == "想"


def test_missing_phrase_or_boundary_anchor_is_not_fabricated():
    words = timed("女孩开门警察赶到")
    aligned, report = align_recognized_words("女孩开门，怪人突然追了上来，警察赶到。", words, 3)
    assert aligned == []
    assert report["reason"] == "asr_text_discrepancy"
    aligned, report = align_recognized_words("女孩退后男人跑了。", timed("孩退后男人跑了"), 2)
    assert aligned == []
    assert report["reason"] == "missing_boundary_anchor"


def test_short_internal_asr_gap_is_marked_as_inferred():
    # Local ASR skipped the character 快, but the measured gap bounds its time.
    words = timed("女孩很退后男人跑了", step=.2)
    for word in words[3:]:
        word["start"] += .25
        word["end"] += .25
    aligned, report = align_recognized_words("女孩很快退后男人跑了", words, 3)
    assert aligned[3]["text"] == "快"
    assert aligned[3]["granularity"] == "inferred_gap"
    assert report["confidence"] == "medium"
    assert aligned[2]["end"] <= aligned[3]["start"] < aligned[3]["end"] <= aligned[4]["start"]


@pytest.mark.parametrize("expected,recognized", [("2026", "二零二六"), ("21", "二十一"), ("3", "三")])
def test_numeral_notation_preserves_actual_clock_and_reports_uncertainty(expected, recognized):
    from factory.eval.narration_audit import compare_transcript
    text = "到了"+expected+"年女孩终于获救"
    actual = "到了"+recognized+"年女孩终于获救"
    words = timed(actual)
    aligned, report = align_recognized_words(text, words, 4)
    assert "".join(w["text"] for w in aligned) == text
    assert report["confidence"] == "medium"
    assert report["uncertain_spans"][0]["kind"] == "number_notation"
    audit = compare_transcript(text, actual)
    assert audit["status"] == "pass"
    assert audit["warnings"][0]["kind"] == "number_notation"


def test_repeated_sentence_is_not_aligned_as_a_clean_original():
    expected = "女孩赶紧退后男人突然跑开警察很快赶到"
    actual = expected+"警察很快赶到"
    aligned, report = align_recognized_words(expected, timed(actual), 6)
    assert aligned == []
    assert report["confidence"] == "unavailable"
    assert report["reason"] == "asr_text_discrepancy"


def test_long_silence_stays_in_audio_clock_and_is_reported():
    words = timed("女孩退后男人跑了")
    for word in words[4:]:
        word["start"] += 3
        word["end"] += 3
    aligned, report = align_recognized_words("女孩退后男人跑了", words, 5)
    assert aligned[4]["start"] == 3.8
    assert report["silence_gaps"][0]["duration"] == 3
    sentences = [ScriptSentence(index=1, text="女孩退后。"), ScriptSentence(index=2, text="男人跑了。")]
    timings = sentence_boundaries(sentences, aligned, 5)
    assert timings[0]["end"] == timings[1]["start"] == 3.8
    assert timings[-1]["end"] == 5


def test_sentence_boundary_inside_multi_character_word_is_interpolated():
    sentences = [ScriptSentence(index=1, text="女孩退后。"), ScriptSentence(index=2, text="男人跑了。")]
    result = sentence_boundaries(sentences, [{"text": "女孩退后男人跑了", "start": .2, "end": 1.8}], 2)
    assert result[1]["start"] == pytest.approx(1)
    assert result[0]["start"] == 0
    assert result[-1]["end"] == 2


def test_provider_clock_drift_uses_cached_measured_asr_and_resume_does_not_rescale():
    sentences = [ScriptSentence(index=1, text="女孩退后。"), ScriptSentence(index=2, text="男人跑了。")]
    provider = timed("女孩退后男人跑了", start=.2, step=.25)  # ends 2.2, audio is 2 s
    local = timed("女孩退后男人跑了", start=.1, step=.2)
    first = align_continuous_result(sentences, {"word_timings": provider,
        "timing_source": "provider_word_timestamps", "content_check": {"asr_word_timings": local}}, 2)
    assert first["timing_source"] == "local_asr_words"
    assert first["sentence_timings"][1]["start"] == pytest.approx(.9)
    assert first["timestamp_duration_scale"] == 1
    second = align_continuous_result(sentences, first, 2)
    assert second["word_timings"] == first["word_timings"]
    assert second["sentence_timings"] == first["sentence_timings"]


def test_legacy_content_verdict_does_not_gate_time_location():
    sentences = [ScriptSentence(index=1, text="女孩退后。"), ScriptSentence(index=2, text="男人跑了。")]
    result = {"word_timings": timed("女孩退后男人跑了"), "timing_source": "provider_word_timestamps",
              "content_check": {"status": "fail", "issues": [{"kind": "repeated_phrase", "text": "男人跑了男人跑了"}]}}
    aligned = align_continuous_result(sentences, result, 2)
    assert aligned["timing_source"] == "provider_word_timestamps"
    assert aligned["sentence_timings"][1]["start"] == pytest.approx(.8)


def test_long_chinese_local_asr_substitution_keeps_measured_times_outside_region():
    prefix = "女孩看着门外的男人他始终没有离开"*8+"她转身发现床上有人"
    suffix = "站着红帽男人正在向窗外望去"+"女孩随后沿着楼梯向下跑同伴紧跟在后面"*8
    expected = prefix+"抱紧毯子"+suffix
    actual = prefix+"报警谈则"+suffix
    words = timed(actual, step=.1)
    aligned, report = align_recognized_words(expected, words, len(actual)*.1+1)
    assert len(expected) > 200
    assert len(aligned) == len(expected)
    assert report["matched_ratio"] == pytest.approx((len(expected)-4)/len(expected))
    assert report["confidence"] == "medium"
    assert report["uncertain_spans"][0]["kind"] == "bounded_substitution"
    assert aligned[len(prefix)+4]["start"] == words[len(prefix)+4]["start"]
    assert aligned[-1]["end"] == words[-1]["end"]
    assert aligned[len(prefix)]["granularity"] == "bounded_substitution"


def test_rejected_local_region_reports_whole_text_ratio_and_does_not_fabricate_clock():
    prefix = "女孩看着门外的男人他始终没有离开"*8+"她转身发现床上有人"
    suffix = "站着红帽男人正在向窗外望去"+"女孩随后沿着楼梯向下跑同伴紧跟在后面"*8
    expected = prefix+"抱紧毯子"+suffix
    actual = prefix+"报警谈则"+suffix
    words = timed(actual, step=.1)
    for word in words[len(prefix)+1:]:
        word["start"] += 2
        word["end"] += 2
    aligned, report = align_recognized_words(expected, words, len(actual)*.1+4)
    assert aligned == []  # A four-character region lasting over 2 s is not locally trustworthy.
    assert report["matched_ratio"] == pytest.approx((len(expected)-4)/len(expected))
    assert report["confidence"] == "unavailable"


def test_local_asr_repeat_keeps_first_observed_clock_and_labels_unmapped_interval():
    prefix = "女孩看着门外的男人他始终没有离开"*8+"她站在门口犹豫不决"
    phrase = "女孩靠近锅口查看"
    suffix = "看清里面的东西后她马上退回门边"+"同伴在外面等待她她赶紧走向同伴"*8
    expected = prefix+phrase+suffix
    actual = prefix+phrase+phrase+suffix
    words = timed(actual, step=.1)
    aligned, report = align_recognized_words(expected, words, len(actual)*.1+1)
    assert len(aligned) == len(expected)
    assert report["confidence"] == "medium"
    assert report["uncertain_spans"][0]["kind"] == "bounded_asr_repeat"
    assert aligned[len(prefix)]["start"] == words[len(prefix)]["start"]
    assert aligned[len(prefix)+len(phrase)]["start"] == words[len(prefix)+2*len(phrase)]["start"]
    assert report["unmapped_audio_spans"][0]["start"] == words[len(prefix)+len(phrase)]["start"]
    assert report["unmapped_audio_spans"][0]["end"] == words[len(prefix)+2*len(phrase)-1]["end"]


@pytest.mark.parametrize("words", [
    [{"text": "女", "start": float("nan"), "end": .2}],
    [{"text": "女", "start": .5, "end": .7}, {"text": "孩", "start": .1, "end": .2}],
    [{"text": c, "start": 0, "end": 0} for c in "女孩退后男人跑了"],
])
def test_invalid_local_clocks_are_rejected(words):
    aligned, report = align_recognized_words("女孩退后男人跑了", words, 2)
    assert aligned == []
    assert report["confidence"] == "unavailable"


def test_content_audit_caches_words_and_rechecks_edited_script_without_retranscribing(tmp_path, monkeypatch):
    from factory.eval import narration_audit
    audio = tmp_path / "voice.wav"
    audio.write_bytes(b"intact audio")
    output = tmp_path / "audit.json"
    calls = []

    def transcribe(path, model):
        calls.append((path, model))
        return {"text": "她们赶紧往后想跑", "segments": [{"words": [
            {"word": "她们", "start": .2, "end": .5, "probability": .9},
            {"word": "赶紧往后想跑", "start": .5, "end": 1.7}]}]}

    monkeypatch.setattr(narration_audit, "local_transcribe", transcribe)
    first = audit_narration(audio, "她们赶紧往后巷跑。", output, require_word_timings=True)
    assert first["asr_cache_policy"] == ASR_CACHE_POLICY
    assert len(first["asr_word_timings"]) == 2
    assert first["status"] == "pass"
    assert audit_narration(audio, "她们赶紧往后巷跑。", output, require_word_timings=True) == first
    changed = audit_narration(audio, "她们赶紧往后想跑。", output, require_word_timings=True)
    assert changed["warnings"] == []
    assert len(calls) == 1


def test_old_audit_adds_missing_word_clock_once_without_resynthesizing(tmp_path, monkeypatch):
    from factory.eval import narration_audit
    audio = tmp_path / "voice.wav"
    audio.write_bytes(b"intact audio")
    output = tmp_path / "audit.json"
    original = {"audio_sha256": hashlib.sha256(audio.read_bytes()).hexdigest(),
                "expected_text": "女孩退后", "status": "pass"}
    write_json(output, original)
    assert audit_narration(audio, "女孩退后。", output) == original
    calls = []
    monkeypatch.setattr(narration_audit, "local_transcribe", lambda *args: calls.append(args) or {
        "text": "女孩退后", "segments": [{"words": [{"word": "女孩退后", "start": .1, "end": 1}]}]})
    upgraded = audit_narration(audio, "女孩退后。", output, require_word_timings=True)
    assert upgraded["asr_word_timings"][0]["end"] == 1
    audit_narration(audio, "女孩退后。", output, require_word_timings=True)
    assert len(calls) == 1
    assert audio.read_bytes() == b"intact audio"


def test_complete_synthesis_reuses_single_local_audit_for_every_output(tmp_path, monkeypatch):
    from factory.config import Config
    from factory.eval import narration_audit
    from factory.models import ScriptVariant
    from factory.providers.tts.base import write_wav
    from factory.providers.tts.continuous import synth_continuous
    texts = ["女孩发现门外有人。", "她赶紧后退锁好了门。", "窗外忽然出现一个人影。", "警察最后把她安全救走。"]
    text = "".join(texts)
    calls = {"synth": 0, "asr": 0}

    class Recorder:
        config = Config({"workspace": str(tmp_path / "ws"), "tts": {"speed": 1.2}})

        def synth(self, submitted, output, speed=None):
            assert submitted == text
            assert speed == 1.2
            calls["synth"] += 1
            write_wav(output, .1 * np.sin(np.arange(24000*8) * .1), 24000)
            return {"audio_path": str(output), "provider": "fixture_real_voice", "timing_source": "estimated"}

    def transcribe(*args):
        from factory.script.quality import compact
        calls["asr"] += 1
        return {"text": text, "segments": [{"words": timed(compact(text), step=.15)}]}

    monkeypatch.setattr(narration_audit, "local_transcribe", transcribe)
    variant = ScriptVariant(sentences=[ScriptSentence(index=i+1, text=t, estimated_seconds=2) for i, t in enumerate(texts)])
    one = synth_continuous(Recorder(), variant, tmp_path / "one")
    two = synth_continuous(Recorder(), variant, tmp_path / "two")
    assert one["timing_source"] == "local_asr_words"
    assert one["word_timings"] == two["word_timings"]
    assert one["voice_sha256"] == one["original_audio_sha256"] == two["voice_sha256"]
    assert one["audio_edits"] == []
    assert calls == {"synth": 1, "asr": 1}


def test_portable_dtw_returns_a_monotonic_minimum_cost_path():
    from functools import lru_cache
    from factory.eval.word_clock import numpy_dtw
    matrix = np.array([[0., 1., 4., 8.], [5., 2., 1., 5.], [8., 4., 2., 0.]])
    rows, columns = numpy_dtw(matrix)
    assert (rows[0], columns[0]) == (0, 0)
    assert (rows[-1], columns[-1]) == (2, 3)
    assert all((int(b-a), int(d-c)) in {(1, 1), (1, 0), (0, 1)}
               for a, b, c, d in zip(rows, rows[1:], columns, columns[1:]))

    @lru_cache(None)
    def minimum(row, column):
        if row < 0 or column < 0:
            return float("inf")
        if row == column == 0:
            return matrix[0, 0]
        return matrix[row, column]+min(minimum(row-1, column), minimum(row, column-1), minimum(row-1, column-1))

    assert matrix[rows, columns].sum() == minimum(2, 3)


def test_local_gpu_oom_falls_back_to_cpu_and_restores_timing_backend(tmp_path, monkeypatch):
    import os
    import sys
    from types import SimpleNamespace
    from factory.eval import narration_audit
    from factory.eval.word_clock import numpy_dtw
    checkpoint = tmp_path / ".cache/whisper/small.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"cached model fixture")
    monkeypatch.setattr(narration_audit.Path, "home", lambda: tmp_path)
    monkeypatch.setattr(narration_audit, "_MODELS", {})
    devices, released = [], []

    class OutOfMemory(Exception):
        pass

    original = object()
    timing = SimpleNamespace(dtw_cpu=original)

    def load_model(path, device):
        devices.append(device)

        def transcribe(*args, **kwargs):
            assert kwargs["word_timestamps"] is True
            assert timing.dtw_cpu is (numpy_dtw if os.name == "nt" else original)
            if device == "cuda":
                raise OutOfMemory()
            return {"text": "女孩退后", "segments": []}

        return SimpleNamespace(transcribe=transcribe)

    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(set_num_threads=lambda n: None,
        cuda=SimpleNamespace(is_available=lambda: True, empty_cache=lambda: released.append(True),
                             OutOfMemoryError=OutOfMemory)))
    monkeypatch.setitem(sys.modules, "whisper", SimpleNamespace(timing=timing, load_model=load_model))
    result = narration_audit.local_transcribe(tmp_path / "voice.wav")
    assert result["alignment_device"] == "cpu"
    assert devices == ["cuda", "cpu"]
    assert released
    assert timing.dtw_cpu is original
    narration_audit.release_local_models()
