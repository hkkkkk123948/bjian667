import json

import pytest

from factory.config import Config
from factory.script.duration import (duration_state, estimate_duration, record_duration,
                                     spoken_text, word_budget)


def config(tmp_path, **tts):
    return Config({"workspace": str(tmp_path), "tts": {"type": "bailian", "model": "cosyvoice-v2",
                   "voice": "cosyvoice-v2-a", "speed": 1.2, **tts},
                   "params": {"min_duration": 35, "max_duration": 50}})


def result(seconds, **kwargs):
    return {"provider": "bailian-cosyvoice", "duration": seconds,
            "voice": "cosyvoice-v2-a", "playback_speed": 1.2,
            "content_check": {"status": "pass"}, **kwargs}


def test_comments_count_as_speech_and_brackets_labels_do_not(tmp_path):
    c = config(tmp_path)
    plain = "女孩出门忘带玫瑰花司机急刹"
    marked = "女孩出门（忘带玫瑰花）司机急刹"
    assert estimate_duration(c, marked).seconds == estimate_duration(c, plain).seconds
    assert estimate_duration(c, marked).spoken_units == len(plain)
    assert "评论话术" not in spoken_text("【评论话术｜不参与匹配】忘带花")
    assert estimate_duration(c, "女孩出门(忘带玫瑰花)司机急刹").seconds == estimate_duration(c, plain).seconds


def test_real_history_calibrates_and_retries_do_not_inflate_samples(tmp_path):
    c = config(tmp_path)
    a, b = "女孩提着灯在走廊里寻找声音" * 8, "男孩发现门外的脚步声越来越近" * 8
    assert record_duration(c, a, result(len(a) / 5))
    assert record_duration(c, a, result(len(a) / 5))
    one = estimate_duration(c, a)
    assert one.sample_count == 1 and not one.calibrated
    assert record_duration(c, b, result(len(b) / 5))
    estimate = estimate_duration(c, "灯" * 200)
    assert estimate.sample_count == 2 and estimate.calibrated
    assert estimate.seconds == 40
    budget = word_budget(c, reserve_text="（忘带玫瑰花）")
    assert budget["minimum"] == 175 - 5
    assert budget["maximum"] == 250 - 5
    assert estimate_duration(c, a + "。\n").sample_count == 2


@pytest.mark.parametrize("changes", [{"voice": "cosyvoice-v2-b"}, {"speed": 1.1},
    {"type": "edge"}, {"cosyvoice_model": "cosyvoice-v3"}, {"base_url": "https://other.example"},
    {"emotion": "sad"}, {"rate": "0.8"}])
def test_different_voice_model_speed_provider_are_isolated(tmp_path, changes):
    c = config(tmp_path)
    assert record_duration(c, "灯" * 120, result(24))
    assert estimate_duration(config(tmp_path, **changes), "灯" * 120).sample_count == 0


def test_voice_alias_resolves_to_same_actual_clone(tmp_path):
    c = config(tmp_path)
    assert record_duration(c, "灯" * 120, result(24))
    alias = config(tmp_path, voice="我的音色", voices=[{"name": "我的音色", "voice_id": "cosyvoice-v2-a",
                                                   "model": "cosyvoice-v2"}])
    assert estimate_duration(alias, "灯" * 120).sample_count == 1


@pytest.mark.parametrize("changes", [{"content_check": {"status": "fail"}}, {"provider": "mock"},
    {"provider": "test_native"}, {"voice": "wrong"}, {"playback_speed": 0.8},
    {"audio_edits": ["cut"]}, {"duration": float("nan")}, {"duration": 0.2}])
def test_invalid_or_rejected_audio_never_calibrates(tmp_path, changes):
    c = config(tmp_path)
    assert not record_duration(c, "灯" * 120, result(24, **changes))
    assert estimate_duration(c, "灯" * 120).sample_count == 0


def test_bad_history_and_metadata_are_tolerated_and_import_is_once(tmp_path, monkeypatch):
    c = config(tmp_path)
    history = tmp_path / ".cache/narration_duration_v1.json"
    history.parent.mkdir(parents=True)
    history.write_text("{broken", encoding="utf-8")
    cache = tmp_path / ".cache/tts_continuous"
    cache.mkdir()
    (cache / "broken.json").write_text("{broken", encoding="utf-8")
    audio = result(24, content_check={"status": "pass", "expected_text": "灯" * 120})
    (cache / "valid.json").write_text(json.dumps(audio), encoding="utf-8")
    assert estimate_duration(c, "灯" * 120).sample_count == 1
    (cache / "valid.json").unlink()
    # Subsequent keystrokes use the compact history, not the directory scan.
    def no_scan(*args, **kwargs):
        raise AssertionError("repeated TTS cache scan")
    monkeypatch.setattr(type(cache), "glob", no_scan)
    assert estimate_duration(c, "灯" * 125).sample_count == 1
    saved = json.loads(history.read_text(encoding="utf-8"))
    assert "voice" not in str(saved) and "expected_text" not in str(saved)


def test_unknown_model_cache_is_not_borrowed(tmp_path):
    c = config(tmp_path, model="new-model", voice="other")
    cache = tmp_path / ".cache/tts_continuous"
    cache.mkdir(parents=True)
    metadata = result(24, voice="other", content_check={"status": "pass", "expected_text": "灯" * 120})
    (cache / "old.json").write_text(json.dumps(metadata), encoding="utf-8")
    assert estimate_duration(c, "灯" * 120).sample_count == 0


def test_estimate_ranges_distinguish_strong_and_borderline_overlength(tmp_path):
    c = config(tmp_path)
    near = estimate_duration(c, "灯" * 340)
    assert duration_state(near, 35, 50) == "near_long"
    huge = estimate_duration(c, "灯" * 600)
    assert duration_state(huge, 35, 50) == "too_long"
    assert duration_state(estimate_duration(c, "灯" * 100), 35, 50) == "too_short"
    assert duration_state(estimate_duration(c, ""), 35, 50) == "empty"


def test_outlier_does_not_move_calibrated_rate(tmp_path):
    c = config(tmp_path)
    for count in range(100, 105):
        rate = 15 if count == 104 else 5
        assert record_duration(c, "灯" * count, result(count / rate))
    estimate = estimate_duration(c, "灯" * 200)
    assert estimate.rate == 5 and estimate.seconds == 40
    assert estimate.sample_count == 4


def test_optional_history_write_failure_does_not_fail_synthesis(tmp_path, monkeypatch):
    c = config(tmp_path)
    def cannot_replace(*args, **kwargs):
        raise PermissionError("read-only optional history")
    monkeypatch.setattr(type(tmp_path), "replace", cannot_replace)
    assert not record_duration(c, "灯" * 120, result(24))
    assert estimate_duration(c, "灯" * 120).seconds > 0
