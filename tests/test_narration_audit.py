from factory.eval.narration_audit import compare_transcript


def test_extra_spoken_tail_is_rejected():
    expected = "他把女孩们救了出来，警察随后赶到调查。"
    actual = "他把女孩们救了出来警察随后赶到调查警察随后赶到调查"
    result = compare_transcript(expected, actual)
    assert result["status"] == "fail"
    assert result["issues"] == [{"kind": "repeated_phrase", "text": "警察随后赶到调查",
                                "transcript_start": 17, "transcript_end": 25}]
    assert result["audio_edits"] == []


def test_missing_sentence_is_rejected():
    result = compare_transcript("女孩打开房门。她看见了陌生男人。警察赶到现场。", "女孩打开房门警察赶到现场")
    assert result["status"] == "fail"
    assert any(issue["kind"] == "missing_phrase" for issue in result["issues"])


def test_single_character_asr_homophone_is_a_warning():
    result = compare_transcript("她们赶紧往后巷跑。", "她们赶紧往后想跑")
    assert result["status"] == "pass"
    assert result["warnings"]


def test_asr_memory_is_released_before_hd_render(monkeypatch):
    import sys
    from types import SimpleNamespace
    from factory.eval import narration_audit
    released = []
    monkeypatch.setattr(narration_audit, "_MODELS", {"fixture": object()})
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(
        is_available=lambda: True, empty_cache=lambda: released.append(True))))
    narration_audit.release_local_models()
    assert narration_audit._MODELS == {}
    assert released == [True]
