"""Compare what was actually spoken with the script. Never edit the audio."""
from difflib import SequenceMatcher
from pathlib import Path
from threading import RLock

from ..script.quality import compact
from ..utils import read_json, write_json
from .speech_alignment import equivalent_number_notation

_MODELS = {}
ASR_CACHE_POLICY = "transcript_and_words_v1"
_TRANSCRIBE_LOCK = RLock()


def release_local_models():
    """Return ASR memory before HD video filters allocate their working frames."""
    with _TRANSCRIBE_LOCK:
        if not _MODELS:
            return
        import gc
        import torch
        _MODELS.clear()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def compare_transcript(expected, transcript, words=None):
    expected, actual = compact(expected), compact(transcript)
    issues, warnings = [], []
    for op, a, b, x, y in SequenceMatcher(None, expected, actual, autojunk=False).get_opcodes():
        want, got = expected[a:b], actual[x:y]
        if op == "equal":
            continue
        if op == "replace" and equivalent_number_notation(want, got):
            warnings.append({"kind": "number_notation", "expected": want, "actual": got})
            continue
        if op == "insert" and len(got) >= 5 and got in expected:
            issues.append({"kind": "repeated_phrase", "text": got, "transcript_start": x, "transcript_end": y})
        elif op == "delete" and len(want) >= 5:
            issues.append({"kind": "missing_phrase", "text": want})
        elif max(len(want), len(got)) >= 5:
            issues.append({"kind": "text_mismatch", "expected": want, "actual": got})
        else:
            # A one-character homophone is often ASR error, not a TTS error.
            warnings.append({"expected": want, "actual": got})
    return {"status": "fail" if issues else "pass", "issues": issues, "warnings": warnings,
            "expected_text": expected, "recognized_text": actual,
            "method": "local_asr_script_comparison", "audio_edits": []}


def local_transcribe(audio_path, model_name="small"):
    with _TRANSCRIBE_LOCK:
        return _local_transcribe(audio_path, model_name)


def _local_transcribe(audio_path, model_name):
    import os
    import torch
    import whisper

    checkpoint = Path.home() / ".cache" / "whisper" / (model_name + ".pt")
    if not checkpoint.is_file():
        raise RuntimeError("本地语音核对模型未安装: %s" % model_name)
    torch.set_num_threads(4)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    def transcribe_on(selected_device):
        key = (str(checkpoint), selected_device)
        if key not in _MODELS:
            # Loading small and turbo together can exceed a 4 GB GPU. Only one
            # narration ASR model needs to be resident at a time.
            if _MODELS:
                release_local_models()
            _MODELS[key] = whisper.load_model(str(checkpoint), device=selected_device)
        # Keep this scoped to our narration ASR, not the user's installed
        # Whisper package. The lock protects the temporary timing backend.
        with _TRANSCRIBE_LOCK:
            original_dtw = whisper.timing.dtw_cpu
            try:
                if os.name == "nt":
                    from .word_clock import numpy_dtw
                    whisper.timing.dtw_cpu = numpy_dtw
                result = _MODELS[key].transcribe(str(audio_path), language="zh", fp16=selected_device == "cuda",
                                                temperature=0, condition_on_previous_text=False,
                                                word_timestamps=True, verbose=False)
            finally:
                whisper.timing.dtw_cpu = original_dtw
        result["alignment_device"] = selected_device
        return result

    try:
        return transcribe_on(device)
    except torch.cuda.OutOfMemoryError:
        if device != "cuda":
            raise
        release_local_models()
        return transcribe_on("cpu")


def _transcription_words(result):
    """Keep the measured ASR word clock, not segment-wide proportional times."""
    return [{"text": str(word.get("word", word.get("text", ""))),
             "start": float(word["start"]), "end": float(word["end"]),
             **({"probability": float(word["probability"])} if word.get("probability") is not None else {})}
            for segment in result.get("segments", []) for word in segment.get("words", [])
            if word.get("start") is not None and word.get("end") is not None]


def audit_narration(audio_path, expected_text, output_path, model_name="small", require_word_timings=False):
    import hashlib
    audio_path = Path(audio_path)
    signature = hashlib.sha256(audio_path.read_bytes()).hexdigest()
    cached = read_json(output_path) or {}
    same_audio = cached.get("audio_sha256") == signature
    if same_audio and cached.get("expected_text") == compact(expected_text):
        if not require_word_timings or cached.get("asr_cache_policy") == ASR_CACHE_POLICY:
            return cached
    # The audio transcript is independent of the script. Rechecking an edited
    # punctuation mark or script against the same audio must not rerun ASR.
    if same_audio and cached.get("asr_cache_policy") == ASR_CACHE_POLICY:
        transcript = cached.get("asr_transcript", cached.get("recognized_text", ""))
        words = cached.get("asr_word_timings", [])
        selected_model = cached.get("asr_model", model_name)
        device = cached.get("asr_device", "")
    else:
        result = local_transcribe(audio_path, model_name)
        transcript, words = result["text"], _transcription_words(result)
        selected_model, device = model_name, result.get("alignment_device", "")
    report = compare_transcript(expected_text, transcript)
    report.update(audio_sha256=signature, asr_model=selected_model, model_tokens=0,
                  asr_cache_policy=ASR_CACHE_POLICY, asr_transcript=transcript,
                  asr_word_timings=words, asr_device=device)
    write_json(output_path, report)
    return report
