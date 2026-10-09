"""One text submission, one intact narration. Only the video is edited."""
import hashlib
import json
import math
import shutil
from pathlib import Path

from ...script.quality import compact
from ...config import DEFAULT_TTS_SPEED
from ...utils import ensure_dir, read_json, write_json, media_duration

SYNTHESIS_CACHE_POLICY = "selected_synthesis_identity_v2"


def synthesis_identity(conf, speed=None):
    """Resolve the parameters actually used by the selected TTS implementation.

    Registry labels, unrelated clones, credentials and transport timeouts do
    not change the voice. This intentionally follows the request builders in
    bailian.py, edge.py and rest.py instead of hashing the whole config.
    """
    provider = str(conf.get("type") or "edge").lower()
    if provider not in {"edge", "bailian", "openai", "elevenlabs", "gemini", "mock"}:
        provider = "edge"  # create_tts uses Edge for unknown types.
    speed = float(conf.get("speed", DEFAULT_TTS_SPEED) if speed is None else speed)
    if not math.isfinite(speed) or speed <= 0:
        raise ValueError("配音速度必须是大于零的有效数字")
    voice, model = str(conf.get("voice") or ""), str(conf.get("model") or "")
    params = {}
    if provider == "bailian":
        from .bailian import DEFAULT_BASE, DEFAULT_PATH, WS_URL
        voice, model = voice or "Cherry", model or "qwen3-tts-flash"
        for item in conf.get("voices") or []:
            if isinstance(item, dict) and item.get("name") == voice and item.get("voice_id"):
                voice = str(item["voice_id"])
                break
        selected = next((item for item in conf.get("voices") or [] if isinstance(item, dict)
                         and voice in (item.get("voice_id"), item.get("name"))), None)
        if selected is not None:
            model = str(selected.get("model") or model)
        elif voice.startswith("cosyvoice"):
            model = str(conf.get("cosyvoice_model") or "cosyvoice-v2")
        elif voice.startswith("qwen-tts-vc-"):
            model = str(conf.get("voice_model") or "qwen3-tts-vc-2026-01-22")
        if model.startswith("cosyvoice") or voice.startswith("cosyvoice"):
            model = str(conf.get("cosyvoice_model") or "cosyvoice-v2")
            endpoint = str(conf.get("ws_url") or WS_URL)
            params = {"rate": speed, "seed": int(conf.get("seed", 0)), "sample_rate": 24000, "format": "wav"}
        else:
            endpoint = str(conf.get("base_url") or DEFAULT_BASE).rstrip("/")
            if "/api/" not in endpoint and not endpoint.endswith("/generation"):
                endpoint += DEFAULT_PATH
            params = {key: conf[key] for key in ("rate", "pitch", "volume", "language", "emotion")
                      if conf.get(key) not in (None, "")}
            params.setdefault("rate", round(speed, 2))
    elif provider == "edge":
        voice, model, endpoint = voice or "zh-CN-YunxiNeural", "edge", "edge-tts"
        params = {"rate": "%+d%%" % int(round((speed-1) * 100))}
    elif provider == "openai":
        voice, model = voice or "alloy", model or "gpt-4o-mini-tts"
        endpoint = str(conf.get("base_url") or "https://api.openai.com/v1").rstrip("/")+"/audio/speech"
        params = {"speed": speed, "response_format": "mp3"}
    elif provider == "elevenlabs":
        voice, model = voice or "21m00Tcm4TlvDq8ikWAM", model or "eleven_multilingual_v2"
        endpoint = "https://api.elevenlabs.io/v1/text-to-speech/"+voice
        params = {"stability": .5, "similarity_boost": .75}
    elif provider == "gemini":
        # The current Gemini adapter sends no voice/speed/emotion setting.
        voice, model = "gemini", model or "gemini-2.5-flash-preview-tts"
        endpoint = str(conf.get("base_url") or "https://generativelanguage.googleapis.com").rstrip("/")
        endpoint += "/v1beta/models/"+model+":generateContent"
    else:
        endpoint = "mock"
        params = {"speed": speed}
    return {"provider": provider, "voice": voice, "model": model, "endpoint": endpoint,
            "requested_speed": speed, "parameters": params}


def _cache_key(text, identity):
    return hashlib.sha256(json.dumps({"text": text, "identity": identity, "mode": SYNTHESIS_CACHE_POLICY},
                                    sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def _legacy_cache_key(text, conf, speed):
    identity = {k: conf.get(k) for k in ("type", "model", "voice", "voices", "base_url", "emotion", "language")}
    return hashlib.sha256(json.dumps({"text": text, "voice": identity, "speed": speed, "mode": "continuous_v1"},
                                    sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def _safe_metadata(path):
    try:
        result = read_json(path) or {}
        return result if isinstance(result, dict) else {}
    except (OSError, ValueError, UnicodeError):
        return {}


def _same_legacy_identity(result, identity):
    """Only infer old identities that actually have enough recorded evidence."""
    # The old metadata did not record HTTP model/rate/pitch/volume. A native
    # CosyVoice v2 clone has an unambiguous model prefix, speed and seed.
    from .bailian import WS_URL
    if (identity["provider"] != "bailian" or identity["model"] != "cosyvoice-v2"
            or identity["endpoint"] != WS_URL
            or not str(result.get("provider", "")).startswith("bailian")
            or result.get("voice") != identity["voice"]
            or not identity["voice"].startswith("cosyvoice-v2-")
            or result.get("model") not in (None, "", identity["model"])):
        return False
    try:
        return (abs(float(result["playback_speed"])-identity["requested_speed"]) < .0001
                and int(result["seed"]) == identity["parameters"]["seed"])
    except (KeyError, TypeError, ValueError):
        return False


def _verified_result(result, identity, text, audio, legacy_key_proves_text=False):
    if not result or not audio.is_file():
        return None
    text_hash = hashlib.sha256(text.encode()).hexdigest()
    if result.get("text") != text and result.get("synthesis_text_sha256") != text_hash and not legacy_key_proves_text:
        return None
    if result.get("synthesis_identity"):
        if result["synthesis_identity"] != identity:
            return None
    elif not _same_legacy_identity(result, identity):
        return None
    expected_hashes = [result.get(key) for key in ("audio_sha256", "voice_sha256", "original_audio_sha256")
                       if result.get(key)]
    audit_hash = (result.get("content_check") or {}).get("audio_sha256")
    if audit_hash:
        expected_hashes.append(audit_hash)
    if not expected_hashes:
        return None
    digest = hashlib.sha256(audio.read_bytes()).hexdigest()
    if any(item != digest for item in expected_hashes):
        return None
    return {**result, "audio_path": str(audio.resolve()), "text": text,
            "synthesis_text_sha256": text_hash, "synthesis_identity": identity,
            "synthesis_cache_policy": SYNTHESIS_CACHE_POLICY, "audio_sha256": digest}


def _recover_result(out_dir, cache, key, text, identity, conf, speed):
    job = _safe_metadata(out_dir / "tts.json")
    if job:
        suffix = Path(str(job.get("audio_path", "voice.wav"))).suffix or ".wav"
        audio = out_dir / ("voice"+suffix)
        recovered = _verified_result(job, identity, text, audio)
        if recovered:
            return recovered, None
    metadata = cache / (key+".json")
    cached = _safe_metadata(metadata)
    recovered = _verified_result(cached, identity, text, Path(str(cached.get("audio_path", ""))))
    if recovered:
        return recovered, cache / (key+".audit.json")
    legacy_key = _legacy_cache_key(text, conf, speed)
    legacy = _safe_metadata(cache / (legacy_key+".json"))
    recovered = _verified_result(legacy, identity, text, Path(str(legacy.get("audio_path", ""))), True)
    return (recovered, cache / (legacy_key+".audit.json")) if recovered else ({}, None)


class NarrationContentError(RuntimeError):
    def __init__(self, report):
        self.report = report
        details = "；".join(item.get("text", item.get("actual", "")) for item in report.get("issues", []))
        super().__init__("完整配音未通过本地核对: %s；必须整段重新合成" % details)


def synth_continuous(tts, variant, out_dir):
    if not variant.sentences or not ''.join(s.text for s in variant.sentences).strip():
        raise ValueError('配音文本不能为空')
    config = getattr(tts, "config", None)
    conf = config.tts if config else getattr(tts, "conf", {})
    speed = float(conf.get("speed", DEFAULT_TTS_SPEED))
    identity = synthesis_identity(conf, speed)
    text = "".join(s.text for s in variant.sentences)
    key = _cache_key(text, identity)
    out_dir = ensure_dir(out_dir)
    cache = ensure_dir(config.workspace_path() / ".cache" / "tts_continuous") if config else ensure_dir(out_dir / "cache")
    metadata = cache / (key + ".json")
    result, previous_audit = _recover_result(out_dir, cache, key, text, identity, conf, speed)
    if result:
        raw = Path(result["audio_path"])
        cached_audio = cache / (key+raw.suffix)
        if raw.resolve() != cached_audio.resolve():
            shutil.copyfile(raw, cached_audio)
        result["audio_path"] = str(cached_audio.resolve())
        write_json(metadata, result)
        audit = _safe_metadata(previous_audit) if previous_audit else result.get("content_check") or {}
        if audit.get("audio_sha256") == result["audio_sha256"]:
            write_json(cache / (key+".audit.json"), audit)
    retries = max(0, min(2, int(config.param("tts_content_retries", 1)))) if config else 0
    failures = 0
    for attempt in range(retries + 1):
        failed = False
        if not Path(result.get("audio_path", "")).is_file() or failed:
            kwargs = {"speed": speed}
            if failed:
                old = Path(result["audio_path"])
                if old.is_file():
                    rejected = ensure_dir(cache / "rejected")
                    shutil.copyfile(old, rejected / (hashlib.sha256(old.read_bytes()).hexdigest() + old.suffix))
                kwargs["seed"] = (int(result.get("seed", 0)) + 1) % 65536
            result = tts.synth(text, cache / (key + ".wav"), **kwargs)
            result.update(synthesis_identity=identity, synthesis_cache_policy=SYNTHESIS_CACHE_POLICY,
                          synthesis_text_sha256=hashlib.sha256(text.encode()).hexdigest(), text=text,
                          audio_sha256=hashlib.sha256(Path(result["audio_path"]).read_bytes()).hexdigest())
            result["content_failure_count"] = failures
            if str(result.get("provider", "")).startswith("mock"):
                raise RuntimeError("真实配音失败，拒绝Mock测试音: %s" % result.get("fallback_reason", ""))
            write_json(metadata, result)
        raw = Path(result["audio_path"])
        if config and config.param("narration_audit", "local") == "local" and not str(result.get("provider", "")).startswith("test_"):
            from ...eval.narration_audit import audit_narration
            report = audit_narration(raw, text, cache / (key + ".audit.json"),
                                     model_name=config.param("narration_audit_model", "small"),
                                     require_word_timings=True)
            result["content_check"] = report
            report['gate_enabled'] = False
            report['purpose'] = 'word_clock_only'
            write_json(metadata, result)
        break
    target = out_dir / ("voice" + raw.suffix)
    if raw.resolve() != target.resolve():
        shutil.copyfile(raw, target)
    duration = media_duration(target)
    aligned = align_continuous_result(variant.sentences, result, duration)
    return {**aligned, "audio_path": str(target.resolve()), "duration": duration, "text": text,
            "playback_speed": speed,
            "audio_mode": "single_continuous_synthesis", "audio_edits": [],
            "original_audio_sha256": hashlib.sha256(raw.read_bytes()).hexdigest(),
            "voice_sha256": hashlib.sha256(target.read_bytes()).hexdigest()}


def align_continuous_result(sentences, result, duration, audit_report=None):
    """Refresh alignment for new or resumed narration, with no synthesis calls.

    ``audit_report`` can be a saved audit_narration report. Returned word and
    sentence timestamps use seconds on the actual intact audio, scale = 1.
    """
    from ...eval.speech_alignment import ALIGNMENT_POLICY, select_word_alignment
    expected = "".join(s.text for s in sentences)
    audit = audit_report if audit_report is not None else result.get("content_check") or {}
    previous = result.get("alignment_report") or {}
    words = result.get("word_timings", [])
    if (audit_report is None and previous.get("policy") == ALIGNMENT_POLICY
            and abs(float(previous.get("audio_duration", duration))-duration) <= .04
            and "".join(compact(w.get("text", "")) for w in words) == compact(expected)):
        # Existing canonical results already live on the audio clock. Do not
        # apply their original provider calibration a second time on resume.
        aligned, report = words, previous
    else:
        provider_source = result.get("timing_source", "")
        if result.get("timestamp_events"):
            from .bailian import _cosy_word_timings
            words = _cosy_word_timings(result["timestamp_events"])
            provider_source = "provider_word_timestamps" if words else "estimated"
        aligned, report = select_word_alignment(expected, duration, words, provider_source,
                                                 audit.get("asr_word_timings", []))
    timings = sentence_boundaries(sentences, aligned, duration)
    if not timings:
        from ...render.timeline import _align_proportional
        timings = _align_proportional(sentences, duration, 1.0)
        aligned = []
        report = {**report, "source": "estimated", "confidence": "low",
                  "reason": report.get("reason", "unreliable_sentence_boundaries")}
    report = {**report, "audio_duration": duration,
              "asr_model": audit.get("asr_model", ""), "asr_audio_sha256": audit.get("audio_sha256", "")}
    return {**result, "word_timings": aligned, "sentence_timings": timings,
            "timing_source": report["source"], "alignment_report": report,
            "timestamp_duration_scale": 1.0}


def sentence_boundaries(sentences, words, duration):
    spoken = [(compact(w.get("text", "")), float(w.get("start", 0)), float(w.get("end", 0))) for w in words]
    spoken = [w for w in spoken if w[0]]
    expected = "".join(compact(s.text) for s in sentences)
    if not spoken or "".join(w[0] for w in spoken) != expected:
        return []
    offsets = []
    cursor = 0
    for text, start, end in spoken:
        offsets.append((cursor, cursor + len(text), start, end))
        cursor += len(text)
    boundaries = [0.0]
    cursor = 0
    for sent in sentences[:-1]:
        cursor += len(compact(sent.text))
        next_word = next((w for w in offsets if w[0] <= cursor < w[1]), None)
        if next_word is None:
            return []
        next_start = next_word[2] + (next_word[3]-next_word[2]) * (cursor-next_word[0]) / (next_word[1]-next_word[0])
        if next_start <= boundaries[-1] or next_start >= duration:
            return []
        boundaries.append(next_start)
    boundaries.append(duration)
    return [{"index": s.index, "text": s.text, "role": s.role, "start": boundaries[i], "end": boundaries[i + 1]}
            for i, s in enumerate(sentences)]
