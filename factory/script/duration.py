"""Local narration duration estimates, calibrated per actual TTS voice.

No model requests or audio decoding are performed here. Only successful intact
narrations contribute, and repeated synthesis of the same text counts once.
"""
from dataclasses import dataclass
from hashlib import sha256
from itertools import islice
import json
import math
from pathlib import Path
from statistics import median
import threading
import time
import uuid

from .quality import compact
from ..config import DEFAULT_TTS_SPEED


_LOCK = threading.RLock()
_CACHE = {}
_MAX_RECORDS = 256
_MAX_IMPORT_FILES = 256
_MAX_FILE_BYTES = 2_000_000


@dataclass(frozen=True)
class DurationEstimate:
    seconds: float
    low_seconds: float
    high_seconds: float
    sample_count: int
    calibrated: bool
    spoken_units: int
    rate: float
    profile_id: str


def spoken_text(text):
    """Count comments as speech; never count bracket/legacy editor labels."""
    return str(text or "").replace("【评论话术｜不参与匹配】", "").translate(
        str.maketrans("", "", "（）()"))


def _number(value, default):
    try:
        value = float(value)
        return value if math.isfinite(value) else default
    except (TypeError, ValueError):
        return default


def _profile(config):
    conf = config.tts
    provider = str(conf.get("type") or "edge").lower()
    voice = str(conf.get("voice") or "")
    model = str(conf.get("model") or "")
    for item in conf.get("voices") or []:
        if isinstance(item, dict) and voice in (item.get("name"), item.get("voice_id")):
            voice = str(item.get("voice_id") or voice)
            model = str(item.get("model") or model)
            break
    if provider == "bailian":
        model = model or "qwen3-tts-flash"
        if model.startswith("cosyvoice") or voice.startswith("cosyvoice"):
            # This is the actual model used by BailianTTS._synth_cosyvoice.
            model = str(conf.get("cosyvoice_model") or "cosyvoice-v2")
    speed = max(0.25, min(4.0, _number(conf.get("speed"), DEFAULT_TTS_SPEED)))
    identity = {"provider": provider, "model": model, "voice": voice,
                "speed": round(speed, 4), "endpoint": str(conf.get("base_url") or "").rstrip("/"),
                "rate": str(conf.get("rate") or ""), "pitch": str(conf.get("pitch") or ""),
                "emotion": str(conf.get("emotion") or ""), "language": str(conf.get("language") or "")}
    key = sha256(json.dumps(identity, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    return key, identity


def _history_path(config, history_path=None):
    return Path(history_path) if history_path else config.workspace_path() / ".cache" / "narration_duration_v1.json"


def _safe_read(path):
    try:
        if path.stat().st_size > _MAX_FILE_BYTES:
            return {}
        result = json.loads(path.read_text(encoding="utf-8"))
        return result if isinstance(result, dict) else {}
    except (OSError, ValueError, UnicodeError):
        return {}


def _stamp(path):
    try:
        stat = path.stat()
        return stat.st_mtime_ns, stat.st_size
    except OSError:
        return None


def _load(path):
    key = str(path.resolve())
    stamp = _stamp(path)
    cached = _CACHE.get(key)
    if cached and cached[0] == stamp:
        return cached[1]
    raw = _safe_read(path)
    rows = raw.get("records", [])
    profiles = raw.get("imported_profiles", [])
    data = {"version": 1, "records": [r for r in rows if isinstance(r, dict)][-_MAX_RECORDS:]
            if isinstance(rows, list) else [],
            "imported_profiles": [p for p in profiles if isinstance(p, str)][-64:]
            if isinstance(profiles, list) else []}
    _CACHE[key] = stamp, data
    return data


def _save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    _CACHE[str(path.resolve())] = _stamp(path), data


def _record(profile_id, identity, text, result):
    if not isinstance(result, dict) or not isinstance(text, str):
        return None
    provider = str(result.get("provider") or "").lower()
    if not provider or provider.startswith(("mock", "test")) or not provider.startswith(identity["provider"]):
        return None
    audit = result.get("content_check") or {}
    if isinstance(audit, dict) and audit.get("status") not in (None, "pass"):
        return None
    if result.get("audio_edits"):
        return None
    actual_voice = str(result.get("voice") or "")
    if actual_voice and actual_voice != identity["voice"]:
        return None
    actual_speed = result.get("playback_speed")
    if actual_speed is not None and abs(_number(actual_speed, -1) - identity["speed"]) > 0.001:
        return None
    if result.get("model") and result["model"] != identity["model"]:
        return None
    clean = compact(spoken_text(text))
    duration = _number(result.get("duration"), 0)
    if len(clean) < 20 or not 3 <= duration <= 600 or not 1.5 <= len(clean) / duration <= 16:
        return None
    text_hash = sha256(clean.encode()).hexdigest()
    return {"profile_id": profile_id, "text_hash": text_hash, "units": len(clean),
            "duration": duration, "recorded_at": time.time()}


def record_duration(config, text, result, *, history_path=None):
    """Record a successful whole narration. Call after content validation.

    A retry with the same spoken text replaces its sample instead of inflating
    confidence. Stored data contains hashes/counts only, never keys or copy.
    """
    profile_id, identity = _profile(config)
    row = _record(profile_id, identity, text, result)
    if not row:
        return False
    path = _history_path(config, history_path)
    with _LOCK:
        data = _load(path)
        data["records"] = [r for r in data["records"]
                           if (r.get("profile_id"), r.get("text_hash")) != (profile_id, row["text_hash"])]
        data["records"] = (data["records"] + [row])[-_MAX_RECORDS:]
        try:
            _save(path, data)
        except OSError:
            return False
    return True


def _import_existing(config, path, data, profile_id, identity):
    """One bounded metadata scan per profile; unknown identity is not borrowed."""
    if profile_id in data["imported_profiles"]:
        return
    directory = config.workspace_path() / ".cache" / "tts_continuous"
    for item in islice(directory.glob("*.json"), _MAX_IMPORT_FILES):
        if item.name.endswith(".audit.json"):
            continue
        result = _safe_read(item)
        audit = result.get("content_check") or {}
        if not isinstance(audit, dict) or audit.get("status") != "pass":
            continue
        # Older cache entries did not preserve model names. Only the native
        # CosyVoice v2 path can be identified unambiguously from that format.
        if not result.get("model") and not (identity["provider"] == "bailian"
                and identity["model"] == "cosyvoice-v2"
                and str(result.get("voice", "")).startswith("cosyvoice")
                and str(result.get("provider", "")) == "bailian-cosyvoice"
                and result.get("playback_speed") is not None):
            continue
        text = result.get("text") or audit.get("expected_text")
        row = _record(profile_id, identity, text, result)
        if row and not any(r.get("profile_id") == profile_id and r.get("text_hash") == row["text_hash"]
                           for r in data["records"]):
            data["records"].append(row)
    data["records"] = data["records"][-_MAX_RECORDS:]
    data["imported_profiles"] = (data["imported_profiles"] + [profile_id])[-64:]
    try:
        _save(path, data)
    except OSError:
        # A read-only cache must not prevent editing or story generation.
        pass


def estimate_duration(config, text, *, history_path=None):
    """Estimate full speech, including parenthesized comments, without API use."""
    profile_id, identity = _profile(config)
    path = _history_path(config, history_path)
    with _LOCK:
        data = _load(path)
        _import_existing(config, path, data, profile_id, identity)
        rates = []
        for row in data["records"]:
            if row.get("profile_id") != profile_id:
                continue
            duration = _number(row.get("duration"), 0)
            units = _number(row.get("units"), 0)
            if duration >= 3 and units >= 20 and 1.5 <= units / duration <= 16:
                rates.append(units / duration)
    rate = median(rates) if rates else 5.5 * identity["speed"]
    if len(rates) >= 5:
        spread = median(abs(r - rate) for r in rates)
        rates = [r for r in rates if abs(r - rate) <= max(0.25 * rate, 3 * spread)]
        rate = median(rates)
    count = len(rates)
    relative_spread = (median(abs(r - rate) for r in rates) / rate) if count else 0
    uncertainty = max(0.08 if count >= 5 else 0.12 if count >= 2 else 0.25,
                      min(0.4, relative_spread * 2.5))
    units = len(compact(spoken_text(text)))
    seconds = units / rate
    return DurationEstimate(round(seconds, 2), round(max(0, seconds * (1 - uncertainty)), 2),
                            round(seconds * (1 + uncertainty), 2), count, count >= 2, units,
                            round(rate, 4), profile_id)


def word_budget(config, min_seconds=None, max_seconds=None, reserve_text=""):
    """Chinese spoken-character budget shared by writing and review.

    Comments reserve speech time but impose no plot/visual requirements.
    """
    estimate = estimate_duration(config, "")
    minimum_seconds = max(1.0, _number(min_seconds if min_seconds is not None
                                     else config.param("min_duration", 15), 15))
    maximum_seconds = max(minimum_seconds, _number(max_seconds if max_seconds is not None
                                                   else config.param("max_duration", 35), 35))
    reserve = len(compact(spoken_text(reserve_text)))
    minimum = max(1, math.ceil(minimum_seconds * estimate.rate) - reserve)
    maximum = max(minimum, math.floor(maximum_seconds * estimate.rate) - reserve)
    return {"minimum": minimum, "maximum": maximum, "target": (minimum + maximum) // 2,
            "rate": estimate.rate, "calibrated": estimate.calibrated,
            "sample_count": estimate.sample_count, "reserved_units": reserve}


def duration_state(estimate, minimum, maximum):
    """Only an entirely over-limit estimate is a strong pre-TTS rejection."""
    if estimate.spoken_units == 0:
        return "empty"
    if estimate.low_seconds > float(maximum):
        return "too_long"
    if estimate.high_seconds < float(minimum):
        return "too_short"
    if estimate.seconds < float(minimum):
        return "near_short"
    if estimate.seconds > float(maximum):
        return "near_long"
    return "in_range"
