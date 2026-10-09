"""Per-clip production checkpoints for incremental script approval.

Unselected clips are parked, never deleted. Narration reuse requires the
actual synthesis identity, exact spoken text and a matching audio digest.
"""

import hashlib
import json
import re
from pathlib import Path

from .models import ClipResult, to_dict
from .utils import read_json, write_json

PICTURE_PARAMS = (
    "min_duration", "max_duration", "scene_align", "scene_align_role", "language",
    "visual_motion", "min_shot_seconds", "min_required_shot_seconds", "max_freeze_seconds",
    "comment_visual_mode", "evidence_cut",
)
RENDER_PARAMS = (
    "aspect_ratio", "target_width", "target_height", "crop_focus", "reframe_faces",
    "short_transitions", "transition_duration", "motion_oversample", "picture_enhance", "picture_grade",
    "voice_volume", "bg_volume", "sound_design", "sound_bed_db", "render_crf", "finish_crf",
    "remove_subtitles", "subtitle_band", "auto_crop_black", "normalize_loudness",
)
QUALITY_PARAMS = ("qc_semantic_min", "qc_visual_min")
STAGES = ["TTS_GENERATED", "MATCHING", "TIMELINE_ALIGNED", "RENDERING",
          "FINAL_MATCH_CHECK", "QUALITY_CHECK", "DRAFT_EXPORT", "COMPLETED"]
REVIEW_POLICY = "single_visual_review_only_v3"


def _digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     default=str).encode("utf-8")).hexdigest()


def file_digest(path):
    path = Path(path)
    if not path.is_file():
        return ""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read(path):
    try:
        data = read_json(path)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError, UnicodeError):
        return {}


def clip_dir(pipeline, clip):
    return Path(pipeline.job_dir) / "clips" / clip.clip_id


def synthesis_key(pipeline, candidate, variant):
    from .providers.tts.continuous import synthesis_identity
    return _digest({"candidate": candidate.candidate_id, "variant": variant.variant_id,
                    "text": variant.full_text, "identity": synthesis_identity(pipeline.config.tts),
                    "mode": pipeline.config.param("voice_mode", "continuous")})


def _sentence_inputs(variant):
    sentences = []
    for sentence in variant.sentences:
        data = to_dict(sentence)
        data.pop("estimated_seconds", None)
        sentences.append(data)
    return sentences


def verified_voice(pipeline, clip, variant):
    from .providers.tts.continuous import _verified_result, synthesis_identity
    result = _read(clip_dir(pipeline, clip) / "tts.json")
    identity = synthesis_identity(pipeline.config.tts)
    if result.get("provider") == "mock" and identity["provider"] != "mock":
        return None
    saved_mode = result.get("production_voice_mode")
    if saved_mode and saved_mode != pipeline.config.param("voice_mode", "continuous"):
        return None
    if pipeline.config.param("voice_mode", "continuous") == "continuous" and result.get("audio_edits"):
        return None
    audio = Path(str(result.get("audio_path") or ""))
    if not audio.is_file():
        # Older metadata may still refer to an external cache that was moved.
        audio = clip_dir(pipeline, clip) / ("voice" + (audio.suffix or ".wav"))
    recovered = _verified_result(result, identity,
                                 variant.full_text, audio)
    if recovered:
        return recovered
    return None


def picture_key(pipeline, clip, candidate, variant):
    source = Path(str(pipeline.state.artifacts.get("video_path") or ""))
    stat = source.stat() if source.is_file() else None
    return _digest({
        "source": [str(source), stat.st_size if stat else None, stat.st_mtime_ns if stat else None],
        "span": [candidate.source_start, candidate.source_end],
        "voice": clip.scores.get("voice_digest", ""),
        "sentences": _sentence_inputs(variant),
        "scene_matches": variant.scene_matches, "comment_visual_mode": variant.comment_visual_mode,
        "params": {key: pipeline.config.param(key) for key in PICTURE_PARAMS},
    })


def render_key(pipeline, clip, policy):
    params = {key: pipeline.config.param(key) for key in RENDER_PARAMS}
    if params["aspect_ratio"] == "auto":
        size = pipeline.state.artifacts.get("source_size") or []
        meta = pipeline.state.artifacts.get("meta") or {}
        width, height = (size if len(size) == 2 else [meta.get("width"), meta.get("height")])
        if width and height:
            params.update(target_width=int(width), target_height=int(height))
    return _digest({"policy": policy, "params": params,
                    "timeline": _read(clip_dir(pipeline, clip) / "timeline.json"),
                    "regions": clip.scores.get("mask_regions"),
                    "region_method": clip.scores.get("mask_method")})


def quality_key(pipeline, clip, candidate, variant):
    from .versioning import PROMPT_VERSIONS
    provider = pipeline.config.data.get("providers", {}).get("qc", {})
    return _digest({"media": file_digest(clip.path) if clip.path else "", "text": variant.full_text,
                    "sentences": _sentence_inputs(variant),
                    "candidate": [candidate.source_start, candidate.source_end],
                    "params": {key: pipeline.config.param(key) for key in QUALITY_PARAMS},
                    "provider": {key: provider.get(key) for key in ("type", "model", "base_url")},
                    "policy": PROMPT_VERSIONS.get("qc_semantic"),
                    "review_policy": REVIEW_POLICY, "copy_review": "disabled_for_production_queue"})


def timeline_ready(pipeline, clip, candidate, variant):
    return (clip.scores.get("picture_key") == picture_key(pipeline, clip, candidate, variant)
            and bool(_read(clip_dir(pipeline, clip) / "timeline.json")))


def render_ready(pipeline, clip, policy):
    return (clip.scores.get("render_policy") == policy and clip.path and Path(clip.path).is_file()
            and clip.scores.get("render_key") == render_key(pipeline, clip, policy)
            and clip.scores.get("render_media_digest") == file_digest(clip.path))


def final_check_ready(pipeline, clip, candidate, variant):
    return bool(clip.gate2) and clip.scores.get("final_check_key") == quality_key(pipeline, clip, candidate, variant)


def quality_ready(pipeline, clip, candidate, variant):
    # An unavailable model or inconclusive review must remain retryable.
    return (clip.qc.get("status") in ("pass", "fail")
            and clip.scores.get("quality_key") == quality_key(pipeline, clip, candidate, variant))


def checkpoint_voice(pipeline, clip, candidate, variant, result):
    from .providers.tts.continuous import synthesis_identity
    result = dict(result, text=variant.full_text,
                  synthesis_identity=result.get("synthesis_identity") or synthesis_identity(pipeline.config.tts),
                  production_voice_mode=pipeline.config.param("voice_mode", "continuous"))
    digest = file_digest(result["audio_path"])
    if digest:
        result.setdefault("audio_sha256", digest)
    write_json(clip_dir(pipeline, clip) / "tts.json", result)
    clip.variant_id, clip.script_text = variant.variant_id, variant.full_text
    clip.source_start, clip.source_end = candidate.source_start, candidate.source_end
    clip.scores.update(tts_path=result["audio_path"], tts_duration=result["duration"],
                       voice_digest=digest,
                       synthesis_key=synthesis_key(pipeline, candidate, variant))
    clip.status = "tts_done"
    pipeline._save_clips()


def reconcile_production_queue(pipeline, queue, policy):
    """Activate the selected stories and invalidate only their missing stages.

    Returns changed/reused/parked clip IDs and the task stages invalidated.
    No synthesis, model request or destructive filesystem operation occurs.
    """
    registry = Path(pipeline.job_dir) / "clips" / "parked_clips.json"
    parked_data = read_json(registry, []) or []
    parked = [ClipResult(**{key: value for key, value in data.items()
                           if key in ClipResult.__annotations__}) for data in parked_data]
    all_clips = {clip.clip_id: clip for clip in parked}
    all_clips.update({clip.clip_id: clip for clip in pipeline.clips})
    numbers = [int(match.group(1)) for name in all_clips
               if (match := re.fullmatch(r"clip_(\d+)", name))]
    for folder in (Path(pipeline.job_dir) / "clips").glob("clip_*"):
        match = re.fullmatch(r"clip_(\d+)", folder.name)
        if match:
            numbers.append(int(match.group(1)))
    next_number = max(numbers, default=0) + 1
    active, changed, reused = [], [], []
    earliest = len(STAGES)
    for candidate in queue:
        variant = pipeline._pick_variant(candidate)
        if not variant:
            continue
        wanted = synthesis_key(pipeline, candidate, variant)
        choices = [clip for clip in all_clips.values()
                   if clip.candidate_id == candidate.candidate_id and clip.variant_id == variant.variant_id
                   and clip.script_text == variant.full_text]
        chosen, result = None, None
        for clip in choices:
            cached = verified_voice(pipeline, clip, variant)
            if cached:
                chosen, result = clip, cached
                break
        if chosen is None:
            # A saved pending record keeps its directory stable across failed attempts.
            chosen = next((clip for clip in choices if clip.status == "pending"
                           and clip.scores.get("synthesis_key") == wanted), None)
        if chosen is None:
            chosen = ClipResult(clip_id="clip_%03d" % next_number,
                                candidate_id=candidate.candidate_id, variant_id=variant.variant_id,
                                script_text=variant.full_text, source_start=candidate.source_start,
                                source_end=candidate.source_end, scores={"synthesis_key": wanted})
            next_number += 1
            all_clips[chosen.clip_id] = chosen
        active.append(chosen)
        if result is None:
            earliest = min(earliest, 0)
            changed.append(chosen.clip_id)
            continue
        reused.append(chosen.clip_id)
        chosen.scores.update(synthesis_key=wanted, voice_digest=result["audio_sha256"],
                             tts_path=result["audio_path"], tts_duration=result["duration"])
        picture = picture_key(pipeline, chosen, candidate, variant)
        # Adopt complete legacy records once; an explicitly outdated script is never adopted.
        if (not chosen.scores.get("picture_key") and chosen.status != "outdated_script"
                and chosen.scores.get("render_policy") == policy
                and _read(clip_dir(pipeline, chosen) / "timeline.json")):
            chosen.scores["picture_key"] = picture
        if not timeline_ready(pipeline, chosen, candidate, variant):
            earliest = min(earliest, 1)
            changed.append(chosen.clip_id)
            continue
        if (not chosen.scores.get("render_key") and chosen.scores.get("render_policy") == policy
                and chosen.path and Path(chosen.path).is_file()):
            chosen.scores["render_key"] = render_key(pipeline, chosen, policy)
            chosen.scores["render_media_digest"] = file_digest(chosen.path)
        if not render_ready(pipeline, chosen, policy):
            earliest = min(earliest, 3)
            changed.append(chosen.clip_id)
            continue
        quality = quality_key(pipeline, chosen, candidate, variant)
        if (not chosen.scores.get("final_check_key") and chosen.gate2
                and chosen.scores.get("quality_review_policy") == REVIEW_POLICY):
            chosen.scores["final_check_key"] = quality
        if (not chosen.scores.get("quality_key") and chosen.qc
                and chosen.scores.get("quality_review_policy") == REVIEW_POLICY):
            chosen.scores["quality_key"] = quality
        if not final_check_ready(pipeline, chosen, candidate, variant):
            earliest = min(earliest, 4)
            changed.append(chosen.clip_id)
        elif not quality_ready(pipeline, chosen, candidate, variant):
            earliest = min(earliest, 5)
            changed.append(chosen.clip_id)
    active_ids = [clip.clip_id for clip in active]
    inactive = [clip for key, clip in all_clips.items() if key not in active_ids]
    previous_ids = pipeline.state.artifacts.get("active_production_clips")
    if previous_ids != active_ids:
        earliest = min(earliest, 6)
    reset = STAGES[earliest:]
    pipeline.state.completed_stages = [stage for stage in pipeline.state.completed_stages if stage not in reset]
    pipeline.state.artifacts["active_production_clips"] = active_ids
    pipeline.state.artifacts["parked_production_clips"] = [clip.clip_id for clip in inactive]
    pipeline.clips = active
    write_json(registry, [to_dict(clip) for clip in inactive])
    pipeline._save_clips()
    pipeline.save_state()
    return {"changed_clip_ids": changed, "reused_clip_ids": reused,
            "parked_clip_ids": [clip.clip_id for clip in inactive], "reset_stages": reset}
