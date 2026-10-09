import json
import os
import re
import time
import uuid
from pathlib import Path

from ..log import get_logger
from ..utils import ensure_dir

DRAFT_RELATIVE_PATHS = (
    "JianyingPro/User Data/Projects/com.lveditor.draft",
    "CapCut/User Data/Projects/com.lveditor.draft",
    "JianyingPro/User Data/Projects",
    "CapCut/User Data/Projects",
)

DEFAULT_META_VERSION = "9.2.0"

_INVALID_CHARS = re.compile(r'[\\/:*?"<>|\s]+')


def detect_draft_root(override=""):
    if override:
        path = Path(override).expanduser()
        if path.exists():
            return str(path)
        return ""
    local = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~")
    for rel in DRAFT_RELATIVE_PATHS:
        candidate = Path(local) / rel
        if candidate.exists():
            return str(candidate)
    return ""


def export_draft(video_path, root, title, duration, width, height, folder_name=None,
                 source_start=0.0, source_duration=None, voice_path=None, orig_audio_path=None):
    logger = get_logger()
    video_path = Path(video_path).resolve()
    if not video_path.exists():
        raise FileNotFoundError("成片文件不存在: %s" % video_path)
    root = Path(root)
    ensure_dir(root)
    name = folder_name or _safe_name(title or "cutclip")
    draft_dir = root / name
    ensure_dir(draft_dir)

    duration = max(0.1, float(duration))
    width = int(width) or 1080
    height = int(height) or 1920
    src_duration = float(source_duration) if source_duration else duration
    material_id = "material_%s" % uuid.uuid4().hex[:12]
    segment_id = "segment_%s" % uuid.uuid4().hex[:12]
    path_str = _posix_path(video_path)

    template_content, meta_version = _load_template(root)
    content = _build_content(template_content, path_str, material_id, segment_id,
                             title, duration, width, height,
                             source_start=float(source_start or 0.0), source_duration=src_duration,
                             voice_path=voice_path, orig_audio_path=orig_audio_path)
    write_json(draft_dir / "draft_content.json", content)

    now_ms = int(time.time() * 1000)
    meta = {
        "tm_draft_create": now_ms,
        "tm_draft_modified": now_ms,
        "tm_draft_name": title or "成片",
        "tm_draft_folder_path": _posix_path(draft_dir),
        "tm_draft_removable": False,
        "tm_draft_type": 1,
        "tm_draft_cover": "",
        "tm_draft_version": meta_version or DEFAULT_META_VERSION,
        "tm_draft_use_platform": "",
        "tm_need_download": False,
        "tm_project_id": str(uuid.uuid4()),
        "tm_template_id": "newTemplate",
        "tm_extends": "{\"draft_mode\":\"normal\"}",
    }
    write_json(draft_dir / "draft_meta_info.json", meta)
    logger.info("剪映草稿已导出: %s", draft_dir)
    return draft_dir


def _load_template(root):
    candidates = list(sorted(root.glob("*/draft_meta_info.json")))
    for meta_path in candidates[:1]:
        content_path = meta_path.parent / "draft_content.json"
        if content_path.exists():
            try:
                content = json.loads(content_path.read_text(encoding="utf-8"))
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                version = meta.get("tm_draft_version") or DEFAULT_META_VERSION
                if isinstance(content, dict):
                    return content, version
            except (OSError, ValueError):
                pass
    return None, None


def _build_content(template, path_str, material_id, segment_id, title, duration, width, height,
                   source_start=0.0, source_duration=None, voice_path=None, orig_audio_path=None):
    ms = int(round(duration * 1000))
    src_ms = int(round(float(source_start) * 1000))
    src_dur_ms = int(round(float(source_duration or duration) * 1000))
    video = {
        "id": material_id,
        "type": "video",
        "path": path_str,
        "sound_path": path_str,
        "height": int(height),
        "width": int(width),
        "duration": duration,
        "material_name": title or "clip",
        "material_id": material_id,
        "similarity_id": "",
        "mute": False,
        "audio_volume": 1.0,
        "edit_type": "none",
    }
    clip = {
        "alpha": 1,
        "transform": {"x": 0, "y": 0, "scale": {"x": 1.0, "y": 1.0}, "rotation": 0},
        "track_scale": {"x": 1.0, "y": 1.0},
        "flip": {"horizontal": False, "vertical": False},
        "render_index": 0,
        "source_render_index": 0,
        "visible": True,
    }
    video_track = {
        "type": "video",
        "is_default_visible": True,
        "segments": [{
            "id": segment_id,
            "material_id": material_id,
            "speed": 1.0,
            "reverse": False,
            "extra_type_option": "normal",
            "target_timerange": {"duration": ms, "start": 0, "end": ms},
            "source_timerange": {"duration": src_dur_ms, "start": src_ms, "end": src_ms + src_dur_ms},
            "clip": clip,
        }],
    }
    audios, audio_tracks = _build_audio_tracks(video, material_id,
                                               voice_path=voice_path, orig_audio_path=orig_audio_path,
                                               src_ms=src_ms, src_dur_ms=src_dur_ms)

    if template and isinstance(template, dict) and isinstance(template.get("all_tracks"), list):
        content = dict(template)
        materials = content.get("materials") or {}
        if not isinstance(materials, dict):
            materials = {}
        materials["videos"] = [video]
        materials["audios"] = audios
        content["materials"] = materials
        content["all_tracks"] = [video_track] + audio_tracks
        _set_canvas(content, width, height)
        return content

    return {
        "canvas_config": {"canvas_size": {"width": width, "height": height},
                          "canvas_ratio": "custom", "color_space": 0,
                          "ratio": {"width": width, "height": height},
                          "mode": 2, "standard": 0, "fps": 25},
        "all_tracks": [video_track] + audio_tracks,
        "materials": {
            "videos": [video],
            "texts": [], "stickers": [], "audios": audios, "transitions": [],
            "effects": [], "filters": [], "audio_effects": [],
            "materials_audio_mixes": [], "materials_behaviors": [],
            "materials_video_effects": [], "materials_text_effects": [],
            "materials_interactive_alt_texts": [], "materials_labels": [],
            "materials_subtitle": [], "materials_plugs": [], "templates": [],
        },
        "keyframes": [],
        "track_group_container": [],
        "group_container": [],
        "audio_mixes": [],
        "primary_ufc": [],
        "motion_models": [],
        "watermark_config": {"watermark_mode": -1},
        "draft_materials": [],
        "chunk_data": [],
        "timeline_extend_models": [],
        "tracks": [],
        "text_effects": [],
        "video_effects": [],
        "custom_video_effects": [],
        "filters": [],
        "colors": [],
        "audio_effects": [],
        "audio_editor_transitions": [],
        "audio_mix": [],
        "behaviors": [],
        "version": "v1_1_0",
    }


def _build_audio_tracks(video, video_material_id, voice_path, orig_audio_path, src_ms, src_dur_ms):
    """额外音轨：TTS 旁白 + 原片声音，便于在剪映里重新混音（视频轨静音）。"""
    audios = []
    tracks = []

    def _add(path, name, source_ms):
        pid = Path(path).resolve()
        if not pid.exists():
            return
        aid = "audio_%s" % uuid.uuid4().hex[:12]
        seg = "audio_seg_%s" % uuid.uuid4().hex[:12]
        audios.append({
            "id": aid, "type": "audio", "path": _posix_path(pid),
            "name": name, "material_name": name, "duration": src_dur_ms / 1000.0,
            "music_id": "", "app_id": 0, "category_name": "", "formula_id": "",
        })
        tracks.append({
            "type": "audio",
            "is_default_visible": True,
            "segments": [{
                "id": seg, "material_id": aid, "speed": 1.0, "reverse": False,
                "extra_type_option": "normal", "volume": 1.0,
                "target_timerange": {"duration": src_dur_ms, "start": 0, "end": src_dur_ms},
                "source_timerange": {"duration": src_dur_ms, "start": source_ms, "end": source_ms + src_dur_ms},
            }],
        })

    if voice_path:
        _add(voice_path, "旁白", 0)
    if orig_audio_path:
        _add(orig_audio_path, "原片声音", src_ms)
    video['mute'] = bool(tracks)
    video['audio_volume'] = 0.0 if tracks else 1.0
    return audios, tracks


def _set_canvas(content, width, height):
    canvas = content.get("canvas_config") or {}
    if not isinstance(canvas, dict):
        canvas = {}
    canvas["canvas_size"] = {"width": width, "height": height}
    canvas["ratio"] = {"width": width, "height": height}
    content["canvas_config"] = canvas


def _posix_path(path):
    text = str(path)
    if "\\" in text:
        text = text.replace("\\", "/")
    return text


def _safe_name(text):
    text = _INVALID_CHARS.sub("_", text).strip("_")
    return (text or "cutclip")[:80]


def write_json(path, data):
    path = Path(path)
    ensure_dir(path.parent)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
