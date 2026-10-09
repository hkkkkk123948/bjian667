import copy
from pathlib import Path

from .utils import ensure_dir, read_json, write_json

DEFAULT_TTS_SPEED = 1.1

ROLES = [
    "video_analysis",
    "story_analysis",
    "reference_analysis",
    "script_generation",
    "match_judge",
    "qc",
]

ROLE_LABELS = {
    "video_analysis": "视频理解",
    "story_analysis": "剧情分析",
    "reference_analysis": "文案分析",
    "script_generation": "文案生成",
    "match_judge": "首次镜头匹配",
    "qc": "画面匹配审核",
}

LLM_TYPES = ["mock", "gemini", "openai", "custom_http"]
TTS_TYPES = ["mock", "edge", "bailian", "elevenlabs", "openai", "gemini"]

DEFAULT_CONFIG = {
    "workspace": "workspace",
    "providers": {
        "video_analysis": {"type": "openai", "base_url": "https://ai.t8star.org/v1", "api_key": "", "model": "gpt-5.6-sol", "backup_models": ["qwen3-vl-flash"], "timeout": 600, "max_retry": 2},
        "story_analysis": {"type": "openai", "base_url": "https://ai.t8star.org/v1", "api_key": "", "model": "gpt-5.6-sol", "backup_models": ["qwen3-vl-flash"], "timeout": 300, "max_retry": 2},
        "reference_analysis": {"type": "openai", "base_url": "https://ai.t8star.org/v1", "api_key": "", "model": "gpt-5.6-sol", "backup_models": ["qwen3-vl-flash"], "timeout": 180, "max_retry": 2},
        "script_generation": {"type": "openai", "base_url": "https://ai.t8star.org/v1", "api_key": "", "model": "gpt-5.6-sol", "backup_models": ["qwen3-vl-flash"], "timeout": 300, "max_retry": 2},
        "match_judge": {"type": "openai", "base_url": "https://ai.t8star.org/v1", "api_key": "", "model": "gpt-5.6-sol", "backup_models": ["qwen3-vl-flash"], "timeout": 300, "max_retry": 2},
        "qc": {"type": "openai", "base_url": "https://ai.t8star.org/v1", "api_key": "", "model": "gpt-5.6-sol", "backup_models": ["qwen3-vl-flash"], "timeout": 300, "max_retry": 2},
    },
    "tts": {"type": "edge", "voice": "zh-CN-YunxiNeural", "speed": DEFAULT_TTS_SPEED, "api_key": "", "base_url": "", "model": "",
            "rate": "", "volume": "", "language": "", "emotion": "", "voices": []},
    "judge": {"type": "openai", "base_url": "https://ai.t8star.org/v1", "api_key": "",
              "model": "gemini-3.1-flash-lite-preview", "timeout": 300, "max_retry": 1},
    "judge_weights": {"match": 0.30, "visual": 0.20, "story": 0.20, "copy": 0.15,
                      "pacing": 0.10, "originality": 0.05},
    "asr": {"base_url": "", "api_key": "", "model": "whisper-1", "language": "auto"},
    "ytdlp": {"proxy": ""},
    "params": {
        "min_duration": 24.0,
        "max_duration": 36.0,
        "min_match": 75.0,
        "max_clips": 0,
        "candidate_pool": 12,
        "scripts_per_candidate": 3,
        "script_candidates": 6,
        "analyze_max_candidates": 0,
        "rewrite_margin": 15.0,
        "max_rewrite_attempts": 3,
        "review_mode": "semi",
        "content_mode": "creative",
        "creative_strategy": "visual_story",
        "sentence_tts": "off",
        "voice_mode": "continuous",
        "narration_audit": "local",
        "narration_audit_model": "small",
        "tts_content_retries": 0,
        "tts_text_repair": "off",
        "visual_motion": "on",
        "motion_oversample": 3,
        "min_shot_seconds": 0.7,
        "short_transitions": "on",
        "transition_duration": 0.125,
        "picture_enhance": "on",
        "picture_grade": "on",
        "render_crf": 17,
        "finish_crf": 16,
        "max_freeze_seconds": 0.7,
        "story_complexity": "simple",
        "comment_hook_text": "",
        "comment_hook_position": "auto",
        "comment_hook_role": "story_analysis",
        "comment_visual_mode": "auto",
        "approved_plan_reuse": "on",
        "director_overview_frames": 48,
        "director_detail_frames": 40,
        "director_rewrite_attempts": 1,
        "qc_semantic_min": 80,
        "reference_local_only": "on",
        "sound_design": "on",
        "sound_bed_db": -29.0,
        "aspect_ratio": "auto",
        "target_width": 1080,
        "target_height": 1920,
        "crop_focus": 0.5,
        "bg_volume": 0.0,
        "voice_volume": 1.0,
        "scene_threshold": 0.35,
        "frame_l1_fps": 0.5,
        "frame_l2_fps": 4.0,
        "frame_l3_fps": 8.0,
        "max_l1_frames": 1800,
        "transcribe": "force",
        "whisper_model": "small",
        "visual_frames_per_scene": 1,
        "visual_batch_scenes": 6,
        "visual_workers": 3,
        "language": "zh",
        "subtitle_font": "Microsoft YaHei",
        "subtitle_fontsize": 64,
        "dedupe_threshold": 0.72,
        "gate2_drop_tolerance": 8.0,
        "llm_cache": "on",
        "transcribe_full": "auto",
        "window_transcribe_min": 1200,
        "window_transcribe_pad": 3.0,
        "mine_prompt_chars": 24000,
        "reframe_faces": "on",
        "normalize_loudness": "on",
        "cleanup_frames": "on",
        "match_workers": 3,
        "gate_fallback": "off",
        "story_rebuild": "on",
        "ocr": "auto",
        "pre_tts_compress": "off",
        "remove_subtitles": "off",
        "subtitle_band": 0.16,
        "auto_crop_black": "on",
        "llm_price_input": 0.0,
        "llm_price_output": 0.0,
    },
}


def deep_merge(base, extra):
    out = copy.deepcopy(base)
    for key, value in (extra or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = value
    return out


class Config:
    def __init__(self, data=None, path=None):
        self.data = deep_merge(DEFAULT_CONFIG, data or {})
        self.path = Path(path) if path else None

    @classmethod
    def load(cls, path):
        p = Path(path)
        data = read_json(p, {})
        return cls(data, p)

    def save(self, path=None):
        p = Path(path or self.path or Path(self.workspace) / "config.json")
        ensure_dir(p.parent)
        write_json(p, self.data)
        self.path = p
        return p

    @property
    def workspace(self):
        return self.data.get("workspace", "workspace")

    def workspace_path(self):
        p = Path(self.workspace)
        if not p.is_absolute():
            p = Path(__file__).resolve().parent.parent / p
        ensure_dir(p)
        return p

    @property
    def params(self):
        return self.data["params"]

    def param(self, key, default=None):
        return self.data["params"].get(key, default)

    def provider(self, role):
        providers = self.data.get("providers", {})
        conf = dict(providers.get(role) or providers.get("story_analysis") or {})
        conf.setdefault("type", "mock")
        return conf

    def set_provider(self, role, conf):
        self.data.setdefault("providers", {})[role] = conf

    @property
    def tts(self):
        return self.data.get("tts", {})

    @property
    def asr(self):
        return self.data.get("asr", {})

    def job_dir(self, job_id):
        p = self.workspace_path() / "jobs" / job_id
        ensure_dir(p)
        return p

    def clone(self):
        return Config(copy.deepcopy(self.data), self.path)


def new_job_id(url=""):
    import hashlib
    import time

    raw = "%s|%s" % (time.time(), url)
    return time.strftime("%Y%m%d_%H%M%S") + "_" + hashlib.md5(raw.encode("utf-8")).hexdigest()[:6]
