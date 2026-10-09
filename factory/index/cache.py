"""跨任务媒体索引缓存（文档 §89）：同一视频+同参数只分析一次。

缓存键 = 视频身份(路径/大小/mtime) + 影响索引的关键参数。
命中后直接复用场景、转写、音频事件与场景画面描述，跳过
场景检测 / Whisper / 抽帧 / 视觉分析。
"""

import hashlib
import json
from pathlib import Path

from ..utils import read_json, write_json

CACHE_VERSION = 4


def index_cache_key(video_path, params):
    p = Path(video_path)
    try:
        st = p.stat()
        ident = "%s|%d|%d" % (p.resolve(), st.st_size, int(st.st_mtime))
    except OSError:
        ident = str(video_path)
    signature = {
        "v": CACHE_VERSION,
        "scene_threshold": params.get("scene_threshold"),
        "language": params.get("language"),
        "whisper_model": params.get("whisper_model"),
        "transcribe": params.get("transcribe"),
        "visual_frames": params.get("visual_frames_per_scene"),
        "asr_language": (params.get("asr_language") or "auto"),
    }
    h = hashlib.sha1()
    h.update(ident.encode("utf-8"))
    h.update(b"\x00")
    h.update(json.dumps(signature, sort_keys=True, ensure_ascii=False).encode("utf-8"))
    return h.hexdigest()


def cache_path(cache_dir, key):
    return Path(cache_dir) / (key + ".json")


def load_cached(cache_dir, key):
    return read_json(cache_path(cache_dir, key))


def save_cached(cache_dir, key, index):
    from ..models import to_dict

    write_json(cache_path(cache_dir, key), to_dict(index))
