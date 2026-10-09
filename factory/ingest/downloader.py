import os
import shutil
from pathlib import Path

from ..log import get_logger
from ..models import VideoMeta
from ..utils import ensure_dir, write_json
from .probe import probe_video

SUB_LANGS = ["zh-Hans", "zh-CN", "zh", "zh-Hant", "en", "en-US"]


def is_local_source(source):
    try:
        p = Path(source)
        return p.exists() and p.is_file() and p.suffix.lower() in {".mp4", ".mkv", ".mov", ".webm", ".avi", ".flv", ".m4v"}
    except OSError:
        return False


def _resolve_proxy(proxy_hint=None):
    if proxy_hint:
        return proxy_hint
    for name in ("HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy", "ALL_PROXY", "all_proxy"):
        value = (os.environ.get(name) or "").strip()
        if value:
            return value
    return None


def fetch_info(url, proxy=None):
    import yt_dlp

    opts = {"quiet": True, "no_warnings": True, "noplaylist": True, "skip_download": True}
    resolved = _resolve_proxy(proxy)
    if resolved:
        opts["proxy"] = resolved
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=False)
    if info.get("_type") == "playlist" and info.get("entries"):
        info = info["entries"][0]
    return info


def _format_info(info):
    meta = VideoMeta()
    meta.video_id = info.get("id") or ""
    meta.title = info.get("title") or ""
    meta.channel = info.get("channel") or info.get("uploader") or ""
    meta.uploader = info.get("uploader") or ""
    meta.duration = float(info.get("duration") or 0.0)
    meta.url = info.get("webpage_url") or info.get("original_url") or ""
    meta.extractor = info.get("extractor_key") or info.get("extractor") or ""
    meta.description = (info.get("description") or "")[:2000]
    return meta


def download(url, out_dir, progress_cb=None, subtitle_langs=None, proxy=None):
    import yt_dlp

    out_dir = ensure_dir(out_dir)
    subtitle_langs = subtitle_langs or SUB_LANGS

    def hook(d):
        if not progress_cb:
            return
        status = d.get("status")
        if status == "downloading":
            total = d.get("total_bytes") or d.get("total_bytes_estimate") or 0
            done = d.get("downloaded_bytes") or 0
            if total:
                progress_cb(done / max(1, total), d.get("filename") or "")
            else:
                progress_cb(-1, d.get("filename") or "")
        elif status == "finished":
            progress_cb(1.0, d.get("filename") or "")

    opts = {
        "format": "bv*[height<=1080][ext=mp4]+ba[ext=m4a]/bv*[height<=1080]+ba/b[height<=1080]/b",
        "merge_output_format": "mp4",
        "outtmpl": str(out_dir / "%(id)s.%(ext)s"),
        "writesubtitles": True,
        "writeautomaticsub": True,
        "subtitleslangs": subtitle_langs,
        "subtitlesformat": "json3",
        "writeinfojson": True,
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "retries": 5,
        "fragment_retries": 5,
        "progress_hooks": [hook],
        "geo_bypass": True,
        "cachedir": False,
    }
    resolved_proxy = _resolve_proxy(proxy)
    if resolved_proxy:
        opts["proxy"] = resolved_proxy
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=True)
    if info.get("_type") == "playlist" and info.get("entries"):
        info = info["entries"][0]
    video_id = info.get("id") or "video"
    video_path = _find_video(out_dir, video_id, info)
    subtitle_path = _find_subtitle(out_dir, video_id, info)
    return video_path, info, subtitle_path


_VIDEO_EXTS = {".mp4", ".mkv", ".webm", ".mov"}


def _find_video(out_dir, video_id, info):
    for item in info.get("requested_downloads") or []:
        fp = item.get("filepath")
        if fp and Path(fp).exists():
            return Path(fp)
    for candidate in sorted(out_dir.glob(video_id + ".*")):
        if candidate.suffix.lower() in _VIDEO_EXTS:
            return candidate
    raise FileNotFoundError("下载完成但找不到视频文件: %s" % video_id)


def _find_subtitle(out_dir, video_id, info):
    subs = info.get("requested_subtitles") or {}
    for lang, item in subs.items():
        fp = item.get("filepath")
        if fp and Path(fp).exists():
            return Path(fp)
    for lang in SUB_LANGS:
        for ext in ("json3", "vtt", "srt"):
            candidate = out_dir / ("%s.%s.%s" % (video_id, lang, ext))
            if candidate.exists():
                return candidate
    for candidate in sorted(out_dir.glob(video_id + ".*")):
        if candidate.suffix.lower() in {".json3", ".vtt", ".srt"}:
            return candidate
    return None


def _find_sidecar_subtitle(video_path):
    for ext in (".srt", ".vtt", ".ass", ".ssa", ".json3"):
        candidate = video_path.with_suffix(ext)
        if candidate.exists():
            return candidate
    return None


def ingest(url, workspace, progress_cb=None, log=None, proxy=None):
    logger = log or get_logger()
    workspace = ensure_dir(workspace)
    raw_dir = ensure_dir(workspace / "source")
    if is_local_source(url):
        target = raw_dir / Path(url).name
        if Path(url).resolve() != target.resolve():
            if not target.exists():
                shutil.copy2(url, target)
        sidecar = _find_sidecar_subtitle(Path(url))
        subtitle_target = None
        if sidecar:
            subtitle_target = raw_dir / sidecar.name
            if sidecar.resolve() != subtitle_target.resolve() and not subtitle_target.exists():
                shutil.copy2(sidecar, subtitle_target)
        info = {"id": Path(url).stem, "title": Path(url).stem, "duration": None,
                "webpage_url": str(Path(url).resolve()), "extractor": "local"}
        meta = _format_info(info)
        probe = probe_video(target)
        meta.duration = probe["duration"]
        meta.width = probe["width"]
        meta.height = probe["height"]
        meta.fps = probe["fps"]
        meta.path = str(target)
        meta.subtitle_path = str(subtitle_target) if subtitle_target else ""
        write_json(workspace / "source" / "meta.json", {
            "meta": meta.__dict__, "info": {"id": info["id"], "title": info["title"]},
        })
        logger.info("使用本地视频: %s (字幕: %s)", target, subtitle_target.name if subtitle_target else "无")
        return meta, target, subtitle_target

    logger.info("获取视频信息: %s", url)
    video_path, info, subtitle_path = download(url, raw_dir, progress_cb=progress_cb, proxy=proxy)
    meta = _format_info(info)
    probe = probe_video(video_path)
    meta.path = str(video_path)
    meta.duration = probe["duration"] or meta.duration
    meta.width = probe["width"]
    meta.height = probe["height"]
    meta.fps = probe["fps"]
    meta.subtitle_path = str(subtitle_path) if subtitle_path else ""
    write_json(workspace / "source" / "meta.json", {"meta": meta.__dict__, "info": info})
    logger.info("下载完成: %s (%.1f MB)", video_path.name, (video_path.stat().st_size / 1e6))
    return meta, video_path, subtitle_path
