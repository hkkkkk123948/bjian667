"""Conventional picture finishing after assembly, with the audio stream copied."""
from pathlib import Path
import subprocess

from ..utils import ffmpeg_bin, run_cmd, write_json

BASE_PICTURE_FILTER = "hqdn3d=0.8:0.6:1.2:0.8,unsharp=5:5:0.22:3:3:0.0"
GRADE_FILTER = "eq=contrast=1.035:gamma=0.985"
# Kept for callers/tests that import the historical name.
PICTURE_FILTER = BASE_PICTURE_FILTER


def picture_filter(params=None):
    params = params or {}
    parts = []
    if str(params.get("picture_grade", "on")).lower() == "on":
        parts.append(GRADE_FILTER)
    parts.append(BASE_PICTURE_FILTER)
    return ",".join(parts)


def finish_picture(source, target, params):
    params = params or {}
    filt = picture_filter(params)
    crf = str(int(params.get("finish_crf", params.get("render_crf", 16))))
    before = _audio_hash(source)
    run_cmd([ffmpeg_bin(), "-hide_banner", "-loglevel", "error", "-y", "-i", str(source),
             "-map", "0:v:0", "-map", "0:a:0", "-vf", filt,
             "-c:v", "libx264", "-preset", "veryfast", "-tune", "zerolatency", "-crf", crf, "-threads", "1",
             "-x264-params", "rc-lookahead=0:sync-lookahead=0:ref=2:bframes=0",
             "-pix_fmt", "yuv420p", "-c:a", "copy", "-movflags", "+faststart", str(target)], timeout=1800)
    after = _audio_hash(target)
    if before != after:
        raise RuntimeError("画质后处理意外改变了音频流")
    write_json(Path(target).parent / "picture_finish.json", {
        "filters": filt, "method": "mild_denoise_luma_sharpen_optional_grade",
        "output_crf": int(crf), "picture_grade": str(params.get("picture_grade", "on")),
        "resolution_change": False, "audio_stream": "copy", "audio_time_edits": [],
        "source_audio_sha256": before, "output_audio_sha256": after,
        "note": "轻去噪、亮度锐化与可选轻度反差；保留原始分辨率，音频流原样拷贝。"})


def _audio_hash(path):
    return subprocess.check_output([ffmpeg_bin(), "-v", "error", "-i", str(path), "-map", "0:a:0",
                                    "-c:a", "copy", "-f", "hash", "-hash", "sha256", "-"],
                                   text=True, timeout=60).strip().split("=", 1)[-1]
