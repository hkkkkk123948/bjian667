import json

from ..utils import ffprobe_bin, run_cmd


def probe_video(path):
    out, _, _ = run_cmd([
        ffprobe_bin(), "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height,r_frame_rate,avg_frame_rate,duration,codec_name",
        "-of", "json", str(path)
    ], check=True, timeout=60)
    stream = {}
    try:
        streams = json.loads(out).get("streams") or []
        stream = streams[0] if streams else {}
    except json.JSONDecodeError:
        stream = {}

    out2, _, _ = run_cmd([
        ffprobe_bin(), "-v", "error",
        "-show_entries", "format=duration,size,bit_rate",
        "-of", "json", str(path)
    ], check=True, timeout=60)
    fmt = {}
    try:
        fmt = json.loads(out2).get("format") or {}
    except json.JSONDecodeError:
        fmt = {}

    width = int(stream.get("width") or 0)
    height = int(stream.get("height") or 0)
    fps = _parse_rate(stream.get("avg_frame_rate")) or _parse_rate(stream.get("r_frame_rate")) or 0.0
    duration = _to_float(stream.get("duration")) or _to_float(fmt.get("duration")) or 0.0
    return {
        "width": width,
        "height": height,
        "fps": round(fps, 3),
        "duration": round(duration, 3),
        "codec": stream.get("codec_name", ""),
        "size": _to_float(fmt.get("size")),
        "bit_rate": _to_float(fmt.get("bit_rate")),
    }


def has_audio_stream(path):
    out, _, code = run_cmd([
        ffprobe_bin(), "-v", "error", "-select_streams", "a",
        "-show_entries", "stream=index", "-of", "csv=p=0", str(path)
    ], check=False)
    return code == 0 and out.strip() != ""


def _parse_rate(value):
    if not value:
        return 0.0
    text = str(value)
    if "/" in text:
        num, den = text.split("/", 1)
        try:
            den_f = float(den)
            return float(num) / den_f if den_f else 0.0
        except ValueError:
            return 0.0
    try:
        return float(text)
    except ValueError:
        return 0.0


def _to_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0
