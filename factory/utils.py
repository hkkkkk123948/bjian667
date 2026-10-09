import json
import math
import os
import re
import shutil
import subprocess
import sys
import unicodedata
from pathlib import Path

FFMPEG = "ffmpeg"
FFPROBE = "ffprobe"
FFPLAY = "ffplay"

CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
LATIN_WORD_RE = re.compile(r"[A-Za-z][A-Za-z'\-]*")
PUNCT_PAUSE = "，、；：,;:"
PUNCT_BREAK = "。！？!?\n"
_SENT_CJK = re.compile(r"(?<=[。！？!?；;])")
_SENT_EN = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'（(])")
_ABBREV = re.compile(r"\b(?:Mr|Mrs|Ms|Dr|St|vs|etc|e\.g|i\.e|No)\.$", re.I)


def split_sentences(text):
    """中英双语分句：中文按全角标点，英文按句号/问号/叹号（跳过常见缩写）。"""
    text = normalize_space(text)
    if not text:
        return []
    out = []
    for part in _SENT_CJK.split(text):
        part = part.strip()
        if not part:
            continue
        if re.search(r"[A-Za-z]", part) and re.search(r"[.!?]\s", part + " "):
            for sub in _SENT_EN.split(part):
                sub = sub.strip()
                if sub and not _ABBREV.search(sub):
                    out.append(sub)
        else:
            out.append(part)
    return out


def audio_fit_score(ratio):
    """旁白时长 / 原片段时长 的适配分（第一、二闸门共用同一模型）。"""
    if 0.72 <= ratio <= 1.08:
        return 95.0
    if 0.6 <= ratio <= 1.25:
        return 80.0
    if 0.45 <= ratio <= 1.5:
        return 60.0
    return 40.0


def ensure_utf8_console():
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


def ensure_dir(path):
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def read_json(path, default=None):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return default
    except json.JSONDecodeError:
        return default


def write_json(path, data):
    p = Path(path)
    ensure_dir(p.parent)
    tmp = p.with_suffix(p.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, p)
    return p


def run_cmd(args, timeout=None, check=True, encoding="utf-8", errors="replace"):
    if isinstance(args, str):
        args = [args]
    creationflags = 0
    if sys.platform == "win32":
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    from .process import register, unregister

    proc = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, creationflags=creationflags)
    register(proc)
    try:
        try:
            stdout, stderr = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            stdout, stderr = proc.communicate()
            raise RuntimeError("command timeout (%ss): %s" % (timeout, " ".join(map(str, args))[:400]))
    finally:
        unregister(proc)
    out = (stdout or b"").decode(encoding, errors)
    err = (stderr or b"").decode(encoding, errors)
    if check and proc.returncode != 0:
        raise RuntimeError(
            "command failed (%s): %s\n%s" % (proc.returncode, " ".join(map(str, args))[:400], err[-2000:])
        )
    return out, err, proc.returncode


def which(name):
    return shutil.which(name)


def ffmpeg_bin():
    return which(FFMPEG) or FFMPEG


def ffprobe_bin():
    return which(FFPROBE) or FFPROBE


def ffplay_bin():
    return which(FFPLAY) or None


def parse_timecode(value):
    if value is None:
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text:
        return 0.0
    if re.fullmatch(r"\d+(\.\d+)?", text):
        return float(text)
    parts = text.split(":")
    seconds = 0.0
    for part in parts:
        seconds = seconds * 60 + float(part)
    return seconds


def fmt_timecode(seconds, millis=True):
    seconds = max(0.0, float(seconds))
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    if millis:
        return "%02d:%02d:%06.3f" % (h, m, s)
    return "%02d:%02d:%02d" % (h, m, int(round(s)))


def fmt_clock(seconds):
    seconds = max(0.0, int(round(float(seconds))))
    return "%02d:%02d:%02d" % (seconds // 3600, (seconds % 3600) // 60, seconds % 60)


def count_cjk(text):
    return len(CJK_RE.findall(text))


def split_text_units(text):
    cjk = CJK_RE.findall(text)
    latin = LATIN_WORD_RE.findall(text)
    return cjk, latin


def estimate_speech_seconds(text, cjk_rate=4.4, word_rate=2.9, pause_break=0.22, pause_soft=0.1):
    if not text:
        return 0.0
    cjk, latin = split_text_units(text)
    base = len(cjk) / cjk_rate + len(latin) / word_rate
    pauses = 0.0
    for ch in text:
        if ch in PUNCT_BREAK:
            pauses += pause_break
        elif ch in PUNCT_PAUSE:
            pauses += pause_soft
    return max(0.35, base + pauses)


def strip_control(text):
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = "".join(ch for ch in text if ch == "\n" or ch == "\t" or unicodedata.category(ch)[0] != "C")
    return text.strip()


def normalize_space(text):
    return re.sub(r"\s+", " ", strip_control(text or "")).strip()


def tokenize(text):
    text = (text or "").lower()
    tokens = []
    for word in LATIN_WORD_RE.findall(text):
        tokens.append(word)
    cjk_chars = CJK_RE.findall(text)
    for i in range(len(cjk_chars)):
        tokens.append(cjk_chars[i])
        if i + 1 < len(cjk_chars):
            tokens.append(cjk_chars[i] + cjk_chars[i + 1])
    return tokens


def tfidf_vectors(docs):
    vectors = []
    df = {}
    for doc in docs:
        tf = {}
        for tok in tokenize(doc):
            tf[tok] = tf.get(tok, 0) + 1
        vectors.append(tf)
        for tok in tf:
            df[tok] = df.get(tok, 0) + 1
    total = len(docs) or 1
    out = []
    for tf in vectors:
        vec = {}
        for tok, count in tf.items():
            idf = math.log((1 + total) / (1 + df.get(tok, 0))) + 1.0
            vec[tok] = (1 + math.log(count)) * idf
        norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
        out.append({k: v / norm for k, v in vec.items()})
    return out


def cosine_sim(vec_a, vec_b):
    if not vec_a or not vec_b:
        return 0.0
    if len(vec_b) < len(vec_a):
        vec_a, vec_b = vec_b, vec_a
    score = sum(v * vec_b.get(k, 0.0) for k, v in vec_a.items())
    return max(0.0, min(1.0, score))


def jaccard(a, b):
    a, b = set(a or []), set(b or [])
    if not a and not b:
        return 0.0
    return len(a & b) / max(1, len(a | b))


def clamp(value, low, high):
    return max(low, min(high, value))


def overlap_ratio(a_start, a_end, b_start, b_end):
    inter = max(0.0, min(a_end, b_end) - max(a_start, b_start))
    union = max(a_end, b_end) - min(a_start, b_start)
    if union <= 0:
        return 0.0
    return inter / union


def slugify(text, max_len=48):
    text = normalize_space(text)
    text = re.sub(r"[^\w\u3400-\u9fff]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return (text or "clip")[:max_len]


def human_size(num):
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(num) < 1024:
            return "%.1f%s" % (num, unit)
        num /= 1024
    return "%.1fPB" % num


def open_path(path):
    path = str(path)
    if sys.platform == "win32":
        os.startfile(path)
    elif sys.platform == "darwin":
        subprocess.Popen(["open", path])
    else:
        subprocess.Popen(["xdg-open", path])


def grab_frame(video_path, time_sec, out_path, width=960):
    """抽取指定时间点的一帧图片，用于区域框选预览。"""
    out_path = Path(out_path)
    ensure_dir(out_path.parent)
    args = [ffmpeg_bin(), "-hide_banner", "-loglevel", "error", "-y",
            "-ss", "%.3f" % max(0.0, float(time_sec)), "-i", str(video_path),
            "-frames:v", "1", "-vf", "scale=%d:-2" % int(width), str(out_path)]
    _, err, code = run_cmd(args, check=False, timeout=120)
    return out_path if code == 0 and out_path.exists() else None


def media_duration(path):
    out, _, _ = run_cmd([ffprobe_bin(), "-v", "error", "-show_entries", "format=duration",
                         "-of", "default=noprint_wrappers=1:nokey=1", str(path)], check=True, timeout=60)
    try:
        return float(out.strip())
    except ValueError:
        return 0.0


def media_size(path):
    out, _, _ = run_cmd([ffprobe_bin(), "-v", "error", "-select_streams", "v:0",
                         "-show_entries", "stream=width,height",
                         "-of", "default=noprint_wrappers=1", str(path)], check=False, timeout=60)
    width = height = 0
    for line in (out or "").splitlines():
        key, _, value = line.partition("=")
        if key == "width":
            width = int(value or 0)
        elif key == "height":
            height = int(value or 0)
    return width, height
