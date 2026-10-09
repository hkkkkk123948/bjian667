from pathlib import Path

from ..utils import ensure_dir


def split_display_lines(text, max_len=16):
    text = (text or "").strip()
    if len(text) <= max_len:
        return [text]
    best = None
    for i, ch in enumerate(text):
        if ch in "，。！？、；：,!?;:" and 4 <= i <= len(text) - 4:
            best = i + 1
            break
    if best is None:
        best = max(1, len(text) // 2)
    a, b = text[:best].rstrip("，, "), text[best:].lstrip()
    if len(b) > max_len:
        mid = len(b) // 2
        return [a, b[:mid], b[mid:]]
    if len(a) > max_len:
        mid = len(a) // 2
        return [a[:mid], a[mid:], b] if b else [a[:mid], a[mid:]]
    return [x for x in (a, b) if x]


def build_ass(sentences, params, path):
    width = int(params.get("target_width", 1080))
    height = int(params.get("target_height", 1920))
    font = params.get("subtitle_font", "Microsoft YaHei")
    fontsize = int(params.get("subtitle_fontsize", 64))
    header = """[Script Info]
ScriptType: v4.00+
PlayResX: %d
PlayResY: %d
WrapStyle: 0
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,%s,%d,&H00FFFFFF,&H000000FF,&H00101010,&H96000000,-1,0,0,0,100,100,1.5,0,1,5,2,2,70,70,400,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
""" % (width, height, font, fontsize)
    lines = [header]
    for item in sentences:
        start = item["start"]
        end = item["end"]
        if end <= start:
            end = start + 0.5
        pieces = split_display_lines(item["text"])
        if len(pieces) <= 1:
            lines.append(_dialogue(start, end, pieces[0] if pieces else item["text"]))
        else:
            total_chars = sum(len(p) for p in pieces) or 1
            cursor = start
            span = end - start
            for piece in pieces:
                share = len(piece) / total_chars
                dur = span * share
                piece_end = cursor + dur if cursor + dur < end else end
                lines.append(_dialogue(cursor, piece_end, piece))
                cursor = piece_end
    path = Path(path)
    ensure_dir(path.parent)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _dialogue(start, end, text):
    text = (text or "").replace("\n", " ").replace("{", "(").replace("}", ")")
    return "Dialogue: 0,%s,%s,Default,,0,0,0,,%s" % (
        _ass_time(start), _ass_time(end), text)


def _ass_time(seconds):
    seconds = max(0.0, float(seconds))
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return "%d:%02d:%05.2f" % (h, m, s)


def build_srt(sentences, path):
    lines = []
    for i, item in enumerate(sentences, 1):
        lines.append(str(i))
        lines.append("%s --> %s" % (_srt_time(item["start"]), _srt_time(item["end"])))
        lines.append("\n".join(split_display_lines(item["text"])))
        lines.append("")
    path = Path(path)
    ensure_dir(path.parent)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _srt_time(seconds):
    seconds = max(0.0, float(seconds))
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return ("%02d:%02d:%06.3f" % (h, m, s)).replace(".", ",")
