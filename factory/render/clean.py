"""画面清理：去除原片硬字幕/水印/黑边（参考解说猫“遮盖原字幕”的思路）。

在成片渲染的滤镜链最前面处理源画面：
- 去黑边：用 cropdetect 采样估计并裁掉黑边；
- 去字幕：裁剪底部字幕带（crop）/ 模糊（blur）/ 抹除（delogo）；
- 手动框选区域：用户在成片界面画矩形，选择 模糊/马赛克/遮挡/抹除。

两种去字幕方式可并存；手动区域优先按用户框选处理。
"""

import re

from ..utils import clamp, ffmpeg_bin, media_duration, run_cmd

_CROPDETECT_RE = re.compile(r"crop=(\d+):(\d+):(\d+):(\d+)")
SUBTITLE_MODES = ("off", "crop", "blur", "delogo")
REGION_METHODS = ("blur", "mosaic", "cover", "delogo")


def detect_black_bars(video_path, params=None, sample_sec=3.0):
    """采样若干时间段用 cropdetect 估计黑边裁剪框，返回 (w, h, x, y) 或 None。"""
    params = params or {}
    if str(params.get("auto_crop_black", "on")).lower() in ("off", "0", "false"):
        return None
    try:
        duration = media_duration(video_path)
    except Exception:
        duration = 0.0
    points = [0.1]
    if duration > 20:
        points = [duration * 0.25, duration * 0.5, duration * 0.75]
    boxes = {}
    for t in points:
        cmd = [ffmpeg_bin(), "-hide_banner", "-loglevel", "info",
               "-ss", "%.2f" % t, "-t", "%.2f" % sample_sec, "-i", str(video_path),
               "-vf", "cropdetect=limit=24:round=2:reset=0", "-an", "-f", "null", "-"]
        _, err, _ = run_cmd(cmd, check=False, timeout=180)
        for m in _CROPDETECT_RE.findall(err or ""):
            key = tuple(int(x) for x in m)
            boxes[key] = boxes.get(key, 0) + 1
    if not boxes:
        return None
    best = max(boxes.items(), key=lambda kv: kv[1])[0]
    w, h, x, y = best
    if w <= 16 or h <= 16:
        return None
    return best


def _band(params):
    return clamp(float(params.get("subtitle_band", 0.16)), 0.0, 0.5)


def _region_px(region, src_size):
    if not src_size or not src_size[0] or not src_size[1]:
        return None
    sw, sh = int(src_size[0]), int(src_size[1])
    x = int(round(clamp(float(region.get("x", 0.0)), 0.0, 1.0) * sw))
    y = int(round(clamp(float(region.get("y", 0.0)), 0.0, 1.0) * sh))
    w = int(round(clamp(float(region.get("w", 0.0)), 0.0, 1.0) * sw))
    h = int(round(clamp(float(region.get("h", 0.0)), 0.0, 1.0) * sh))
    w = max(2, min(w, sw - x - 1))
    h = max(2, min(h, sh - y - 1))
    if w < 2 or h < 2:
        return None
    return x, y, w, h


def build_video_graph(params, base_filter, black_box=None, tail="", src_size=None,
                      regions=None, region_method=None):
    """构造 [0:v]...[vout] 滤镜图：手动区域 + 去黑边 + 去字幕 + 裁幅重构。"""
    regions = regions or []
    method = str(region_method or params.get("region_method", "blur")).lower()
    lines = []
    label = "0:v"
    pending = []
    idx = [0]

    def flush(final=False):
        if pending:
            nxt = "vout" if final else "v%d" % idx[0]
            idx[0] += 1
            lines.append("[%s]%s[%s]" % (label, ",".join(pending), nxt))
            return nxt
        if final:
            lines.append("[%s]null[vout]" % label)
            return "vout"
        return label

    def subgraph(build):
        """build(a, b, bb, out) 生成 split/crop/.../overlay 的 label 链。"""
        nonlocal label
        pending_flushed = flush()
        label = pending_flushed
        a, b, bb, out = "a%d" % idx[0], "b%d" % idx[0], "bb%d" % idx[0], "v%d" % idx[0]
        idx[0] += 1
        lines.append(build(label, a, b, bb, out))
        label = out

    # 1) 手动框选区域（作用在原始帧，坐标与用户所见一致）
    for region in regions:
        box = _region_px(region, src_size)
        if not box:
            continue
        x, y, w, h = box
        if method in ("cover", "遮挡", "black"):
            pending.append("drawbox=x=%d:y=%d:w=%d:h=%d:color=black@1:t=fill" % (x, y, w, h))
        elif method in ("delogo", "抹除"):
            pending.append("delogo=x=%d:y=%d:w=%d:h=%d" % (x, y, w, h))
        elif method in ("mosaic", "马赛克"):
            subgraph(lambda cur, a, b, bb, out, x=x, y=y, w=w, h=h: (
                "[%s]split=2[%s][%s];[%s]crop=%d:%d:%d:%d,scale=iw/16:ih/16,scale=%d:%d:flags=neighbor[%s];"
                "[%s][%s]overlay=%d:%d[%s]" % (cur, a, b, b, w, h, x, y, w, h, bb, a, bb, x, y, out)))
        else:  # blur
            subgraph(lambda cur, a, b, bb, out, x=x, y=y, w=w, h=h: (
                "[%s]split=2[%s][%s];[%s]crop=%d:%d:%d:%d,boxblur=luma_radius=16:luma_power=1[%s];"
                "[%s][%s]overlay=%d:%d[%s]" % (cur, a, b, b, w, h, x, y, bb, a, bb, x, y, out)))

    # 2) 去黑边
    if black_box:
        pending.append("crop=%d:%d:%d:%d" % tuple(black_box))

    # 3) 去字幕带
    mode = str(params.get("remove_subtitles", "off")).lower()
    band = _band(params)
    if mode in ("crop", "裁剪") and band > 0:
        pending.append("crop=iw:ih*%.4f:0:0" % (1.0 - band))
    elif mode in ("blur", "模糊") and band > 0:
        subgraph(lambda cur, a, b, bb, out, band=band: (
            "[%s]split=2[%s][%s];[%s]crop=iw:ih*%.4f:0:ih*(1-%.4f),boxblur=luma_radius=16:luma_power=1[%s];"
            "[%s][%s]overlay=0:H-h[%s]" % (cur, a, b, b, band, band, bb, a, bb, out)))
    elif mode in ("delogo", "抹除") and band > 0 and src_size and src_size[1]:
        sh = int(src_size[1])
        y = max(1, int(sh * (1.0 - band)))
        pending.append("delogo=x=1:y=%d:w=iw-2:h=%d" % (y, max(2, int(sh * band) - 2)))

    # 4) 裁幅/缩放 + 末尾
    if base_filter:
        pending.append(base_filter)
    if tail:
        pending.append(tail)
    flush(final=True)
    return ";".join(lines)


def clean_preview_filter(params, black_box=None):
    mode = str(params.get("remove_subtitles", "off")).lower()
    if mode in ("off", "", "0", "false"):
        return "去字幕: 关闭"
    return "去字幕: %s (底部 %.0f%%)%s" % (
        mode, _band(params) * 100, "，去黑边" if black_box else "")
