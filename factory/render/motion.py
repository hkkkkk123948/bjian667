"""Subpixel push and pan.

The picture path samples with a float cubic warp. A crop origin rounded to
whole pixels alternates during a slow push, so that filter is not used. The
scale/crop graph below is only the fallback when OpenCV is not installed.
Crop width cannot change per frame, and crop's in_w stays on the first frame
when scale size changes, so the origin is written from the frame index.
"""
import math

import numpy as np


def choose_motion(role, index, duration=4):
    if duration < 1.5:
        return "none"
    role = str(role or "")
    if role in ("twist", "payoff", "爆点", "转折"):
        return "push_strong"
    if role in ("hook", "escalation"):
        return "push_in"
    # A lateral move on a short shot feels like a slide. Keep it rare.
    if duration >= 3.6 and index % 8 == 0:
        return "pan_right" if index % 16 == 0 else "pan_left"
    return "push_in"


def motion_amount(duration, returning=False, kind="push_in"):
    """Peak zoom above 1. Small on purpose: a push enlarges source shake."""
    if returning:
        return min(.018, max(0, float(duration)) * .003)
    strong = kind == "push_strong"
    cap = .026 if strong else .016
    rate = .0055 if strong else .0042
    return min(cap, max(0, float(duration)) * rate)


def _smoothstep(u):
    return u * u * (3 - 2 * u)


def motion_sample(kind, time, duration, returning=False):
    """Numerical counterpart used to check smoothness independently of ffmpeg."""
    u = max(0., min(1., float(time) / max(.001, duration)))
    ease = math.sin(math.pi * u) ** 2 if returning else _smoothstep(u)
    zoom = 1 + motion_amount(duration, returning, kind) * ease
    if kind == "pan_right":
        fraction = .15 + .7 * ease
    elif kind == "pan_left":
        fraction = .85 - .7 * ease
    else:
        fraction = .5
    return zoom, fraction if str(kind).startswith("pan_") else .5


def supersample_factor(oversample):
    """Fallback filter only. The render path samples with a float warp."""
    try:
        factor = int(oversample)
    except (TypeError, ValueError):
        factor = 8
    return 8 if factor < 8 else min(12, factor)


def warp_available():
    try:
        import cv2  # noqa: F401
        return True
    except Exception:
        return False


def frame_motion(kind, index, frames, duration, local_spans=None):
    """Zoom and horizontal window fraction for one output frame."""
    frames = int(frames)
    index = int(index)
    if local_spans:
        zoom, pan = 1.0, 0.0
        for span in local_spans:
            start, end = int(span["start_frame"]), int(span["end_frame"]) - 1
            mode = span.get("motion", "push_in")
            if mode == "none" or end <= start or not start <= index <= end:
                continue
            pulse = math.sin(math.pi * (index - start) / (end - start)) ** 2
            zoom += motion_amount((end - start) / 24.0, returning=True, kind=mode) * pulse
            if str(mode).startswith("pan_"):
                pan += (1 if mode == "pan_right" else -1) * 0.22 * pulse
        return zoom, 0.5 + pan
    if kind not in ("push_in", "push_strong", "pan_left", "pan_right") or frames <= 1:
        return 1.0, 0.5
    u = index / max(1, frames - 1)
    return motion_sample(kind, u * float(duration), duration, returning=False)


def warp_frame(frame, zoom, fraction):
    """Sample `frame` with a float crop. Cubic, so the edge does not stair-step."""
    import cv2
    height, width = frame.shape[:2]
    zoom = max(1.0, float(zoom))
    view_w, view_h = width / zoom, height / zoom
    left = min(max(0.0, (width - view_w) * float(fraction)), max(0.0, width - view_w))
    top = max(0.0, (height - view_h) * 0.5)
    matrix = np.array([[view_w / width, 0.0, left], [0.0, view_h / height, top]], dtype=np.float32)
    return cv2.warpAffine(frame, matrix, (width, height),
                          flags=cv2.INTER_CUBIC | cv2.WARP_INVERSE_MAP,
                          borderMode=cv2.BORDER_REPLICATE)


def hold_spans(frames, kind, freeze_frames=0):
    """A spoken aside eases out and back to the same frame."""
    frames = int(frames)
    if frames < 2 or str(kind or "none") == "none":
        return []
    start = min(frames - 1, max(0, int(freeze_frames)))
    if frames - start < 2:
        return []
    return [{"start_frame": start, "end_frame": frames, "motion": kind}]


def _ease_expr(index_expr, steps):
    u = "min(%s/%d,1)" % (index_expr, max(1, int(steps)))
    return "(3*pow(%s,2)-2*pow(%s,3))" % (u, u)


def _px(dim, factor, zoom):
    return "trunc(%d*%d*(%s)+0.5)" % (int(dim), int(factor), zoom)


def motion_filter(kind, width, height, duration, frames, oversample=4, local_spans=None):
    if not local_spans and kind not in ("push_in", "push_strong", "pan_left", "pan_right"):
        return "null"
    frames = int(frames)
    if frames <= 1:
        return "null"
    width = max(2, int(width) // 2 * 2)
    height = max(2, int(height) // 2 * 2)
    factor = supersample_factor(oversample)
    steps = max(1, frames - 1)
    if local_spans:
        pulses, pans = [], []
        for span in local_spans:
            start, end = int(span["start_frame"]), int(span["end_frame"]) - 1
            mode = span.get("motion", "push_in")
            if mode == "none" or end <= start:
                continue
            span_steps = max(1, end - start)
            progress = "(n-%d)/%d" % (start, span_steps)
            pulse = "if(between(n,%d,%d),pow(sin(PI*(%s)),2),0)" % (start, end, progress)
            amount = motion_amount(span_steps / 24.0, returning=True, kind=mode)
            pulses.append("%.8f*(%s)" % (amount, pulse))
            if str(mode).startswith("pan_"):
                sign = 1 if mode == "pan_right" else -1
                pans.append("%d*0.22*(%s)" % (sign, pulse))
        if not pulses:
            return "null"
        zoom = "(1+" + "+".join(pulses) + ")"
        fraction = "(0.5+" + "+".join(pans) + ")" if pans else "0.5"
    else:
        ease = _ease_expr("n", steps)
        zoom = "(1+%.8f*%s)" % (motion_amount(duration, kind=kind), ease)
        if kind == "pan_right":
            fraction = "(.15+.7*%s)" % ease
        elif kind == "pan_left":
            fraction = "(.85-.7*%s)" % ease
        else:
            fraction = "0.5"
    scaled_w = _px(width, factor, zoom)
    scaled_h = _px(height, factor, zoom)
    # Origin from n, not in_w. in_w stays at the first frame when scale's
    # output size changes every frame. max() keeps a 1.0 zoom inside the frame.
    x_expr = "max(0,((%s)-%d)*(%s))" % (scaled_w, width * factor, fraction)
    y_expr = "max(0,((%s)-%d)/2)" % (scaled_h, height * factor)
    flags = "lanczos+accurate_rnd+full_chroma_int"
    return (
        "format=yuv444p,setsar=1,"
        "scale=w='%s':h='%s':eval=frame:flags=%s,"
        "crop=%d:%d:x='%s':y='%s',"
        "scale=%d:%d:flags=%s,setsar=1,format=yuv420p"
    ) % (scaled_w, scaled_h, flags, width * factor, height * factor, x_expr, y_expr,
         width, height, flags)
