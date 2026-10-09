from ..utils import clamp


def resolve_output_size(src_w, src_h, params, max_edge=1920):
    """解析输出尺寸：vertical→1080x1920；auto→严格跟随源片比例（最大边缩到 max_edge）。"""
    aspect = str(params.get("aspect_ratio", "auto")).lower()
    if aspect in ("vertical", "9:16", "portrait"):
        return 1080, 1920
    if src_w and src_h:
        src_w, src_h = int(src_w), int(src_h)
        scale = min(1.0, max_edge / max(src_w, src_h))
        return max(2, int(round(src_w * scale / 2.0)) * 2), max(2, int(round(src_h * scale / 2.0)) * 2)
    return int(params.get("target_width", 1920)), int(params.get("target_height", 1080))


def suggest_output_size(src_w, src_h, params, max_edge=1920):
    width = int(params.get("target_width", 1080))
    height = int(params.get("target_height", 1920))
    if not src_w or not src_h:
        return width, height
    src_w = int(src_w)
    src_h = int(src_h)
    if src_w <= 0 or src_h <= 0:
        return width, height
    scale = min(1.0, max_edge / max(src_w, src_h))
    w = int(round(src_w * scale / 2)) * 2
    h = int(round(src_h * scale / 2)) * 2
    for common_w, common_h in ((1920, 1080), (1080, 1920), (1920, 1080)):
        if abs(w - common_w) < 8:
            w = common_w
        if abs(h - common_h) < 8:
            h = common_h
    return w, h


def video_filter(params, focus=None, focus_expr=None):
    width = int(params.get("target_width", 1080))
    height = int(params.get("target_height", 1920))
    if focus_expr:
        x_expr = "(%s)*(iw-%d)" % (focus_expr, width)
    else:
        if focus is None:
            focus = clamp(float(params.get("crop_focus", 0.5)), 0.0, 1.0)
        focus = clamp(float(focus), 0.0, 1.0)
        x_expr = "(iw-%d)*%.4f" % (width, focus)
    return (
        "scale=%d:%d:force_original_aspect_ratio=increase,"
        "crop=%d:%d:%s:0,"
        "setsar=1"
    ) % (width, height, width, height, x_expr)


def resolution_label(params):
    return "%dx%d" % (int(params.get("target_width", 1080)), int(params.get("target_height", 1920)))


def _focus_samples(video_path, start, end, params, samples=5):
    try:
        import cv2
    except Exception:
        return []
    cap = None
    centers = []
    try:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return []
        cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        if cascade.empty():
            return []
        span = max(0.1, float(end) - float(start))
        for i in range(max(1, samples)):
            t = float(start) + span * (i + 1) / (samples + 1)
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            width = frame.shape[1] or 1
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = cascade.detectMultiScale(gray, 1.1, 5, minSize=(40, 40))
            if len(faces):
                fx, _fy, fw, _fh = max(faces, key=lambda r: r[2] * r[3])
                centers.append((fx + fw / 2.0) / width)
    except Exception:
        return []
    finally:
        try:
            if cap is not None:
                cap.release()
        except Exception:
            pass
    return centers


def estimate_focus(video_path, start, end, params, samples=5):
    """用 OpenCV 人脸检测估计裁剪焦点 x 比例，避免主体/人脸被裁掉。"""
    default = clamp(float(params.get("crop_focus", 0.5)), 0.0, 1.0)
    if str(params.get("reframe_faces", "on")).lower() in ("off", "0", "false"):
        return default
    samples_found = _focus_samples(video_path, start, end, params, samples)
    if samples_found:
        return clamp(sum(samples_found) / len(samples_found), 0.12, 0.88)
    return default


def focus_expression(video_path, start, end, params, samples=6):
    """按时间平滑跟随人脸的水平平移表达式（reframe_faces=track 时用）。"""
    if str(params.get("reframe_faces", "on")).lower() != "track":
        return None
    pts = _focus_samples(video_path, start, end, params, samples)
    if len(pts) < 2:
        return None
    f0 = clamp(pts[0], 0.12, 0.88)
    f1 = clamp(pts[-1], 0.12, 0.88)
    span = max(0.5, float(end) - float(start))
    return "max(0.10,min(0.90,%.4f+(%.4f-%.4f)*t/%.3f))" % (f0, f1, f0, span)


def focus_hint(video_path, start, end, params, index=None):
    if not index:
        return 0.5
    focus = float(params.get("crop_focus", 0.5))
    return clamp(focus, 0.0, 1.0)
