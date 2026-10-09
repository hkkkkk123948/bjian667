"""可选的画面文字识别（OCR，文档 §43）。

用于提取屏幕上的文字/数字（监控时间、门牌、聊天记录等），
作为事实校验与写稿的补充证据。未安装 pytesseract/tesseract 时静默跳过。
"""

from pathlib import Path

from ..log import get_logger
from ..utils import normalize_space


def ocr_available():
    try:
        import pytesseract  # noqa: F401
    except Exception:
        return False
    try:
        import pytesseract

        pytesseract.get_tesseract_version()
        return True
    except Exception:
        return False


def ocr_frames(frames, language="zh", max_chars=400):
    try:
        import pytesseract
        from PIL import Image
    except Exception:
        return ""
    lang = "chi_sim+eng" if language == "zh" else "eng"
    texts = []
    for frame in frames:
        path = Path(frame)
        if not path.exists():
            continue
        try:
            with Image.open(path) as img:
                text = normalize_space(pytesseract.image_to_string(img, lang=lang))
        except Exception:
            continue
        if text and len(text) >= 2:
            texts.append(text)
    return normalize_space(" ".join(texts))[:max_chars]


def analyze_scene_ocr(index, language="zh", workers=2, progress_cb=None):
    """对每个场景的关键帧做 OCR，回填 scene.ocr。返回成功数。"""
    logger = get_logger()
    if not index.scenes:
        return 0
    todo = [s for s in index.scenes if not s.ocr and s.keyframes]
    if not todo:
        return 0
    from concurrent.futures import ThreadPoolExecutor

    def work(scene):
        return scene, ocr_frames(scene.keyframes, language=language)

    done = 0
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        for scene, text in pool.map(work, todo):
            if text:
                scene.ocr = text
                done += 1
    logger.info("场景 OCR 完成: %d/%d", done, len(todo))
    if progress_cb:
        progress_cb(100.0, "画面文字识别 %d/%d" % (done, len(todo)))
    return done
