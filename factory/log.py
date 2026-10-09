import logging
import time
from pathlib import Path

from .utils import ensure_dir

LOGGER_NAME = "ytsf"
_configured = False


def setup_logging(log_file=None, level=logging.INFO):
    global _configured
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(level)
    if _configured:
        return logger
    fmt = logging.Formatter("[%(asctime)s] %(levelname)s %(message)s", "%H:%M:%S")
    handler = logging.StreamHandler()
    handler.setFormatter(fmt)
    logger.addHandler(handler)
    if log_file:
        ensure_dir(Path(log_file).parent)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s %(message)s"))
        logger.addHandler(fh)
    _configured = True
    return logger


def get_logger():
    logger = logging.getLogger(LOGGER_NAME)
    if not logger.handlers:
        setup_logging()
    return logger


class EventEmitter:
    def __init__(self, sink=None):
        self.sink = sink
        self.history = []

    def emit(self, kind, **payload):
        event = {"kind": kind, "ts": time.time(), **payload}
        self.history.append(event)
        if len(self.history) > 5000:
            del self.history[:2000]
        if self.sink:
            try:
                self.sink(event)
            except Exception:
                pass
        return event

    def log(self, message, level="info"):
        get_logger().log(getattr(logging, level.upper(), logging.INFO), message)
        return self.emit("log", message=message, level=level)

    def stage(self, status, stage="", detail=""):
        return self.emit("stage", status=status, stage=stage, detail=detail)

    def progress(self, value, detail=""):
        return self.emit("progress", value=float(value), detail=detail)

    def update(self, **payload):
        return self.emit("update", **payload)
