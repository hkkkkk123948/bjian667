import math

import numpy as np

from ..log import get_logger
from ..models import AudioEvent


def load_pcm(video_path, sample_rate=8000, max_seconds=0):
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(video_path), "-vn", "-ac", "1",
           "-ar", str(sample_rate), "-f", "s16le", "-"]
    if max_seconds:
        cmd = cmd[:1] + ["-t", str(int(max_seconds))] + cmd[1:]
    import subprocess
    import sys

    creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if sys.platform == "win32" else 0
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, creationflags=creationflags)
    if proc.returncode != 0:
        raise RuntimeError("音频解码失败: %s" % proc.stderr.decode("utf-8", "replace")[-400:])
    data = np.frombuffer(proc.stdout, dtype="<i2")
    if data.size == 0:
        return np.zeros(0, dtype=np.float64), sample_rate
    return data.astype(np.float64) / 32768.0, sample_rate


def rms_curve(pcm, sample_rate, window=0.05):
    if pcm.size == 0:
        return np.zeros(0), window
    step = max(1, int(sample_rate * window))
    n = pcm.size // step
    if n == 0:
        return np.array([float(np.sqrt(np.mean(pcm ** 2)))]), window
    trimmed = pcm[: n * step].reshape(n, step)
    return np.sqrt(np.mean(trimmed ** 2, axis=1)), window


def dbfs(values, floor=-90.0):
    out = []
    for v in np.atleast_1d(values):
        v = float(v)
        db = 20.0 * math.log10(v + 1e-9)
        out.append(max(floor, db))
    return np.array(out)


def stream_window_rms(video_path, sample_rate=8000, window=0.05, max_seconds=0, timeout=1800):
    """流式解码音频，逐窗返回 RMS，避免将整条音轨载入内存（长视频友好）。"""
    import subprocess
    import sys

    step = max(1, int(sample_rate * window))
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(video_path), "-vn",
           "-ac", "1", "-ar", str(sample_rate), "-f", "s16le", "-"]
    if max_seconds:
        cmd = cmd[:4] + ["-t", str(int(max_seconds))] + cmd[4:]
    creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if sys.platform == "win32" else 0
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                            creationflags=creationflags)
    from ..process import register, unregister

    register(proc)
    chunk_bytes = step * 2 * 256
    values = []
    buf = b""
    try:
        while True:
            chunk = proc.stdout.read(chunk_bytes)
            if not chunk:
                break
            buf += chunk
            n_windows = len(buf) // (step * 2)
            if n_windows:
                arr = np.frombuffer(buf[:n_windows * step * 2], dtype="<i2").astype(np.float64) / 32768.0
                arr = arr.reshape(n_windows, step)
                values.append(np.sqrt(np.mean(arr ** 2, axis=1)))
                buf = buf[n_windows * step * 2:]
        if len(buf) >= step * 2:
            arr = np.frombuffer(buf[: (len(buf) // (step * 2)) * step * 2], dtype="<i2").astype(np.float64) / 32768.0
            values.append(np.sqrt(np.mean(arr.reshape(-1, step) ** 2, axis=1)))
        proc.wait(timeout=max(1, timeout))
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass
        raise
    finally:
        try:
            proc.stdout.close()
        except Exception:
            pass
        unregister(proc)
    if not values:
        return np.zeros(0)
    return np.concatenate(values)


def analyze_audio(video_path, scenes=None, sample_rate=8000, window=0.05, silence_db=-45.0):
    logger = get_logger()
    try:
        curve = stream_window_rms(video_path, sample_rate=sample_rate, window=window)
    except Exception as exc:
        logger.warning("音频分析失败: %s", exc)
        return [], {}
    if curve.size == 0:
        return [], {}
    win = window
    dbs = dbfs(curve)
    events = []
    in_silence = False
    start_idx = 0
    for i, db in enumerate(dbs):
        if db < silence_db and not in_silence:
            in_silence = True
            start_idx = i
        elif db >= silence_db and in_silence:
            in_silence = False
            if (i - start_idx) * win >= 0.6:
                events.append(AudioEvent(kind="silence", start=round(start_idx * win, 3),
                                         end=round(i * win, 3), value=round(float(dbs[start_idx:i].min()), 2)))
    if in_silence and (len(dbs) - start_idx) * win >= 0.6:
        events.append(AudioEvent(kind="silence", start=round(start_idx * win, 3),
                                 end=round(len(dbs) * win, 3), value=round(float(dbs[start_idx:].min()), 2)))
    peak_db = float(np.percentile(dbs, 98))
    threshold = max(float(np.percentile(dbs, 25)) + 14.0, -40.0)
    for i, db in enumerate(dbs):
        if db >= threshold and db >= peak_db - 6.0:
            events.append(AudioEvent(kind="loud", start=round(i * win, 3), end=round((i + 1) * win, 3),
                                     value=round(float(db), 2)))
    stats = {
        "median_db": round(float(np.median(dbs)), 2),
        "p95_db": round(float(np.percentile(dbs, 95)), 2),
        "speech_threshold_db": round(threshold, 2),
        "duration": round(len(dbs) * win, 3),
        "speech_ratio": round(float(np.mean(dbs > threshold)), 3),
    }
    if scenes:
        for scene in scenes:
            a = int(scene.start / win)
            b = int(math.ceil(scene.end / win))
            seg = dbs[a:max(a + 1, b)]
            scene.rms = round(float(np.mean(seg)), 2)
            scene.speech_ratio = round(float(np.mean(seg > threshold)), 3)
    return events, stats
