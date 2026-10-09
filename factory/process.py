"""全局活动子进程登记表：支持任务取消时中断正在运行的 ffmpeg/whisper。"""

import threading

_ACTIVE = set()
_LOCK = threading.Lock()


def register(proc):
    with _LOCK:
        _ACTIVE.add(proc)


def unregister(proc):
    with _LOCK:
        _ACTIVE.discard(proc)


def kill_active():
    with _LOCK:
        procs = list(_ACTIVE)
    killed = 0
    for proc in procs:
        try:
            if proc.poll() is None:
                proc.kill()
                killed += 1
        except Exception:
            pass
    return killed
