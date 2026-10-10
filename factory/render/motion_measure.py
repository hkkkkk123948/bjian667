"""Before/after measurement for a slow push.

Run from the repo root:

    python -m factory.render.motion_measure

The legacy sampler rounds the crop origin to whole pixels. That string lives
only here, so the picture path can be checked without putting it back into
a render. The current sampler is the float cubic warp used by export.
"""
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from ..utils import ffmpeg_bin
from .motion import frame_motion, warp_frame

WIDTH, HEIGHT, FRAMES, DURATION = 320, 180, 48, 2.0
EDGE, LEVEL = 100, 128
# Historical two-second push: min(0.02, 2 * 0.005).
LEGACY_AMOUNT = 0.01
LEGACY_FILTER = (
    "zoompan=z='1+%.6f*on/%d':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)'"
    ":d=1:s=%dx%d:fps=24"
) % (LEGACY_AMOUNT, FRAMES - 1, WIDTH, HEIGHT)


def synthetic_edge(width=WIDTH, height=HEIGHT, edge=EDGE):
    frame = np.zeros((height, width), np.uint8)
    frame[:, edge:] = 255
    return frame


def edge_position(gray, level=LEVEL):
    """Subpixel x where a mid-frame row crosses `level`."""
    row = gray[gray.shape[0] // 2]
    above = np.flatnonzero(row >= level)
    if above.size == 0:
        return float(gray.shape[1])
    x = int(above[0])
    if x == 0:
        return 0.0
    left, right = float(row[x - 1]), float(row[x])
    if right == left:
        return float(x)
    return (x - 1) + (level - left) / (right - left)


def track_stats(positions):
    positions = np.asarray(positions, dtype=float)
    steps = np.diff(positions)
    second = np.diff(steps) if steps.size else np.array([])
    signed = [float(step) for step in steps if abs(step) > 1e-4]
    reversals = sum(1 for left, right in zip(signed, signed[1:]) if left * right < 0)
    return {
        "start": float(positions[0]),
        "end": float(positions[-1]),
        "max_step": float(np.max(np.abs(steps))) if steps.size else 0.0,
        "second_rms": float(np.sqrt(np.mean(second ** 2))) if second.size else 0.0,
        "reversals": reversals,
    }


def _legacy_zoom(index):
    u = index / max(1, FRAMES - 1)
    ease = u * u * (3 - 2 * u)
    return 1 + LEGACY_AMOUNT * ease, 0.5


def measure_warp(kind="push_in", legacy_amount=False):
    import cv2
    source = cv2.cvtColor(synthetic_edge(), cv2.COLOR_GRAY2BGR)
    positions = []
    for index in range(FRAMES):
        if legacy_amount:
            zoom, fraction = _legacy_zoom(index)
        else:
            zoom, fraction = frame_motion(kind, index, FRAMES, DURATION)
        sampled = warp_frame(source, zoom, fraction)
        positions.append(edge_position(sampled[:, :, 0]))
    return track_stats(positions)


def measure_legacy_zoompan():
    """Integer crop of the same edge. Requires ffmpeg on PATH."""
    import cv2
    with tempfile.TemporaryDirectory(prefix="motion_measure_") as folder:
        image = Path(folder) / "edge.png"
        cv2.imwrite(str(image), synthetic_edge())
        raw = subprocess.check_output([
            ffmpeg_bin(), "-v", "error", "-filter_threads", "1", "-loop", "1", "-framerate", "24",
            "-i", str(image), "-vf", LEGACY_FILTER, "-frames:v", str(FRAMES),
            "-f", "rawvideo", "-pix_fmt", "gray", "-",
        ])
    frames = np.frombuffer(raw, dtype=np.uint8).reshape(-1, HEIGHT, WIDTH)
    if len(frames) != FRAMES:
        raise RuntimeError("legacy sampler returned %d frames" % len(frames))
    return track_stats([edge_position(frame) for frame in frames])


def _line(name, stats):
    return ("%-22s  edge %7.3f -> %7.3f   max|step| %.4f   second-rms %.4f   reversals %d" % (
        name, stats["start"], stats["end"], stats["max_step"], stats["second_rms"], stats["reversals"]))


def report():
    legacy = measure_legacy_zoompan()
    current = measure_warp(legacy_amount=True)
    shipped = measure_warp(legacy_amount=False)
    lines = [
        "Slow push, %dx%d, %d frames, vertical edge at x=%d, level %d." % (
            WIDTH, HEIGHT, FRAMES, EDGE, LEVEL),
        "Legacy filter: %s" % LEGACY_FILTER,
        _line("legacy integer crop", legacy),
        _line("cubic warp, same zoom", current),
        _line("cubic warp, shipped", shipped),
    ]
    return "\n".join(lines)


def main():
    print(report())


if __name__ == "__main__":
    main()
