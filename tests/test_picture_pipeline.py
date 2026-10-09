import hashlib
import json
import subprocess

import numpy as np
import pytest

from factory.render.motion import motion_filter
from factory.render.transitions import transition_plan
from factory.render.exporter import export_clip
from factory.providers.tts.base import write_wav
from factory.utils import ffmpeg_bin, media_duration


def test_transition_handles_do_not_shorten_narration():
    plan = transition_plan([2.0, 3.0, 2.5])
    assert plan["output_frames"] == 180
    assert all(t["duration_frames"] == 3 for t in plan["transitions"])
    rendered = sum(h["nominal_frames"] + h["before"] + h["after"] for h in plan["handles"])
    assert rendered - sum(t["duration_frames"] for t in plan["transitions"]) == 180
    assert max(t["duration"] for t in transition_plan([2, 2], seconds=1)["transitions"]) <= .18


def test_subpixel_pan_does_not_jump_by_whole_pixels(tmp_path):
    cv2 = pytest.importorskip("cv2")
    im = np.zeros((180, 320), dtype=np.uint8)
    cv2.circle(im, (175, 90), 25, 240, -1)
    cv2.rectangle(im, (75, 45), (100, 140), 120, -1)
    source = tmp_path / "grid.png"
    cv2.imwrite(str(source), im)
    data = subprocess.check_output([ffmpeg_bin(), "-v", "error", "-filter_threads", "1", "-loop", "1", "-framerate", "24",
                                    "-i", str(source), "-vf", motion_filter("pan_right", 320, 180, 2, 48),
                                    "-frames:v", "48", "-f", "rawvideo", "-pix_fmt", "gray", "-"])
    frames = np.frombuffer(data, dtype=np.uint8).reshape(-1, 180, 320)
    shifts = [cv2.phaseCorrelate(a.astype(np.float32), b.astype(np.float32))[0][0]
              for a, b in zip(frames, frames[1:])]
    assert max(abs(s) for s in shifts) < .8
    assert max(shifts) < .2


def test_finished_video_keeps_length_and_audio(tmp_path):
    source = tmp_path / "source.mp4"
    subprocess.run([ffmpeg_bin(), "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc2=size=320x180:rate=24:duration=9",
                    "-an", "-c:v", "libx264", "-threads", "2", str(source)], check=True)
    voice = tmp_path / "voice.wav"
    t = np.arange(6 * 24000) / 24000
    write_wav(voice, .15 * np.sin(2 * np.pi * 180 * t), 24000)
    original_hash = hashlib.sha256(voice.read_bytes()).hexdigest()
    timeline = {"source_duration": 6, "speed_factor": 1, "sentence_timings": [], "segments": [
        {"src_start": i * 3, "src_duration": 2, "dur": 2, "out_start": i * 2,
         "motion": "pan_right" if i % 2 else "push_in", "freeze_duration": 0} for i in range(3)]}
    output = tmp_path / "clip.mp4"
    export_clip(source, voice, None, timeline, {"target_width": 320, "target_height": 180, "prefer_nvenc": "off",
                "motion_oversample": 3, "short_transitions": "on", "transition_duration": .125, "picture_enhance": "on"}, output)
    assert media_duration(output) == pytest.approx(6, abs=.06)
    assert hashlib.sha256(voice.read_bytes()).hexdigest() == original_hash
    finish = json.loads((tmp_path / "picture_finish.json").read_text(encoding="utf-8"))
    assert finish["source_audio_sha256"] == finish["output_audio_sha256"]
    count = subprocess.check_output(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                                     "stream=nb_frames", "-of", "csv=p=0", str(output)], text=True)
    assert int(count.strip()) == 144
