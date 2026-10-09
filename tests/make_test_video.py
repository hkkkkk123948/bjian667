import math
import random
import shutil
import struct
import subprocess
import sys
import wave
from pathlib import Path

ASSETS = Path(__file__).resolve().parent / "assets"

SCENES = [
    ("0x121a2e", 0),
    ("0x2e1220", 1),
    ("0x0e2418", 2),
    ("0x2e2610", 3),
    ("0x101010", 4),
    ("0x1e1030", 5),
]

SRT = """1
00:00:01,000 --> 00:00:05,000
深夜十一点，我独自走进了废弃三年的北山疗养院。

2
00:00:05,200 --> 00:00:09,800
走廊尽头的灯还亮着，可这里早就断电了。

3
00:00:10,200 --> 00:00:14,500
护士站的登记簿上，最后一行名字还在渗墨。

4
00:00:14,700 --> 00:00:19,500
那是我姐姐的笔迹，写着今晚十一点半。

5
00:00:20,000 --> 00:00:24,500
地下室的铁门半开着，里面传来规律的敲击声。

6
00:00:24,700 --> 00:00:29,500
三长两短，是姐姐和我小时候的暗号。

7
00:00:30,000 --> 00:00:34,500
我顺着台阶往下走，手电筒照见了一排病床。

8
00:00:34,700 --> 00:00:39,500
每张床上都放着一台还在转动的老式录音机。

9
00:00:40,000 --> 00:00:44,500
录音机里循环播放着同一句话：别相信她。

10
00:00:44,700 --> 00:00:49,500
身后传来脚步声，和我刚才的节奏一模一样。

11
00:00:49,700 --> 00:00:54,000
镜子里映出两个人，可病房里只有我一个。

12
00:00:54,200 --> 00:00:58,800
登记簿上的名字，此刻变成了我自己的。
"""

STYLE = """解说词风格参考：
1. 短句为主，一句不超过十八个字，避免从句堆叠。
2. 全程使用第二人称"你"，制造压迫感和代入感。
3. 每三到四句必须抛出一个未解之谜，钩住观众。
4. 禁止科普式解释，情绪优先于信息量。
5. 悬念结尾：最后一句永远比上一句更安静。
6. 用具体的数字和物件建立真实感，例如"十一点半""三长两短"。
7. 情绪曲线：压抑铺垫 → 逐步加速 → 瞬间静止 → 反转。
"""


def run(cmd):
    print(" ".join(str(c) for c in cmd))
    proc = subprocess.run([str(c) for c in cmd], capture_output=True)
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr.decode("utf-8", "replace")[-3000:])
        sys.exit(1)


def make_scenes(tmp):
    for i, (color, idx) in enumerate(SCENES):
        out = tmp / ("scene_%d.mp4" % i)
        box_x = "mod(%d+t*180,1620)" % (100 + idx * 250)
        vf = "drawbox=x='%s':y=%d:w=300:h=300:color=white@0.55:t=fill,drawbox=x=0:y=%d:w=1920:h=6:color=0xff2222@0.8:t=fill" % (
            box_x, 300 + idx * 90, 90 + idx * 160)
        run(["ffmpeg", "-y", "-f", "lavfi", "-i",
             "color=c=%s:s=1920x1080:d=10:r=25" % color,
             "-vf", vf, "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p", out])
    return [tmp / ("scene_%d.mp4" % i) for i in range(len(SCENES))]


def concat(scenes, out):
    listing = out / "list.txt"
    listing.write_text("".join("file '%s'\n" % s.as_posix() for s in scenes), encoding="utf-8")
    run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", listing,
         "-c", "copy", out / "video_noaudio.mp4"])


def make_audio(path, duration=60.0, rate=44100):
    random.seed(7)
    loud = [(7.0, 12.5), (24.0, 30.0), (44.0, 50.5), (55.0, 59.0)]
    total = int(duration * rate)
    frames = bytearray()
    for n in range(total):
        t = n / rate
        noise = (random.random() * 2 - 1)
        is_loud = any(a <= t <= b for a, b in loud)
        if is_loud:
            tone = math.sin(2 * math.pi * (180 + 40 * math.sin(t * 2)) * t)
            sample = 0.22 * tone + 0.10 * noise
        else:
            sample = 0.006 * noise
        frames += struct.pack("<h", int(max(-1.0, min(1.0, sample)) * 32000))
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(bytes(frames))


def main():
    ASSETS.mkdir(parents=True, exist_ok=True)
    tmp = ASSETS / "tmp"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    scenes = make_scenes(tmp)
    concat(scenes, tmp)
    make_audio(tmp / "audio.wav")
    final = ASSETS / "source_video.mp4"
    run(["ffmpeg", "-y", "-i", tmp / "video_noaudio.mp4", "-i", tmp / "audio.wav",
         "-c:v", "copy", "-c:a", "aac", "-shortest", final])
    (ASSETS / "source_video.srt").write_text(SRT, encoding="utf-8")
    (ASSETS / "reference_style.txt").write_text(STYLE, encoding="utf-8")
    shutil.rmtree(tmp)
    print("OK:", final)


if __name__ == "__main__":
    main()
