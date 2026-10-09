"""Reproducible checks on exported media, independent of LLM opinions."""
import hashlib
import json
import re
from pathlib import Path

from ..utils import ffmpeg_bin, run_cmd, read_json, write_json, media_duration
from .judge import audit_timeline


def audit_media(path, timeline=None):
    path = Path(path)
    timeline = timeline or read_json(path.parent / "timeline.json") or {}
    _, errors, code = run_cmd([ffmpeg_bin(), "-v", "error", "-xerror", "-i", str(path),
                               "-f", "null", "-"], check=False, timeout=180)
    _, log, _ = run_cmd([ffmpeg_bin(), "-hide_banner", "-nostats", "-i", str(path),
                         "-af", "loudnorm=I=-16:TP=-1.5:LRA=7:print_format=json", "-f", "null", "-"],
                        check=False, timeout=180)
    match = re.search(r'\{\s*"input_i".*?\}', log or "", re.S)
    loudness = json.loads(match.group()) if match else {}
    _, blacklog, _ = run_cmd([ffmpeg_bin(), "-hide_banner", "-nostats", "-i", str(path),
                              "-vf", "blackdetect=d=0.18:pix_th=0.08:pic_th=0.98", "-an", "-f", "null", "-"],
                             check=False, timeout=180)
    blacks = re.findall(r"black_start:([\d.]+) black_end:([\d.]+) black_duration:([\d.]+)", blacklog or "")
    provenance = audit_timeline(timeline)
    duration = media_duration(path)
    out,_,_ = run_cmd(['ffprobe','-v','error','-show_entries','stream=codec_type,duration','-of','json',str(path)],timeout=60)
    streams = json.loads(out).get('streams',[])
    video_duration = next((float(s['duration']) for s in streams if s.get('codec_type')=='video' and s.get('duration') not in (None,'N/A')),None)
    audio_duration = next((float(s['duration']) for s in streams if s.get('codec_type')=='audio' and s.get('duration') not in (None,'N/A')),None)
    issues = list(provenance["issues"])
    decisions = read_json(path.parent/'edit_decisions.json') or {}
    shots = decisions.get('shots') or []
    short_shots = [s for s in shots if float(s.get('duration',0)) < .35]
    if short_shots:
        issues.append('存在低于0.35秒的独立碎镜头，需调整文案或镜头承接')
    if video_duration is None or abs(video_duration-float(timeline.get('narration_duration',duration))) > .16:
        issues.append('视频轨时长不足或与旁白不符：视频%s秒，旁白%.3f秒' %
                      ('未知' if video_duration is None else '%.3f'%video_duration,float(timeline.get('narration_duration',duration))))
    if code:
        issues.append("完整解码失败")
    if blacks:
        issues.append("成片存在持续黑场")
    if abs(duration - float(timeline.get("narration_duration", duration))) > .16:
        issues.append("画面时长和实测旁白不一致")
    if not loudness:
        issues.append("无法测量实际音轨响度")
    elif float(loudness["input_tp"]) > -.7:
        issues.append("编码后真峰值余量过小")
    report = {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
              "status": "pass" if not issues else "fail", "issues": issues,
              "duration": duration, "decode_exit": code, "decode_errors": errors,
              "video_duration":video_duration,"audio_duration":audio_duration,
              "integrated_lufs": loudness.get("input_i"), "true_peak_dbtp": loudness.get("input_tp"),
              "black_periods": blacks, "timeline": provenance,
              "pacing": {'rendered_shots':len(shots),'short_shots':short_shots,
                         'context_joins':decisions.get('context_joins',0)},
              "alignment": timeline.get('alignment_report') or {},
              "note": "机器审计只验证文件、响度、黑场和剪辑区间；故事及听感需要审阅。"}
    write_json(path.parent / "media_audit.json", report)
    return report
