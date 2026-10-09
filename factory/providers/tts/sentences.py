"""Legacy per-sentence synthesis; the production default is continuous.py."""
from pathlib import Path

from ...script.quality import script_issues
from ...utils import ensure_dir, ffmpeg_bin, run_cmd, read_json, write_json, media_duration


def synth_sentences(tts, variant, out_dir):
    issues = script_issues(variant.sentences)
    if issues:
        raise RuntimeError("配音前文案检查失败: " + "；".join(issues))
    out_dir = ensure_dir(out_dir)
    chunks = ensure_dir(out_dir / "sentence_audio")
    timings, files, providers = [], [], []
    cursor = 0.0
    for i, sent in enumerate(variant.sentences, 1):
        metadata = chunks / ("%03d.json" % i)
        cached = read_json(metadata) or {}
        if cached.get("text") == sent.text and Path(cached.get("audio_path", "")).is_file():
            result = cached
        else:
            result = tts.synth(sent.text, chunks / ("%03d.wav" % i))
            if str(result.get("provider", "")).startswith("mock"):
                raise RuntimeError("拒绝Mock配音")
            result["text"] = sent.text
            write_json(metadata, result)
        path = Path(result["audio_path"]).resolve()
        duration = media_duration(path)
        timings.append({"index": sent.index, "text": sent.text, "role": sent.role,
                        "start": cursor, "end": cursor + duration})
        cursor += duration
        files.append(path)
        providers.append(result.get("provider"))
    manifest = chunks / "concat.txt"
    manifest.write_text("\n".join("file '%s'" % str(p).replace("\\", "/").replace("'", "'\\''") for p in files), encoding="utf-8")
    out = Path(out_dir) / "voice.wav"
    run_cmd([ffmpeg_bin(), "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(manifest),
             "-ac", "1", "-ar", "24000", "-c:a", "pcm_s16le", str(out)], timeout=180)
    return {"audio_path": str(out), "duration": media_duration(out), "sentence_timings": timings,
            "timing_source": "measured_sentence_audio", "providers": providers, "provider": providers[0]}
