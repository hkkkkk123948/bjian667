def audio_graph(has_source_audio, has_voice, params, speed=1.0):
    bg = float(params.get("bg_volume", 0.0))
    voice = float(params.get("voice_volume", 1.0))
    normalize = str(params.get("normalize_loudness", "on")).lower() not in ("off", "0", "false")
    tail = "alimiter=limit=0.97"
    if normalize:
        tail += ",loudnorm=I=-14:TP=-1.5:LRA=11"
    if not has_voice:
        if has_source_audio:
            return "[0:a]%s[aout]" % tail
        return None
    voice_prefix = "[1:a]"
    if abs(speed - 1.0) > 0.01:
        voice_prefix = "[1:a]atempo=%.4f," % min(4.0, max(0.5, speed))
    if not has_source_audio or bg <= 0.0:
        return "%svolume=%.3f,%s[aout]" % (voice_prefix, voice, tail)
    return (
        "[0:a]volume=%.3f[bg];"
        "%svolume=%.3f[vo];"
        "[bg][vo]amix=inputs=2:duration=first:dropout_transition=2:normalize=0[mix];"
        "[mix]%s[aout]"
    ) % (bg, voice_prefix, voice, tail)
