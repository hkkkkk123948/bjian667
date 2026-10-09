"""Frame-counted short dissolves; handles preserve every narration cut point."""

# Seconds by incoming shot role. Always capped at 0.18 by the planner.
ROLE_TRANSITION_SECONDS = {
    "hook": 0.10,
    "setup": 0.08,
    "escalation": 0.14,
    "twist": 0.16,
    "payoff": 0.18,
    "爆点": 0.16,
    "转折": 0.16,
}


def transition_seconds_for_role(role, default=0.125):
    return float(ROLE_TRANSITION_SECONDS.get(str(role or ""), default))


def transition_plan(durations, enabled=True, seconds=0.125, fps=24, roles=None):
    lengths, elapsed, previous = [], 0.0, 0
    for duration in durations:
        elapsed += float(duration)
        end = round(elapsed * fps)
        lengths.append(max(1, end - previous))
        previous = end
    roles = list(roles or [])
    counts = []
    for i, (left, right) in enumerate(zip(lengths, lengths[1:])):
        incoming = roles[i + 1] if i + 1 < len(roles) else ""
        sec = transition_seconds_for_role(incoming, seconds)
        desired = round(max(0.0, min(float(sec), 0.18)) * fps) if enabled else 0
        counts.append(max(0, min(desired, min(left, right) // 3)))
    handles = [{"nominal_frames": n, "before": counts[i - 1] // 2 if i else 0,
                "after": counts[i] - counts[i] // 2 if i < len(counts) else 0}
               for i, n in enumerate(lengths)]
    boundary = 0
    transitions, boundaries = [], []
    for i, n in enumerate(lengths[:-1]):
        boundary += n
        count = counts[i]
        before = count // 2
        item = {"index": i + 1, "boundary_frame": boundary,
                "start_frame": boundary - before, "duration_frames": count,
                "duration": count / fps, "type": "fade" if count else "cut",
                "role": roles[i + 1] if i + 1 < len(roles) else ""}
        boundaries.append(item)
        if count:
            transitions.append(item)
    return {"fps": fps, "output_frames": sum(lengths), "handles": handles,
            "transitions": transitions, "boundaries": boundaries}


def xfade_graph(plan):
    n = len(plan["handles"])
    parts = ["[%d:v]settb=AVTB,setpts=PTS-STARTPTS,format=yuv420p[v%d]" % (i, i) for i in range(n)]
    last = "v0"
    for i, tr in enumerate(plan["transitions"], 1):
        label = "blend%d" % i
        parts.append("[%s][v%d]xfade=transition=fade:duration=%.9f:offset=%.9f[%s]" % (
            last, i, tr["duration"], tr["start_frame"] / plan["fps"], label))
        last = label
    parts.append("[%s]trim=end_frame=%d,setpts=PTS-STARTPTS[vout]" % (last, plan["output_frames"]))
    return ";".join(parts)


def pair_xfade_graph(transition, output_frames, fps=24):
    # Matroska rounds timestamps to milliseconds. Rebuild exact frame times
    # before each dissolve or a one-frame transition may start after EOF.
    return ("[0:v]settb=AVTB,setpts=N/(%d*TB),format=yuv444p[left];"
            "[1:v]settb=AVTB,setpts=N/(%d*TB),format=yuv444p[right];"
            "[left][right]xfade=transition=fade:duration=%.9f:offset=%.9f,"
            "trim=end_frame=%d,setpts=PTS-STARTPTS[vout]") % (
                fps, fps, transition["duration"], transition["start_frame"] / fps, output_frames)


def pair_concat_graph(output_frames, fps=24):
    """A zero-length transition still consumes its input in a mixed chain."""
    return ("[0:v]settb=AVTB,setpts=N/(%d*TB),format=yuv444p[left];"
            "[1:v]settb=AVTB,setpts=N/(%d*TB),format=yuv444p[right];"
            "[left][right]concat=n=2:v=1:a=0,trim=end_frame=%d,setpts=PTS-STARTPTS[vout]") % (
                fps, fps, output_frames)
