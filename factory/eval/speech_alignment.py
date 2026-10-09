"""Map the confirmed text onto measured speech without changing any audio.

All returned timestamps use the actual audio clock. Character timings inside a
multi-character ASR word are interpolated and labelled as such; they are never
presented as independently measured phoneme boundaries.
"""
import math
from difflib import SequenceMatcher

from ..script.quality import compact

ALIGNMENT_POLICY = "measured_speech_v2_local_anchors"


def _number_value(text):
    if text.isdecimal():
        return int(text)
    digits = {c: i for i, c in enumerate("零一二三四五六七八九")}
    digits.update({"〇": 0, "两": 2})
    if text and all(c in digits for c in text):
        return int("".join(str(digits[c]) for c in text))
    units = {"十": 10, "百": 100, "千": 1000, "万": 10000, "亿": 100000000}
    if not text or any(c not in digits and c not in units for c in text):
        return None
    total, section, number, last_small = 0, 0, 0, 10000
    for index, char in enumerate(text):
        if char in digits:
            number = digits[char]
        elif units[char] < 10000:
            if units[char] >= last_small:
                return None
            section += (number or 1) * units[char]
            number = 0
            last_small = units[char]
        else:
            if index and text[index-1] in units:
                return None
            total += (section + number or 1) * units[char]
            section, number = 0, 0
            last_small = 10000
    return total + section + number


def equivalent_number_notation(expected, recognized):
    """Accept 21/二十一 and 2026/二零二六, without accepting changed values."""
    left, right = _number_value(expected), _number_value(recognized)
    return left is not None and right is not None and left == right


def _characters(words, duration, scale=1.0):
    characters = []
    previous_start = 0.0
    for word in words:
        text = compact(word.get("text", word.get("word", "")))
        if not text:
            continue
        try:
            start, end = float(word["start"]) * scale, float(word["end"]) * scale
        except (KeyError, TypeError, ValueError):
            return [], "invalid_word_timestamp"
        if (not math.isfinite(start) or not math.isfinite(end) or start < -.05
                or start < previous_start - .05 or end < start or start > duration or end > duration + .12):
            return [], "invalid_word_clock"
        start, end = max(0.0, start), min(duration, end)
        previous_start = start
        for index, character in enumerate(text):
            characters.append({"text": character,
                               "start": start + (end-start) * index / len(text),
                               "end": start + (end-start) * (index+1) / len(text),
                               "granularity": "character" if len(text) == 1 else "within_word"})
    if not characters:
        return [], "missing_word_timestamps"
    if sum(item["end"] <= item["start"] for item in characters) > max(2, len(characters) * .15):
        return [], "too_many_zero_length_words"
    # Overlapping neighbouring provider/Whisper words should not create an
    # overlapping aside. Keep the later word start and clip the previous end.
    for left, right in zip(characters, characters[1:]):
        if left["end"] > right["start"]:
            left["end"] = max(left["start"], right["start"])
    return characters, ""


def align_recognized_words(expected_text, words, duration):
    """Locate the given text on the measured ASR word clock, monotonically.

    A local equal-length ASR replacement can retain measured times when it is
    short in time and bounded by exact neighbouring words. Its text mapping is
    explicitly uncertain. Long discrepancies have no trustworthy text clock;
    returning no alignment expresses that limitation, never a voice verdict.
    """
    expected = compact(expected_text)
    chars, reason = _characters(words, duration)
    base = {"policy": ALIGNMENT_POLICY, "source": "local_asr_words", "audio_clock": True,
            "confidence": "unavailable", "matched_ratio": 0, "uncertain_spans": []}
    if reason or not expected:
        return [], {**base, "reason": reason or "empty_text"}
    actual = "".join(item["text"] for item in chars)
    mapped = [None] * len(expected)
    # Calculate the complete comparison before considering any local failure.
    # An early return used to report only the matching prefix as the ratio of
    # the entire narration, turning a four-character ASR error into a false
    # 20% match and discarding accurate clocks for the rest of a long script.
    opcodes = SequenceMatcher(None, expected, actual, autojunk=False).get_opcodes()
    exact_matched = sum(b-a for op, a, b, x, y in opcodes if op == "equal")
    matched = exact_matched
    base.update(matched_ratio=matched/len(expected), exact_matched_ratio=matched/len(expected),
                expected_characters=len(expected), recognized_characters=len(actual))
    uncertain = []
    first_occurrences = {}
    unmapped_audio_spans = []
    for position, (op, a, b, x, y) in enumerate(opcodes):
        if op == "equal":
            for index in range(b-a):
                measured = first_occurrences.get(a+index, chars[x+index])
                mapped[a+index] = dict(measured, text=expected[a+index])
            continue
        uncertain.append({"kind": op, "start_char": a, "end_char": b,
                          "expected": expected[a:b], "recognized": actual[x:y]})
        if x < y:
            uncertain[-1].update(start=chars[x]["start"], end=chars[y-1]["end"])
        if op == "replace" and equivalent_number_notation(expected[a:b], actual[x:y]):
            uncertain[-1]["kind"] = "number_notation"
            start, end = chars[x]["start"], chars[y-1]["end"]
            for index in range(b-a):
                mapped[a+index] = {"text": expected[a+index],
                                   "start": start+(end-start)*index/(b-a),
                                   "end": start+(end-start)*(index+1)/(b-a),
                                   "granularity": "number_notation"}
            # Numeral notation is text-equivalent but its inner character
            # boundaries remain inferred and therefore medium confidence.
            matched += b-a
            continue
        bounded_repeat = False
        if op == "insert" and 2 < y-x <= 12:
            left = opcodes[position-1] if position else None
            right = opcodes[position+1] if position+1 < len(opcodes) else None
            left_size = left[2]-left[1] if left and left[0] == "equal" else 0
            right_size = right[2]-right[1] if right and right[0] == "equal" else 0
            interval = chars[y-1]["end"]-chars[x]["start"]
            repeat_before = (left_size >= y-x and right_size >= 3 and a >= y-x
                             and actual[x:y] == expected[a-(y-x):a])
            repeat_after = (left_size >= 3 and right_size >= y-x
                            and actual[x:y] == expected[a:a+y-x])
            bounded_repeat = (base["exact_matched_ratio"] >= .9 and 0 < interval <= 2
                              and (repeat_before or repeat_after))
            if bounded_repeat:
                # Match the first observed occurrence. The second occurrence
                # stays an explicitly unmapped audio interval, naturally
                # covered by the surrounding picture's intact audio clock.
                # This is time location, not a verdict or a repair of speech.
                if repeat_before:
                    start, stop = chars[x]["start"], chars[y-1]["end"]
                else:
                    for index in range(y-x):
                        first_occurrences[a+index] = dict(chars[x+index], granularity="first_asr_occurrence")
                    start, stop = chars[y]["start"], chars[y+y-x-1]["end"]
                unmapped_audio_spans.append({"start": start, "end": stop,
                                             "recognized": actual[x:y], "kind": "repeated_asr_interval"})
                uncertain[-1].update(kind="bounded_asr_repeat", mapping_uses="first_observed_occurrence",
                                      measured_interval=interval)
        bounded_replacement = False
        if op == "replace" and b-a == y-x and 2 < b-a <= 6:
            left = opcodes[position-1] if position else None
            right = opcodes[position+1] if position+1 < len(opcodes) else None
            left_size = left[2]-left[1] if left and left[0] == "equal" else 0
            right_size = right[2]-right[1] if right and right[0] == "equal" else 0
            interval = chars[y-1]["end"]-chars[x]["start"]
            bounded_replacement = (base["exact_matched_ratio"] >= .9
                                   and left_size >= 3 and right_size >= 3 and 0 < interval <= 1.2)
            if bounded_replacement:
                uncertain[-1].update(kind="bounded_substitution", left_anchor_characters=left_size,
                                      right_anchor_characters=right_size, measured_interval=interval)
        if max(b-a, y-x) > 2 and not bounded_replacement and not bounded_repeat:
            return [], {**base, "reason": "asr_text_discrepancy", "uncertain_spans": uncertain,
                        "matched_ratio": matched / len(expected)}
        if op == "replace" and b-a == y-x:
            for index in range(b-a):
                mapped[a+index] = dict(chars[x+index], text=expected[a+index],
                                       granularity="bounded_substitution" if bounded_replacement else "asr_substitution")
    ratio = matched / len(expected)
    lexical = [s for s in uncertain if s["kind"] != "number_notation"]
    if ratio < .8 or sum(s["end_char"]-s["start_char"] for s in lexical) > max(2, len(expected)*.12):
        return [], {**base, "reason": "insufficient_matching_speech", "matched_ratio": ratio,
                    "uncertain_spans": uncertain}
    # At most two omitted ASR characters may be inferred between measured
    # anchors. A missing opening/ending has no anchors and stays unaligned.
    index = 0
    while index < len(mapped):
        if mapped[index] is not None:
            index += 1
            continue
        first = index
        while index < len(mapped) and mapped[index] is None:
            index += 1
        if first == 0 or index == len(mapped):
            return [], {**base, "reason": "missing_boundary_anchor", "matched_ratio": ratio,
                        "uncertain_spans": uncertain}
        start, end = mapped[first-1]["end"], mapped[index]["start"]
        if end-start <= .015 or end-start > 1.0:
            return [], {**base, "reason": "unreliable_gap_duration", "matched_ratio": ratio,
                        "uncertain_spans": uncertain}
        for offset in range(index-first):
            mapped[first+offset] = {"text": expected[first+offset],
                                    "start": start+(end-start)*offset/(index-first),
                                    "end": start+(end-start)*(offset+1)/(index-first),
                                    "granularity": "inferred_gap"}
    # Exact full text is now available to both sentence and comment alignment.
    return mapped, {**base, "source": "local_asr_interpolated" if uncertain else "local_asr_words",
                    "confidence": "medium" if uncertain else "high", "matched_ratio": ratio,
                    "uncertain_spans": uncertain, "word_count": len(words),
                    "character_count": len(mapped), "aligned_ratio": 1.0,
                    "uncertain_character_ratio": sum(s["end_char"]-s["start_char"] for s in uncertain)/len(expected),
                    "unmapped_audio_spans": unmapped_audio_spans,
                    "silence_gaps": [{"start": a["end"], "end": b["start"],
                                      "duration": b["start"]-a["end"]}
                                     for a, b in zip(chars, chars[1:]) if b["start"]-a["end"] >= .7],
                    "within_word_interpolation": any(c["granularity"] == "within_word" for c in mapped)}


def select_word_alignment(expected_text, duration, provider_words=None, provider_source="", asr_words=None):
    """Prefer verified provider clocks; use already available local ASR otherwise.

    Some providers publish timestamps on a slightly longer synthesis clock.
    When the shift exceeds 120 ms, measured ASR takes priority over stretching
    the provider clock. A bounded calibrated provider fallback remains explicit.
    """
    expected = compact(expected_text)
    provider_words = provider_words or []
    provider = []
    provider_scale = 1.0
    provider_reason = "provider_did_not_supply_measured_words"
    calibrated = False
    if provider_source == "provider_word_timestamps" and provider_words:
        try:
            last = max(float(w.get("end", 0)) for w in provider_words)
            provider_scale = duration / last if last > duration else 1.0
            calibrated = last > duration + .12
            if provider_scale < .9:
                provider_reason = "provider_clock_drift_exceeds_limit"
            else:
                provider, provider_reason = _characters(provider_words, duration, provider_scale)
                if "".join(c["text"] for c in provider) != expected:
                    provider, provider_reason = [], "incomplete_provider_text"
        except (TypeError, ValueError):
            provider_reason = "invalid_provider_timestamp"
    if provider and not calibrated:
        return provider, {"policy": ALIGNMENT_POLICY, "source": "provider_word_timestamps",
                          "confidence": "high", "matched_ratio": 1.0, "audio_clock": True,
                          "provider_timestamp_scale": provider_scale, "uncertain_spans": []}
    local, report = align_recognized_words(expected_text, asr_words or [], duration)
    if local:
        return local, {**report, "provider_fallback_reason": provider_reason or "provider_clock_requires_calibration"}
    if provider:
        return provider, {"policy": ALIGNMENT_POLICY, "source": "provider_words_calibrated",
                          "confidence": "medium", "matched_ratio": 1.0, "audio_clock": True,
                          "provider_timestamp_scale": provider_scale, "uncertain_spans": [],
                          "local_fallback_reason": report.get("reason")}
    return [], {**report, "source": "estimated", "confidence": "low",
                "provider_fallback_reason": provider_reason}


def evidence_window_for_span(alignment_chars, start_char, end_char, pad=0.04):
    """Map a script character span onto measured audio clock bounds.

    Returns None when confidence is too low to tighten picture evidence safely.
    """
    if not alignment_chars:
        return None
    chosen = [c for c in alignment_chars
              if c.get("start") is not None and c.get("end") is not None
              and not (c.get("end_char", -1) <= start_char or c.get("start_char", 10**9) >= end_char)]
    if not chosen:
        return None
    uncertain = sum(1 for c in chosen if c.get("granularity") == "within_word" or c.get("uncertain"))
    if uncertain > max(1, len(chosen) // 2):
        return None
    lo = min(float(c["start"]) for c in chosen) - pad
    hi = max(float(c["end"]) for c in chosen) + pad
    if hi - lo < 0.08:
        return None
    return {"start": max(0.0, lo), "end": hi, "confidence": "high" if uncertain == 0 else "medium"}
