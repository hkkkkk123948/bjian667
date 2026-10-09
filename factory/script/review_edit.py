"""Editable story rows and a marked, independent comment aside."""
import copy
import math
import re
from difflib import SequenceMatcher

from .comment_hook import story_text, apply_comment_hook
from .quality import compact
from .duration import estimate_duration
from ..models import Evidence, ScriptSentence

COMMENT_MARKER = "【评论话术｜不参与匹配】"


def sentence_text(text):
    text = str(text or "").strip()
    if text and text[-1] not in "。！？!?":
        text = text.rstrip("，,；; ") + "。"
    return text


def review_text(variant):
    lines = []
    for sent in variant.sentences:
        body = story_text(sent)
        hooks = getattr(sent,'comment_hooks',[]) or ([{"text":sent.comment_hook, "offset":len(body.rstrip("。！？!?"))}] if getattr(sent,'comment_hook','') else [])
        for hook in reversed(sorted(hooks, key=lambda h:h["offset"])):
            offset = hook["offset"]
            body = body[:offset] + "（" + hook["text"] + "）" + body[offset:]
        lines.append(body)
    return "\n".join(lines)


def parse_review(text):
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    body, hooks, leading = [], [], []
    for line in lines:
        if line.startswith(COMMENT_MARKER):
            phrase = line[len(COMMENT_MARKER):].strip()
            if phrase:
                if body:
                    hooks.append({"text":phrase, "after_sentence":len(body), "offset":len(body[-1].rstrip("。！？!?"))})
                else:
                    leading.append(phrase)
            continue
        chars, comments, cursor = [], [], 0
        while cursor < len(line):
            char = line[cursor]
            if char in "（(":
                end = cursor + 1
                while end < len(line) and line[end] not in "）)":
                    if line[end] in "（(":
                        raise ValueError("评论话术括号请勿嵌套")
                    end += 1
                if end == len(line):
                    raise ValueError("评论话术括号未配对，请补全（）或()")
                if line[end] != ("）" if char == "（" else ")"):
                    raise ValueError("中英文括号请成对使用：（）或()")
                phrase = line[cursor+1:end].strip()
                if phrase:
                    comments.append({"text":phrase,"offset":len(chars)})
                cursor = end + 1
                continue
            if char in "）)":
                raise ValueError("评论话术括号未配对，请补全（）或()")
            chars.append(char)
            cursor += 1
        raw = "".join(chars)
        clean = raw.strip()
        if not clean:
            for hook in comments:
                if body:
                    hooks.append(dict(hook,after_sentence=len(body),offset=len(body[-1].rstrip("。！？!?"))))
                else:
                    leading.append(hook["text"])
            continue
        body.append(clean)
        hooks.extend({"text":p,"offset":0,"after_sentence":len(body)} for p in leading)
        leading.clear()
        prefix = len(raw) - len(raw.lstrip())
        hooks.extend(dict(h,after_sentence=len(body),offset=max(0,h["offset"]-prefix)) for h in comments)
    if leading:
        raise ValueError("评论话术之外至少需要故事正文")
    if len(body) == 1:
        parts = [s for s in re.split(r"(?<=[。！？!?])", body[0]) if s.strip()]
        for hook in hooks:
            start = 0
            for index, part in enumerate(parts,1):
                if hook["offset"] <= start+len(part) or index == len(parts):
                    hook.update(after_sentence=index,offset=max(0,hook["offset"]-start))
                    break
                start += len(part)
        body = [s.strip() for s in parts]
    return [sentence_text(s) for s in body], sorted(hooks,key=lambda h:(h["after_sentence"],h["offset"]))


def edit_variant(pipeline, variant, text):
    texts, hooks = parse_review(text)
    if not texts:
        raise ValueError('请填写非空故事正文；评论话术用（）或()标记')
    old = variant.sentences
    fresh = copy.deepcopy(variant)
    old_texts = [story_text(s) for s in old]
    changed = [compact(s) for s in texts] != [compact(s) for s in old_texts]
    # Equal row counts keep every existing evidence range and shot handle.
    # Splitting/merging allocates the previous selected windows, never the
    # whole source span or invented footage.
    if len(texts) == len(old):
        fresh.sentences = copy.deepcopy(old)
    else:
        fresh.sentences, fresh.scene_matches = _remap_rows(variant, old_texts, texts)
    for i, (sent, line) in enumerate(zip(fresh.sentences, texts), 1):
        sent.index, sent.text = i, line
        sent.narrative_text = sent.comment_hook = sent.comment_hook_mode = ""
        sent.comment_hooks = []
    missing = []
    registered = {match['sentence_index']: match.get('source_windows') for match in fresh.scene_matches
                  if isinstance(match, dict) and type(match.get('sentence_index')) is int}
    for sent in fresh.sentences:
        ev = sent.source_evidence
        valid = ev is not None
        try:
            valid = valid and math.isfinite(ev.start+ev.end) and ev.end-ev.start >= .15
            windows = registered.get(sent.index)
            if windows:
                valid = valid and isinstance(windows, list) and all(
                    math.isfinite(float(window['start'])+float(window['end'])) and
                    float(window['end'])-float(window['start']) >= .15 for window in windows)
        except (TypeError, KeyError, ValueError):
            valid = False
        if not valid:
            sent.source_evidence = None
            missing.append(sent.index)
    if missing:
        fresh.scene_matches = [match for match in fresh.scene_matches if not isinstance(match, dict) or
                               match.get('sentence_index') not in missing]
        fresh.scene_matches.extend({'sentence_index': number, 'source_windows': [], 'needs_rematch': True,
                                    'reason': '文案已保存，原镜头无法可靠承接该句；制作时重新匹配，不修改正文'} for number in missing)
    fresh.review_comment_hooks = hooks
    fresh.review_comment_hook = ({"text":hooks[0]["text"],"after_sentence":hooks[0]["after_sentence"]}
                                 if len(hooks)==1 else {"text":"","after_sentence":0})
    apply_comment_hook(pipeline, fresh)
    # Use exactly the same voice-specific clock as the review panel, after
    # comments have been inserted into the actual spoken text.
    for sent in fresh.sentences:
        sent.estimated_seconds = estimate_duration(pipeline.config, sent.text).seconds
    fresh.estimated_seconds = estimate_duration(pipeline.config, fresh.full_text).seconds
    return fresh, changed


def _remap_rows(variant, old_texts, texts):
    groups = []
    for kind, a, b, x, y in SequenceMatcher(None, [compact(s) for s in old_texts],
                                         [compact(s) for s in texts], autojunk=False).get_opcodes():
        if x == y:
            continue
        if kind == "equal":
            groups.extend([([row], [target]) for row, target in zip(range(a, b), range(x, y))])
            continue
        if a == b:
            if not old_texts:
                groups.append([[], list(range(x, y))])
                continue
            neighbor = max(0, min(len(old_texts) - 1, a - 1))
            found = next((g for g in groups if neighbor in g[0]), None)
            if found:
                found[1].extend(range(x, y))
            else:
                groups.append([list(range(neighbor, neighbor + 1)), list(range(x, y))])
        else:
            groups.append([list(range(a, b)), list(range(x, y))])
    # A leading insertion and the following old row share one source pool.
    for i in range(len(groups) - 1, -1, -1):
        for j in range(i):
            if set(groups[i][0]) & set(groups[j][0]):
                groups[j] = [sorted(set(groups[i][0] + groups[j][0])), sorted(set(groups[i][1] + groups[j][1]))]
                groups.pop(i)
                break
    matches = {m['sentence_index']: m.get('source_windows') for m in variant.scene_matches or []
               if isinstance(m, dict) and type(m.get('sentence_index')) is int}
    out, aligned = [None] * len(texts), []
    for source_rows, target_rows in groups:
        if len(source_rows) == len(target_rows) == 1 and compact(old_texts[source_rows[0]]) == compact(texts[target_rows[0]]):
            old_sent = variant.sentences[source_rows[0]]
            out[target_rows[0]] = copy.deepcopy(old_sent)
            if matches.get(old_sent.index):
                aligned.append({"sentence_index": target_rows[0] + 1,
                                "source_windows": copy.deepcopy(matches[old_sent.index])})
            continue
        windows = []
        missing_source = not source_rows
        for row in source_rows:
            sent = variant.sentences[row]
            ev = sent.source_evidence
            if not ev:
                missing_source = True
                break
            rows = matches.get(sent.index) or [{'start': ev.start, 'end': ev.end}]
            if not isinstance(rows, list):
                missing_source = True
                break
            for window in rows:
                try:
                    start, end = float(window['start']), float(window['end'])
                except (TypeError, ValueError, KeyError):
                    missing_source = True
                    break
                if not math.isfinite(start+end) or start < 0 or end-start < .15:
                    missing_source = True
                    break
                windows.append({'start': start, 'end': end})
        if missing_source or not windows:
            for target in target_rows:
                sent = copy.deepcopy(variant.sentences[source_rows[0]]) if source_rows else ScriptSentence(role='context')
                sent.source_evidence = None
                out[target] = sent
            continue
        total = sum(w["end"] - w["start"] for w in windows)
        weight = sum(max(1, len(compact(texts[k]))) for k in target_rows)
        cursor, window_index = windows[0]["start"], 0
        for pos, target in enumerate(target_rows):
            need = total * max(1, len(compact(texts[target]))) / weight
            pieces = []
            while need > 1e-8 and window_index < len(windows):
                window = windows[window_index]
                take = min(need, window["end"] - cursor)
                if take > 1e-8:
                    pieces.append({"start": cursor, "end": cursor + take, "weight": take})
                cursor += take
                need -= take
                if cursor >= window["end"] - 1e-8:
                    window_index += 1
                    if window_index < len(windows):
                        cursor = windows[window_index]["start"]
            if not pieces or any(w['end']-w['start'] < .15 for w in pieces):
                sent = copy.deepcopy(variant.sentences[source_rows[0]])
                sent.source_evidence = None
                out[target] = sent
                continue
            sent = copy.deepcopy(variant.sentences[source_rows[0]])
            sent.source_evidence = Evidence(start=min(w["start"] for w in pieces), end=max(w["end"] for w in pieces),
                                             scene_ids=sorted({sid for r in source_rows for sid in variant.sentences[r].source_evidence.scene_ids}))
            out[target] = sent
            aligned.append({"sentence_index": target + 1, "source_windows": pieces})
    for index, sentence in enumerate(out):
        if sentence is None:
            out[index] = ScriptSentence(index=index+1, text=texts[index], role='context', source_evidence=None)
    return out, aligned
