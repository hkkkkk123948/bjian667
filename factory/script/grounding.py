"""Small, reusable story checks against observations already read from the source.

Facts are extracted once by the visual reader. The writer and this editorial
check receive text only; neither can create new source times. These checks are
draft assistance, not a claim that sparsely sampled frames prove every action.
"""
import copy
import json
import math
import re


POLICY = "story_grounding_v1"


def narrative(text):
    """The user's parenthetical comments are spoken asides, not story claims."""
    return re.sub(r"（[^（）]*）|\([^()]*\)", "", str(text or "")).strip()


def normalize_facts(data, lo, hi):
    facts = data.get("facts", []) if isinstance(data, dict) else []
    if not 6 <= len(facts) <= 24:
        raise ValueError("分镜事实表需6至24条可见事件，不能用剧情推测代替画面事实")
    out, ids = [], set()
    for fact in facts:
        try:
            fid = str(fact["id"])
            start, end = float(fact["start"]), float(fact["end"])
            visible = str(fact["visible"]).strip()
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("分镜事实表缺少编号、时间或可见事件") from exc
        if (not fid or fid in ids or not visible or len(visible) > 160 or
                not math.isfinite(start + end) or not lo <= start < end <= hi or end - start < 1):
            raise ValueError("分镜事实表含重复编号、无效时间或不明确事件")
        ids.add(fid)
        out.append({"id": fid, "start": start, "end": end, "visible": visible})
    ordered = sorted(out, key=lambda fact: fact["start"])
    if any(previous["end"] - current["start"] > .1 for previous, current in zip(ordered, ordered[1:])):
        raise ValueError("分镜事实表的事件区间互相重叠，需重新核对，不能重复复用同一镜头")
    return ordered


def fact_prompt(lo, hi):
    return ("只做原片观察，不编故事。按时间顺序读所附已有分镜（标有源片秒数），整理12至20条可见事件，"
            "覆盖主角遇险、主动应对、危险升级和最后真实结果。若可见事件不足，宁可少列，最少6条；"
            "同一人的特写不要当作新角色。每条用稳定称呼说明谁在哪里做什么，20至50字。"
            "记录看见的姿态、动作、物品；独自拾石不写被人带到林地，门外站人不写跟随众人出门。"
            "静止表情不能证明击打、绑架或抓捕；看不清的动作明确写不确定。"
            "不引用原片字幕、心理、对白或编造因果。时间须来自图片标签，取同一事件的可用区间，"
            "通常3至8秒、相互不重叠；别覆盖黑幕、片头或片尾。范围%.3f至%.3f秒。"
            "只输出JSON {\"facts\":[{\"id\":\"F01\",\"start\":秒,\"end\":秒,\"visible\":\"可见事件\"}]}。") % (lo, hi)


def bind_facts(data, facts):
    """Only the independent fact catalog supplies source bounds and visuals."""
    if not isinstance(data, dict) or not isinstance(data.get('sentences'), list):
        return {'sentences': []}
    out = copy.deepcopy(data)
    catalog = {fact["id"]: fact for fact in facts}
    for beat in out.get("sentences", []):
        if not isinstance(beat, dict):
            return {'sentences': []}
        ids = beat.get("fact_ids", [])
        if not isinstance(ids, list) or not ids or len(ids) > 3 or len(set(map(str, ids))) != len(ids):
            beat.update(evidence_start=-1, evidence_end=-1, visual="")
            continue
        chosen = [catalog[fid] for fid in ids if isinstance(fid, str) and fid in catalog]
        if len(chosen) != len(ids):
            beat.update(evidence_start=-1, evidence_end=-1, visual="")
            continue
        chosen.sort(key=lambda fact: fact["start"])
        beat['fact_ids'] = [fact['id'] for fact in chosen]
        beat.update(evidence_start=chosen[0]["start"], evidence_end=max(fact["end"] for fact in chosen),
                    visual="；".join(fact["visible"] for fact in chosen))
        beat.pop("source_windows", None)
        if len(chosen) > 1:
            beat["source_windows"] = [{"start": fact["start"], "end": fact["end"], "weight": 1 / len(chosen)}
                                      for fact in chosen]
    return out


def check_prompt(data, facts):
    catalog = {fact["id"]: fact for fact in facts}
    beats = []
    for index, beat in enumerate(data.get("sentences", []), 1):
        ids = beat.get("fact_ids") or []
        observations = [catalog[fid]["visible"] for fid in ids if fid in catalog]
        # Known reviewed outlines do not need new IDs to retain their exact times.
        if not observations and beat.get("visual"):
            observations = [beat["visual"]]
        beats.append({"index": index, "text": narrative(beat.get("narrative_text") or beat.get("text")),
                      "visible": observations})
    return ("你审核尚未配音的惊悚故事，依据独立画面事实逐句核对，不要求逐字描述镜头。"
            "允许合理心理、目的、常识性连接，以及相邻求助/援手/警察镜头推断得到帮助；"
            "具体动作、道具、地点变化、胜负和结局必须有事实依据。不能把画面没证明等同于没有发生，"
            "证据不清标unclear；明确越过事实的具体动作标unsupported。"
            "完整故事需危险→主角应对→新发现或升级→解决办法→结果，援手如何帮助需交代；"
            "按整条因果判断，不机械要求每项单独占一句。不强加抓捕或所有人平安结局。"
            "评论括号已排除，不用给它找物件或编剧情。逐句输出全部index，status仅supported/inference/unsupported/unclear，"
            "reason只为问题句写具体缺少什么证据（不超过35字）。若有明确问题，可给safe_rewrites，"
            "只改有问题句及必要衔接句，保持现有事实、主角、句数和完整结果，不加新事实、评论或括号。"
            "不为了讨好而给通过，不写分数。只返回JSON {\"beats\":[{\"index\":1,\"status\":\"supported\",\"reason\":\"\"}],"
            "\"causal_complete\":true,\"missing_links\":[{\"index\":句号,\"reason\":\"缺少的因果\"}],"
            "\"safe_rewrites\":[{\"index\":句号,\"text\":\"替换旁白\"}]}。故事：" +
            json.dumps(beats, ensure_ascii=False, separators=(",", ":")))


def normalize_report(response, count):
    """Reject partial or malformed answers instead of manufacturing a pass."""
    report = {"policy": POLICY, "basis": "cached_storyboard_facts", "status": "review_required",
              "causal_complete": False, "issues": [], "model_score": None, "checked_complete": False,
              "evidence_limit": "依据已采样分镜的可见观察预检，保留成片实际画面质检；不确定动作需人工确认"}
    if not isinstance(response, dict):
        report["issues"] = [{"index": 0, "kind": "unclear", "reason": "文案核验未返回有效结果"}]
        return report
    checks = response.get("beats")
    valid = isinstance(checks, list) and len(checks) == count
    seen = set()
    for check in checks if valid else []:
        if (not isinstance(check, dict) or type(check.get("index")) is not int or
                not 1 <= check["index"] <= count or check["index"] in seen or
                check.get("status") not in ("supported", "inference", "unsupported", "unclear")):
            valid = False
            break
        seen.add(check["index"])
        if check["status"] in ("unsupported", "unclear"):
            report["issues"].append({"index": check["index"], "kind": check["status"],
                                     "reason": str(check.get("reason") or "该句动作与现有画面依据不明确")[:160]})
    links = response.get("missing_links")
    if not valid or not isinstance(response.get("causal_complete"), bool) or not isinstance(links, list):
        report["issues"] = [{"index": 0, "kind": "unclear", "reason": "核验没有完整覆盖全部句子和故事因果，需人工确认"}]
        return report
    report['checked_complete'] = True
    for link in links:
        reason = str(link.get("reason", "") if isinstance(link, dict) else link).strip()
        index = link.get("index", 0) if isinstance(link, dict) else 0
        if type(index) is not int or not 0 <= index <= count:
            index = 0
        if reason:
            report["issues"].append({"index": index, "kind": "causal_gap", "reason": reason[:160]})
    report["causal_complete"] = response["causal_complete"] and not links
    if not report["causal_complete"] and not any(issue["kind"] == "causal_gap" for issue in report["issues"]):
        report["issues"].append({"index": 0, "kind": "causal_gap", "reason": "遇险到结果之间缺少明确连接"})
    if not report["issues"] and report["causal_complete"]:
        report["status"] = "pass"
    return report


def apply_safe_rewrites(data, response):
    """Return a new *draft* or None. Source times and factual mapping never change."""
    changes = response.get("safe_rewrites") if isinstance(response, dict) else None
    beats = data.get("sentences", [])
    if not isinstance(changes, list) or not changes or len(changes) > len(beats):
        return None
    out, seen = copy.deepcopy(data), set()
    for change in changes:
        if not isinstance(change, dict):
            return None
        index, text = change.get("index"), change.get("text")
        if (type(index) is not int or not 1 <= index <= len(beats) or index in seen or
                not isinstance(text, str) or not text.strip() or narrative(text) != text.strip()):
            return None
        seen.add(index)
        out["sentences"][index - 1]["text"] = text.strip()
    return out if out != data else None
