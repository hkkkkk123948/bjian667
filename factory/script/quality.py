"""Local editorial checks. Matching a picture is insufficient to make a story."""
import re
from difflib import SequenceMatcher


def compact(text):
    return re.sub(r"[^\w\u4e00-\u9fff]", "", text or "").lower()


def script_issues(sentences):
    texts = [((getattr(s, "narrative_text", "") or s.text) if hasattr(s, "text")
              else str(s.get("narrative_text") or s.get("text", ""))) for s in sentences]
    issues = []
    for i, a in enumerate(texts):
        for j, b in enumerate(texts[:i]):
            ca, cb = compact(a), compact(b)
            if min(len(ca), len(cb)) >= 10 and SequenceMatcher(None, ca, cb).ratio() > 0.79:
                issues.append("第%d句与第%d句内容重复" % (j + 1, i + 1))
    if len(texts) < 4:
        issues.append("故事少于四个叙事节拍")
    # Denser causal floors live in style.short_story.simple_story_issues.
    if texts and len(compact(texts[0])) > 30:
        issues.append("开头过长，危险出现太晚")
    return issues


# Order matters: danger opens, the protagonist acts because of it, the danger
# changes, and the last beats say what that action caused.
_DANGER = ("怕", "追", "逃", "躲", "怪", "危险", "杀", "盯", "扑", "抓", "吓", "藏", "偷拍", "跟踪", "威胁")
_ACTION = ("跑", "躲", "喊", "报警", "逃", "求", "跟", "退", "冲", "锁", "打电话", "拉开", "推开", "求助", "上车")
_ESCALATION = ("发现", "才知道", "却", "突然", "照片", "原来", "追上来", "跟上来", "抓住", "拖",
               "里面", "更多", "还是", "再次", "又", "怎料", "不料", "下一秒")
_OUTCOME = ("救", "逃出", "安全", "失踪", "真相", "回家", "获救", "警察", "带走", "消失",
            "到家", "回了家", "停住", "没再跟", "抓住了")


def _beat_texts(sentences):
    return [((getattr(s, "narrative_text", "") or s.text) if hasattr(s, "text")
             else str(s.get("narrative_text") or s.get("text", ""))) for s in sentences]


def _indexes(texts, words):
    return [i for i, text in enumerate(texts) if any(word in text for word in words)]


def causal_chain_issues(sentences):
    """Require 遇险 → 应对 → 升级 → 结果 in that order, not a bag of scare words."""
    texts = _beat_texts(sentences)
    issues = []
    if len(texts) < 6:
        issues.append("因果链偏短，简单故事建议至少六个节拍")
    if not texts:
        return issues
    danger, action = _indexes(texts, _DANGER), _indexes(texts, _ACTION)
    escalation, outcome = _indexes(texts, _ESCALATION), _indexes(texts, _OUTCOME)
    if not danger:
        issues.append("缺少可见危险或压迫感，开头应直接进入遇险")
    elif danger[0] > min(2, len(texts) - 1):
        issues.append("危险出现太晚，首段应直接进入遇险")
    if not action:
        issues.append("缺少应对动作，需写出主角如何行动")
    elif danger and action[0] < danger[0]:
        issues.append("应对写在遇险之前，因果顺序反了")
    if not outcome:
        issues.append("缺少明确结果（逃脱/获救/失踪/发现真相）")
    elif outcome[-1] < max(0, len(texts) - 2):
        issues.append("结果出现过早，结尾没有收束")
    if len(texts) >= 6:
        start = action[0] if action else (danger[0] if danger else 0)
        if not any(start < index < len(texts) - 1 for index in escalation):
            issues.append("危险没有升级，只是重复惊吓或直接跳到结果")
    return issues


def narration_spine(facts):
    """Writing-order hint built from visible facts. It adds no events and makes no model call."""
    generic = ("第1句直接写可见危险；第2句写主角为什么要行动；中间写一次由前一事件触发的升级；"
               "最后一句写逃脱、获救、失踪或真相，并点明这个结果如何由前面的行动造成。")
    rows = []
    for fact in facts or []:
        if not isinstance(fact, dict):
            continue
        visible = str(fact.get("visible") or "").strip()
        actions = fact.get("actions") or []
        action_bits = []
        for action in actions[:2]:
            action_bits.append(str(action.get("text") or "") if isinstance(action, dict) else str(action))
        text = "；".join(part for part in (visible, "；".join(bit for bit in action_bits if bit)) if part)
        if text:
            rows.append((str(fact.get("id") or fact.get("shot_id") or ""), text))
    if len(rows) < 2:
        return generic
    count = len(rows)
    bounds = [0, max(1, count // 4), max(2, count // 2), max(3, (3 * count) // 4), count]
    for index in range(1, len(bounds)):
        if bounds[index] <= bounds[index - 1]:
            bounds[index] = min(count, bounds[index - 1] + 1)
    bounds[-1] = count
    labels = ("遇险，用作首句", "行动理由和应对", "由前因触发的危险升级或新发现", "结果，用作末句")
    lines = [generic]
    for label, left, right in zip(labels, bounds, bounds[1:]):
        if left >= right:
            continue
        bits = []
        for fid, text in rows[left:right][:3]:
            bits.append(("%s %s" % (fid, text[:36])).strip())
        lines.append("%s：%s" % (label, " / ".join(bits)))
    return "\n".join(lines)
