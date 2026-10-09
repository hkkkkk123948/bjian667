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


def causal_chain_issues(sentences):
    """Lightweight local check: prefer 遇险→行动→升级→结果 density."""
    texts = [((getattr(s, "narrative_text", "") or s.text) if hasattr(s, "text")
              else str(s.get("narrative_text") or s.get("text", ""))) for s in sentences]
    issues = []
    if len(texts) < 6:
        issues.append("因果链偏短，简单故事建议至少六个节拍")
    joined = "".join(texts)
    danger = any(k in joined for k in ("怕", "追", "逃", "躲", "怪", "危险", "杀", "盯", "扑", "抓", "吓", "藏"))
    action = any(k in joined for k in ("跑", "躲", "喊", "门", "报警", "逃", "求", "跟", "退", "冲"))
    outcome = any(k in joined for k in ("救", "逃出", "安全", "失踪", "真相", "回家", "离开", "获救", "警察", "死"))
    if texts and not danger:
        issues.append("缺少可见危险或压迫感，开头应直接进入遇险")
    if texts and not action:
        issues.append("缺少应对动作，需写出主角如何行动")
    if texts and not outcome:
        issues.append("缺少明确结果（逃脱/获救/失踪/发现真相）")
    return issues
