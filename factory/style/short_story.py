"""Shared editorial policy used by the GUI, automatic director and local presets."""
VERSION = "simple_story_v5_causal_order"
# LENGTH_HINT is stripped by some rewrite prompts that only need causal rules.
LENGTH_HINT = "7至8句，每句18至26字，整段150至190字；目标成片约24至36秒，句句推进事件，避免注水。"
STORY_RULES = (
    "模仿参考示例的直白故事：一名主角、一种危险、一次发现、一个结果。"
    "按时间顺序讲，" + LENGTH_HINT +
    "请逐句核对长度，不能用8句不到100字的提纲代替解说稿。"
    "只用女孩、男人等固定称呼，不堆人物身份。每句只推进一件事，少用逗号，不用省略号和破折号。"
    "禁止复杂回忆、反复转折、离开后又闪回吓人；结尾明确逃脱、获救、失踪或发现真相。"
    "动作、人物和道具要与镜头对应，不编造需画面证明的事。首句直接进入事件，不介绍标题，不用空泛疑问句。"
    "优先讲遇险、逃跑、获救；禁止用短信、录音界面、商品标签、物品替换或小字谜题制造反转。"
)
REFERENCE_EXAMPLES = [
    "女孩独自等公交，一个陌生男人一直盯着她。女人下车时她赶紧跟了过去，告诉女人自己害怕被跟踪。女人把她带回家，让家人陪着她。",
    "女孩去老师家学吉他，无意间看见一张合照。照片里的男人正是偷拍她的人。她马上离开并报警，警方随后找到更多偷拍照片。",
]

STORY_RULES += (
    "正文必须形成遇险→行动理由→行动→新发现或危险升级→解决办法→结果的因果链。"
    "每个转折由前一事件触发，不能只串联惊吓镜头。"
    "有人帮助时交代主角如何求助或援手为何帮助，最后交代人是否安全或失踪，别以警方突然出现草草收尾。"
    "用户的评论引导话术属于独立插话，由软件另外加入；正文不要模仿参考中的玫瑰花插话，不为它编造物件、折返、对话或支线。"
    "密度要求：24至36秒内至少完成一次遇险、一次主动应对、一次危险升级、一个有画面的结果；禁止注水形容外貌或重复同一动作。"
)


def simple_story_issues(beats):
    issues = []
    texts = [((getattr(b, "narrative_text", "") or b.text) if hasattr(b, "text")
              else b.get("narrative_text") or b.get("text", "")) for b in beats]
    if not 6 <= len(texts) <= 9:
        issues.append("简单故事须6至9个节拍")
    for i, text in enumerate(texts, 1):
        if any(term in text for term in ("短信", "录音界面", "署名", "标签", "替换", "镜头切")):
            issues.append("第%d句引入复杂小字谜题，改为直接的遇险逃跑故事" % i)
        import re
        if "……" in text or "——" in text or any(part.count("，") > 1 for part in re.split(r"[。！？!?]", text)):
            issues.append("第%d句停顿符号过多" % i)
    from ..script.quality import causal_chain_issues
    for issue in causal_chain_issues([{"text": t} for t in texts]):
        if issue not in issues:
            issues.append(issue)
    return issues
