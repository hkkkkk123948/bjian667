"""解说句 ↔ 原片场景 的语义匹配（参考“解说猫”的 script_video_matcher 思路）。

“解说猫”用 Chinese-CLIP 计算“句子文本 vs 场景关键帧”余弦相似度，再做贪心分配，把每句解说
匹配到一个最合适的场景，然后按匹配场景生成时间线（并按 TTS 时长适配场景时长）。
这里我们用 LLM（gpt-5.6-sol，具备画面描述）做等价的句子→场景语义匹配，避免依赖大模型权重。
"""


from ..log import get_logger
from ..utils import clamp, normalize_space
from ..script.comment_hook import story_text


def candidate_scenes(index, candidate, margin=8.0):
    """候选时段（含少量外扩）内的场景，带画面描述，供匹配使用。"""
    if not index or not index.scenes:
        return []
    lo = candidate.source_start - margin
    hi = candidate.source_end + margin
    out = []
    for scene in index.scenes:
        if scene.end <= lo or scene.start >= hi:
            continue
        visual = normalize_space(getattr(scene, "visual", "") or "")
        if 0.0 <= getattr(scene, "brightness", -1.0) < 15.0 and not visual:
            continue
        out.append(scene)
    return out


def align_sentences_to_scenes(candidate, sentences, index, router, role="match_judge",
                              language="zh", margin=8.0):
    """句子→场景匹配：让 LLM 输出 NxM 相似度矩阵，再做“贪心唯一分配+相邻惩罚”。

    借“解说猫”的 script_video_matcher 思路（余弦相似度 + 贪心分配，每句挑相似度最高的
    未使用场景，并避免相邻句子用相邻/重复场景），保证分镜多样、画面推进不倒退。
    返回 [scene_id,...]，与 sentences 等长，缺失为 None。失败返回 []。
    """
    logger = get_logger()
    if not sentences or not index:
        return []
    if not router or router.is_mock(role):
        return []
    scenes = candidate_scenes(index, candidate, margin)
    if len(scenes) < 1:
        return []
    scenes = sorted(scenes, key=lambda s: (s.start, s.scene_id))
    scene_lines = []
    for j, sc in enumerate(scenes):
        visual = normalize_space(getattr(sc, "visual", "") or "")[:100]
        text = normalize_space(getattr(sc, "transcript", "") or "")[:70]
        scene_lines.append("S%d: %.1f-%.1fs 画面:%s 台词:%s" % (j, sc.start, sc.end, visual, text))
    prompt = (
        "你是解说与画面的匹配器。给下面每一句解说，对每个候选场景 S%d 打一个 0-1 的语义匹配分"
        "（1=完全吻合）。相同或相邻场景不要给多句都打高分。分数行=句子，列=场景，输出完整矩阵。\n"
        "只输出 JSON：{\"scores\":[[0.9,0.1,...],...]}\n\n"
        "解说句子：\n%s\n\n候选场景：\n%s"
    ) % (len(scenes) - 1, "\n".join("%d. %s" % (i + 1, normalize_space(story_text(s))[:80])
                                     for i, s in enumerate(sentences)), "\n".join(scene_lines))
    try:
        data = router.get(role).complete_json(prompt, json_mode=True, default=None)
    except Exception as exc:
        logger.warning("句子-场景匹配失败: %s", exc)
        return []
    if not isinstance(data, dict):
        return []
    raw = data.get("scores")
    if not isinstance(raw, list) or len(raw) != len(sentences):
        return []
    m = len(sentences)
    n = len(scenes)
    scores = [[0.0]] * m
    for i, row in enumerate(raw):
        if not isinstance(row, list):
            continue
        scores[i] = [0.0] * n
        for j, v in enumerate(row):
            try:
                scores[i][j] = min(1.0, max(0.0, float(v)))
            except (TypeError, ValueError):
                scores[i][j] = 0.0
    # 贪心唯一分配：按解说顺序，挑未使用场景里得分最高的，遇相邻场景打折
    used = [False] * n
    prev_j = -1
    matches: list = [None] * m
    for i in range(m):
        best_j, best_v = -1, -1.0
        for j in range(n):
            if used[j]:
                continue
            v = scores[i][j]
            if prev_j >= 0 and abs(j - prev_j) <= 1:
                v -= 0.45
            if j < prev_j - 1:  # 倒退惩罚（保持画面时间递增）
                v -= 0.15 * (prev_j - 1 - j)
            if v > 0.05 and v > best_v:
                best_v, best_j = v, j
        if best_j >= 0:
            matches[i] = scenes[best_j].scene_id
            used[best_j] = True
            prev_j = best_j
        else:
            prev_j = -1
    if not any(m is not None for m in matches):
        return []
    used_count = sum(1 for v in matches if v is not None)
    need = max(2, int(round(0.6 * m)))
    if used_count < need:
        return []
    used_ids = {scenes[sc].scene_id for sc in range(n) if used[sc]}
    logger.info("句子-场景匹配: %d/%d 唯一命中（%d 个场景被用）", used_count, m, len(used_ids))
    return matches


def scene_time_range(index, scene_id, candidate, min_len=0.4):
    scene = next((s for s in index.scenes if s.scene_id == scene_id), None)
    if not scene:
        return None
    lo, hi = candidate.source_start, candidate.source_end
    start = clamp(scene.start, lo, hi)
    end = clamp(scene.end, lo, hi)
    if end - start < min_len:
        end = min(hi, start + min(min_len + 1.0, hi - start))
    if end - start < min_len:
        return None
    return start, end
