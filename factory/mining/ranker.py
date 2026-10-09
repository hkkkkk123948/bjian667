from ..utils import clamp


def rank_candidates(candidates, top_n=None):
    scored = []
    for cand in candidates:
        base = (cand.story_completeness * 0.40 + cand.visual_quality * 0.25 + cand.short_video_potential * 0.35)
        dup_penalty = clamp(cand.diversity_similarity, 0.0, 1.0) * 18.0
        value = clamp(base - dup_penalty, 0.0, 100.0)
        scored.append((value, cand))
    scored.sort(key=lambda x: (-x[0], x[1].source_start))
    ranked = [cand for _, cand in scored]
    for value, cand in scored:
        cand.story_score = int(round(value))
    if top_n:
        ranked = ranked[:top_n]
    return ranked
