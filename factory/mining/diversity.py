from ..log import get_logger
from ..utils import cosine_sim, jaccard, overlap_ratio, tfidf_vectors, tokenize


def filter_diversity(candidates, threshold=0.72, top_k=0):
    if len(candidates) < 2:
        return candidates
    logger = get_logger()
    scored = sorted(candidates, key=lambda c: c.story_score, reverse=True)
    texts = ["%s %s" % (c.story_summary, " ".join(c.tags or [])) for c in scored]
    vectors = tfidf_vectors(texts)
    kept = []
    for i, cand in enumerate(scored):
        similarity = 0.0
        for other, j in kept:
            s = (
                0.34 * cosine_sim(vectors[i], vectors[j])
                + 0.30 * overlap_ratio(cand.source_start, cand.source_end, other.source_start, other.source_end)
                + 0.18 * jaccard(cand.characters, other.characters)
                + 0.18 * _event_overlap(cand, other)
            )
            similarity = max(similarity, s)
        cand.diversity_similarity = round(similarity, 3)
        if similarity >= threshold:
            cand.status = "rejected_duplicate"
            cand.reject_reason = "与已选候选相似度 %.0f%% 超过阈值 %.0f%%" % (similarity * 100, threshold * 100)
            logger.info("淘汰重复候选 %s (sim=%.2f)", cand.candidate_id, similarity)
            continue
        kept.append((cand, i))
    kept_cands = [cand for cand, _ in kept]
    if top_k and len(kept_cands) > top_k:
        for cand in kept_cands[top_k:]:
            cand.status = "rejected_rank"
            cand.reject_reason = "故事价值排名靠后"
        kept_cands = kept_cands[:top_k]
    return kept_cands


def _event_overlap(a, b):
    a_events = set(t.lower() for t in (a.tags or []))
    b_events = set(t.lower() for t in (b.tags or []))
    if a_events and b_events:
        return jaccard(a_events, b_events)
    a_sent = tokenize(a.story_summary)
    b_sent = tokenize(b.story_summary)
    if a_sent and b_sent:
        return jaccard(a_sent, b_sent)
    return 0.0


def pair_similarity(candidates, threshold=0.72):
    texts = ["%s %s" % (c.story_summary, " ".join(c.tags or [])) for c in candidates]
    vectors = tfidf_vectors(texts)
    result = []
    for i in range(len(candidates)):
        for j in range(i + 1, len(candidates)):
            s = (
                0.34 * cosine_sim(vectors[i], vectors[j])
                + 0.30 * overlap_ratio(candidates[i].source_start, candidates[i].source_end,
                                       candidates[j].source_start, candidates[j].source_end)
                + 0.18 * jaccard(candidates[i].characters, candidates[j].characters)
            )
            result.append((candidates[i].candidate_id, candidates[j].candidate_id, round(s, 3)))
    return [t for t in result if t[2] >= threshold]
