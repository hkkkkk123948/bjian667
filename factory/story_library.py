"""Exact-source reviewed story reuse. A changed video never inherits another video's evidence."""
import hashlib
from pathlib import Path

from .utils import read_json

LIBRARY = Path(__file__).parent / "presets" / "reviewed_stories.json"


def source_digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def find_reviewed_story(source, params, reference_text=""):
    if str(params.get("approved_plan_reuse", "on")) != "on":
        return None
    path = Path(source)
    if not path.is_file():
        return None
    library = read_json(LIBRARY) or {}
    prefix = library.get("reference_prefix", "")
    reference_hash = library.get("reference_sha256")
    if reference_text and reference_hash and hashlib.sha256(reference_text.encode()).hexdigest() != reference_hash:
        return None
    if reference_text and prefix and not reference_text.strip().startswith(prefix):
        return None
    size = path.stat().st_size
    possible = [item for item in library.get("stories", []) if item.get("source_size") == size
                and item.get("review_status") != "rejected_incomplete"]
    if not possible:
        return None
    digest = source_digest(path)
    return next((item for item in possible if item["source_sha256"] == digest), None)


def find_story_outline(source):
    """Reuse only source identity and story span, including a rejected old script.

    A rejected narrative is never approved by this lookup. The director must
    read the story's frames and write a fresh script under the current policy.
    Source facts remain reusable when the reference style changes.
    """
    path = Path(source)
    if not path.is_file():
        return None
    library = read_json(LIBRARY) or {}
    possible = [item for item in library.get("stories", []) if item.get("source_size") == path.stat().st_size]
    if not possible:
        return None
    digest = source_digest(path)
    item = next((x for x in possible if x["source_sha256"] == digest), None)
    if not item:
        return None
    script = item.get("script") or {}
    beats = item.get("verified_shots") or script.get("sentences") or []
    if not beats:
        return None
    return {"title": script.get("title", "重新编写"),
            "source_start": min(float(b["evidence_start"]) for b in beats),
            "source_end": max(float(b["evidence_end"]) for b in beats),
            "story_summary": script.get("story_logic", "") + "。用已核对镜头重写完整因果，不复用旧文案。",
            "shots": [{"shot_id": i + 1, "start": b["evidence_start"], "end": b["evidence_end"],
                       "visual": b["visual"], "source_windows": b.get("source_windows", [])} for i, b in enumerate(beats)],
            "script_shot_ids": item.get("script_shot_ids") or list(range(1, len(beats) + 1)),
            "source_sha256": digest}


def refresh_evidence_windows(source, variant):
    """Apply corrected shot handles only to byte-identical source evidence."""
    outline = find_story_outline(source)
    if not outline:
        return
    matches = {m.get("sentence_index"): m for m in variant.scene_matches}
    for sentence in variant.sentences:
        ev = sentence.source_evidence
        if not ev:
            continue
        shot = next((s for s in outline["shots"] if abs(s["start"] - ev.start) < .01
                     and abs(s["end"] - ev.end) < .01), None)
        if shot and shot.get("source_windows"):
            match = matches.setdefault(sentence.index, {"sentence_index": sentence.index})
            match["source_windows"] = shot["source_windows"]
    variant.scene_matches = list(matches.values())
