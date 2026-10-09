"""Inject trusted observed shots at the new reader boundary, never ffprobe fake bytes."""
from factory.utils import write_json


def shots_from_facts(facts):
    rows = []
    for fact in facts:
        mid = (float(fact["start"]) + float(fact["end"])) / 2
        fid = "%s_mid" % fact["id"]
        rows.append({"shot_id": fact["id"], "start": fact["start"], "end": fact["end"],
                     "visible": fact["visible"], "characters": [],
                     "frames": [{"frame_id": fid, "time": mid, "position": "middle"}],
                     "observations": [{"frame_id": fid, "visible": fact["visible"]}],
                     "actions": [{"text": fact["visible"], "frame_ids": [fid]}]})
    return rows

def inject_observer(monkeypatch, director):
    import factory.mining.shot_evidence as evidence
    import factory.mining.source_context as context
    import factory.mining.chapter_hints as chapters
    monkeypatch.setattr(context, 'ensure_source_context', lambda *args, **kw: {'available': False})
    monkeypatch.setattr(chapters, 'load_chapter_hints', lambda *args, **kw: [])
    def observe(p, video, lo, hi, folder, role=None):
        raw = director._cached_call(p, 'facts', '范围%.3f至%.3f秒' % (lo,hi), [], p.router.get('video_analysis'))
        return shots_from_facts(raw['facts'])
    monkeypatch.setattr(evidence, 'build_and_observe', observe)


def write_trusted_catalog(p, candidate, facts):
    import hashlib
    import json
    from factory.mining.shot_evidence import POLICY
    from factory.mining.source_context import source_context
    context_signature = hashlib.sha256(json.dumps(source_context(p,candidate.source_start,candidate.source_end,max_chars=6000),
                                                  ensure_ascii=False,sort_keys=True).encode()).hexdigest()
    write_json(p.job_dir/'analysis/director/story_01/shot_evidence/catalog.json', {
        'policy': POLICY, 'settings': {'source_start':candidate.source_start,'source_end':candidate.source_end},
        'reader': {'source_context_signature':context_signature}, 'shots': shots_from_facts(facts)})
