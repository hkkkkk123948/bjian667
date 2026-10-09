import json

from factory.config import Config, deep_merge
from factory.ingest.downloader import is_local_source
from factory.matching import run_gates, score_variant, select_variant_best
from factory.mining.diversity import filter_diversity
from factory.models import (Evidence, MatchReport, ScriptSentence, ScriptVariant, StoryCandidate,
                            TranscriptSegment, candidate_from_dict, to_dict)
from factory.render.subtitles import build_ass, build_srt, split_display_lines
from factory.script.writer import write_scripts
from factory.statemachine import STAGES, stage_progress
from factory.style.analyzer import _default_suspense_profile, style_fingerprint
from factory.utils import estimate_speech_seconds, fmt_clock, tokenize
import factory.index.transcript as transcript_mod


def make_candidate(cid="C001", start=10.0, end=40.0):
    return StoryCandidate(
        candidate_id=cid, source_start=start, source_end=end,
        story_summary="深夜，你走进废弃疗养院，发现录音机循环着同一句话。",
        title="疗养院的录音机", hook="你敢相信，每张床上都放着一台还在转动的录音机。",
        conflict="地下室的铁门半开着，里面传来规律的敲击声。",
        turning_point="录音机里循环播放着同一句话：别相信她。",
        payoff="登记簿上的名字，此刻变成了我自己的。",
        evidence=[Evidence(start=start, end=end, quote="深夜十一点，我独自走进了北山疗养院。")],
        story_completeness=85, visual_quality=80, short_video_potential=90, story_score=88,
    )


def make_transcript():
    lines = [
        (1.0, 5.0, "深夜十一点，我独自走进了废弃三年的北山疗养院。"),
        (5.2, 9.8, "走廊尽头的灯还亮着，可这里早就断电了。"),
        (10.2, 14.5, "护士站的登记簿上，最后一行名字还在渗墨。"),
        (14.7, 19.5, "那是我姐姐的笔迹，写着今晚十一点半。"),
        (20.0, 24.5, "地下室的铁门半开着，里面传来规律的敲击声。"),
        (24.7, 29.5, "三长两短，是姐姐和我小时候的暗号。"),
        (30.0, 34.5, "我顺着台阶往下走，手电筒照见了一排病床。"),
        (34.7, 39.5, "每张床上都放着一台还在转动的老式录音机。"),
        (40.0, 44.5, "录音机里循环播放着同一句话：别相信她。"),
        (44.7, 49.5, "身后传来脚步声，和我刚才的节奏一模一样。"),
        (49.7, 54.0, "镜子里映出两个人，可病房里只有我一个。"),
        (54.2, 58.8, "登记簿上的名字，此刻变成了我自己的。"),
    ]
    return [TranscriptSegment(start=a, end=b, text=t) for a, b, t in lines]


def test_match_report_weights_and_compute():
    assert abs(sum(MatchReport.WEIGHTS.values()) - 1.0) < 1e-9
    report = MatchReport()
    report.visual_alignment = 100.0
    report.narrative_alignment = 100.0
    report.reference_style_match = 100.0
    report.temporal_alignment = 100.0
    report.fact_alignment = 100.0
    report.information_coverage = 100.0
    report.audio_tts_fit = 100.0
    report.story_completeness = 100.0
    assert report.compute() == 100.0
    data = report.as_dict()
    assert data["final_score"] == 100.0
    assert "weights" in data


def test_candidate_roundtrip():
    cand = make_candidate()
    cand.scripts = [ScriptVariant(
        variant_id="S1", candidate_id="C001", flavor="suspense",
        sentences=[ScriptSentence(index=1, text="第一句。", source_evidence=Evidence(start=10.0, end=12.0, quote="第一句。"),
                                  estimated_seconds=2.0)],
        full_text="第一句。", estimated_seconds=2.0)]
    data = to_dict(cand)
    restored = candidate_from_dict(json.loads(json.dumps(data, ensure_ascii=False)))
    assert restored.candidate_id == "C001"
    assert restored.evidence[0].start == 10.0
    assert restored.scripts[0].sentences[0].source_evidence.end == 12.0
    assert restored.scripts[0].flavor == "suspense"


def test_tokenize_and_speech_estimate():
    tokens = tokenize("深夜十一点")
    assert any(len(t) == 2 for t in tokens)
    seconds = estimate_speech_seconds("深夜十一点，我独自走进了废弃疗养院。")
    assert 2.0 < seconds < 8.0
    assert estimate_speech_seconds("") == 0.0


def test_statemachine_stages():
    assert len(STAGES) == len(set(STAGES)) == 18
    assert STAGES[0] == "CREATED" and STAGES[-1] == "COMPLETED"
    assert "VISUAL_ANALYSIS" in STAGES
    assert STAGES.index("VISUAL_ANALYSIS") == STAGES.index("MEDIA_INDEXING") + 1
    values = [stage_progress(s) for s in STAGES]
    assert values[0] == 0 and values[-1] == 100
    assert values == sorted(values)
    assert fmt_clock(65.0) == "00:01:05"


def test_transcript_parsers(tmp_path):
    srt = tmp_path / "a.srt"
    srt.write_text("1\n00:00:01,000 --> 00:00:03,500\n第一句话。\n\n2\n00:00:04,000 --> 00:00:06,000\n第二句话。\n",
                   encoding="utf-8")
    segments = transcript_mod.from_subtitle_file(str(srt))
    assert len(segments) == 2
    assert segments[0].start == 1.0 and segments[0].end == 3.5
    assert segments[1].text == "第二句话。"

    vtt = tmp_path / "a.vtt"
    vtt.write_text("WEBVTT\n\n00:01.000 --> 00:03.000\n你好世界\n", encoding="utf-8")
    segments = transcript_mod.from_subtitle_file(str(vtt))
    assert len(segments) == 1 and segments[0].text == "你好世界"

    json3 = tmp_path / "a.json3"
    json3.write_text(json.dumps({"events": [{"tStartMs": 2000, "dDurationMs": 1500,
                                             "segs": [{"utf8": "测试字幕"}]}]}), encoding="utf-8")
    segments = transcript_mod.from_subtitle_file(str(json3))
    assert len(segments) == 1 and segments[0].start == 2.0


def test_rule_variant_evidence_bound_and_monotonic():
    cand = make_candidate(start=10.0, end=45.0)
    transcript = make_transcript()
    profile = _default_suspense_profile()
    router = Config()
    from factory.providers import LLMRouter
    variants = write_scripts(cand, profile, transcript, Config().params, LLMRouter(router))
    assert len(variants) == 3
    for variant in variants:
        assert variant.full_text.strip()
        assert variant.sentences
        previous = None
        for sent in variant.sentences:
            ev = sent.source_evidence
            assert ev is not None
            assert 10.0 <= ev.start <= ev.end <= 45.0
            if previous is not None:
                assert ev.start >= previous - 1e-6
            previous = ev.start
        assert variant.estimated_seconds > 0
        assert 5.0 <= variant.estimated_seconds <= 60.0
    flavors = {v.flavor for v in variants}
    assert flavors == {"suspense", "conflict", "twist"}


def test_score_variant_and_gates():
    cand = make_candidate()
    transcript = make_transcript()
    profile = _default_suspense_profile()
    from factory.providers import LLMRouter
    router = LLMRouter(Config())
    variants = write_scripts(cand, profile, transcript, Config().params, router)
    for variant in variants:
        score_variant(cand, variant, profile, transcript, router=router)
        final = variant.match_report.get("final_score", 0.0)
        assert 0.0 < final <= 100.0
        for key in MatchReport.WEIGHTS:
            assert 0.0 <= variant.match_report.get(key, -1) <= 100.0
    best, score, passed = select_variant_best(variants, 0.0)
    assert best is not None and score > 0 and passed

    gate1, best_variant, passed = run_gates(cand, variants, profile, transcript,
                                            dict(Config().params, min_match=10.0), router)
    first_hard = gate1["attempts"][0].get("hard_gates", "pass")
    if first_hard != "pass":
        # 文档硬门槛（画面≥75/事实≥90/时间≥80/叙事≥80）未达时走人工复核
        assert not passed
        assert gate1["status"] == "review_required"
        assert cand.status == "review_required"
        assert first_hard and any("visual_alignment" in h for h in first_hard)
    else:
        assert passed and gate1["status"] == "pass"
        assert cand.status == "gate1_pass"
        assert cand.selected_script == best_variant.variant_id
    assert cand.match_report.get("gate1")["best_score"] == gate1["best_score"]


def test_gates_reject_low_threshold():
    cand = make_candidate()
    transcript = make_transcript()
    profile = _default_suspense_profile()
    from factory.providers import LLMRouter
    router = LLMRouter(Config())
    variants = write_scripts(cand, profile, transcript, Config().params, router)
    gate1, _best, passed = run_gates(cand, variants, profile, transcript,
                                     dict(Config().params, min_match=99.0), router)
    assert not passed
    assert gate1["status"] in ("reject", "review_required")
    assert cand.status in ("rejected_low_match", "review_required")
    assert cand.reject_reason


def test_shorten_variant_to_fit():
    from factory.models import Evidence, ScriptSentence, ScriptVariant
    from factory.render.timeline import shorten_variant_to_fit

    ev = Evidence(start=0.0, end=1.0, quote="x")
    sents = [ScriptSentence(index=i + 1, text="第%d句解说内容。" % i, source_evidence=ev,
                            estimated_seconds=4.0, role=("detail" if i == 2 else "story"))
             for i in range(6)]
    v = ScriptVariant(variant_id="V1", candidate_id="C001", flavor="suspense",
                      sentences=sents, full_text="".join(s.text for s in sents), estimated_seconds=24.0)
    # Length targets no longer delete approved narration; helper is advisory only.
    fit = shorten_variant_to_fit(v, 15.0)
    assert fit is v
    assert fit.estimated_seconds == 24.0
    assert len(fit.sentences) == 6
    assert shorten_variant_to_fit(v, 2.0) is v


def test_diversity_dedup():
    a = make_candidate("C001", 10.0, 40.0)
    b = make_candidate("C002", 10.5, 40.5)
    b.hook = a.hook
    b.conflict = a.conflict
    b.turning_point = a.turning_point
    b.payoff = a.payoff
    c = make_candidate("C003", 40.0, 60.0)
    c.story_summary = "另一段完全无关的监控画面，出现了第二个来访者。"
    c.hook = "监控里出现了第二个来访者。"
    c.conflict = "他和我走的是同一条路线。"
    c.turning_point = "原来他十年前也来过这里。"
    c.payoff = "录像带的日期是明天。"
    kept = filter_diversity([a, b, c], threshold=0.72)
    ids = {x.candidate_id for x in kept}
    assert "C001" in ids
    assert "C002" not in ids or "C001" not in ids
    assert "C003" in ids
    assert b.status.startswith("rejected") or a.status.startswith("rejected")


def test_subtitles_build(tmp_path):
    sentences = [
        {"start": 0.0, "end": 2.5, "text": "你敢相信，每张床上都放着一台还在转动的老式录音机。"},
        {"start": 2.5, "end": 4.0, "text": "别信她。"},
    ]
    params = {"target_width": 1080, "target_height": 1920, "subtitle_font": "Microsoft YaHei",
              "subtitle_fontsize": 64}
    ass_path = build_ass(sentences, params, tmp_path / "out.ass")
    content = ass_path.read_text(encoding="utf-8")
    assert "PlayResX: 1080" in content
    assert "PlayResY: 1920" in content
    assert content.count("Dialogue:") >= 3
    srt_path = build_srt(sentences, tmp_path / "out.srt")
    srt_content = srt_path.read_text(encoding="utf-8")
    assert "00:00:00,000 --> 00:00:02,500" in srt_content
    assert len(split_display_lines("一二三四五六七八九十一二三四五六七八九十")) == 2


def test_style_fingerprint():
    fp = style_fingerprint("短句先行。你敢看到最后吗？再给你三秒。")
    assert fp["chars_per_second"] > 0
    assert 0.0 <= fp["short_sentence_ratio"] <= 1.0


def test_config_merge_and_local_source(tmp_path):
    merged = deep_merge({"a": {"b": 1, "c": 2}}, {"a": {"b": 9}})
    assert merged["a"] == {"b": 9, "c": 2}
    config = Config()
    assert config.param("min_match") == 75.0
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"00")
    assert is_local_source(str(video))
    assert not is_local_source(str(tmp_path / "missing.mp4"))
    assert not is_local_source("https://www.youtube.com/watch?v=abc")
