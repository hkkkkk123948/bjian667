"""Editing-feel invariants: continuous motion, cuts, fill, and comment holds."""
from pathlib import Path

import numpy as np
import pytest

from factory.eval.judge import audit_timeline
from factory.matching.initial_plan import _frames_for_phrase, _merge_repeated_shots
from factory.matching.scene_align import _assign_matches
from factory.render.motion import (
    choose_motion, frame_motion, hold_spans, motion_amount, motion_filter, warp_frame,
)
from factory.render.motion_measure import edge_position, measure_warp, synthetic_edge
from factory.render.pacing import boundary_is_jump, fill_plan
from factory.render.transitions import transition_plan


ROOT = Path(__file__).resolve().parents[1]


def test_picture_graphs_do_not_use_integer_crop_filter():
    for name in ("factory/render/motion.py", "factory/render/exporter.py"):
        text = (ROOT / name).read_text(encoding="utf-8")
        assert "zoompan" not in text
    graph = motion_filter("push_in", 320, 180, 2, 48, oversample=2)
    assert "zoompan" not in graph
    assert "eval=frame" in graph and "n" in graph


def test_shipped_push_and_hold_are_continuous():
    pytest.importorskip("cv2")
    push = measure_warp(legacy_amount=False)
    assert push["reversals"] == 0
    assert push["end"] < push["start"] - 0.15
    assert push["max_step"] < 0.05
    source = synthetic_edge()
    spans = hold_spans(48, "push_in")
    held = []
    import cv2
    color = cv2.cvtColor(source, cv2.COLOR_GRAY2BGR)
    zooms = []
    for index in range(48):
        zoom, fraction = frame_motion("none", index, 48, 2.0, spans)
        zooms.append(zoom)
        held.append(edge_position(warp_frame(color, zoom, fraction)[:, :, 0]))
    assert zooms[0] == pytest.approx(1) and zooms[-1] == pytest.approx(1)
    assert max(zooms) > 1
    assert held[0] == pytest.approx(held[-1], abs=0.02)
    steps = np.diff(held)
    assert max(abs(steps)) < 0.05


def test_fill_keeps_the_action_and_refuses_a_slideshow():
    full = fill_plan(3.0, 2.0)
    assert full["src_used"] == pytest.approx(3.0) and full["freeze"] == 0 and full["interpolate"] is False
    mild = fill_plan(1.5, 3.0)
    assert mild["speed"] == pytest.approx(0.5) and mild["freeze"] == 0 and mild["interpolate"] is True
    severe = fill_plan(0.4, 3.0)
    assert severe["freeze"] > 0.5 and severe["speed"] >= 0.34
    assert severe["active"] + severe["freeze"] == pytest.approx(3.0)


def test_impacts_and_jump_cuts_are_hard_and_flashes_are_not_dissolves():
    roles = ["setup", "escalation", "payoff"]
    plan = transition_plan([2, 2, 2], roles=roles)
    assert [item["type"] for item in plan["boundaries"]] == ["cut", "fade"]
    assert plan["boundaries"][1]["duration_frames"] >= 3
    forced = transition_plan([2, 2, 2], force_hard=[0])
    assert forced["boundaries"][0]["type"] == "cut"
    assert forced["boundaries"][1]["type"] == "fade"
    short = transition_plan([0.08, 2], seconds=0.125)
    assert short["boundaries"][0]["duration_frames"] == 0
    left = {"src_start": 1, "src_duration": 1, "evidence_start": 0, "evidence_end": 4, "shot_id": "DOOR"}
    right = {"src_start": 2.1, "src_duration": 1, "evidence_start": 0, "evidence_end": 4, "shot_id": "DOOR"}
    assert boundary_is_jump(left, right) is True
    comment = dict(right, visual_kind="comment_still", shot_id="DOOR")
    assert boundary_is_jump(left, comment) is False


def test_comment_hold_stays_on_the_current_sentence():
    from factory.render.comment_visuals import apply_comment_picture
    segments = [
        {"sentence_index": 1, "out_start": 0, "dur": 4, "src_start": 10, "src_duration": 4,
         "evidence_start": 10, "evidence_end": 16, "motion": "push_in", "shot_id": "NOW"},
        {"sentence_index": 2, "out_start": 4, "dur": 4, "src_start": 30, "src_duration": 4,
         "evidence_start": 30, "evidence_end": 36, "motion": "push_in", "shot_id": "NEXT"},
    ]
    comment = {"sentence_index": 1, "text": "评论", "start": 2, "end": 3}
    out = apply_comment_picture(segments, [comment], "auto", None)
    still = next(seg for seg in out if seg.get("visual_kind") == "comment_still")
    assert still["sentence_index"] == 1
    assert still["evidence_start"] == 10 and still["src_start"] < 16
    assert still["shot_id"] == "NOW"
    assert still["motion"] == "push_in"
    assert all(seg["shot_id"] != "NEXT" or seg["out_start"] >= 3 for seg in out)


def test_borrowed_dwell_still_covers_every_spoken_line():
    segments = [
        {"sentence_index": 1, "out_start": 0, "dur": 2.5, "src_start": 0, "src_duration": 2,
         "evidence_start": 0, "evidence_end": 5},
        {"sentence_index": 2, "out_start": 2.5, "dur": 0.85, "src_start": 5, "src_duration": 0.8,
         "evidence_start": 5, "evidence_end": 8},
        {"sentence_index": 3, "out_start": 3.35, "dur": 2.775, "src_start": 8, "src_duration": 2,
         "evidence_start": 8, "evidence_end": 12},
    ]
    timings = [
        {"index": 1, "start": 0, "end": 3},
        {"index": 2, "start": 3, "end": 3.125},
        {"index": 3, "start": 3.125, "end": 6.125},
    ]
    assert audit_timeline({"segments": segments, "sentence_timings": timings})["issues"] == []
    gapped = [dict(segments[1], out_start=4.0)]
    assert any("空隙" in issue for issue in audit_timeline(
        {"segments": [segments[0], *gapped, segments[2]], "sentence_timings": timings})["issues"])


def test_same_angle_is_not_split_and_weak_jumps_reuse_a_strong_shot():
    assert _merge_repeated_shots([
        {"shot_id": "DOOR", "frame_ids": ["a"], "target_text": "门口"},
        {"shot_id": "DOOR", "frame_ids": ["b"], "target_text": "推门"},
        {"shot_id": "DOG", "frame_ids": ["c"], "target_text": "恶犬"},
    ]) == [
        {"shot_id": "DOOR", "frame_ids": ["a", "b"], "target_text": "门口"},
        {"shot_id": "DOG", "frame_ids": ["c"], "target_text": "恶犬"},
    ]
    shot = {"actions": [
        {"text": "恶犬", "frame_ids": ["short"]},
        {"text": "恶犬扑倒怪人", "frame_ids": ["long"]},
    ], "frames": [{"frame_id": "mid"}], "observations": []}
    assert _frames_for_phrase(shot, "恶犬扑倒怪人", {"short", "long", "mid"}) == ["long"]
    assert _assign_matches([[0.2, 0.92], [0.3, 0.88]]) == [1, 1]
    assert _assign_matches([[0.2, 0.92], [0.5, 0.99]]) == [1, 0]


def test_motion_stays_small_and_rare_on_short_shots():
    assert choose_motion("twist", 1, 1.2) == "none"
    assert choose_motion("hook", 1, 2) == "push_in"
    assert choose_motion("payoff", 3, 4) == "push_strong"
    assert motion_amount(4) <= 0.016
    assert motion_amount(4, kind="push_strong") <= 0.026
    assert choose_motion("setup", 16, 4) == "pan_right"
    assert choose_motion("setup", 8, 4) == "pan_left"
    assert choose_motion("setup", 1, 4) == "push_in"
