import pytest
from factory.mining.visual_story import _sample_step,_valid_story_span


@pytest.mark.parametrize('duration,lo,hi',[(20,0,19.9),(45,5,40),(600,20,380),(1800,400,1200)])
def test_complete_short_animation_and_long_chapter_are_both_valid(duration,lo,hi):
    assert _valid_story_span(lo,hi,duration)


@pytest.mark.parametrize('span,count',[(20,48),(20,40),(240,40),(1200,48)])
def test_sampling_budget_is_bounded_and_short_animation_keeps_event_coverage(span,count):
    step = _sample_step(span,count)
    assert span/step <= count
    if span == 20:
        assert step <= .5


def test_invalid_or_trivial_source_windows_still_reject():
    assert not _valid_story_span(-1,20,30)
    assert not _valid_story_span(0,31,30)
    assert not _valid_story_span(0,2,20)
