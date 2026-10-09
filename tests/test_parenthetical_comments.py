import copy

import pytest

from factory.models import to_dict, variant_from_dict
from factory.script.comment_hook import story_text, strip_hook, comment_phrases
from factory.script.review_edit import parse_review, edit_variant, review_text
from test_review_edit import fixture


def test_inline_chinese_and_english_parentheses_are_comments(tmp_path):
    p,c,v = fixture(tmp_path)
    text = "女孩（忘记带玫瑰花了）发现门外一直有人。\n她赶紧躲到垃圾箱后面(男孩忘记点灯了)。\n怪人却从旁边的门缝探出头。\n她向男人求助后等来了警察。"
    fresh,changed = edit_variant(p,v,text)
    assert not changed
    assert story_text(fresh.sentences[0]) == "女孩发现门外一直有人。"
    assert fresh.sentences[0].text.index("忘记带玫瑰花了") < fresh.sentences[0].text.index("发现门外")
    assert "忘记带玫瑰花了" in fresh.full_text and "男孩忘记点灯了" in fresh.full_text
    assert not any(ch in fresh.full_text for ch in "（）()")
    assert fresh.scene_matches == v.scene_matches
    restored = variant_from_dict(to_dict(fresh))
    assert restored == fresh
    assert "（忘记带玫瑰花了）" in review_text(restored)
    body = copy.deepcopy(fresh)
    strip_hook(body)
    assert all(not s.comment_hooks for s in body.sentences)
    assert "玫瑰花" not in body.full_text and "点灯" not in body.full_text


def test_multiple_comments_same_sentence_standalone_and_legacy_markers():
    body,hooks = parse_review("女孩（玫瑰花）出门(点灯)。\n（另一句插话）\n【评论话术｜不参与匹配】旧格式兼容\n她看见怪人。")
    assert body == ["女孩出门。","她看见怪人。"]
    assert [h['text'] for h in hooks] == ['玫瑰花','点灯','另一句插话','旧格式兼容']
    assert [h['offset'] for h in hooks[:2]] == [2,4]


@pytest.mark.parametrize('text',["女孩（忘带花。", "女孩忘带花）。", "女孩（花(灯)）出门。", "女孩（忘带花)出门。"])
def test_invalid_or_nested_parentheses_have_clear_error(text):
    with pytest.raises(ValueError,match="括号"):
        parse_review(text)


def test_full_paragraph_parentheses_keep_position_after_sentence_split():
    body,hooks = parse_review("女孩出门。（忘带花）她看见怪人(没点灯)，赶紧跑开。")
    assert body == ["女孩出门。","她看见怪人，赶紧跑开。"]
    assert hooks[0]['after_sentence']==1 and hooks[1]['after_sentence']==2


def test_two_comment_phrases_are_both_retained_exactly_once(tmp_path):
    p,c,v = fixture(tmp_path)
    base = "\n".join(story_text(s) for s in v.sentences)
    base = base.replace('女孩发现','女孩（第一句插话）（第二句插话）发现')
    fresh,_ = edit_variant(p,v,base)
    assert comment_phrases(fresh.sentences[0]) == ['第一句插话','第二句插话']
    assert fresh.full_text.count('第一句插话') == fresh.full_text.count('第二句插话') == 1


def test_parentheses_comments_are_in_one_intact_tts_submission(tmp_path):
    import numpy as np
    from factory.providers.tts.base import write_wav
    from factory.providers.tts.continuous import synth_continuous
    from factory.script.comment_hook import apply_comment_hook
    p,c,v = fixture(tmp_path)
    doc="\n".join(story_text(s) for s in v.sentences).replace('女孩发现','女孩（没带花）发现').replace('她赶紧','她赶紧(忘点灯)')
    fresh,_=edit_variant(p,v,doc)
    strip_hook(fresh)
    apply_comment_hook(p,fresh)

    class Recorder:
        config=p.config
        calls=[]

        def synth(self,text,out_path,**kwargs):
            self.calls.append((text,kwargs['speed']))
            write_wav(out_path,.1*np.sin(np.arange(24000)*.1),24000)
            return {'audio_path':str(out_path),'provider':'test_fixture','duration':1}

    tts=Recorder()
    result=synth_continuous(tts,fresh,tmp_path/'clip')
    assert len(tts.calls)==1 and tts.calls[0][1]==1.1
    assert tts.calls[0][0].count('没带花')==tts.calls[0][0].count('忘点灯')==1
    assert not any(ch in tts.calls[0][0] for ch in '（）()')
    assert result['audio_edits']==[] and result['voice_sha256']==result['original_audio_sha256']
