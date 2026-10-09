import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
pytest.importorskip("PySide6")
from PySide6.QtTest import QTest
from PySide6.QtGui import QTextCursor
from factory.gui.qt import create_app
from factory.config import Config
from factory.models import ScriptSentence, ScriptVariant, StoryCandidate
from factory.pipeline import ProductionPipeline
from factory.utils import read_json


def editor(tmp_path):
    config = Config({"workspace":str(tmp_path / "ws"),"params":{"review_mode":"semi"}})
    config_path = tmp_path / "config.json"
    config.save(config_path)
    app,window = create_app(config_path=str(config_path))
    p = ProductionPipeline(config,"fixture.mp4")
    v = ScriptVariant(variant_id="v",sentences=[ScriptSentence(index=1,text="女孩出门（忘带花）继续往前走(忘点灯)。")])
    p.candidates = [StoryCandidate(candidate_id="c",scripts=[v],selected_script="v",status="review_required")]
    p.state.status="SCRIPT_GENERATED"
    window.running=True
    page=window.pages['review']
    page.set_pipeline(p)
    return app,window,page,p


def test_actual_keyboard_input_and_draft_survive_background_refresh(tmp_path):
    app,window,page,p=editor(tmp_path)
    try:
        assert not page.script.isReadOnly()
        page.script.moveCursor(QTextCursor.MoveOperation.End)
        QTest.keyClicks(page.script," EDIT")
        app.processEvents()
        assert page.script.toPlainText().endswith(' EDIT')
        assert read_json(p.job_dir/'analysis/ui_drafts.json')['c'].endswith(' EDIT')
        page.refresh()
        assert page.script.toPlainText().endswith(' EDIT')
        window.running=False
        page.refresh()
        assert not page.script.isReadOnly() and page.script.toPlainText().endswith(' EDIT')
    finally:
        window.close()


def test_both_parenthesis_styles_are_highlighted(tmp_path):
    app,window,page,p=editor(tmp_path)
    try:
        page.highlighter.rehighlight()
        app.processEvents()
        spans=page.script.document().firstBlock().layout().formats()
        highlights=[f for f in spans if f.format.foreground().color().name()=='#ffd479']
        assert len(highlights)==2
        text=page.script.toPlainText()
        assert [text[f.start:f.start+f.length] for f in highlights]==['（忘带花）','(忘点灯)']
    finally:
        window.close()


def test_duration_updates_from_unsaved_text_including_comment_and_preserves_draft(tmp_path):
    app,window,page,p=editor(tmp_path)
    try:
        p.config.params.update(min_duration=35,max_duration=50)
        page.script.setPlainText("女孩出门（忘带玫瑰花）。")
        QTest.qWait(220)
        app.processEvents()
        assert "口播共 9 字（含评论话术）" in page.duration_label.text()
        assert "目标 35–50 秒" in page.duration_label.text()
        assert "暂无同设置样本" in page.duration_label.text()
        assert "明显过短" in page.duration_label.text()
        page.refresh()
        assert page.script.toPlainText() == "女孩出门（忘带玫瑰花）。"
        assert read_json(p.job_dir/'analysis/ui_drafts.json')['c'] == page.script.toPlainText()
    finally:
        window.close()


def test_long_user_copy_is_confirmable_but_unpaired_comment_needs_syntax_fix(tmp_path, monkeypatch):
    from PySide6.QtWidgets import QMessageBox
    warnings = []
    monkeypatch.setattr(QMessageBox, "warning", lambda *args: warnings.append(args[-1]))
    app,window,page,p=editor(tmp_path)
    try:
        window.running=False
        p.config.params.update(min_duration=35,max_duration=50)
        long_text = "女孩提着灯在走廊里寻找声音。" * 40
        page.script.setPlainText(long_text)
        page._review('approve')
        assert not warnings
        assert p.candidates[0].status == "approved"
        assert page.script.toPlainText().replace('\n', '') == long_text
        assert "明显超长" in page.duration_label.text()
        page.script.setPlainText("女孩出门（忘带玫瑰花。")
        page._review('approve')
        assert p.candidates[0].status == "approved"
        assert "括号未配对" in page.duration_label.text()
        assert read_json(p.job_dir/'analysis/ui_drafts.json')['c'] == page.script.toPlainText()
    finally:
        window.close()


def test_continue_after_stop_clears_cancel_only_when_worker_has_finished(tmp_path, monkeypatch):
    app,window,page,p=editor(tmp_path)
    try:
        window.pipeline=p
        p.cancel_requested=True
        calls=[]
        monkeypatch.setattr(window,"_spawn",lambda fn:calls.append(fn))
        window.running=True
        window.continue_job()
        assert p.cancel_requested and not calls
        window.running=False
        window.continue_job()
        assert not p.cancel_requested and calls == [p.run]
        p.cancel_requested=True
        window.produce_now()
        assert not p.cancel_requested and calls[-1] == p.produce
    finally:
        window.close()


def test_continue_with_unsaved_approved_draft_returns_to_review_and_preserves_it(tmp_path, monkeypatch):
    app,window,page,p=editor(tmp_path)
    try:
        window.pipeline=p
        window.running=False
        p.candidates[0].status="approved"
        draft="女孩独自出门（忘带玫瑰花），看见门外的陌生人。"
        page.script.setPlainText(draft)
        calls=[]
        monkeypatch.setattr(window,"_spawn",lambda fn:calls.append(fn))
        window.continue_job()
        assert not calls
        assert window.stack.currentWidget() is page
        assert page.script.toPlainText() == draft
        assert read_json(p.job_dir/'analysis/ui_drafts.json')['c'] == draft
        assert not page.review_notice.isHidden()
        assert "尚未继续制作" in window.status_label.text()
        assert "未保存修改" in page.review_notice.text()
        assert p.candidates[0].status == "approved"
    finally:
        window.close()


def test_confirm_current_draft_does_not_silently_produce_other_unsaved_approved_draft(tmp_path, monkeypatch):
    app,window,page,p=editor(tmp_path)
    try:
        window.pipeline=p
        window.running=False
        original=["女孩走进屋子。", "她发现门没有关。", "陌生人从门后出现。", "女孩立刻跑出房间。"]
        def candidate(identifier):
            variant=ScriptVariant(variant_id='v'+identifier,candidate_id=identifier,
                full_text=''.join(original),sentences=[ScriptSentence(index=i,text=t) for i,t in enumerate(original,1)])
            return StoryCandidate(candidate_id=identifier,scripts=[variant],selected_script=variant.variant_id,status='approved')
        p.candidates=[candidate('c'),candidate('d')]
        page.refresh()
        first='\n'.join(original).replace("走进", "悄悄走进")
        second='\n'.join(original).replace("陌生人", "怪人")
        page.script.setPlainText(first)
        page.tree.setCurrentItem(page.tree.topLevelItem(1))
        page.script.setPlainText(second)
        page.tree.setCurrentItem(page.tree.topLevelItem(0))
        calls=[]
        monkeypatch.setattr(window,"_spawn",lambda fn:calls.append(fn))
        page._produce()
        assert not calls
        assert window.stack.currentWidget() is page
        assert page._current_id == 'd' and page.script.toPlainText() == second
        assert '悄悄走进' in p._pick_variant(p.candidates[0]).full_text
        assert '怪人' not in p._pick_variant(p.candidates[1]).full_text
        assert read_json(p.job_dir/'analysis/ui_drafts.json') == {'d':second}
        assert not page.review_notice.isHidden()
    finally:
        window.close()


def test_reload_failed_job_marks_actual_failed_stage_without_marking_completed_job(tmp_path):
    from factory.gui.qt.theme import chip_qss
    app,window,page,p=editor(tmp_path)
    try:
        progress=window.pages['progress']
        p.state.status='FAILED'
        p.state.completed_stages=['TTS_GENERATED']
        p.state.artifacts['failed_stage']='MATCHING'
        progress.reset(p)
        assert progress.chips['MATCHING'].styleSheet() == chip_qss('failed')
        assert progress.chips['TTS_GENERATED'].styleSheet() == chip_qss('done')
        p.state.status='COMPLETED'
        progress.reset(p)
        assert progress.chips['MATCHING'].styleSheet() == chip_qss('pending')
    finally:
        window.close()
