import os
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

from factory.config import Config
from factory.pipeline import ProductionPipeline


def test_new_configuration_defaults_to_all_qualified_stories():
    assert Config().param("max_clips") == 0
    assert Config({"params": {"max_clips": 3}}).param("max_clips") == 3


@pytest.mark.parametrize("saved_limit", [0, 1, 3, 15])
def test_resuming_saved_task_retains_its_story_limit(tmp_path, saved_limit):
    original = Config({"workspace": str(tmp_path / "ws"), "params": {"max_clips": saved_limit}})
    task = ProductionPipeline(original, "source.mp4")
    task.save_state()
    current = Config({"workspace": str(tmp_path / "ws")})
    resumed = ProductionPipeline(current, "", job_id=task.job_id)
    assert resumed.config.param("max_clips") == saved_limit
    assert current.param("max_clips") == 0


@pytest.fixture
def app():
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


def task_page(tmp_path, limit=None):
    from factory.gui.qt.pages.task import TaskPage
    params = {} if limit is None else {"max_clips": limit}
    config = Config({"workspace": str(tmp_path / "ws"), "params": params}, tmp_path / "config.json")
    calls = []
    main = SimpleNamespace(config_obj=config, request_cancel=lambda: None,
                           start_job=lambda url, refs: calls.append((url, refs)), resume_job=lambda _: None)
    return TaskPage(main), config, calls


def test_new_task_displays_automatic_story_count_and_explains_manual_limit(app, tmp_path):
    from PySide6.QtWidgets import QLabel
    page, _, _ = task_page(tmp_path)
    try:
        assert page.spin_max_clips.value() == 0
        assert page.spin_max_clips.text() == "自动：全部合格故事"
        hints = " ".join(label.text() for label in page.findChildren(QLabel))
        assert "完整、不重复" in hints and "质量不足" in hints
        assert "填 1 只做 1 条" in hints and "配音前等待确认" in hints
    finally:
        page.close()


@pytest.mark.parametrize("limit", [0, 1, 15])
def test_task_submission_preserves_zero_auto_and_explicit_manual_limits(app, tmp_path, limit):
    page, config, calls = task_page(tmp_path, limit)
    try:
        assert page.spin_max_clips.value() == limit
        page.url_edit.setText("https://www.youtube.com/watch?v=fixture")
        page._start()
        assert config.param("max_clips") == limit
        assert Config.load(config.path).param("max_clips") == limit
        assert calls == [("https://www.youtube.com/watch?v=fixture", [])]
    finally:
        page.close()


def test_configuration_page_marks_zero_as_automatic(app, tmp_path):
    from PySide6.QtWidgets import QLabel
    from factory.gui.qt.pages.config_page import ConfigPage
    config = Config({"workspace": str(tmp_path / "ws")})
    page = ConfigPage(SimpleNamespace(config_obj=config))
    try:
        widget, _ = page.param_widgets["max_clips"]
        assert widget.text() == "0"
        assert "全部合格故事" in widget.toolTip()
        assert "成片数量（0 自动）" in [label.text() for label in page.findChildren(QLabel)]
    finally:
        page.close()


@pytest.mark.parametrize("invalid", ["-1", "1.5"])
def test_configuration_page_rejects_invalid_count_without_replacing_auto(app, tmp_path, monkeypatch, invalid):
    from PySide6.QtWidgets import QMessageBox
    from factory.gui.qt.pages.config_page import ConfigPage
    config = Config({"workspace": str(tmp_path / "ws")}, tmp_path / "config.json")
    warnings = []
    monkeypatch.setattr(QMessageBox, "warning", lambda *args: warnings.append(args[-1]))
    page = ConfigPage(SimpleNamespace(config_obj=config))
    try:
        page.param_widgets["max_clips"][0].setText(invalid)
        page._save()
        assert warnings and "max_clips" in warnings[0]
        assert config.param("max_clips") == 0
        assert not config.path.exists()
    finally:
        page.close()


def test_review_distinguishes_discovery_from_scripts_and_user_approval(app, tmp_path):
    from factory.gui.qt.pages.review import ReviewPage
    from factory.models import ScriptSentence, ScriptVariant, StoryCandidate
    config = Config({"workspace": str(tmp_path / "ws")})
    pipeline = ProductionPipeline(config, "source.mp4")
    pipeline.candidates = [StoryCandidate(candidate_id=str(index), status=status,
        scripts=[ScriptVariant(variant_id=str(index), sentences=[ScriptSentence(index=1, text="女孩走出房间。")])])
        for index, status in enumerate(("approved", "review_required", "rejected"))]
    pipeline.state.artifacts["story_discovery"] = {"mode": "auto", "discovered": 8,
                                                  "selected": 4, "complete": False}
    pipeline.state.artifacts["pending_stories"] = [{"title": "待生成甲"}, {"title": "待生成乙"}]
    page = ReviewPage(SimpleNamespace(running=False, continue_job=lambda: None))
    try:
        page.set_pipeline(pipeline)
        label = page.queue_label.text()
        assert "发现 8 个故事" in label
        assert "已生成文案 3" in label and "待审 1" in label and "已确认 1" in label
        assert "待生成 2" in label and "故事分析待继续" in label
        assert "合格成片" not in label and "已通过" not in label
        assert not page.resume_scripts.isHidden() and "继续分析故事" in page.resume_scripts.text()
    finally:
        page.close()


def test_review_does_not_display_old_story_criticism_or_scores_as_barriers(app, tmp_path):
    from factory.gui.qt.pages.review import ReviewPage
    from factory.models import ScriptSentence, ScriptVariant, StoryCandidate
    config = Config({"workspace": str(tmp_path / "ws")})
    pipeline = ProductionPipeline(config, "source.mp4")
    pipeline.candidates = [StoryCandidate(candidate_id="c", status="review_required", story_score=12,
        final_match=18, reject_reason="旧文案预检拒绝原因", match_report={"story_grounding": {
            "status": "fail", "issues": [{"index": 1, "reason": "旧剧情因果不足"}]}, "story_completeness": 10},
        scripts=[ScriptVariant(variant_id="v", sentences=[ScriptSentence(index=1, text="女孩走出房间。")])])]
    page = ReviewPage(SimpleNamespace(running=False, continue_job=lambda: None))
    try:
        page.set_pipeline(pipeline)
        detail = page.detail.toPlainText()
        assert "旧文案预检" not in detail and "旧剧情因果" not in detail
        assert "故事画面依据" not in detail and "18.0" not in detail
        assert "确认后直接配音与匹配" in detail
        assert page.tree.headerItem().text(1) == "文案准备"
        assert page.tree.headerItem().text(2) == "你的确认"
        assert page.tree.topLevelItem(0).text(1) == "已生成"
        assert page.tree.topLevelItem(0).text(2) == "待确认"
    finally:
        page.close()
