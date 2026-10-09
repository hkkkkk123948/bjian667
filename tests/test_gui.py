import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PySide6")

from factory.gui.qt import create_app  # noqa: E402


@pytest.fixture(scope="module")
def qt():
    app, win = create_app()
    win.show()
    app.processEvents()
    yield app, win
    win.close()
    app.processEvents()


def test_window_structure(qt):
    _app, win = qt
    assert win.stack.count() == 5
    assert len(win.nav_buttons) == 5
    for key in ("task", "progress", "review", "config", "output"):
        assert key in win.pages


def test_task_history_loaded(qt):
    _app, win = qt
    win.pages["task"].refresh_history()
    assert win.pages["task"].history.topLevelItemCount() >= 1


def test_config_widgets_populated(qt):
    _app, win = qt
    config = win.pages["config"]
    first_role = config.role_widgets[next(iter(config.role_widgets))]
    assert first_role["type"].currentText() in ("mock", "gemini", "openai", "custom_http")
    assert config.tts_type.currentText() in ("edge", "mock", "bailian", "elevenlabs", "openai", "gemini")
    assert "min_duration" in config.param_widgets
