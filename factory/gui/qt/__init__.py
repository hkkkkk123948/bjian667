import sys

from PySide6.QtWidgets import QApplication

from .main_window import MainWindow
from .theme import build_qss


def create_app(config_path=None):
    app = QApplication.instance() or QApplication(sys.argv)
    app.setStyleSheet(build_qss())
    window = MainWindow(config_path=config_path)
    return app, window


def launch_gui(config_path=None):
    app, window = create_app(config_path=config_path)
    window.show()
    return app.exec()
