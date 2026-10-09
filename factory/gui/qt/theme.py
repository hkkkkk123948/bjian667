from pathlib import Path

BG = "#070d16"
PANEL = "#0f1927"
CARD = "#0f1927"
CARD2 = "#141f30"
BORDER = "#24364c"
TEXT = "#edf5ff"
MUTED = "#8da0b8"
ACCENT = "#20d4ee"
ACCENT_HOVER = "#45dcf0"
ACCENT_DOWN = "#16b6cf"
ACCENT_TEXT = "#04121d"
DANGER = "#ff5f63"
WARN = "#f5a524"
SUCCESS = "#30d77e"
LOG_INFO = "#c8d6e5"
LOG_WARN = "#f5c877"
LOG_ERROR = "#ff8a8e"

STAGE_COLORS = {
    "done": ("#0e3b2a", "#7df0b0"),
    "active": ("#0a3d4c", "#7be6f7"),
    "pending": ("#16283a", "#7f93ab"),
    "failed": ("#3a1722", "#ff9aa3"),
    "blocked": ("#3a1722", "#ff9aa3"),
}

BODY_QSS = """
QMainWindow, QWidget#root { background: %(bg)s; color: %(text)s; }
QWidget { font-family: "Microsoft YaHei UI"; font-size: 13px; color: %(text)s; }

QToolTip { background: %(card2)s; color: %(text)s; border: 1px solid %(border)s; padding: 4px; }

QDialog, QMessageBox { background: %(card)s; color: %(text)s; }
QMessageBox { background: %(card)s; }
QMessageBox QLabel { color: %(text)s; background: transparent; }
QMessageBox QPushButton { min-width: 84px; padding: 6px 14px; }
QDialog QLabel { color: %(text)s; background: transparent; }

QLabel { color: %(text)s; background: transparent; }
QLabel#muted { color: %(muted)s; }
QLabel#title { font-size: 19px; font-weight: 600; letter-spacing: 1px; }
QLabel#section { font-size: 15px; font-weight: 600; color: %(accent)s; }
QLabel#hint { color: %(muted)s; font-size: 12px; }

QLineEdit, QTextEdit, QPlainTextEdit, QSpinBox, QDoubleSpinBox, QComboBox {
    background: %(bg)s;
    border: 1px solid %(border)s;
    border-radius: 6px;
    padding: 6px 10px;
    color: %(text)s;
    selection-background-color: %(accent)s;
    selection-color: %(accent_text)s;
}
QLineEdit:focus, QTextEdit:focus, QPlainTextEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {
    border: 1px solid %(accent)s;
}
QLineEdit:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled { color: %(muted)s; background: %(card2)s; }
QComboBox::drop-down { border: none; width: 24px; }
QComboBox::down-arrow { image: url("%(arrow)s"); width: 12px; height: 8px; }
QComboBox QAbstractItemView {
    background: %(card2)s;
    border: 1px solid %(border)s;
    selection-background-color: %(accent)s;
    selection-color: %(accent_text)s;
    color: %(text)s;
    outline: none;
}

QPushButton {
    background: #172538;
    border: 1px solid #30465f;
    border-radius: 7px;
    padding: 7px 16px;
    color: %(text)s;
}
QPushButton:hover { background: #1d2f45; border-color: %(accent)s; }
QPushButton:pressed { background: #14202f; }
QPushButton:disabled { color: %(muted)s; background: %(card)s; border-color: %(border)s; }

QPushButton#accent {
    background: %(accent)s; color: %(accent_text)s; border: 1px solid %(accent_hover)s; font-weight: 700;
}
QPushButton#accent:hover { background: %(accent_hover)s; }
QPushButton#accent:pressed { background: %(accent_down)s; }
QPushButton#accent:disabled { background: #12414f; color: #8fb9c2; border-color: #12414f; }

QPushButton#danger { background: #321923; border: 1px solid #8d3841; color: #ff9194; }
QPushButton#danger:hover { background: %(danger)s; color: #ffffff; border-color: %(danger)s; }

QPushButton#nav {
    text-align: left;
    padding: 11px 16px;
    border: 1px solid transparent;
    border-radius: 8px;
    background: transparent;
    color: #9daec1;
    font-size: 14px;
}
QPushButton#nav:hover { background: #111d2b; color: #dce7f3; }
QPushButton#nav:checked { background: #0a3443; border-color: #155a79; color: %(accent)s; font-weight: 700; }

QFrame#card {
    background: %(card)s;
    border: 1px solid %(border)s;
    border-radius: 10px;
}
QFrame#sidebar { background: #080e18; border-right: 1px solid %(border)s; }
QLabel#statusbar {
    background: #0c1521;
    border-top: 1px solid %(border)s;
    color: %(muted)s;
    padding: 8px 14px;
}

QTreeWidget {
    background: %(bg)s;
    border: 1px solid %(border)s;
    border-radius: 8px;
    outline: none;
    alternate-background-color: #0b1421;
}
QTreeWidget::item { height: 30px; padding-left: 6px; }
QTreeWidget::item:hover { background: #122234; }
QTreeWidget::item:selected { background: %(accent)s; color: %(accent_text)s; }

QListWidget, QListView, QTableWidget {
    background: %(bg)s;
    border: 1px solid %(border)s;
    border-radius: 8px;
    color: %(text)s;
    outline: none;
    alternate-background-color: #0b1421;
}
QListWidget::item, QListView::item, QTableWidget::item { padding: 4px 6px; }
QListWidget::item:hover, QListView::item:hover { background: #122234; }
QListWidget::item:selected, QListView::item:selected, QTableWidget::item:selected {
    background: %(accent)s; color: %(accent_text)s;
}

QMenu { background: %(card2)s; color: %(text)s; border: 1px solid %(border)s; }
QMenu::item { padding: 5px 18px; }
QMenu::item:selected { background: %(accent)s; color: %(accent_text)s; }

QHeaderView::section {
    background: #0f1c2b;
    color: %(muted)s;
    border: none;
    border-bottom: 1px solid %(border)s;
    padding: 7px 6px;
    font-weight: 600;
}

QProgressBar {
    background: %(bg)s;
    border: 1px solid %(border)s;
    border-radius: 6px;
    text-align: center;
    color: %(text)s;
    height: 18px;
}
QProgressBar::chunk { background: %(accent)s; border-radius: 5px; }

QScrollArea { border: none; background: transparent; }
QScrollBar:vertical { background: transparent; width: 10px; margin: 0; }
QScrollBar::handle:vertical { background: %(border)s; border-radius: 5px; min-height: 30px; }
QScrollBar::handle:vertical:hover { background: #31506b; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical { background: transparent; }
QScrollBar:horizontal { background: transparent; height: 10px; }
QScrollBar::handle:horizontal { background: %(border)s; border-radius: 5px; }
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal { width: 0; }

QRadioButton, QCheckBox { spacing: 8px; }
QRadioButton::indicator, QCheckBox::indicator { width: 16px; height: 16px; }
QRadioButton::indicator:unchecked, QCheckBox::indicator:unchecked {
    border: 1px solid %(border)s; background: %(bg)s; border-radius: 8px;
}
QRadioButton::indicator:checked {
    border: 2px solid %(accent)s; background: %(accent)s; border-radius: 8px;
}

QGroupBox {
    border: 1px solid %(border)s;
    border-radius: 8px;
    margin-top: 12px;
    padding-top: 10px;
}
QGroupBox::title { subcontrol-origin: margin; left: 12px; padding: 0 6px; color: %(accent)s; }
"""


def _ensure_arrow():
    path = Path(__file__).with_name("_arrow.png")
    if path.exists():
        return path
    try:
        from PIL import Image, ImageDraw

        img = Image.new("RGBA", (12, 8), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        draw.polygon([(1, 2), (11, 2), (6, 7)], fill=(141, 160, 184, 255))
        img.save(path)
    except Exception:
        pass
    return path


def build_qss():
    arrow = _ensure_arrow().as_posix()
    return BODY_QSS % {
        "bg": BG, "text": TEXT, "muted": MUTED, "card": CARD, "card2": CARD2,
        "border": BORDER, "accent": ACCENT, "accent_hover": ACCENT_HOVER,
        "accent_down": ACCENT_DOWN, "accent_text": ACCENT_TEXT, "danger": DANGER,
        "arrow": arrow,
    }


def chip_qss(state):
    bg, fg = STAGE_COLORS.get(state, STAGE_COLORS["pending"])
    return (
        "QLabel { background: %s; color: %s; border-radius: 5px; padding: 6px 8px;"
        " font-size: 12px; font-weight: 600; }" % (bg, fg))
