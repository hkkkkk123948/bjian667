from PySide6.QtWidgets import QFrame, QLabel, QVBoxLayout


def card():
    frame = QFrame()
    frame.setObjectName("card")
    lay = QVBoxLayout(frame)
    lay.setContentsMargins(20, 18, 20, 18)
    lay.setSpacing(12)
    return frame, lay


def section_title(text):
    label = QLabel(text)
    label.setObjectName("section")
    return label


def hint_label(text):
    label = QLabel(text)
    label.setObjectName("hint")
    label.setWordWrap(True)
    return label
