from PySide6.QtCore import QRect, Qt
from PySide6.QtGui import QColor, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (QComboBox, QDialog, QHBoxLayout, QLabel, QPushButton,
                               QVBoxLayout, QWidget)

REGION_LABELS = [("blur", "模糊"), ("mosaic", "马赛克"), ("cover", "黑块遮挡"), ("delogo", "智能抹除")]


class RegionCanvas(QWidget):
    """显示一帧画面，鼠标拖拽画矩形；区域存归一化坐标 (0~1)。"""

    def __init__(self, image_path, regions=None):
        super().__init__()
        self.pixmap = QPixmap(str(image_path)) if image_path else QPixmap()
        self.regions = list(regions or [])
        self._start = None
        self._current = None
        self.setMinimumSize(520, 300)
        self.setMouseTracking(True)

    def _image_rect(self):
        if self.pixmap.isNull():
            return QRect(0, 0, 0, 0)
        aw, ah = self.width(), self.height()
        iw, ih = self.pixmap.width(), self.pixmap.height()
        scale = min(aw / iw, ah / ih)
        w, h = int(iw * scale), int(ih * scale)
        return QRect(int((aw - w) / 2), int((ah - h) / 2), w, h)

    def _to_widget(self, region):
        r = self._image_rect()
        return QRect(r.x() + int(region["x"] * r.width()), r.y() + int(region["y"] * r.height()),
                     int(region["w"] * r.width()), int(region["h"] * r.height()))

    def _to_norm(self, rect):
        r = self._image_rect()
        if r.width() <= 0 or r.height() <= 0:
            return None
        x = (rect.left() - r.x()) / r.width()
        y = (rect.top() - r.y()) / r.height()
        w = rect.width() / r.width()
        h = rect.height() / r.height()
        x = max(0.0, min(1.0, x))
        y = max(0.0, min(1.0, y))
        w = max(0.0, min(1.0 - x, w))
        h = max(0.0, min(1.0 - y, h))
        return {"x": round(x, 4), "y": round(y, 4), "w": round(w, 4), "h": round(h, 4)}

    def clear(self):
        self.regions = []
        self.update()

    def pop(self):
        if self.regions:
            self.regions.pop()
            self.update()

    def paintEvent(self, _event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#0b1119"))
        r = self._image_rect()
        if not self.pixmap.isNull() and r.width() > 0:
            painter.drawPixmap(r, self.pixmap)
        pen = QPen(QColor("#20d4ee"), 2)
        painter.setPen(pen)
        for region in self.regions:
            rect = self._to_widget(region)
            painter.fillRect(rect, QColor(32, 212, 238, 60))
            painter.drawRect(rect)
        if self._current is not None:
            pen = QPen(QColor("#ffd54f"), 2, Qt.PenStyle.DashLine)
            painter.setPen(pen)
            painter.drawRect(self._current)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.RightButton:
            pos = event.position().toPoint()
            for i in range(len(self.regions) - 1, -1, -1):
                if self._to_widget(self.regions[i]).contains(pos):
                    self.regions.pop(i)
                    self.update()
                    break
            return
        self._start = event.position().toPoint()
        self._current = QRect(self._start, self._start)

    def mouseMoveEvent(self, event):
        if self._start is not None:
            self._current = QRect(self._start, event.position().toPoint()).normalized()
            self.update()

    def mouseReleaseEvent(self, event):
        if self._start is None:
            return
        rect = QRect(self._start, event.position().toPoint()).normalized()
        self._start = None
        self._current = None
        r = self._image_rect()
        if rect.width() > 6 and rect.height() > 6 and r.intersects(rect):
            region = self._to_norm(rect.intersected(r))
            if region and region["w"] > 0.01 and region["h"] > 0.01:
                self.regions.append(region)
        self.update()


class RegionDialog(QDialog):
    def __init__(self, parent, image_path, regions=None, method="blur"):
        super().__init__(parent)
        self.setWindowTitle("框选要去除的区域")
        self.resize(760, 560)
        self.regions = list(regions or [])
        self.method = method

        lay = QVBoxLayout(self)
        tip = QLabel("在画面上按住鼠标拖拽即可框选要去除的字幕/水印区域；可框多处；右键点区域可删除。")
        tip.setObjectName("hint")
        lay.addWidget(tip)
        self.canvas = RegionCanvas(image_path, self.regions)
        lay.addWidget(self.canvas, 1)

        row = QHBoxLayout()
        row.addWidget(QLabel("处理方式"))
        self.method_combo = QComboBox()
        for key, label in REGION_LABELS:
            self.method_combo.addItem(label, key)
        idx = self.method_combo.findData(method)
        self.method_combo.setCurrentIndex(idx if idx >= 0 else 0)
        row.addWidget(self.method_combo)
        clear_btn = QPushButton("清空")
        clear_btn.clicked.connect(self.canvas.clear)
        pop_btn = QPushButton("删除上一个")
        pop_btn.clicked.connect(self.canvas.pop)
        ok_btn = QPushButton("应用并重渲染")
        ok_btn.setObjectName("accent")
        ok_btn.clicked.connect(self._accept)
        cancel_btn = QPushButton("取消")
        cancel_btn.clicked.connect(self.reject)
        row.addWidget(pop_btn)
        row.addWidget(clear_btn)
        row.addStretch(1)
        row.addWidget(cancel_btn)
        row.addWidget(ok_btn)
        lay.addLayout(row)

    def _accept(self):
        self.regions = list(self.canvas.regions)
        self.method = self.method_combo.currentData() or "blur"
        self.accept()
