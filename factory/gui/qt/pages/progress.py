from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (QGridLayout, QHBoxLayout, QLabel, QPlainTextEdit, QProgressBar,
                               QPushButton, QVBoxLayout, QWidget)

from ....statemachine import STAGES, stage_label, stage_progress
from ....utils import open_path
from ..theme import LOG_ERROR, LOG_INFO, LOG_WARN, chip_qss
from .common import section_title
import html


class ProgressPage(QWidget):
    def __init__(self, main):
        super().__init__()
        self.main = main
        self.chips = {}

        root = QVBoxLayout(self)
        root.setContentsMargins(24, 20, 24, 20)
        root.setSpacing(14)

        head = QHBoxLayout()
        head.addWidget(section_title("生产进度"))
        head.addStretch(1)
        self.job_label = QLabel("未开始")
        self.job_label.setObjectName("muted")
        head.addWidget(self.job_label)
        root.addLayout(head)

        grid = QGridLayout()
        grid.setSpacing(6)
        visible_stages = [stage for stage in STAGES if stage != 'FINAL_MATCH_CHECK']
        for i, stage in enumerate(visible_stages):
            chip = QLabel(stage_label(stage))
            chip.setStyleSheet(chip_qss("pending"))
            grid.addWidget(chip, i // 4, i % 4)
            self.chips[stage] = chip
        for col in range(4):
            grid.setColumnStretch(col, 1)
        root.addLayout(grid)

        bar_row = QHBoxLayout()
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setFormat("%v%")
        self.pct_label = QLabel("0%")
        self.pct_label.setMinimumWidth(44)
        self.pct_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        bar_row.addWidget(self.progress, 1)
        bar_row.addWidget(self.pct_label)
        root.addLayout(bar_row)

        self.detail_label = QLabel("")
        self.detail_label.setObjectName("muted")
        root.addWidget(self.detail_label)

        btn_row = QHBoxLayout()
        stop_btn = QPushButton("停止")
        stop_btn.setObjectName("danger")
        stop_btn.clicked.connect(self.main.request_cancel)
        cont_btn = QPushButton("继续执行")
        cont_btn.clicked.connect(self.main.continue_job)
        open_btn = QPushButton("打开任务目录")
        open_btn.clicked.connect(self._open_job_dir)
        clear_btn = QPushButton("清空日志")
        clear_btn.clicked.connect(self._clear_log)
        btn_row.addWidget(stop_btn)
        btn_row.addWidget(cont_btn)
        btn_row.addWidget(open_btn)
        btn_row.addWidget(clear_btn)
        btn_row.addStretch(1)
        root.addLayout(btn_row)

        self.log_text = QPlainTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setFont(QFont("Consolas", 9))
        self.log_text.setStyleSheet("background:#101318; border:1px solid #2c3342; border-radius:6px;")
        root.addWidget(self.log_text, 1)

    def reset(self, pipeline):
        self.job_label.setText("任务 %s" % pipeline.job_id)
        completed = set(pipeline.state.completed_stages)
        for stage in STAGES:
            self._paint(stage, "done" if stage in completed else "pending")
        failed_stage = pipeline.state.artifacts.get("failed_stage")
        if pipeline.state.status in ("FAILED", "CANCELLED") and failed_stage in STAGES:
            self._paint(failed_stage, "failed")
        current = pipeline.state.status
        if current in ('FAILED','CANCELLED'):
            current = failed_stage
        elif current == 'REVIEW_REQUIRED':
            current = pipeline.state.artifacts.get('review_reason_stage', 'QUALITY_CHECK')
            self._paint(current,'blocked')
        if current in STAGES:
            value = round(stage_progress(current))
            self.progress.setValue(value)
            self.pct_label.setText('%d%%' % value)
            self.detail_label.setText(stage_label(current))
            if pipeline.state.status in STAGES and current not in completed:
                self._paint(current,'active')
        else:
            self.progress.setValue(0)
            self.pct_label.setText('0%')
            self.detail_label.setText('')
        self.append_log("任务 %s 已创建/载入，状态 %s" % (pipeline.job_id, pipeline.state.status))

    def _paint(self, stage, state):
        chip = self.chips.get(stage)
        if chip:
            chip.setStyleSheet(chip_qss(state))

    def on_event(self, event):
        kind = event.get("kind")
        if kind == "stage":
            status = event.get("status")
            stage = event.get("stage", "")
            if stage in STAGES:
                if status == "start":
                    self._paint(stage, "active")
                elif status == "done":
                    self._paint(stage, "done")
                elif status == "failed":
                    self._paint(stage, "failed")
                elif status == "blocked":
                    self._paint(stage, "blocked")
            detail = event.get("detail", "")
            if status == "start":
                self.append_log("▶ %s %s" % (stage, detail))
            elif status == "done":
                self.append_log("✓ %s %s" % (stage, detail))
            elif status in ("failed", "blocked"):
                self.append_log("✗ %s %s" % (stage, detail), "error")
        elif kind == "progress":
            value = max(0.0, min(100.0, float(event.get("value", 0))))
            self.progress.setValue(int(value))
            self.pct_label.setText("%d%%" % int(value))
            detail = event.get("detail", "")
            if detail:
                self.detail_label.setText(detail)
        elif kind == "log":
            self.append_log(event.get("message", ""), event.get("level", "info"))

    def append_log(self, message, level="info"):
        color = {"info": LOG_INFO, "warning": LOG_WARN, "error": LOG_ERROR}.get(level, LOG_INFO)
        text = html.escape(str(message)).replace(" ", "&nbsp;")
        self.log_text.appendHtml('<span style="color:%s">%s</span>' % (color, text))

    def _clear_log(self):
        self.log_text.clear()

    def _open_job_dir(self):
        if self.main.pipeline:
            open_path(self.main.pipeline.job_dir)
