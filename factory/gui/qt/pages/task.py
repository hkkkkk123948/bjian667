import json
from pathlib import Path

from PySide6.QtWidgets import (QButtonGroup, QComboBox, QDoubleSpinBox, QFileDialog, QHBoxLayout, QLabel,
                               QLineEdit, QMessageBox, QPlainTextEdit, QPushButton, QRadioButton,
                               QSpinBox, QTreeWidget, QTreeWidgetItem, QVBoxLayout, QWidget)

from .common import card, hint_label, section_title

HISTORY_LIMIT = 200


class TaskPage(QWidget):
    def __init__(self, main):
        super().__init__()
        self.main = main

        root = QVBoxLayout(self)
        root.setContentsMargins(24, 20, 24, 20)
        root.setSpacing(16)

        title = QLabel("新建生产任务")
        title.setObjectName("title")
        root.addWidget(title)

        form, form_lay = card()

        src_row = QHBoxLayout()
        src_row.setSpacing(8)
        src_label = QLabel("视频来源")
        src_label.setMinimumWidth(70)
        self.url_edit = QLineEdit()
        self.url_edit.setPlaceholderText("YouTube 链接或本地视频路径，例如 D:\\videos\\source.mp4")
        browse_btn = QPushButton("浏览本地…")
        browse_btn.clicked.connect(self._browse_video)
        src_row.addWidget(src_label)
        src_row.addWidget(self.url_edit, 1)
        src_row.addWidget(browse_btn)
        form_lay.addLayout(src_row)
        form_lay.addWidget(hint_label(
            "支持 YouTube 链接或本地 mp4/mkv/mov/webm；本地视频放同名 .srt/.vtt/.ass/.json3 字幕可自动识别"))

        ref_label_row = QHBoxLayout()
        ref_label_row.addWidget(QLabel("AI 仿写示例（风格源，直接粘贴文本）"))
        ref_label_row.addStretch(1)
        load_example_btn = QPushButton("从文件载入到文本框")
        load_example_btn.clicked.connect(self._load_example_file)
        clear_example_btn = QPushButton("清空")
        clear_example_btn.clicked.connect(lambda: self.ref_text.clear())
        ref_label_row.addWidget(load_example_btn)
        ref_label_row.addWidget(clear_example_btn)
        form_lay.addLayout(ref_label_row)
        self.ref_text = QPlainTextEdit()
        self.ref_text.setPlaceholderText(
            "把你想模仿的示例解说文案直接粘贴到这里（可多段）。\n"
            "AI 会严格模仿它的题材、句式、节奏、语气与结构来仿写；留空则使用内置模板。")
        self.ref_text.setMinimumHeight(90)
        self.ref_text.setMaximumHeight(150)
        example = str(self.main.config_obj.data.get("reference_example", "") or "")
        if example:
            self.ref_text.setPlainText(example)
        form_lay.addWidget(self.ref_text)
        form_lay.addWidget(hint_label("AI 仿写：根据示例文案风格 + 原片画面创作，非逐句解说"))

        hook_row = QHBoxLayout()
        hook_row.addWidget(QLabel("评论引导话术"))
        self.comment_hook_edit = QLineEdit()
        self.comment_hook_edit.setMaxLength(60)
        self.comment_hook_edit.setPlaceholderText("例如：忘记带玫瑰花了（留空不添加）")
        self.comment_hook_edit.setText(str(self.main.config_obj.param("comment_hook_text", "")))
        self.comment_hook_position = QComboBox()
        for label, value in (("AI选择前段位置", "auto"), ("开场事件后", "early"), ("结尾前", "late")):
            self.comment_hook_position.addItem(label, value)
        idx = self.comment_hook_position.findData(self.main.config_obj.param("comment_hook_position", "auto"))
        self.comment_hook_position.setCurrentIndex(max(0, idx))
        hook_row.addWidget(self.comment_hook_edit, 1)
        hook_row.addWidget(self.comment_hook_position)
        form_lay.addLayout(hook_row)
        form_lay.addWidget(hint_label("保留话术原句，短暂插入后接回故事；不改成剧情或新增道具。随整篇文案一次配音。"))

        mode_row = QHBoxLayout()
        mode_row.setSpacing(16)
        mode_row.addWidget(QLabel("生产模式"))
        self.mode_group = QButtonGroup(self)
        self.mode_auto = QRadioButton("自动一键出片")
        self.mode_semi = QRadioButton("先生成文案，我修改确认后制作")
        (self.mode_semi if self.main.config_obj.param("review_mode", "semi") == "semi" else self.mode_auto).setChecked(True)
        self.mode_group.addButton(self.mode_auto, 0)
        self.mode_group.addButton(self.mode_semi, 1)
        mode_row.addWidget(self.mode_auto)
        mode_row.addWidget(self.mode_semi)
        mode_row.addStretch(1)
        form_lay.addLayout(mode_row)

        param_row = QHBoxLayout()
        param_row.setSpacing(20)
        self.spin_max_clips = QSpinBox()
        self.spin_max_clips.setRange(0, 9999)
        self.spin_max_clips.setSpecialValueText("自动：全部合格故事")
        self.spin_max_clips.setToolTip("自动按原片中完整、不重复的合格故事数量决定；填 1 只做 1 条，正数为成片上限。")
        self.spin_max_clips.setValue(int(self.main.config_obj.param("max_clips", 0)))
        self.spin_min_dur = QDoubleSpinBox()
        self.spin_min_dur.setRange(5, 60)
        self.spin_min_dur.setValue(float(self.main.config_obj.param("min_duration", 25)))
        self.spin_max_dur = QDoubleSpinBox()
        self.spin_max_dur.setRange(10, 90)
        self.spin_max_dur.setValue(float(self.main.config_obj.param("max_duration", 35)))
        self.spin_min_match = QDoubleSpinBox()
        self.spin_min_match.setRange(40, 95)
        self.spin_min_match.setValue(float(self.main.config_obj.param("min_match", 85)))
        for text, spin in (("成片数量", self.spin_max_clips), ("最短时长(s)", self.spin_min_dur),
                           ("最长时长(s)", self.spin_max_dur), ("匹配阈值", self.spin_min_match)):
            col = QHBoxLayout()
            col.addWidget(QLabel(text))
            col.addWidget(spin)
            param_row.addLayout(col)
        param_row.addStretch(1)
        form_lay.addLayout(param_row)
        form_lay.addWidget(hint_label(
            "自动按原片中完整、不重复的故事数决定，质量不足的故事跳过；填 1 只做 1 条，正数为上限。"
            "选择先审核时，每份文案仍在配音前等待确认。"))

        action_row = QHBoxLayout()
        action_row.setSpacing(10)
        self.start_btn = QPushButton("▶  开始生产")
        self.start_btn.setObjectName("accent")
        self.start_btn.clicked.connect(self._start)
        self.mode_semi.toggled.connect(lambda checked: self.start_btn.setText("▶  生成文案" if checked else "▶  开始生产"))
        if self.mode_semi.isChecked():
            self.start_btn.setText("▶  生成文案")
        self.stop_btn = QPushButton("停止")
        self.stop_btn.setObjectName("danger")
        self.stop_btn.clicked.connect(self.main.request_cancel)
        self.hint = QLabel("")
        self.hint.setObjectName("hint")
        action_row.addWidget(self.start_btn)
        action_row.addWidget(self.stop_btn)
        action_row.addWidget(self.hint)
        action_row.addStretch(1)
        form_lay.addLayout(action_row)
        root.addWidget(form)

        hist_card, hist_lay = card()
        hist_head = QHBoxLayout()
        hist_head.addWidget(section_title("历史任务（断点续跑）"))
        hist_head.addStretch(1)
        refresh_btn = QPushButton("刷新")
        refresh_btn.clicked.connect(self.refresh_history)
        resume_btn = QPushButton("载入选中任务")
        resume_btn.setObjectName("accent")
        resume_btn.clicked.connect(self._resume)
        hist_head.addWidget(refresh_btn)
        hist_head.addWidget(resume_btn)
        hist_lay.addLayout(hist_head)

        self.history = QTreeWidget()
        self.history.setColumnCount(4)
        self.history.setHeaderLabels(["任务 ID", "状态", "更新时间", "来源"])
        self.history.setRootIsDecorated(False)
        self.history.setAlternatingRowColors(True)
        self.history.setColumnWidth(0, 210)
        self.history.setColumnWidth(1, 120)
        self.history.setColumnWidth(2, 140)
        self.history.setColumnWidth(3, 420)
        self.history.itemDoubleClicked.connect(lambda _i, _c: self._resume())
        hist_lay.addWidget(self.history)
        root.addWidget(hist_card, 1)

        self.refresh_history()

    def mode_value(self):
        return "semi" if self.mode_group.checkedId() == 1 else "auto"

    def _browse_video(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "选择本地视频", "", "视频文件 (*.mp4 *.mkv *.mov *.webm *.avi *.flv *.m4v);;所有文件 (*.*)")
        if path:
            self.url_edit.setText(path)

    def _load_example_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "选择示例文案", "", "文案文件 (*.txt *.md *.json *.docx);;所有文件 (*.*)")
        if not path:
            return
        try:
            from ....style.loader import _read_one

            text = _read_one(path)
        except Exception:
            try:
                text = Path(path).read_text(encoding="utf-8", errors="replace")
            except OSError:
                text = ""
        if text.strip():
            current = self.ref_text.toPlainText().strip()
            self.ref_text.setPlainText((current + "\n\n" + text.strip()).strip() if current else text.strip())

    def _start(self):
        url = self.url_edit.text().strip()
        if not url:
            QMessageBox.warning(self, "提示", "请填写视频链接或本地路径")
            return
        if not Path(url).exists() and not url.startswith(("http://", "https://")):
            QMessageBox.warning(self, "提示", "本地文件不存在，请检查路径")
            return
        min_dur = self.spin_min_dur.value()
        max_dur = self.spin_max_dur.value()
        if min_dur >= max_dur:
            QMessageBox.warning(self, "提示", "最短时长必须小于最长时长")
            return
        params = {
            "review_mode": self.mode_value(),
            "max_clips": int(self.spin_max_clips.value()),
            "min_duration": min_dur,
            "max_duration": max_dur,
            "min_match": self.spin_min_match.value(),
            "comment_hook_text": self.comment_hook_edit.text().strip(),
            "comment_hook_position": self.comment_hook_position.currentData(),
        }
        self.main.config_obj.data["params"].update(params)
        example = self.ref_text.toPlainText().strip()
        self.main.config_obj.data["reference_example"] = example
        self.main.config_obj.save()
        references = [example] if example else []
        self.main.start_job(url, references)

    def _resume(self):
        item = self.history.currentItem()
        if not item:
            QMessageBox.information(self, "提示", "请先选择历史任务")
            return
        self.main.resume_job(item.text(0))

    def refresh_history(self):
        self.history.clear()
        jobs_dir = self.main.config_obj.workspace_path() / "jobs"
        if not jobs_dir.exists():
            return
        rows = []
        for job in jobs_dir.iterdir():
            data = job / "job.json"
            if not data.exists():
                continue
            try:
                info = json.loads(data.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            rows.append((info.get("updated_at", ""), info.get("job_id", job.name), info))
        rows.sort(reverse=True)
        for _updated, job_id, info in rows[:HISTORY_LIMIT]:
            item = QTreeWidgetItem([job_id, info.get("status", ""),
                                    info.get("updated_at", "").replace("T", " ")[:16],
                                    info.get("url", "")[:120]])
            self.history.addTopLevelItem(item)
