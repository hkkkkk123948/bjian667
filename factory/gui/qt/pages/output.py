from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (QApplication, QHBoxLayout, QLabel, QMessageBox, QPushButton,
                               QTreeWidget, QTreeWidgetItem, QVBoxLayout, QWidget)

from ....utils import fmt_clock, grab_frame, media_duration, open_path
from .common import section_title


class OutputPage(QWidget):
    def __init__(self, main):
        super().__init__()
        self.main = main
        self.pipeline = None

        root = QVBoxLayout(self)
        root.setContentsMargins(24, 20, 24, 20)
        root.setSpacing(14)

        head = QHBoxLayout()
        head.addWidget(section_title("成片库"))
        head.addStretch(1)
        self.summary = QLabel("")
        self.summary.setObjectName("muted")
        head.addWidget(self.summary)
        root.addLayout(head)

        self.tree = QTreeWidget()
        self.tree.setColumnCount(9)
        self.tree.setHeaderLabels(["成片", "候选", "原片区间", "时长", "匹配", "QC", "状态", "草稿", "文件"])
        self.tree.setRootIsDecorated(False)
        self.tree.setAlternatingRowColors(True)
        for col, width in ((0, 100), (1, 80), (2, 150), (3, 70), (4, 70), (5, 60), (6, 100), (7, 150), (8, 0)):
            self.tree.setColumnWidth(col, width)
        self.tree.itemDoubleClicked.connect(lambda _i, _c: self._open_selected())
        root.addWidget(self.tree, 1)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(8)
        refresh_btn = QPushButton("刷新")
        refresh_btn.clicked.connect(self.refresh)
        play_btn = QPushButton("打开选中成片")
        play_btn.setObjectName("accent")
        play_btn.clicked.connect(self._open_selected)
        folder_btn = QPushButton("打开所在文件夹")
        folder_btn.clicked.connect(self._open_folder)
        report_btn = QPushButton("打开生产报告")
        report_btn.clicked.connect(self._open_report)
        draft_btn = QPushButton("打开剪映草稿")
        draft_btn.clicked.connect(self._open_draft)
        job_btn = QPushButton("打开任务目录")
        job_btn.clicked.connect(self._open_job)
        mask_btn = QPushButton("框选去水印/字幕")
        mask_btn.setObjectName("accent")
        mask_btn.clicked.connect(self._edit_regions)
        hint = QLabel("双击列表可直接播放成片")
        hint.setObjectName("hint")
        btn_row.addWidget(refresh_btn)
        btn_row.addWidget(play_btn)
        btn_row.addWidget(mask_btn)
        btn_row.addWidget(folder_btn)
        btn_row.addWidget(report_btn)
        btn_row.addWidget(draft_btn)
        btn_row.addWidget(job_btn)
        btn_row.addStretch(1)
        btn_row.addWidget(hint)
        root.addLayout(btn_row)

    def set_pipeline(self, pipeline):
        self.pipeline = pipeline
        self.refresh()

    def _clips(self):
        if self.pipeline and self.pipeline.clips:
            return self.pipeline.clips
        return []

    def refresh(self):
        self.tree.clear()
        clips = self._clips()
        for clip in clips:
            duration = clip.duration or 0.0
            if Path(clip.path or "").exists():
                try:
                    duration = media_duration(clip.path)
                except Exception:
                    pass
            gate2 = clip.gate2 or {}
            qc = clip.qc or {}
            template = [self.draft_text(clip), clip.path or ""]
            item = QTreeWidgetItem([
                clip.clip_id, clip.candidate_id,
                "%s-%s" % (fmt_clock(clip.source_start), fmt_clock(clip.source_end)),
                "%.1fs" % duration,
                "%.1f" % float((clip.scores or {}).get("final_score", 0.0) or 0.0),
                qc.get("status", gate2.get("status", "-")),
                clip.status] + template)
            self.tree.addTopLevelItem(item)
            if clip.draft_path and clip.status == "completed":
                item.setForeground(7, QColor("#4caf50"))
        passed = sum(1 for c in clips if c.status == "completed")
        self.summary.setText("成片 %d 条（通过 %d）" % (len(clips), passed))

    def _selected_path(self):
        item = self.tree.currentItem()
        if not item:
            return None
        return item.text(8) or None

    def _open_draft(self):
        item = self.tree.currentItem()
        if not item:
            return
        path = item.text(7)
        if Path(path).exists():
            open_path(str(Path(path)))
        else:
            QMessageBox.information(self, "提示", "该成片未导出剪映草稿")

    @staticmethod
    def draft_text(clip):
        if clip.draft_path:
            return "已导出"
        if clip.status == "completed":
            return "未导出"
        return "-"

    def _open_selected(self):
        path = self._selected_path()
        if not path:
            QMessageBox.information(self, "提示", "请先选择成片")
            return
        if not Path(path).exists():
            QMessageBox.warning(self, "提示", "文件不存在: %s" % path)
            return
        open_path(path)

    def _selected_clip(self):
        if not self.pipeline:
            return None
        item = self.tree.currentItem()
        if not item:
            return None
        cid = item.text(0)
        return next((c for c in (self.pipeline.clips or []) if c.clip_id == cid), None)

    def _edit_regions(self):
        import tempfile

        clip = self._selected_clip()
        if not clip:
            QMessageBox.information(self, "提示", "请先在列表选择一条成片")
            return
        if not self.pipeline or not self.pipeline.state.artifacts.get("video_path"):
            QMessageBox.information(self, "提示", "缺少源视频，无法框选")
            return
        source = self.pipeline.state.artifacts["video_path"]
        frame = Path(tempfile.gettempdir()) / ("ytsf_region_%s.jpg" % clip.clip_id)
        if not grab_frame(source, clip.source_start, frame):
            QMessageBox.warning(self, "提示", "无法获取预览帧")
            return
        from .region_dialog import RegionDialog

        regions = (clip.scores or {}).get("mask_regions") or []
        method = (clip.scores or {}).get("mask_method") or "blur"
        dialog = RegionDialog(self, str(frame), regions=regions, method=method)
        if dialog.exec() != RegionDialog.DialogCode.Accepted:
            return
        if not dialog.regions:
            QMessageBox.information(self, "提示", "未框选任何区域")
            return
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            self.pipeline.rerender_clip(clip.clip_id, dialog.regions, dialog.method)
        except Exception as exc:
            QApplication.restoreOverrideCursor()
            QMessageBox.warning(self, "重渲染失败", str(exc)[:400])
            return
        QApplication.restoreOverrideCursor()
        self.refresh()
        self.main.log_ui("已按框选区域重渲染 %s（%d 个区域，%s）"
                         % (clip.clip_id, len(dialog.regions), dialog.method))
        QMessageBox.information(self, "完成", "已重渲染并覆盖 %s" % clip.clip_id)

    def _open_folder(self):
        path = self._selected_path()
        if path and Path(path).exists():
            open_path(str(Path(path).parent))
        else:
            self._open_job()

    def _open_job(self):
        if self.pipeline:
            open_path(self.pipeline.job_dir)

    def _open_report(self):
        if not self.pipeline:
            QMessageBox.information(self, "提示", "请先新建或载入任务")
            return
        report = self.pipeline.job_dir / "reports" / "production_report.json"
        if not report.exists():
            QMessageBox.information(self, "提示", "报告尚未生成")
            return
        open_path(report)

    def on_event(self, event):
        if event.get("kind") == "update" and ("clips" in event or "report" in event):
            self.refresh()
