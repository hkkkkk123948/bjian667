from pathlib import Path
import re

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QFont, QSyntaxHighlighter, QTextCharFormat, QColor
from PySide6.QtWidgets import (QHBoxLayout, QLabel, QMessageBox, QPlainTextEdit, QPushButton, QInputDialog, QFileDialog, QComboBox,
                               QSplitter, QTreeWidget, QTreeWidgetItem, QVBoxLayout, QWidget)

from ....utils import fmt_clock
from ....utils import read_json, write_json
from .common import section_title
from ..theme import MUTED
from ....script.review_edit import COMMENT_MARKER, review_text, parse_review
from ....script.duration import estimate_duration, duration_state, word_budget


class CommentHighlighter(QSyntaxHighlighter):
    def highlightBlock(self, text):
        if text.strip().startswith(COMMENT_MARKER):
            fmt = QTextCharFormat()
            fmt.setForeground(QColor("#ffd479"))
            fmt.setBackground(QColor("#49351e"))
            self.setFormat(0, len(text), fmt)
        else:
            fmt = QTextCharFormat()
            fmt.setForeground(QColor("#ffd479"))
            fmt.setBackground(QColor("#49351e"))
            for match in re.finditer(r"（[^（）]*）|\([^()]*\)", text):
                self.setFormat(match.start(), len(match.group()), fmt)

class ReviewPage(QWidget):
    def __init__(self, main):
        super().__init__()
        self.main = main
        self.pipeline = None
        self._drafts = {}
        self._draft_job_id = None
        self._current_id = None
        self._loading_editor = False
        self._duration_timer = QTimer(self)
        self._duration_timer.setSingleShot(True)
        self._duration_timer.setInterval(180)
        self._duration_timer.timeout.connect(self._update_duration)

        root = QVBoxLayout(self)
        root.setContentsMargins(24, 20, 24, 20)
        root.setSpacing(14)

        head = QHBoxLayout()
        head.addWidget(section_title("文案编辑与确认（配音与画面匹配之前）"))
        head.addStretch(1)
        self.queue_label = QLabel("")
        self.queue_label.setObjectName("muted")
        head.addWidget(self.queue_label)
        self.resume_scripts = QPushButton("继续生成剩余文案")
        self.resume_scripts.clicked.connect(self.main.continue_job)
        self.resume_scripts.setVisible(False)
        head.addWidget(self.resume_scripts)
        root.addLayout(head)
        flow = QLabel("素材分析 → 生成文案 → 【你编辑确认】 → 整段配音 → 匹配画面 → 唯一画面审核 → 成片")
        flow.setObjectName("hint")
        root.addWidget(flow)
        self.review_notice = QLabel("")
        self.review_notice.setWordWrap(True)
        self.review_notice.setStyleSheet("color:#ffd479;")
        self.review_notice.setVisible(False)
        root.addWidget(self.review_notice)

        splitter = QSplitter(Qt.Orientation.Horizontal)

        self.tree = QTreeWidget()
        self.tree.setColumnCount(5)
        self.tree.setHeaderLabels(["文案", "文案准备", "你的确认", "原片区间", "制作状态"])
        self.tree.setRootIsDecorated(False)
        self.tree.setAlternatingRowColors(True)
        self.tree.setColumnWidth(0, 90)
        self.tree.setColumnWidth(1, 70)
        self.tree.setColumnWidth(2, 70)
        self.tree.setColumnWidth(3, 150)
        self.tree.setColumnWidth(4, 150)
        self.tree.currentItemChanged.connect(lambda _a, _b: self._show_detail())
        splitter.addWidget(self.tree)

        right = QWidget()
        right_lay = QVBoxLayout(right)
        right_lay.setContentsMargins(16, 0, 0, 0)
        right_lay.setSpacing(8)
        right_lay.addWidget(QLabel("故事资料"))
        self.detail = QPlainTextEdit()
        self.detail.setReadOnly(True)
        self.detail.setFont(QFont("Microsoft YaHei UI", 10))
        self.detail.setMaximumHeight(200)
        right_lay.addWidget(self.detail)
        editor_head = QHBoxLayout()
        editor_head.addWidget(QLabel("正文可编辑 · 括号内为评论话术（高亮）"))
        editor_head.addStretch(1)
        insert_hook = QPushButton("插入评论话术（）")
        insert_hook.clicked.connect(self._insert_comment)
        editor_head.addWidget(insert_hook)
        right_lay.addLayout(editor_head)
        self.script = QPlainTextEdit()
        self.script.setFont(QFont("Microsoft YaHei UI", 10))
        self.highlighter = CommentHighlighter(self.script.document())
        self.script.textChanged.connect(self._on_edit)
        right_lay.addWidget(self.script)
        self.duration_label = QLabel("预计配音时长：选择文案后显示")
        self.duration_label.setWordWrap(True)
        self.duration_label.setObjectName("durationHint")
        right_lay.addWidget(self.duration_label)
        file_row = QHBoxLayout()
        import_text = QPushButton("导入文案…")
        import_text.clicked.connect(self._import_text)
        export_text = QPushButton("导出标注文案…")
        export_text.clicked.connect(self._export_text)
        file_row.addWidget(import_text)
        file_row.addWidget(export_text)
        file_row.addStretch(1)
        right_lay.addLayout(file_row)
        picture_row = QHBoxLayout()
        picture_row.addWidget(QLabel("话术段画面"))
        self.comment_picture = QComboBox()
        for label,value in (("原位承接（推荐）","auto"),("保持当前画面","follow"),("原位轻推/平移","motion"),("原位短停再缓动","freeze")):
            self.comment_picture.addItem(label,value)
        self.comment_picture.currentIndexChanged.connect(self._set_comment_picture)
        picture_row.addWidget(self.comment_picture,1)
        right_lay.addLayout(picture_row)
        note = QLabel("用中文（）或英文()括起评论话术，自动橙色高亮，可写多处。\n"
                      "例：女孩冲出车道（没带玫瑰花），司机立刻急刹。\n"
                      "括号内的话参与整段配音，符号不读；不按话术字面查找动作或道具。\n"
                      "话术在哪里出现，就只在对应配音时段处理画面；结束后接回正文，不把效果套到整句。")
        note.setWordWrap(True)
        note.setObjectName("hint")
        right_lay.addWidget(note)
        splitter.addWidget(right)
        splitter.setSizes([520, 560])
        root.addWidget(splitter, 1)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(8)
        refresh_btn = QPushButton("刷新")
        refresh_btn.clicked.connect(self.refresh)
        approve_btn = QPushButton("仅确认文案")
        approve_btn.setToolTip('保存并确认当前文案，暂不配音或匹配画面；可逐条审核后再开始制作。')
        approve_btn.setObjectName("accent")
        approve_btn.clicked.connect(lambda: self._review("approve"))
        reject_btn = QPushButton("淘汰")
        reject_btn.setObjectName("danger")
        reject_btn.clicked.connect(lambda: self._review("reject"))
        rewrite_btn = QPushButton("重写脚本")
        rewrite_btn.clicked.connect(lambda: self._review("regenerate"))
        apply_btn = QPushButton("保存文案")
        apply_btn.clicked.connect(self._apply_edit)
        produce_btn = QPushButton("▶  确认文案并制作")
        produce_btn.setToolTip('先确认当前文案，再制作所有已确认文案；按成片数量上限处理，未确认稿不会制作。')
        produce_btn.setObjectName("accent")
        produce_btn.clicked.connect(self._produce)
        btn_row.addWidget(refresh_btn)
        btn_row.addWidget(approve_btn)
        btn_row.addWidget(reject_btn)
        btn_row.addWidget(rewrite_btn)
        btn_row.addWidget(apply_btn)
        btn_row.addStretch(1)
        btn_row.addWidget(produce_btn)
        root.addLayout(btn_row)

    def set_pipeline(self, pipeline):
        self.pipeline = pipeline
        job_id = pipeline.job_id if pipeline else None
        if job_id != self._draft_job_id:
            self._draft_job_id = job_id
            self._current_id = None
            self._drafts = read_json(pipeline.job_dir / "analysis" / "ui_drafts.json") or {} if pipeline else {}
            self.review_notice.setVisible(False)
        self.refresh()

    def _can_edit(self):
        if (self.main.running and getattr(self.main,'rewriting_candidate_id',None) is not None
                and getattr(self.main,'rewriting_candidate_id',None) == self._current_id):
            return False
        if not self.main.running:
            return True
        return bool(self.pipeline and self.pipeline.config.param("review_mode") == "semi"
                    and not self.pipeline.completed("TTS_GENERATED")
                    and self.pipeline.state.status not in ("TTS_GENERATED", "MATCHING", "TIMELINE_ALIGNED", "RENDERING"))

    def _on_edit(self):
        if self._loading_editor or not self._current_id:
            return
        self._drafts[self._current_id] = self.script.toPlainText()
        self._save_drafts()
        if not self.pending_draft_ids():
            self.review_notice.setVisible(False)
        self._duration_timer.start()

    def pending_draft_ids(self):
        """An approved saved script must not hide a different unsaved draft."""
        if not self.pipeline:
            return []
        pending = []
        for candidate in self._rows():
            if candidate.status != "approved" or candidate.candidate_id not in self._drafts:
                continue
            variant = self.pipeline._pick_variant(candidate)
            if variant and self._drafts[candidate.candidate_id].strip() != review_text(variant).strip():
                pending.append(candidate.candidate_id)
        return pending

    def show_pending_draft(self, candidate_id):
        items = self.tree.findItems(candidate_id, Qt.MatchFlag.MatchExactly, 0)
        if items:
            self.tree.setCurrentItem(items[0])
        self.review_notice.setText("已确认的文案还有未保存修改，暂未继续制作。修改已保留；"
                                   "请保存后重新确认，或点击“确认文案并制作”使用当前修改稿。")
        self.review_notice.setVisible(True)
        self.script.setFocus()

    def _update_duration(self):
        if not self.pipeline or not self._current_id:
            self.duration_label.setText("预计配音时长：选择文案后显示")
            return None
        config = self.pipeline.config
        estimate = estimate_duration(config, self.script.toPlainText())
        minimum = float(config.param("min_duration", 15))
        maximum = float(config.param("max_duration", 35))
        state = duration_state(estimate, minimum, maximum)
        basis = ("同音色、模型和语速的 %d 段历史配音校准" % estimate.sample_count
                 if estimate.calibrated else "仅 1 段同设置样本，初步估算" if estimate.sample_count
                 else "暂无同设置样本，按语速估算")
        message = {"empty": "请填写文案。", "too_short": "预计明显过短，请补足故事；不会自动加词或放慢配音。",
                   "near_short": "预计偏短，建议补足故事内容。", "too_long": "预计明显超长，仅供参考；确认后仍按你的文案制作。",
                   "near_long": "预计偏长，建议适当精简。", "in_range": "预计在目标范围内，实际时长以整段配音为准。"}[state]
        try:
            parse_review(self.script.toPlainText())
            bracket_error = ""
        except ValueError as exc:
            bracket_error = str(exc)
            message += "\n括号提示：" + bracket_error
        budget = word_budget(config)
        self.duration_label.setText(
            "预计整段配音 %.1f 秒（约 %.1f–%.1f 秒） · 目标 %g–%g 秒\n"
            "%s；口播共 %d 字（含评论话术），括号不计。建议总字数 %d–%d。\n%s"
            % (estimate.seconds, estimate.low_seconds, estimate.high_seconds, minimum, maximum,
               basis, estimate.spoken_units, budget["minimum"], budget["maximum"], message))
        self.duration_label.setStyleSheet("color:%s;" % (
            "#ff7777" if state == "too_long" or bracket_error else "#ffd479" if state in
            ("too_short", "near_short", "near_long") else MUTED))
        return estimate

    def _duration_ready(self):
        self._update_duration()
        try:
            parse_review(self.script.toPlainText())
        except ValueError:
            self.main.log_ui("评论话术括号未正确配对，请按文案下方提示修正；修改已暂存")
            self.script.setFocus()
            return False
        return True

    def _save_drafts(self):
        if self.pipeline:
            write_json(self.pipeline.job_dir / "analysis" / "ui_drafts.json", self._drafts)

    def _forget_draft(self, candidate_id):
        self._drafts.pop(candidate_id, None)
        self._save_drafts()
        if not self.pending_draft_ids():
            self.review_notice.setVisible(False)

    def on_event(self, event):
        if event.get("kind") == "update" and "candidates" in event:
            self.refresh()

    def _rows(self):
        if self.pipeline:
            return self.pipeline._load_candidates()
        return []

    def refresh(self):
        current = self._selected_id()
        self.tree.blockSignals(True)
        self.tree.clear()
        rows = self._rows()
        pending = (self.pipeline.state.artifacts.get("pending_stories") or []) if self.pipeline else []
        discovery = self.pipeline.state.artifacts.get("story_discovery") if self.pipeline else None
        discovery_pending = isinstance(discovery, dict) and discovery.get("complete") is False
        self.resume_scripts.setVisible(bool(pending) or discovery_pending)
        self.resume_scripts.setText("继续分析故事并生成文案" if discovery_pending else "继续生成剩余文案")
        for cand in rows:
            item = QTreeWidgetItem([
                cand.candidate_id, "已生成" if cand.scripts else "待生成",
                "已确认" if cand.status == "approved" else "已跳过" if cand.status in ("rejected", "rejected_by_user") else "待确认",
                "%s-%s" % (fmt_clock(cand.source_start), fmt_clock(cand.source_end)), self._display_state(cand)])
            self.tree.addTopLevelItem(item)
        queue = [c for c in rows if c.status in ("gate1_pass", "review_required")]
        approved = [c for c in rows if c.status == "approved"]
        script_count = sum(bool(candidate.scripts) for candidate in rows)
        summary = []
        if isinstance(discovery, dict) and isinstance(discovery.get("discovered"), int):
            summary.append("发现 %d 个故事" % discovery["discovered"])
        summary.extend(["已生成文案 %d" % script_count, "待审 %d" % len(queue), "已确认 %d" % len(approved)])
        if pending:
            summary.append("待生成 %d" % len(pending))
        if discovery_pending:
            summary.append("故事分析待继续")
        self.queue_label.setText(" · ".join(summary))
        self.queue_label.setStyleSheet("color:%s;" % MUTED)
        if current and self.tree.findItems(current, Qt.MatchFlag.MatchExactly, 0):
            items = self.tree.findItems(current, Qt.MatchFlag.MatchExactly, 0)
            self.tree.setCurrentItem(items[0])
        elif rows:
            self.tree.setCurrentItem(self.tree.topLevelItem(0))
        self.tree.blockSignals(False)
        self._show_detail()

    def _selected_id(self):
        item = self.tree.currentItem()
        return item.text(0) if item else None

    @staticmethod
    def _display_state(candidate):
        return {"approved": "等待制作", "rejected": "已跳过", "rejected_by_user": "已跳过", "completed": "制作完成",
                "qc_failed": "画面匹配有问题，已保留成片", "failed": "制作异常"}.get(
                    candidate.status, "等待你确认" if candidate.scripts else "文案待生成")

    def _find(self, candidate_id):
        for cand in self._rows():
            if cand.candidate_id == candidate_id:
                return cand
        return None

    def _show_detail(self):
        self.script.setReadOnly(not self._can_edit())
        self._loading_editor = True
        candidate_id = self._selected_id()
        if not candidate_id or not self.pipeline:
            self.detail.clear()
            self.script.clear()
            self._current_id = None
            self._loading_editor = False
            self._update_duration()
            return
        self.script.blockSignals(True)
        cand = self._find(candidate_id)
        self.detail.clear()
        if not cand:
            self.script.clear()
            self.script.blockSignals(False)
            self._current_id = None
            self._loading_editor = False
            self._update_duration()
            return
        lines = [
            "标题: %s" % (cand.title or "-"),
            "梗概: %s" % cand.story_summary,
            "冲突: %s" % cand.conflict,
            "转折: %s" % cand.turning_point,
            "落点: %s" % cand.payoff,
            "钩子: %s" % cand.hook,
            "原片可用区间: %s" % ", ".join("%s-%s" % (fmt_clock(e.start), fmt_clock(e.end))
                                       for e in cand.evidence),
            "文案由你编辑确认；确认后直接配音与匹配，不再进行AI故事审稿。",
            "制作状态: %s" % self._display_state(cand),
        ]
        self.detail.setPlainText("\n".join(lines))
        variant = self.pipeline._pick_variant(cand)
        self.comment_picture.blockSignals(True)
        self.comment_picture.setCurrentIndex(max(0,self.comment_picture.findData(
            getattr(variant,'comment_visual_mode','') or self.pipeline.config.param("comment_visual_mode","auto"))))
        self.comment_picture.blockSignals(False)
        text = self._drafts.get(candidate_id, review_text(variant) if variant else "")
        if self._current_id != candidate_id or self.script.toPlainText() != text:
            self.script.setPlainText(text)
        self.script.blockSignals(False)
        self._current_id = candidate_id
        self._loading_editor = False
        self._update_duration()

    def _set_comment_picture(self):
        if self._loading_editor or not self.pipeline or not self._current_id:
            return
        candidate = self._find(self._current_id)
        variant = self.pipeline._pick_variant(candidate)
        mode = self.comment_picture.currentData()
        if getattr(variant,'comment_visual_mode','') == mode:
            return
        variant.comment_visual_mode = mode
        self.pipeline._save_candidates()
        if self.pipeline.completed("TIMELINE_ALIGNED"):
            stages = {"MATCHING","TIMELINE_ALIGNED","RENDERING","FINAL_MATCH_CHECK","QUALITY_CHECK","DRAFT_EXPORT","COMPLETED"}
            self.pipeline.state.completed_stages = [s for s in self.pipeline.state.completed_stages if s not in stages]
            self.pipeline.state.status = "CANDIDATE_FILTERED"
            self.pipeline.save_state()
        self.main.log_ui("已设置话术画面："+self.comment_picture.currentText()+"；配音保持整段不剪接")

    def _selected(self):
        candidate_id = self._selected_id()
        if not candidate_id or not self.pipeline:
            QMessageBox.information(self, "提示", "请先选择候选")
            return None
        return candidate_id

    def _insert_comment(self):
        if self.script.isReadOnly():
            return
        current = self.script.textCursor().selectedText()
        phrase, accepted = QInputDialog.getText(self, "插入评论话术", "填写原句，软件会自动加（）并高亮：", text=current)
        if not accepted:
            return
        if phrase.strip():
            cursor = self.script.textCursor()
            cursor.insertText("（" + phrase.strip() + "）")

    def _import_text(self):
        if self.script.isReadOnly():
            return
        path, _ = QFileDialog.getOpenFileName(self, "导入标注文案", "", "文案 (*.txt *.md)")
        if path:
            try:
                self.script.setPlainText(Path(path).read_text(encoding="utf-8-sig"))
            except (OSError, UnicodeError) as exc:
                QMessageBox.warning(self, "无法读取文案", str(exc))

    def _export_text(self):
        path, _ = QFileDialog.getSaveFileName(self, "导出标注文案", "文案待审.txt", "文案 (*.txt)")
        if path:
            try:
                Path(path).write_text(self.script.toPlainText(), encoding="utf-8")
            except OSError as exc:
                QMessageBox.warning(self, "无法保存文案", str(exc))

    def _review(self, action):
        candidate_id = self._selected()
        if not candidate_id:
            return
        if self.main.running:
            QMessageBox.information(self, "提示", "修改已自动暂存，请等待当前生成结束后确认。")
            return
        if action == "approve" and not self._duration_ready():
            return
        if action == 'regenerate':
            self._save_drafts()
            self.main.regenerate_script(candidate_id)
            return
        try:
            self.pipeline.apply_review(candidate_id, action, new_text=self.script.toPlainText() if action == "approve" else None)
        except ValueError as exc:
            QMessageBox.warning(self, "文案未保存", str(exc))
            return
        self._forget_draft(candidate_id)
        self.refresh()
        found = self._find(candidate_id)
        self.main.log_ui("%s %s → %s" % (action, candidate_id, found.status if found else ""))

    def _apply_edit(self):
        if self.main.running:
            self._save_drafts()
            self.main.log_ui("修改已自动暂存，当前文案生成结束后可确认制作")
            return
        candidate_id = self._selected()
        if not candidate_id:
            return
        text = self.script.toPlainText().strip()
        try:
            self.pipeline.apply_review(candidate_id, "edit", new_text=text)
        except ValueError as exc:
            QMessageBox.warning(self, "文案未保存", str(exc))
            return
        self._forget_draft(candidate_id)
        self.refresh()
        self.main.log_ui("已保存文案 %s，确认后再制作" % candidate_id)

    def _produce(self):
        if self.main.running:
            return
        if not self.pipeline:
            QMessageBox.information(self, "提示", "请先新建或载入任务")
            return
        selected = self._selected_id()
        if selected:
            if not self._duration_ready():
                return
            try:
                self.pipeline.apply_review(selected, "approve", new_text=self.script.toPlainText())
            except ValueError as exc:
                QMessageBox.warning(self, "文案未确认", str(exc))
                return
            self._forget_draft(selected)
        approved = [c for c in self._rows() if c.status == "approved"]
        if not approved:
            QMessageBox.information(self, "提示", "没有已通过的候选，请先审核")
            return
        self.main.produce_now()
