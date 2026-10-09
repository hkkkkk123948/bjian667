import queue
import threading

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (QButtonGroup, QFrame, QHBoxLayout, QLabel, QMainWindow,
                               QPushButton, QStackedWidget, QVBoxLayout, QWidget)

from ...config import Config
from ...log import EventEmitter
from ...pipeline import ProductionPipeline
from ...utils import read_json, write_json
from .pages.output import OutputPage
from .pages.progress import ProgressPage
from .pages.review import ReviewPage
from .pages.task import TaskPage
from .pages import ConfigPage
from .theme import MUTED, build_qss

APP_NAME = "短片工厂"

NAV_ITEMS = (("task", "✦  任务"), ("progress", "◔  进度"), ("review", "✎  审核"),
             ("config", "⚙  配置"), ("output", "▣  成片"))


class MainWindow(QMainWindow):
    def __init__(self, config_path=None):
        super().__init__()
        self.setWindowTitle(APP_NAME + " V1.2 · 原片剧情与真实镜头版")
        self.resize(1420, 900)
        self.setMinimumSize(1180, 760)
        self.setStyleSheet(build_qss())

        self.config_obj = self._load_config(config_path)
        self.pipeline = None
        self.worker = None
        self.running = False
        self.rewriting_candidate_id = None
        self.event_queue = queue.Queue()
        self.data = {"candidates": [], "clips": [], "meta": {}, "report": {}}

        root_widget = QWidget()
        root_widget.setObjectName("root")
        root = QHBoxLayout(root_widget)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        self.setCentralWidget(root_widget)
        self.nav_buttons = {}

        root.addWidget(self._build_sidebar())

        right = QVBoxLayout()
        right.setContentsMargins(0, 0, 0, 0)
        right.setSpacing(0)
        self.stack = QStackedWidget()
        self.pages = {}

        task = TaskPage(self)
        progress = ProgressPage(self)
        review = ReviewPage(self)
        config = ConfigPage(self)
        output = OutputPage(self)
        for key, page in (("task", task), ("progress", progress), ("review", review),
                          ("config", config), ("output", output)):
            self.pages[key] = page
            self.stack.addWidget(page)
        right.addWidget(self.stack, 1)

        self.status_label = QLabel("就绪")
        self.status_label.setObjectName("statusbar")
        right.addWidget(self.status_label)
        root.addLayout(right, 1)

        self._switch_page("task")

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._pump)
        self._timer.start(80)

    def _load_config(self, path):
        if path:
            return Config.load(path)
        default = Config().workspace_path() / "config.json"
        if default.exists():
            return Config.load(default)
        return Config(path=default)

    def _build_sidebar(self):
        sidebar = QFrame()
        sidebar.setObjectName("sidebar")
        sidebar.setFixedWidth(224)
        lay = QVBoxLayout(sidebar)
        lay.setContentsMargins(14, 22, 14, 16)
        lay.setSpacing(6)

        brand = QLabel("短片工厂")
        brand.setObjectName("section")
        brand.setStyleSheet("font-size:17px; font-weight:700; letter-spacing:2px; color:#ffffff;")
        sub = QLabel("long → short · v1")
        sub.setStyleSheet("color:%s; font-size:11px; letter-spacing:1px;" % MUTED)
        lay.addWidget(brand)
        lay.addWidget(sub)
        lay.addSpacing(22)

        self.nav_group = QButtonGroup(self)
        self.nav_group.setExclusive(True)
        for index, (key, text) in enumerate(NAV_ITEMS):
            btn = QPushButton(text)
            btn.setObjectName("nav")
            btn.setCheckable(True)
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            self.nav_group.addButton(btn, index)
            btn.clicked.connect(lambda _checked, k=key: self._switch_page(k))
            lay.addWidget(btn)
            self.nav_buttons[key] = btn
        lay.addStretch(1)

        hint = QLabel("先确认文案 · 整段配音 · 唯一画面审核\n接口异常保留进度，续跑复用配音")
        hint.setWordWrap(True)
        hint.setStyleSheet("color:%s; font-size:11px;" % (MUTED,))
        lay.addWidget(hint)
        return sidebar

    def _switch_page(self, key):
        if key in self.pages:
            self.stack.setCurrentWidget(self.pages[key])
            self.nav_buttons[key].setChecked(True)
            if key == "task":
                self.pages["task"].refresh_history()
            elif key == "review":
                self.pages["review"].refresh()
            elif key == "output":
                self.pages["output"].refresh()

    def make_emitter(self):
        def sink(event):
            self.event_queue.put(event)
        return EventEmitter(sink)

    def log_ui(self, message, level="info"):
        self.event_queue.put({"kind": "log", "message": message, "level": level, "ts": 0})
        if level == "info":
            self.status_label.setText(message)

    def start_job(self, url, references):
        if self.running:
            self.log_ui("已有任务在运行，请先停止", "warning")
            return
        self.pipeline = ProductionPipeline(self.config_obj, url, references=references,
                                           emitter=self.make_emitter())
        self.data = {"candidates": [], "clips": [], "meta": {}, "report": {}}
        self.pages["progress"].reset(self.pipeline)
        self.pages["review"].set_pipeline(self.pipeline)
        self.pages["output"].set_pipeline(self.pipeline)
        self.pages["task"].refresh_history()
        self.status_label.setText("任务 %s 运行中" % self.pipeline.job_id)
        self._switch_page("progress")
        self._spawn(self.pipeline.run)

    def resume_job(self, job_id):
        if self.running:
            self.log_ui("已有任务在运行，请先停止", "warning")
            return
        job_file = self.config_obj.job_dir(job_id) / "job.json"
        state = read_json(job_file) or {}
        url = state.get("url", "")
        references = state.get("references") or []
        self.pipeline = ProductionPipeline(self.config_obj, url, references=references,
                                           emitter=self.make_emitter(), job_id=job_id)
        self.pipeline._load_clips()
        self.pages["progress"].reset(self.pipeline)
        self.pages["review"].set_pipeline(self.pipeline)
        self.pages["output"].set_pipeline(self.pipeline)
        self.pages["review"].refresh()
        self.pages["output"].refresh()
        self.status_label.setText("已载入任务 %s（%s），点击“继续执行”续跑" % (job_id, self.pipeline.state.status))
        self._switch_page("progress")
        self.log_ui("载入任务 %s，已完成阶段: %s" % (job_id, ", ".join(self.pipeline.state.completed_stages) or "无"))

    def _return_to_pending_draft(self):
        review = self.pages["review"]
        pending = review.pending_draft_ids()
        if pending:
            self._switch_page("review")
            review.show_pending_draft(pending[0])
            self.log_ui("有已确认文案的未保存修改，已返回审核；草稿保留，尚未继续制作")
            return True
        return False

    def continue_job(self):
        if self.running or not self.pipeline:
            return
        if self._return_to_pending_draft():
            return
        self.pipeline.cancel_requested = False
        self.status_label.setText("任务 %s 续跑中" % self.pipeline.job_id)
        self._switch_page("progress")
        self._spawn(self.pipeline.run)

    def produce_now(self):
        if self.running or not self.pipeline:
            return
        if self._return_to_pending_draft():
            return
        self.pipeline.cancel_requested = False
        self.status_label.setText("任务 %s 生产中" % self.pipeline.job_id)
        self._switch_page("progress")
        self._spawn(self.pipeline.produce)

    def request_cancel(self):
        if self.pipeline and self.running:
            self.pipeline.cancel()
            self.log_ui("已请求停止，正在中断当前步骤", "warning")

    def regenerate_script(self, candidate_id):
        """Network writing runs in the worker; the editor keeps its draft safe."""
        if self.running or not self.pipeline:
            return
        pipeline = self.pipeline
        pipeline.cancel_requested = False
        self.rewriting_candidate_id = candidate_id
        self.log_ui('正在依据分镜重写一份文案，完成后返回审核；原稿与配音保留')
        def work():
            pipeline.apply_review(candidate_id,'regenerate')
            pipeline.state.artifacts['script_review_pending'] = True
            pipeline._stage('CANDIDATE_FILTERED','新文案已生成，等待确认；尚未配音或匹配画面')
            self.event_queue.put({'kind':'script_rewritten','candidate_id':candidate_id})
        self._spawn(work)
        self.pages['review'].refresh()

    def save_config(self):
        path = self.config_obj.save()
        self.log_ui("配置已保存: %s" % path)
        self.status_label.setText("配置已保存")

    def _spawn(self, fn):
        self.running = True

        def run():
            try:
                fn()
            except Exception as exc:
                self.event_queue.put({"kind": "log", "message": "执行异常: %s" % exc, "level": "error", "ts": 0})
            finally:
                self.event_queue.put({"kind": "worker_done", "ts": 0})

        self.worker = threading.Thread(target=run, daemon=True)
        self.worker.start()

    def _pump(self):
        try:
            while True:
                self._dispatch(self.event_queue.get_nowait())
        except queue.Empty:
            pass

    def _dispatch(self, event):
        kind = event.get("kind")
        if kind == 'script_rewritten':
            page = self.pages['review']
            candidate_id = event['candidate_id']
            draft = page._drafts.get(candidate_id)
            if draft is not None and self.pipeline:
                path = self.pipeline.job_dir/'analysis'/'ui_draft_history.json'
                history = read_json(path) or []
                if not isinstance(history,list):
                    history = []
                write_json(path,(history+[{'candidate_id':candidate_id,'text':draft}])[-20:])
            page._forget_draft(candidate_id)
            page.refresh()
            return
        if kind == "worker_done":
            self.running = False
            rewriting = self.rewriting_candidate_id is not None
            self.rewriting_candidate_id = None
            if rewriting:
                self.pages['review'].refresh()
                self.pages['output'].refresh()
                self.pages['task'].refresh_history()
                self._switch_page('review')
                return
            if self.pipeline:
                status = self.pipeline.state.status
                self.status_label.setText("任务 %s：%s" % (self.pipeline.job_id, status))
                if status in ("COMPLETED", "REVIEW_REQUIRED", "FAILED", "CANCELLED", "CANDIDATE_FILTERED"):
                    self.pages["review"].refresh()
                    self.pages["output"].refresh()
                    self.pages["task"].refresh_history()
                    if status == "COMPLETED":
                        self._switch_page("output")
                    elif status in ("REVIEW_REQUIRED", "CANDIDATE_FILTERED"):
                        self._switch_page("review")
            return
        self.pages["progress"].on_event(event)
        if kind == "update":
            if "candidates" in event:
                self.data["candidates"] = event["candidates"]
                self.pages["review"].refresh()
            if "clips" in event:
                self.data["clips"] = event["clips"]
                self.pages["output"].refresh()
            if "meta" in event:
                self.data["meta"] = event["meta"]
            if "report" in event:
                self.data["report"] = event["report"]
                self.pages["output"].refresh()
        self.pages["review"].on_event(event)
        self.pages["output"].on_event(event)
