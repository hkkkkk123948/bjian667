from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QApplication, QComboBox, QDoubleSpinBox, QFileDialog, QGridLayout,
                               QHBoxLayout, QLabel, QLineEdit, QListWidget, QListWidgetItem,
                               QMessageBox, QPushButton, QScrollArea, QVBoxLayout, QWidget)

from ....config import LLM_TYPES, ROLES, ROLE_LABELS, TTS_TYPES, DEFAULT_TTS_SPEED
from ....export.capcut import detect_draft_root
from .common import card, section_title

VOICE_HEADER_SYS = "── 系统音色 ──"
VOICE_HEADER_CLONE = "── 克隆音色 ──"

PARAM_SPEC = [
    ("min_duration", "文案目标最短时长(s)", "float"),
    ("max_duration", "文案目标最长时长(s，配音不截断)", "float"),
    ("qc_semantic_min", "最终画面匹配阈值", "float"),
    ("max_clips", "成片数量（0 自动）", "int"),
    ("candidate_pool", "候选池", "int"),
    ("analyze_max_candidates", "分析候选上限(0自动)", "int"),
    ("review_mode", "文案确认方式（semi 手动）", ("auto", "semi")),
    ("content_mode", "内容模式", ("creative", "commentary")),
    ("creative_strategy", "原创流程", ("visual_story", "legacy")),
    ("voice_mode", "配音方式", ("continuous",)),
    ("visual_motion", "画面推近与平移", ("on", "off")),
    ("short_transitions", "短叠化转场", ("on", "off")),
    ("transition_duration", "转场时长(s，最多0.18)", "float"),
    ("picture_enhance", "成片去噪与锐化", ("on", "off")),
    ("picture_grade", "成片轻度反差", ("on", "off")),
    ("approved_plan_reuse", "复用已审查素材", ("on", "off")),
    ("director_overview_frames", "全片看图上限", "int"),
    ("director_detail_frames", "故事看图上限", "int"),
    ("sound_design", "惊悚环境配乐", ("on", "off")),
    ("sound_bed_db", "配乐响度(dB)", "float"),
    ("aspect_ratio", "画幅", ("auto", "vertical")),
    ("target_width", "成片宽", "int"),
    ("target_height", "成片高", "int"),
    ("crop_focus", "画面焦点(0-1)", "float"),
    ("bg_volume", "原声音量", "float"),
    ("voice_volume", "配音音量", "float"),
    ("scene_threshold", "场景阈值", "float"),
    ("dedupe_threshold", "去重阈值", "float"),
    ("transcribe", "字幕转写", ("api", "auto", "force", "off")),
    ("whisper_model", "whisper模型", "str"),
    ("language", "语言", "str"),
    ("reframe_faces", "人脸跟随", ("on", "track", "off")),
    ("remove_subtitles", "去原字幕", ("off", "crop", "blur", "delogo")),
    ("subtitle_band", "字幕带比例", "float"),
    ("auto_crop_black", "自动去黑边", ("on", "off")),
    ("normalize_loudness", "响度归一", ("on", "off")),
]

CHOICE = {
    "short_transitions": ("on", "off"),
    "picture_enhance": ("on", "off"),
            "picture_grade": ("on", "off"),
    "approved_plan_reuse": ("on", "off"),
    "creative_strategy": ("visual_story", "legacy"),
    "voice_mode": ("continuous",),
    "visual_motion": ("on", "off"),
    "sound_design": ("on", "off"),
    "review_mode": ("auto", "semi"),
    "content_mode": ("creative", "commentary"),
    "aspect_ratio": ("auto", "vertical"),
    "transcribe": ("api", "auto", "force", "off"),
    "reframe_faces": ("on", "track", "off"),
    "remove_subtitles": ("off", "crop", "blur", "delogo"),
    "auto_crop_black": ("on", "off"),
    "normalize_loudness": ("on", "off"),
    "gate_fallback": ("on", "off"),
    "story_rebuild": ("on", "off"),
    "pre_tts_compress": ("on", "off"),
}

VOICE_PRESETS = {
    "bailian": ["Cherry", "Serena", "Ethan", "Chelsie"],
    "edge": ["zh-CN-YunxiNeural", "zh-CN-XiaoxiaoNeural", "zh-CN-YunjianNeural", "zh-CN-XiaoyiNeural"],
    "openai": ["alloy", "echo", "fable", "onyx", "nova", "shimmer"],
    "gemini": ["Kore", "Puck", "Charon", "Fenrir"],
    "elevenlabs": ["21m00Tcm4TlvDq8ikWAM", "EXAVITQu4vr4xnSDxMaL"],
}


class ConfigPage(QWidget):
    def __init__(self, main):
        super().__init__()
        self.main = main
        self.role_widgets = {}
        self.param_widgets = {}

        root = QVBoxLayout(self)
        root.setContentsMargins(24, 20, 24, 20)
        root.setSpacing(14)

        title = QLabel("系统配置")
        title.setObjectName("title")
        root.addWidget(title)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        content = QWidget()
        lay = QVBoxLayout(content)
        lay.setContentsMargins(0, 0, 8, 0)
        lay.setSpacing(14)

        lay.addWidget(section_title("分析 API（大模型，按角色）"))
        prov_card, prov_lay = card()
        uni_row = QHBoxLayout()
        uni_row.setSpacing(8)
        uni_row.addWidget(QLabel("类型"))
        self.uni_type = QComboBox()
        self.uni_type.addItems(LLM_TYPES)
        uni_row.addWidget(self.uni_type)
        uni_row.addWidget(QLabel("模型"))
        self.uni_model = QLineEdit()
        self.uni_model.setMinimumWidth(180)
        uni_row.addWidget(self.uni_model)
        uni_row.addWidget(QLabel("Base URL"))
        self.uni_base = QLineEdit()
        self.uni_base.setMinimumWidth(220)
        uni_row.addWidget(self.uni_base, 1)
        uni_row.addWidget(QLabel("API Key"))
        self.uni_key = QLineEdit()
        self.uni_key.setEchoMode(QLineEdit.EchoMode.Password)
        self.uni_key.setMinimumWidth(180)
        uni_row.addWidget(self.uni_key)
        uni_apply = QPushButton("应用到全部")
        uni_apply.clicked.connect(self._apply_unified)
        uni_row.addWidget(uni_apply)
        uni_test = QPushButton("测试")
        uni_test.clicked.connect(lambda: self._test_llm("story_analysis"))
        uni_row.addWidget(uni_test)
        prov_lay.addLayout(uni_row)

        grid = QGridLayout()
        grid.setHorizontalSpacing(12)
        grid.setVerticalSpacing(8)
        prov_lay.addLayout(grid)
        headers = ("角色", "类型", "模型", "API Key", "Base URL", "测试")
        for col, title_text in enumerate(headers):
            cell = QLabel(title_text)
            cell.setObjectName("hint")
            grid.addWidget(cell, 0, col)
        for row, role in enumerate(ROLES, start=1):
            grid.addWidget(QLabel(ROLE_LABELS[role]), row, 0)
            type_combo = QComboBox()
            type_combo.addItems(LLM_TYPES)
            grid.addWidget(type_combo, row, 1)
            model_edit = QLineEdit()
            model_edit.setMinimumWidth(190)
            grid.addWidget(model_edit, row, 2)
            key_edit = QLineEdit()
            key_edit.setEchoMode(QLineEdit.EchoMode.Password)
            key_edit.setMinimumWidth(170)
            grid.addWidget(key_edit, row, 3)
            base_edit = QLineEdit()
            base_edit.setMinimumWidth(220)
            grid.addWidget(base_edit, row, 4)
            test_btn = QPushButton("测试")
            test_btn.clicked.connect(lambda _=False, r=role: self._test_llm(r))
            grid.addWidget(test_btn, row, 5)
            self.role_widgets[role] = {"type": type_combo, "model": model_edit,
                                       "api_key": key_edit, "base_url": base_edit}
        grid.setColumnStretch(2, 2)
        grid.setColumnStretch(3, 2)
        grid.setColumnStretch(4, 3)
        prov_lay.addWidget(QLabel("提示：被内容策略拦截时自动回退 backup_models（见 config.json）。"))
        lay.addWidget(prov_card)

        lay.addWidget(section_title("配音 TTS API"))
        tts_card, tts_lay = card()
        tts_row = QHBoxLayout()
        tts_row.setSpacing(8)
        tts_row.addWidget(QLabel("类型"))
        self.tts_type = QComboBox()
        self.tts_type.addItems(TTS_TYPES)
        self.tts_type.currentTextChanged.connect(self._refresh_voices)
        tts_row.addWidget(self.tts_type)
        tts_row.addWidget(QLabel("音色"))
        self.tts_voice = QComboBox()
        self.tts_voice.setMinimumWidth(200)
        tts_row.addWidget(self.tts_voice)
        refresh_voices = QPushButton("刷新音色")
        refresh_voices.clicked.connect(lambda: self._refresh_voices())
        tts_row.addWidget(refresh_voices)
        tts_row.addWidget(QLabel("模型"))
        self.tts_model = QLineEdit()
        self.tts_model.setMinimumWidth(160)
        tts_row.addWidget(self.tts_model)
        tts_row.addWidget(QLabel("Base URL"))
        self.tts_base = QLineEdit()
        self.tts_base.setMinimumWidth(240)
        tts_row.addWidget(self.tts_base, 1)
        tts_row.addWidget(QLabel("语速"))
        self.tts_speed = QDoubleSpinBox()
        self.tts_speed.setRange(0.5, 2.5)
        self.tts_speed.setSingleStep(0.1)
        self.tts_speed.setValue(DEFAULT_TTS_SPEED)
        self.tts_speed.setDecimals(1)
        tts_row.addWidget(self.tts_speed)
        tts_row.addWidget(QLabel("API Key"))
        self.tts_key = QLineEdit()
        self.tts_key.setEchoMode(QLineEdit.EchoMode.Password)
        self.tts_key.setMinimumWidth(200)
        tts_row.addWidget(self.tts_key)
        tts_test = QPushButton("测试配音")
        tts_test.clicked.connect(self._test_tts)
        tts_row.addWidget(tts_test)
        tts_lay.addLayout(tts_row)
        tts_lay.addWidget(QLabel("百炼：类型选 bailian，模型 qwen3-tts-flash（或自定义音色模型），音色可填 Cherry/Serena/Ethan/Chelsie 或自有音色ID。"))
        lay.addWidget(tts_card)

        lay.addWidget(section_title("音色克隆（百炼声音复刻）"))
        clone_card, clone_lay = card()
        c1 = QHBoxLayout()
        c1.setSpacing(8)
        c1.addWidget(QLabel("参考音频"))
        self.clone_audio = QLineEdit()
        self.clone_audio.setMinimumWidth(320)
        self.clone_audio.setPlaceholderText("选择一段 10~60 秒清晰人声 wav/mp3")
        c1.addWidget(self.clone_audio, 1)
        browse = QPushButton("浏览")
        browse.clicked.connect(self._pick_audio)
        c1.addWidget(browse)
        clone_lay.addLayout(c1)
        c2 = QHBoxLayout()
        c2.setSpacing(8)
        c2.addWidget(QLabel("音色前缀(英文/数字)"))
        self.clone_prefix = QLineEdit("myvoice")
        self.clone_prefix.setMinimumWidth(140)
        c2.addWidget(self.clone_prefix)
        c2.addWidget(QLabel("复刻模型"))
        self.clone_model = QComboBox()
        self.clone_model.addItems(["cosyvoice-v2", "cosyvoice-v1"])
        self.clone_model.setCurrentText("cosyvoice-v2")
        c2.addWidget(self.clone_model)
        clone_btn = QPushButton("克隆并保存音色")
        clone_btn.setObjectName("accent")
        clone_btn.clicked.connect(self._clone_voice)
        c2.addWidget(clone_btn)
        self.clone_status = QLabel("")
        self.clone_status.setObjectName("hint")
        c2.addWidget(self.clone_status, 1)
        clone_lay.addLayout(c2)
        self.clone_list = QLabel("")
        self.clone_list.setObjectName("hint")
        clone_lay.addWidget(self.clone_list)
        self.voice_list = QListWidget()
        self.voice_list.setMinimumHeight(72)
        self.voice_list.setMaximumHeight(120)
        clone_lay.addWidget(self.voice_list)
        clr_row = QHBoxLayout()
        del_btn = QPushButton("删除选中音色")
        del_btn.setObjectName("danger")
        del_btn.clicked.connect(self._delete_voice)
        clr_row.addWidget(del_btn)
        listen_btn = QPushButton("试听选中音色")
        listen_btn.clicked.connect(self._audition_voice)
        clr_row.addWidget(listen_btn)
        clr_row.addStretch(1)
        clone_lay.addLayout(clr_row)
        lay.addWidget(clone_card)

        lay.addWidget(section_title("剪映草稿导出"))
        draft_card, draft_lay = card()
        draft_row = QHBoxLayout()
        draft_row.setSpacing(8)
        self.draft_detected = QLabel("")
        self.draft_detected.setObjectName("hint")
        draft_row.addWidget(QLabel("自动检测到"))
        draft_row.addWidget(self.draft_detected, 1)
        draft_row.addWidget(QLabel("手动指定目录"))
        self.draft_override = QLineEdit()
        self.draft_override.setMinimumWidth(300)
        draft_row.addWidget(self.draft_override)
        draft_lay.addLayout(draft_row)
        lay.addWidget(draft_card)

        lay.addWidget(section_title("网络（yt-dlp 代理）"))
        proxy_card, proxy_lay = card()
        proxy_row = QHBoxLayout()
        proxy_row.setSpacing(8)
        proxy_row.addWidget(QLabel("代理地址"))
        self.proxy_edit = QLineEdit()
        self.proxy_edit.setMinimumWidth(360)
        self.proxy_edit.setPlaceholderText("留空则读系统环境变量，例如 http://127.0.0.1:7890")
        proxy_row.addWidget(self.proxy_edit)
        proxy_row.addStretch(1)
        proxy_lay.addLayout(proxy_row)
        lay.addWidget(proxy_card)

        lay.addWidget(section_title("生产参数"))
        param_card, param_lay = card()
        pgrid = QGridLayout()
        pgrid.setHorizontalSpacing(16)
        pgrid.setVerticalSpacing(8)
        param_lay.addLayout(pgrid)
        for i, (key, label, kind) in enumerate(PARAM_SPEC):
            row, col = i // 3, (i % 3) * 2
            pgrid.addWidget(QLabel(label), row, col)
            if key in CHOICE:
                widget = QComboBox()
                widget.addItems(CHOICE[key])
            else:
                widget = QLineEdit()
                if key == "max_clips":
                    widget.setToolTip("0：自动制作全部合格故事；正整数：成片数量上限，例如 1 只做 1 条。")
            pgrid.addWidget(widget, row, col + 1)
            self.param_widgets[key] = (widget, kind)
        pgrid.setColumnStretch(5, 1)
        save_row = QHBoxLayout()
        save_btn = QPushButton("保存配置")
        save_btn.setObjectName("accent")
        save_btn.clicked.connect(self._save)
        self.save_hint = QLabel("")
        self.save_hint.setObjectName("hint")
        save_row.addWidget(save_btn)
        save_row.addWidget(self.save_hint)
        save_row.addStretch(1)
        param_lay.addLayout(save_row)
        lay.addWidget(param_card)

        lay.addStretch(1)
        scroll.setWidget(content)
        root.addWidget(scroll, 1)

        self.load()

    def load(self):
        config = self.main.config_obj
        for role in ROLES:
            conf = config.provider(role)
            widgets = self.role_widgets[role]
            self._set_combo(widgets["type"], conf.get("type", "mock"))
            widgets["model"].setText(conf.get("model", ""))
            widgets["api_key"].setText(conf.get("api_key", ""))
            widgets["base_url"].setText(conf.get("base_url", ""))
        first = config.provider("story_analysis")
        self._set_combo(self.uni_type, first.get("type", "openai"))
        self.uni_model.setText(first.get("model", ""))
        self.uni_base.setText(first.get("base_url", ""))
        self.uni_key.setText(first.get("api_key", ""))

        tts = config.tts
        self._set_combo(self.tts_type, tts.get("type", "edge"))
        self._refresh_voices()
        self._set_voice(tts.get("voice", ""))
        self.tts_model.setText(tts.get("model", ""))
        self.tts_base.setText(tts.get("base_url", ""))
        self.tts_speed.setValue(float(tts.get("speed", DEFAULT_TTS_SPEED)))
        self.tts_key.setText(tts.get("api_key", ""))

        self.draft_override.setText(str(config.param("capcut_draft_dir", "")))
        self._refresh_draft_detected()
        self.proxy_edit.setText(str(config.data.get("ytdlp", {}).get("proxy", "")))
        for key, (widget, _kind) in self.param_widgets.items():
            current = str(config.param(key, ""))
            if isinstance(widget, QComboBox):
                index = widget.findText(current)
                if index >= 0:
                    widget.setCurrentIndex(index)
            else:
                widget.setText(current)

    def _set_combo(self, combo, value):
        index = combo.findText(value)
        if index >= 0:
            combo.setCurrentIndex(index)

    def _set_voice(self, name):
        name = (name or "").strip()
        if not name:
            return
        index = self.tts_voice.findText(name)
        if index < 0:
            self.tts_voice.addItem(name)
            index = self.tts_voice.findText(name)
        self.tts_voice.setCurrentIndex(index)

    def _refresh_voices(self):
        kind = self.tts_type.currentText()
        presets = list(VOICE_PRESETS.get(kind, []))
        custom = [v for v in (self.main.config_obj.tts.get("voices") or []) if isinstance(v, dict)]
        current = self.tts_voice.currentText()
        self.tts_voice.blockSignals(True)
        self.tts_voice.clear()
        self._add_voice_header(self.tts_voice, VOICE_HEADER_SYS)
        self.tts_voice.addItems(presets)
        if custom:
            self._add_voice_header(self.tts_voice, VOICE_HEADER_CLONE)
            for v in custom:
                self.tts_voice.addItem(v.get("name") or v.get("voice_id"))
        self.tts_voice.blockSignals(False)
        if current and current not in (VOICE_HEADER_SYS, VOICE_HEADER_CLONE):
            self._set_voice(current)
        else:
            self.tts_voice.setCurrentIndex(1 if self.tts_voice.count() > 1 else 0)
        self._update_clone_list()

    @staticmethod
    def _add_voice_header(combo, text):
        combo.addItem(text)
        item = combo.model().item(combo.count() - 1)
        if item is not None:
            item.setEnabled(False)

    def _update_clone_list(self):
        voices = [v for v in (self.main.config_obj.tts.get("voices") or []) if isinstance(v, dict)]
        self.voice_list.clear()
        for v in voices:
            vid = v.get("voice_id", "")
            item = QListWidgetItem("%s    (%s)" % (v.get("name") or vid, vid))
            item.setData(Qt.ItemDataRole.UserRole, vid)
            self.voice_list.addItem(item)
        self.clone_list.setText("已克隆音色 %d 个（可删除）" % len(voices) if voices else "尚未克隆音色")

    def _delete_voice(self):
        item = self.voice_list.currentItem()
        if not item:
            QMessageBox.warning(self, "删除音色", "请先在下方列表选择要删除的克隆音色")
            return
        vid = item.data(Qt.ItemDataRole.UserRole)
        config = self.main.config_obj
        voices = [v for v in (config.tts.get("voices") or [])
                  if not (isinstance(v, dict) and v.get("voice_id") == vid)]
        if len(voices) == len(config.tts.get("voices") or []):
            return
        name = item.text().split(" ")[0]
        if QMessageBox.question(self, "删除音色", "确定删除克隆音色 %s ？" % name) != QMessageBox.StandardButton.Yes:
            return
        config.data.setdefault("tts", {})["voices"] = voices
        if config.data["tts"].get("voice") in (name, vid):
            config.data["tts"]["voice"] = ""
        config.save()
        try:
            from ....providers.tts.bailian import delete_voice
            delete_voice(config.tts.get("api_key", ""), vid, config.tts.get("base_url", ""))
        except Exception:
            pass
        self._refresh_voices()
        self.main.log_ui("已删除克隆音色 %s" % name)

    def _audition_voice(self):
        from ....providers.tts.manager import create_tts
        import tempfile
        from pathlib import Path

        voice = self.tts_voice.currentText()
        if voice in (VOICE_HEADER_SYS, VOICE_HEADER_CLONE) or not voice:
            QMessageBox.warning(self, "试听", "请先选择一个音色")
            return
        conf = {"type": self.tts_type.currentText(), "voice": voice,
                "speed": float(self.tts_speed.value()), "api_key": self.tts_key.text().strip(),
                "base_url": self.tts_base.text().strip(), "model": self.tts_model.text().strip(),
                "voices": self.main.config_obj.tts.get("voices") or []}
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            provider = create_tts(conf, allow_fallback=False)
            out = Path(tempfile.gettempdir()) / "ytsf_audition.wav"
            result = provider.synth("这是当前音色的试听效果，一二三四五。", out)
        except Exception as exc:
            QApplication.restoreOverrideCursor()
            QMessageBox.warning(self, "试听失败", str(exc)[:300])
            return
        QApplication.restoreOverrideCursor()
        from ....utils import open_path
        open_path(result["audio_path"])
        self.main.log_ui("试听 %s: %s (%.2fs)" % (voice, result["audio_path"], result.get("duration", 0)))

    def _pick_audio(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择参考音频", "", "音频 (*.wav *.mp3 *.m4a *.aac *.flac)")
        if path:
            self.clone_audio.setText(path)

    def _clone_voice(self):
        from ....providers.tts.bailian import create_voice

        audio = self.clone_audio.text().strip()
        if not audio:
            QMessageBox.warning(self, "音色克隆", "请先选择参考音频")
            return
        config = self.main.config_obj
        api_key = self.tts_key.text().strip() or config.tts.get("api_key", "")
        base_url = self.tts_base.text().strip() or config.tts.get("base_url", "")
        import re

        raw_prefix = self.clone_prefix.text().strip() or "myvoice"
        prefix = re.sub(r"[^A-Za-z0-9]", "", raw_prefix) or "myvoice"
        if prefix != raw_prefix:
            self.clone_prefix.setText(prefix)
            self.main.log_ui("音色前缀只能含英文字母和数字，已改为: %s" % prefix)
        model = self.clone_model.currentText()
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            voice_id = create_voice(api_key, audio, prefix=prefix, target_model=model, base_url=base_url)
        except Exception as exc:
            QApplication.restoreOverrideCursor()
            QMessageBox.warning(self, "音色克隆失败", str(exc)[:400])
            return
        QApplication.restoreOverrideCursor()
        voices = list(config.tts.get("voices") or [])
        existing = {v.get("name") for v in voices if isinstance(v, dict)}
        name, i = prefix, 2
        while name in existing:
            name = "%s%d" % (prefix, i)
            i += 1
        voices.append({"name": name, "voice_id": voice_id, "model": model})
        config.data.setdefault("tts", {})["voices"] = voices
        config.data["tts"]["type"] = "bailian"
        config.data["tts"]["voice"] = name
        config.save()
        self._refresh_voices()
        if self.tts_type.currentText() != "bailian":
            self._set_combo(self.tts_type, "bailian")
        self._set_voice(name)
        self.clone_status.setText("克隆成功: %s" % voice_id)
        QMessageBox.information(self, "音色克隆",
                                "成功，已保存并同步到「配音 TTS API」的音色下拉：\n%s\n（已自动选中）" % name)

    def _apply_unified(self):
        conf = {
            "type": self.uni_type.currentText(),
            "model": self.uni_model.text().strip(),
            "base_url": self.uni_base.text().strip(),
            "api_key": self.uni_key.text().strip(),
        }
        for role in ROLES:
            widgets = self.role_widgets[role]
            self._set_combo(widgets["type"], conf["type"])
            widgets["model"].setText(conf["model"])
            widgets["base_url"].setText(conf["base_url"])
            widgets["api_key"].setText(conf["api_key"])
        self.main.log_ui("已把统一分析 API 应用到全部角色")

    def _test_llm(self, role):
        from ....providers import LLMRouter

        self._sync_providers_to_config()
        router = LLMRouter(self.main.config_obj)
        provider = router.get(role)
        if provider.is_mock:
            QMessageBox.information(self, "测试", "%s 当前为 mock（未填 API Key）" % ROLE_LABELS[role])
            return
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            ok, msg = provider.test_connection()
        finally:
            QApplication.restoreOverrideCursor()
        QMessageBox.information(self, "测试 %s" % ROLE_LABELS[role], ("成功：\n" if ok else "失败：\n") + str(msg))

    def _test_tts(self):
        from ....providers.tts.manager import create_tts

        voice = self.tts_voice.currentText().strip()
        if voice in (VOICE_HEADER_SYS, VOICE_HEADER_CLONE) or not voice:
            QMessageBox.warning(self, "测试配音", "请先选择一个音色")
            return
        conf = {
            "type": self.tts_type.currentText(),
            "voice": self.tts_voice.currentText().strip(),
            "speed": float(self.tts_speed.value()),
            "api_key": self.tts_key.text().strip(),
            "base_url": self.tts_base.text().strip(),
            "model": self.tts_model.text().strip(),
            "voices": self.main.config_obj.tts.get("voices") or [],
        }
        provider = create_tts(conf, allow_fallback=False)
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            ok, msg = _tts_probe(provider)
        finally:
            QApplication.restoreOverrideCursor()
        QMessageBox.information(self, "测试配音", ("成功：\n" if ok else "失败：\n") + str(msg))

    def _sync_providers_to_config(self):
        config = self.main.config_obj
        for role in ROLES:
            widgets = self.role_widgets[role]
            existing = config.provider(role)
            config.set_provider(role, {
                "type": widgets["type"].currentText(),
                "model": widgets["model"].text().strip(),
                "api_key": widgets["api_key"].text().strip(),
                "base_url": widgets["base_url"].text().strip(),
                "timeout": existing.get("timeout", 300),
                "max_retry": existing.get("max_retry", 3),
                "backup_models": existing.get("backup_models", []),
            })

    def _refresh_draft_detected(self):
        root = detect_draft_root(self.draft_override.text().strip())
        self.main.draft_root = root
        self.draft_detected.setText(root or "未检测到剪映，成片时将跳过导出")

    def _save(self):
        config = self.main.config_obj
        params = config.data["params"]
        for key, (widget, kind) in self.param_widgets.items():
            if isinstance(widget, QComboBox):
                params[key] = widget.currentText()
                continue
            raw = widget.text().strip()
            try:
                if kind == "float":
                    params[key] = float(raw)
                elif kind == "int":
                    value = int(float(raw))
                    if key == "max_clips" and (value < 0 or float(raw) != value):
                        raise ValueError("成片数量必须是 0 或正整数")
                    params[key] = value
                else:
                    params[key] = raw
            except ValueError:
                QMessageBox.warning(self, "配置", "参数 %s 值无效: %s" % (key, raw))
                return
        self._sync_providers_to_config()
        voice = self.tts_voice.currentText().strip()
        if voice in (VOICE_HEADER_SYS, VOICE_HEADER_CLONE):
            voice = config.tts.get("voice", "")
        config.data["tts"] = {
            "type": self.tts_type.currentText(),
            "voice": voice,
            "speed": float(self.tts_speed.value()),
            "api_key": self.tts_key.text().strip(),
            "base_url": self.tts_base.text().strip(),
            "model": self.tts_model.text().strip(),
            "rate": config.tts.get("rate", ""),
            "volume": config.tts.get("volume", ""),
            "language": config.tts.get("language", ""),
            "emotion": config.tts.get("emotion", ""),
            "voices": config.tts.get("voices", []),
        }
        params["capcut_draft_dir"] = self.draft_override.text().strip()
        config.data.setdefault("ytdlp", {})["proxy"] = self.proxy_edit.text().strip()
        self._refresh_draft_detected()
        path = config.save()
        self.save_hint.setText("已保存 %s" % path)
        self.main.status_label.setText("配置已保存: %s" % path)
        self.main.log_ui("配置已保存: %s" % path)


def _tts_probe(provider):
    import tempfile
    from pathlib import Path

    try:
        out = Path(tempfile.gettempdir()) / "ytsf_tts_ui_probe.mp3"
        result = provider.synth("这是一段配音测试。", out)
        return True, "%s / %.2fs" % (result.get("provider"), result.get("duration", 0))
    except Exception as exc:
        return False, str(exc)[:300]
