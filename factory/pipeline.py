from datetime import datetime
from itertools import islice
from pathlib import Path
import time

from .config import Config, new_job_id
from .index import build_index, load_index
from .ingest import ingest
from .log import EventEmitter, get_logger, setup_logging
from .mining.diversity import filter_diversity
from .mining.planner import refine_boundaries
from .mining.ranker import rank_candidates
from .mining.scout import mine_stories
from .models import ClipResult, JobState, candidate_from_dict, to_dict
from .providers import LLMRouter
from .providers.base import reset_usage
from .providers.tts import TTSRouter
from .script import write_scripts
from .statemachine import STAGES, stage_label, stage_progress
from .story_count import automatic_count, clip_limit, limit_items
from .style import analyze_reference_style
from .utils import normalize_space, read_json, write_json


class PipelineError(RuntimeError):
    pass


class PipelineCancelled(PipelineError):
    pass


class ScriptReviewRequired(PipelineError):
    """A paid production step must wait for a concrete script correction."""
    pass


def _weak_sentence_indices(sentences, details, limit=3):
    """（兼容入口）转调 script.repair 的实现。"""
    from .script.repair import weak_sentence_indices
    return weak_sentence_indices(sentences, details, limit=limit)


def _dedupe_segments(segments):
    out = []
    seen = set()
    for seg in segments:
        key = (round(seg.start, 2), round(seg.end, 2), normalize_space(seg.text)[:24])
        if key in seen:
            continue
        seen.add(key)
        out.append(seg)
    return out


_TTS_JOB_FIELDS = ("type", "model", "voice", "speed", "rate", "volume", "language", "emotion",
                   "cosyvoice_model", "voice_model")


def _job_tts_settings(conf):
    """Only synthesis choices belong to a task; credentials stay global."""
    settings = {key: conf.get(key, "") for key in _TTS_JOB_FIELDS}
    voice = str(conf.get("voice") or "")
    settings["voices"] = [
        {key: item.get(key, "") for key in ("name", "voice_id", "model")}
        for item in conf.get("voices") or [] if isinstance(item, dict)
        and voice and voice in (item.get("name"), item.get("voice_id"))]
    return settings


def _legacy_tts_settings(job_dir):
    """Recover an unambiguous native CosyVoice v2 identity, never guess one."""
    identities = []
    for metadata in islice((job_dir / "clips").glob("*/tts.json"), 10):
        try:
            data = read_json(metadata) or {}
        except (OSError, ValueError, UnicodeError):
            continue
        if not isinstance(data, dict):
            continue
        voice = str(data.get("voice") or "")
        if (data.get("provider") != "bailian-cosyvoice" or not voice.startswith("cosyvoice-v2-")
                or data.get("model") not in (None, "", "cosyvoice-v2")):
            return None
        try:
            speed = float(data["playback_speed"])
        except (KeyError, ValueError, TypeError):
            return None
        if not 0.25 <= speed <= 4:
            return None
        identities.append((voice, speed))
    if not identities or len(set(identities)) != 1:
        return None
    voice, speed = identities[0]
    return {"type": "bailian", "model": "cosyvoice-v2", "voice": voice, "speed": speed,
            "cosyvoice_model": "cosyvoice-v2", "voice_model": "", "rate": "",
            "volume": "", "language": "", "emotion": "", "voices": []}


def _restore_job_tts(config, settings):
    if not isinstance(settings, dict):
        return
    conf = config.data.setdefault("tts", {})
    for key in _TTS_JOB_FIELDS:
        if key in settings:
            conf[key] = settings[key]
    # Keep unrelated current registry entries, restoring the selected alias
    # even if that name now points to a different clone in global settings.
    voices = [item for item in conf.get("voices") or [] if isinstance(item, dict)]
    for item in settings.get("voices") or []:
        if not isinstance(item, dict):
            continue
        saved = {key: item.get(key, "") for key in ("name", "voice_id", "model")}
        voices = [current for current in voices if not (
            saved["name"] and current.get("name") == saved["name"] or
            saved["voice_id"] and current.get("voice_id") == saved["voice_id"])]
        voices.append(saved)
    conf["voices"] = voices


class ProductionPipeline:
    def __init__(self, config, url, references=None, emitter=None, job_id=None, params_override=None):
        self.config = (config if isinstance(config, Config) else Config(config)).clone()
        self.url = url
        self.references = list(references or [])
        self.emitter = emitter or EventEmitter()
        self.logger = get_logger()
        self.job_id = job_id or new_job_id(url)
        self.job_dir = self.config.job_dir(self.job_id)
        setup_logging(self.job_dir / "pipeline.log")
        self.state = self._load_state()
        # Saved task choices win over later global defaults. New default keys
        # remain available, and an explicit caller override wins last.
        if isinstance(self.state.params, dict):
            self.config.data["params"].update(self.state.params)
        if params_override:
            self.config.data["params"].update(params_override)
        self.config.params['max_clips'] = clip_limit(self.config.params)
        self.state.params = dict(self.config.params)
        if references is None:
            self.references = list(self.state.references)
        self.url = self.url or self.state.url
        settings = self.state.artifacts.get("tts_settings")
        if not settings:
            settings = _legacy_tts_settings(self.job_dir)
        if settings:
            _restore_job_tts(self.config, settings)
        self.state.artifacts["tts_settings"] = _job_tts_settings(self.config.tts)
        self.router = LLMRouter(self.config)
        self.tts = TTSRouter(self.config)
        self.cancel_requested = False
        self.index = None
        self.candidates = []
        self.style_profile = None
        self.clips = []

    def _load_state(self):
        data = read_json(self.job_dir / "job.json")
        if data:
            state = JobState(**{k: v for k, v in data.items() if k in JobState.__annotations__})
            if state.url != self.url and self.url:
                state.url = self.url
            return state
        now = datetime.now().isoformat(timespec="seconds")
        return JobState(
            job_id=self.job_id, url=self.url, status="CREATED", created_at=now, updated_at=now,
            workspace=str(self.job_dir), references=self.references,
            params=dict(self.config.params),
        )

    def save_state(self):
        self.state.updated_at = datetime.now().isoformat(timespec="seconds")
        self.state.references = self.references
        self.state.params = dict(self.config.params)
        self.state.artifacts["tts_settings"] = _job_tts_settings(self.config.tts)
        write_json(self.job_dir / "job.json", to_dict(self.state))

    def _stage(self, name, detail=""):
        self._check_cancel()
        self.state.status = name
        self.state.stage_index = STAGES.index(name) if name in STAGES else -1
        self.save_state()
        self.emitter.stage("start", stage=name, detail=detail)
        self.emitter.progress(stage_progress(name), stage_label(name))
        self._stage_started = time.time()

    def _done(self, name, detail=""):
        if name in STAGES and name not in self.state.completed_stages:
            self.state.completed_stages.append(name)
        started = getattr(self, "_stage_started", None)
        if started:
            times = self.state.artifacts.setdefault("stage_times", {})
            times[name] = round(time.time() - started, 2)
        self.save_state()
        self.emitter.stage("done", stage=name, detail=detail)

    def completed(self, stage):
        return stage in self.state.completed_stages

    def _check_cancel(self):
        if self.cancel_requested:
            from .process import kill_active

            kill_active()
            raise PipelineCancelled("任务已取消")

    def cancel(self):
        """请求取消并立即中断正在运行的子进程（ffmpeg/whisper）。"""
        self.cancel_requested = True
        from .process import kill_active

        kill_active()

    def run(self):
        try:
            reset_usage()
            self.analyze()
            if (self.state.status == 'REVIEW_REQUIRED' and
                    self.state.artifacts.get('review_reason_stage') == 'STORY_MINING' and
                    not self._load_candidates()):
                return self.state
            pending = self.state.artifacts.get("pending_stories") or []
            discovery = self.state.artifacts.get('story_discovery') or {}
            discovery_wait = (self.state.artifacts.get('discovery_resume_required') or
                              discovery.get('manual_review_required'))
            available = self.review_queue()
            if not available and not pending:
                reason = discovery.get('reason') or '原片未发现符合质量要求的完整独立故事；本次不生成成片，也不凑数'
                return self._await_review(reason, 'STORY_MINING')
            if pending or discovery_wait or (self.config.param("review_mode") == "semi" and any(c.status != "approved" for c in available)):
                self.state.artifacts["script_review_pending"] = True
                detail = ("已保存%d段文案，还有%d段待续跑；可先审核已完成文案" % (len(self.candidates), len(pending))
                          if pending else "文案已生成，等待确认；尚未配音或匹配画面")
                if discovery_wait:
                    detail = discovery.get('reason') or '全片故事分析尚未确认完成，已保存文案可先审核'
                self.state.error = ""
                self._stage("CANDIDATE_FILTERED", detail)
                return self.state
            self.produce()
        except Exception as exc:
            self._fail(exc)
        return self.state

    def analyze(self):
        try:
            from .story_library import find_reviewed_story
            if (self.config.param("content_mode", "creative") == "creative"
                    and self.config.param("creative_strategy", "visual_story") == "visual_story"
                    and (not self.router.is_mock("script_generation")
                         or find_reviewed_story(self.url, self.config.params, self._reference_text()))):
                from .mining.visual_story import analyze
                return analyze(self)
            if not self.completed("URL_ANALYZING"):
                self._stage("URL_ANALYZING")
                self._do_ingest()
                self._done("URL_ANALYZING")
            if not self.completed("MEDIA_INDEXING"):
                self._stage("MEDIA_INDEXING")
                self._do_index()
                self._done("MEDIA_INDEXING")
            if not self.completed("VISUAL_ANALYSIS"):
                self._stage("VISUAL_ANALYSIS")
                self._do_visual()
                self._done("VISUAL_ANALYSIS")
            if not self.completed("STORY_MINING"):
                self._stage("STORY_MINING")
                self._do_mine()
                self._done("STORY_MINING")
            if not self._load_candidates():
                report = self.state.artifacts.get('story_discovery') or {}
                return self._await_review(report.get('reason') or '未发现可独立成片的完整故事，本次不凑数','STORY_MINING')
            if not self.completed("REFERENCE_ANALYZED"):
                self._stage("REFERENCE_ANALYZED")
                self._do_reference()
                self._done("REFERENCE_ANALYZED")
            if not self.completed("SCRIPT_GENERATED"):
                self._stage("SCRIPT_GENERATED")
                self._do_scripts()
                self._done("SCRIPT_GENERATED")
            if not self.completed("CANDIDATE_FILTERED"):
                if self.config.param("review_mode") == "semi":
                    for candidate in self._load_candidates():
                        if candidate.scripts and candidate.status not in ("approved", "rejected_by_user"):
                            candidate.selected_script = candidate.selected_script or candidate.scripts[0].variant_id
                            candidate.status = "review_required"
                    self._save_candidates()
                else:
                    self._stage("CANDIDATE_FILTERED", "选择生成文案；制作后只检查画面匹配")
                    self._do_match()
                self._done("CANDIDATE_FILTERED")
            else:
                self._load_candidates()
        except Exception as exc:
            self._fail(exc)
            raise
        return self.state

    def _await_review(self,detail,stage='CANDIDATE_FILTERED'):
        self.state.error = ''
        self.state.status = 'REVIEW_REQUIRED'
        self.state.stage_index = STAGES.index(stage)
        self.state.artifacts.update(script_review_pending=True,review_reason_stage=stage)
        self.save_state()
        self.emitter.stage('blocked',stage=stage,detail=detail)
        self.emitter.progress(stage_progress(stage),detail)
        self.emitter.update(candidates=[to_dict(c) for c in self.candidates])
        return self.state

    def produce(self):
        try:
            self._load_candidates()
            self._load_index()
            self._load_clips()
            from .production import resume_export_repairs, reconcile_production_queue
            self.state.error = ''
            queue = self.production_queue()
            if not queue:
                return self._await_review('暂无可制作的已确认文案，请先审核故事；本次未配音或匹配画面')
            reconcile_production_queue(self,queue)
            resume_export_repairs(self)
            self.state.artifacts["script_review_pending"] = False
            if not self.completed("TTS_GENERATED"):
                self._stage("TTS_GENERATED")
                self._do_tts(queue)
                self._done("TTS_GENERATED")
            if not self.completed("TIMELINE_ALIGNED"):
                self._stage("MATCHING", "根据整段配音和已确认正文匹配镜头，评论话术不参与剧情匹配")
                self._do_timeline(queue)
                self._done("MATCHING")
                self._stage("TIMELINE_ALIGNED", "保存与完整音频对应的剪辑时间轴")
                self._done("TIMELINE_ALIGNED")
            if not self.completed("RENDERING"):
                self._stage("RENDERING")
                self._do_render(queue)
                self._done("RENDERING")
            if not self.completed("FINAL_MATCH_CHECK"):
                self._stage("FINAL_MATCH_CHECK")
                self._do_final_check(queue)
                self._done("FINAL_MATCH_CHECK")
            if not self.completed("QUALITY_CHECK"):
                self._stage("QUALITY_CHECK")
                ok = self._do_qc(queue)
                self._done("QUALITY_CHECK")
                if not ok:
                    self.state.status = "REVIEW_REQUIRED"
                    self.save_state()
                    self.emitter.stage("blocked", stage="QUALITY_CHECK", detail="本次已生成%d条，暂无合格成片；请查看各条具体检查项" % len(self.clips))
                    self._write_report()
                    self.emitter.update(clips=[to_dict(c) for c in self.clips],
                                        candidates=[to_dict(c) for c in self.candidates])
                    return self.state
            if self.completed("QUALITY_CHECK") and not any(c.status == "completed" for c in self.clips):
                self.state.status = "REVIEW_REQUIRED"
                self.save_state()
                self._write_report()
                return self.state
            if not self.completed("DRAFT_EXPORT"):
                self._stage("DRAFT_EXPORT")
                self._do_draft_export(queue)
                self._done("DRAFT_EXPORT")
            self._stage("COMPLETED")
            self._write_report()
            self._done("COMPLETED")
            self.emitter.update(clips=[to_dict(c) for c in self.clips], candidates=[to_dict(c) for c in self.candidates])
        except ScriptReviewRequired as exc:
            self.state.artifacts['script_review_pending'] = True
            self.state.error = ''
            self._stage('CANDIDATE_FILTERED', str(exc))
            self.emitter.update(candidates=[to_dict(c) for c in self.candidates])
            return self.state
        except Exception as exc:
            self._fail(exc)
            raise
        return self.state

    def _do_ingest(self):
        downloading = [False]

        def cb(pct, name):
            if pct >= 0 and not downloading[0]:
                downloading[0] = True
                self._stage("DOWNLOADING", "下载媒体文件")
            if pct >= 0:
                self.emitter.progress(20.0 + pct * 12.0, "下载 %s" % Path(name).name)

        meta, video_path, subtitle_path = ingest(self.url, self.job_dir, progress_cb=cb,
                                             proxy=self.config.data.get("ytdlp", {}).get("proxy", ""))
        self._done("DOWNLOADING")
        self.state.artifacts["video_path"] = str(video_path)
        self.state.artifacts["subtitle_path"] = str(subtitle_path) if subtitle_path else ""
        self.state.artifacts["meta"] = to_dict(meta)
        self.save_state()
        self.emitter.update(meta=to_dict(meta))

    def _load_index(self):
        if self.index:
            return self.index
        if self.state.artifacts.get("video_path") and self.completed("MEDIA_INDEXING"):
            index_path = self.job_dir / "index" / "video_index.json"
            if index_path.exists():
                self.index = load_index(index_path)
                if self.index:
                    return self.index
        return None

    def _do_index(self):
        from .models import VideoMeta

        meta_data = self.state.artifacts.get("meta") or {}
        meta = VideoMeta(**{k: v for k, v in meta_data.items() if k in VideoMeta.__annotations__})
        video_path = Path(self.state.artifacts["video_path"])
        subtitle_path = self.state.artifacts.get("subtitle_path") or None
        index = build_index(
            meta, video_path, subtitle_path, self.config.params,
            out_dir=self.job_dir / "index",
            progress_cb=lambda p, d: self.emitter.progress(p * 100.0, "媒体索引: %s" % d),
            asr=self.config.asr,
            cache_dir=self.config.workspace_path() / ".cache" / "index",
        )
        self.index = index
        self.state.artifacts["index_path"] = str(self.job_dir / "index" / "video_index.json")
        self.save_state()

    def _do_visual(self):
        """先让视觉模型“看”原片每一段画面，把描述写回索引，供挖故事/写文案使用。"""
        from .index.visual import analyze_scene_visuals

        index = self._load_index()
        if not index:
            raise PipelineError("媒体索引缺失")
        provider = self.router.get("video_analysis")
        done = analyze_scene_visuals(index, provider,
                                     max_frames=int(self.config.param("visual_frames_per_scene", 1)),
                                     language=self.config.param("language", "zh"),
                                     batch_size=int(self.config.param("visual_batch_scenes", 6)),
                                     workers=int(self.config.param("visual_workers", 3)),
                                     progress_cb=lambda p, d: self.emitter.progress(2.0 + (p / 100.0) * 38.0, d))
        from .index import save_index

        if done:
            save_index(index, self.job_dir / "index" / "video_index.json")
            self.emitter.log("原片视觉分析完成: %d 个场景" % done)
        ocr_done = self._do_ocr(index)
        if ocr_done:
            save_index(index, self.job_dir / "index" / "video_index.json")
        self.state.artifacts["visual_scenes"] = done
        self.state.artifacts["ocr_scenes"] = ocr_done
        self.save_state()
        self._cleanup_frames()

    def _do_ocr(self, index):
        """可选 OCR：提取关键帧文字作为事实与写稿证据（需 pytesseract）。"""
        mode = str(self.config.param("ocr", "auto")).lower()
        if mode in ("off", "0", "false"):
            return 0
        from .index.ocr import analyze_scene_ocr, ocr_available

        if mode in ("auto", "on-demand") and not ocr_available():
            self.emitter.log("未安装 pytesseract，跳过画面文字识别")
            return 0
        return analyze_scene_ocr(index, language=self.config.param("language", "zh"),
                                 progress_cb=lambda p, d: self.emitter.log(d))

    def _cleanup_frames(self):
        """视觉分析完成后回收抽帧目录，释放磁盘（不影响索引/成片）。"""
        if str(self.config.param("cleanup_frames", "on")).lower() in ("off", "0", "false"):
            return
        import shutil

        index = self._load_index()
        if not index or not index.frames_dir:
            return
        try:
            shutil.rmtree(index.frames_dir, ignore_errors=True)
        except OSError:
            pass

    def _do_mine(self):
        index = self._load_index()
        if not index:
            raise PipelineError("媒体索引缺失")
        if self.router.is_mock("story_analysis"):
            self.state.artifacts["degraded"] = True
            self.emitter.log("剧情分析模型未配置（当前为 mock），已降级规则挖掘——出片质量会很差，"
                             "请在「配置」页填写分析 API Key/模型", "warning")
        raw = mine_stories(index, self.config.params, self.router, role="story_analysis")
        if getattr(raw,'discovery',None):
            self.state.artifacts['story_discovery'] = raw.discovery
        min_dur = float(self.config.param("min_duration", 15))
        max_dur = float(self.config.param("max_duration", 35))
        if not automatic_count(self.config.params):
            for cand in raw:
                refine_boundaries(cand, index, min_dur, max_dur)
        diverse = filter_diversity(raw, threshold=float(self.config.param("dedupe_threshold", 0.72)))
        ranked = rank_candidates(diverse)
        top_pool = len(ranked) if automatic_count(self.config.params) else int(self.config.param("candidate_pool", 12))
        ranked_ids = {c.candidate_id for c in ranked[:top_pool]}
        for cand in diverse:
            # 通过去重但价值排名靠后的候选
            if cand.candidate_id not in ranked_ids and cand.status == "generated":
                cand.status = "rejected_rank"
                cand.reject_reason = "故事价值排名靠后"
        for cand in raw:
            # 被多样性去重淘汰的候选
            if cand.status == "generated" and cand.candidate_id not in ranked_ids:
                cand.status = "rejected_diversity"
                cand.reject_reason = cand.reject_reason or "多样性淘汰"
        self.candidates = raw
        self._save_candidates()
        if index.transcript_source == "deferred":
            self._deferred_transcribe(index)
        self._done("CANDIDATES_GENERATED", "%d 候选" % len(ranked_ids))
        self.emitter.log("故事挖掘: %d 候选 → 保留 %d，淘汰 %d" % (
            len(raw), len(ranked_ids), len(raw) - len(ranked_ids)))

    def _deferred_transcribe(self, index):
        """无字幕长视频：仅对候选片段做 Whisper，补齐转写后回填场景与索引。"""
        from .index import save_index
        from .index.builder import _attach_transcript_to_scenes
        from .index.transcript import transcribe_windows

        targets = [c for c in self.candidates
                   if c.status in ("generated", "selected", "gate1_pass")]
        if not targets:
            return
        windows = [(c.source_start, c.source_end) for c in targets]
        video_path = index.video.path if index.video else self.state.artifacts.get("video_path", "")
        self.emitter.log("无字幕长视频：仅对 %d 个候选片段做语音转写" % len(windows))
        segs = transcribe_windows(
            video_path, windows,
            model_name=self.config.param("whisper_model", "small"),
            language=self.config.param("language", "zh"),
            pad=float(self.config.param("window_transcribe_pad", 3.0)),
        )
        if not segs:
            return
        merged = sorted(index.transcript + segs, key=lambda s: s.start)
        index.transcript = _dedupe_segments(merged)
        index.transcript_source = "whisper"
        from .index.transcript import quality_check_subtitles
        index.transcript_quality = quality_check_subtitles(index.transcript, index.video.duration if index.video else 0.0)
        _attach_transcript_to_scenes(index)
        save_index(index, self.job_dir / "index" / "video_index.json")
        self.emitter.log("候选片段转写完成: 新增 %d 段" % len(segs))

    def _load_candidates(self):
        if self.candidates:
            return self.candidates
        path = self.job_dir / "analysis" / "story_candidates.json"
        data = read_json(path)
        if data:
            self.candidates = [candidate_from_dict(d) for d in data]
            return self.candidates
        return []

    def _save_candidates(self):
        write_json(self.job_dir / "analysis" / "story_candidates.json", [to_dict(c) for c in self.candidates])
        self.emitter.update(candidates=[to_dict(c) for c in self.candidates])

    def _do_reference(self):
        profile = analyze_reference_style(self.references, self.router, role="reference_analysis")
        self.style_profile = profile
        write_json(self.job_dir / "analysis" / "reference_style.json", to_dict(profile))
        self.state.artifacts["style_profile"] = to_dict(profile)
        self.save_state()

    def _load_style(self):
        if self.style_profile:
            return self.style_profile
        from .models import StyleProfile
        data = self.state.artifacts.get("style_profile") or read_json(self.job_dir / "analysis" / "reference_style.json")
        if data:
            self.style_profile = StyleProfile(**{k: v for k, v in data.items() if k in StyleProfile.__annotations__})
        else:
            self.style_profile = analyze_reference_style(self.references, self.router, role="reference_analysis")
        return self.style_profile

    def _reference_text(self):
        """参考示例原文（供 creative 模式模仿风格）。"""
        try:
            from .style.loader import load_references
            texts, _ = load_references(self.references, max_refs=20)
            return "\n".join(texts)[:1500]
        except Exception:
            return ""

    def _analysis_cap(self):
        """最多深入分析（写稿+评分）多少个候选：按目标成片数决定，避免一股脑全分析。"""
        if automatic_count(self.config.params):
            return len(self._load_candidates())
        explicit = int(self.config.param("analyze_max_candidates", 0) or 0)
        if explicit > 0:
            return explicit
        quota = max(1, clip_limit(self.config.params))
        return max(quota + 2, 3)

    def _do_scripts(self):
        from concurrent.futures import ThreadPoolExecutor

        candidates = self._load_candidates()
        if not candidates:
            raise PipelineError("没有故事候选")
        if self.router.is_mock("script_generation"):
            self.state.artifacts["degraded"] = True
            self.emitter.log("文案生成模型未配置（当前为 mock），已降级兜底文案——出片质量会很差，"
                             "请在「配置」页填写分析 API Key/模型", "warning")
        index = self._load_index()
        style = self._load_style()
        cap = self._analysis_cap()
        pool = [c for c in candidates if c.status in ("generated", "selected", "gate1_pass")]
        targets = sorted(pool, key=lambda c: c.story_score, reverse=True)[:cap]
        for cand in pool:
            if cand not in targets and cand.status == "generated":
                cand.status = "not_selected"
                cand.reject_reason = "故事价值排名超出本次分析上限（目标成片 %d）" % int(self.config.param("max_clips", 1))

        def work(cand):
            self._check_cancel()
            return write_scripts(cand, style, index.transcript if index else [], self.config.params,
                                 self.router, role="script_generation", index=index,
                                 language=self.config.param("language", "zh"),
                                 reference_text=self._reference_text())

        workers = max(1, int(self.config.param("match_workers", 3)))
        if workers > 1 and len(targets) > 1:
            with ThreadPoolExecutor(max_workers=workers) as pool_exec:
                results = list(pool_exec.map(work, targets))
        else:
            results = [work(c) for c in targets]
        for cand, variants in zip(targets, results):
            cand.scripts = variants
        self.emitter.progress(100.0, "脚本生成完成 %d 个候选（分析上限 %d）" % (len(targets), cap))
        self._save_candidates()
        write_json(self.job_dir / "analysis" / "scripts.json", {
            c.candidate_id: [to_dict(v) for v in c.scripts] for c in targets
        })

    def _do_match(self):
        """Legacy candidate selection; matching happens after the full voice exists."""
        candidates = self._load_candidates()
        auto = self.config.param("review_mode", "auto") == "auto"
        with_scripts = sorted((c for c in candidates if c.scripts and c.status != "rejected_by_user"),
                              key=lambda c: c.story_score, reverse=True)
        selected = {c.candidate_id for c in limit_items(with_scripts, self.config.params)}
        for cand in candidates:
            self._check_cancel()
            if cand.status in ("approved", "rejected_by_user"):
                continue
            if not cand.scripts or cand.candidate_id not in selected:
                cand.status = "not_selected"
                cand.reject_reason = "未生成文案或超过手动故事数量上限"
                continue
            cand.selected_script = cand.selected_script or cand.scripts[0].variant_id
            cand.status = "gate1_pass" if auto else "review_required"
            # Keep the legacy status for saved-task compatibility, with an
            # explicit selection basis rather than a fabricated quality score.
            cand.match_report = {"gate1": {"status": "skipped", "basis": "selection_only"}}
            cand.reject_reason = ""
        self._save_candidates()
        self.emitter.log("已选择 %d 份文案；画面匹配将在整段配音后进行" % len(selected))

    def _apply_story_rebuild(self, candidates, index, style):
        """对未达标候选尝试微调故事边界重建（有界，仅前若干名）。"""
        if str(self.config.param("story_rebuild", "on")).lower() in ("off", "0", "false"):
            return
        from .mining.rebuild import rebuild_candidate

        failing = [c for c in candidates
                   if c.status == "review_required" or c.status.startswith("rejected")]
        failing.sort(key=lambda c: c.final_match, reverse=True)
        budget = len(failing) if automatic_count(self.config.params) else max(1, clip_limit(self.config.params))
        for cand in failing[:budget]:
            self._check_cancel()
            try:
                rebuild_candidate(self, cand, index, style, self.config.params, self.router)
            except Exception as exc:
                self.logger.warning("故事重建失败 %s: %s", cand.candidate_id, exc)

    def _apply_gate_fallback(self, candidates):
        """全部候选未过闸门时，取最高分者兜底入选，避免整批白做。"""
        if automatic_count(self.config.params):
            return
        if str(self.config.param("gate_fallback", "on")).lower() in ("off", "0", "false"):
            return
        if any(c.status == "gate1_pass" for c in candidates):
            return
        scored = [c for c in candidates if c.scripts and c.final_match > 0]
        if not scored:
            return
        scored.sort(key=lambda c: c.final_match, reverse=True)
        limit = max(1, clip_limit(self.config.params))
        for cand in scored[:limit]:
            cand.status = "gate1_pass"
            cand.reject_reason = ""
            cand.match_report.setdefault("gate1", {})
            cand.match_report["gate1"]["status"] = "pass"
            cand.match_report["gate1"]["fallback"] = True
            self.emitter.log("闸门兜底: %s 以最高分 %.1f 入选" % (cand.candidate_id, cand.final_match))

    def _regenerate_script(self, candidate, weak_variant, weakest, round_no):
        from .script.repair import regenerate_script
        return regenerate_script(self, candidate, weak_variant, weakest, round_no)

    def review_queue(self):
        return [c for c in self._load_candidates() if c.status in ("gate1_pass", "review_required", "approved")]

    def production_queue(self):
        auto = self.config.param("review_mode", "auto") == "auto"
        queue = []
        for cand in self._load_candidates():
            if cand.status == "approved":
                queue.append(cand)
            elif auto and cand.status == "gate1_pass":
                queue.append(cand)
        queue.sort(key=lambda c: c.final_match, reverse=True)
        return limit_items(queue,self.config.params)

    def apply_review(self, candidate_id, action, new_text=None):
        cand = next((c for c in self._load_candidates() if c.candidate_id == candidate_id), None)
        if not cand:
            return None
        if action == "approve":
            if new_text and new_text.strip() and new_text.strip() != self.selected_variant_text(cand):
                self._apply_script_edit(cand, new_text.strip())
            cand.status = "approved"
        elif action == "edit":
            if new_text and new_text.strip():
                self._apply_script_edit(cand, new_text.strip())
            cand.status = "review_required"
            self.state.artifacts["script_review_pending"] = True
            self.save_state()
        elif action == "reject":
            cand.status = "rejected_by_user"
            cand.reject_reason = "人工审核淘汰"
        elif action == "regenerate":
            self._regenerate_candidate(cand)
        self._save_candidates()
        return cand

    def _regenerate_candidate(self, cand):
        if self.state.artifacts.get("director_local_gate"):
            from .mining.visual_story import regenerate_candidate
            regenerate_candidate(self, cand)
            stages = {"TTS_GENERATED", "MATCHING", "TIMELINE_ALIGNED", "RENDERING", "FINAL_MATCH_CHECK", "QUALITY_CHECK", "DRAFT_EXPORT", "COMPLETED"}
            self.state.completed_stages = [stage for stage in self.state.completed_stages if stage not in stages]
            self.state.status = "CANDIDATE_FILTERED"
            self.state.stage_index = STAGES.index("CANDIDATE_FILTERED")
            self.state.error = ""
            self.state.artifacts["script_review_pending"] = True
            self.state.artifacts.pop("reviewed_source_sha256", None)
            self.state.artifacts.pop("reviewed_story_policy", None)
            for clip in self.clips:
                if clip.candidate_id == cand.candidate_id:
                    clip.status, clip.qc, clip.gate2 = "outdated_script", {}, {}
            if self.clips:
                self._save_clips()
            self._save_candidates()
            self.save_state()
            self.emitter.log("已依据独立分镜事实重写一份草稿，等待确认后才会配音")
            return
        index = self._load_index()
        style = self._load_style()
        # A failed explicit rewrite must leave the approved draft and its
        # already-rendered assets intact. Generate into a detached candidate.
        import copy
        trial = copy.deepcopy(cand)
        variants = write_scripts(trial, style, index.transcript if index else [], self.config.params,
                                 self.router, role="script_generation", index=index,
                                 language=self.config.param("language", "zh"),
                                 reference_text=self._reference_text())
        if not variants or not all(variant.sentences and variant.full_text.strip() for variant in variants):
            raise PipelineError("重写没有返回有效新稿，已保留原文案和已有音视频")
        trial.scripts = variants
        trial.selected_script = variants[0].variant_id
        trial.match_report = {}
        trial.final_match = 0.0
        trial.status = "review_required"
        cand.__dict__.update(copy.deepcopy(trial.__dict__))
        stages = {"TTS_GENERATED", "MATCHING", "TIMELINE_ALIGNED", "RENDERING", "FINAL_MATCH_CHECK", "QUALITY_CHECK", "DRAFT_EXPORT", "COMPLETED"}
        self.state.completed_stages = [stage for stage in self.state.completed_stages if stage not in stages]
        self.state.status = "CANDIDATE_FILTERED"
        self.state.stage_index = STAGES.index("CANDIDATE_FILTERED")
        self.state.error = ""
        self.state.artifacts["script_review_pending"] = True
        self.state.artifacts.pop("reviewed_source_sha256", None)
        self.state.artifacts.pop("reviewed_story_policy", None)
        for clip in self.clips:
            if clip.candidate_id == cand.candidate_id:
                clip.status, clip.qc, clip.gate2 = "outdated_script", {}, {}
        if self.clips:
            self._save_clips()
        self._save_candidates()
        self.save_state()

    def selected_variant_text(self, cand):
        for v in cand.scripts:
            if v.variant_id == cand.selected_script:
                return v.full_text
        return cand.scripts[0].full_text if cand.scripts else ""

    def _apply_script_edit(self, cand, new_text):
        from .script.review_edit import edit_variant
        variant = self._pick_variant(cand)
        if not variant:
            return
        fresh, story_changed = edit_variant(self, variant, new_text)
        spoken_changed = fresh.full_text != variant.full_text
        picture_changed = (fresh.scene_matches != variant.scene_matches or
                           [s.source_evidence for s in fresh.sentences] != [s.source_evidence for s in variant.sentences])
        if spoken_changed or picture_changed:
            stages = {"MATCHING", "TIMELINE_ALIGNED", "RENDERING", "FINAL_MATCH_CHECK", "QUALITY_CHECK", "DRAFT_EXPORT", "COMPLETED"}
            if spoken_changed:
                stages.add('TTS_GENERATED')
            self.state.completed_stages = [s for s in self.state.completed_stages if s not in stages]
            self.state.status = "CANDIDATE_FILTERED"
            self.state.stage_index = STAGES.index("CANDIDATE_FILTERED")
            self.state.error = ""
            for clip in self.clips:
                if clip.candidate_id == cand.candidate_id:
                    clip.status = "outdated_script" if spoken_changed else 'outdated_picture'
                    clip.qc = {}
            if self.clips:
                self._save_clips()
        if story_changed:
            fresh.match_report = {"gate1": {"status": "user_edited_pending_final_semantic_check", "model_score": None}}
            cand.match_report = fresh.match_report
            cand.final_match = 0
            self.state.artifacts.pop("reviewed_source_sha256", None)
            self.state.artifacts.pop("reviewed_story_policy", None)
        self._adopt_variant(cand, fresh)
        self.save_state()

    def _do_tts(self, queue):
        from .production import do_tts
        return do_tts(self, queue)

    def _adopt_variant(self, cand, variant):
        """把新变体持久化回候选并设为选中，保证 TTS 与时间轴一致。"""
        cand.scripts = [v for v in cand.scripts if v.variant_id != variant.variant_id] + [variant]
        cand.selected_script = variant.variant_id
        self._save_candidates()
        return variant

    def _pick_variant(self, cand):
        for v in cand.scripts:
            if v.variant_id == cand.selected_script:
                return v
        if cand.scripts:
            return max(cand.scripts, key=lambda v: (v.match_report or {}).get("final_score", 0.0))
        return None

    def _save_clips(self):
        write_json(self.job_dir / "clips" / "clips.json", [to_dict(c) for c in self.clips])

    def _load_clips(self):
        data = read_json(self.job_dir / "clips" / "clips.json")
        if data:
            self.clips = [ClipResult(**{k: v for k, v in d.items() if k in ClipResult.__annotations__}) for d in data]
        return self.clips

    def _do_timeline(self, queue):
        from .production import do_timeline
        return do_timeline(self, queue)

    def _do_render(self, queue):
        from .production import do_render
        return do_render(self, queue)

    def _do_final_check(self, queue):
        from .production import do_final_check
        return do_final_check(self, queue)

    def _do_qc(self, queue):
        from .production import do_qc
        return do_qc(self, queue)

    def _do_draft_export(self, queue):
        from .production import do_draft_export
        return do_draft_export(self, queue)

    def rerender_clip(self, clip_id, regions, method="blur"):
        from .production import rerender_clip
        return rerender_clip(self, clip_id, regions, method)

    def _by_id(self, candidate_id):
        for c in self._load_candidates():
            if c.candidate_id == candidate_id:
                return c
        raise PipelineError("候选不存在: %s" % candidate_id)

    def _write_report(self):
        from .report import write_report
        return write_report(self)

    def report(self):
        return read_json(self.job_dir / "reports" / "production_report.json") or {}

    def _fail(self, exc):
        if self.state.status in ("FAILED", "CANCELLED") and self.state.error == str(exc):
            return
        failed_stage = (self.state.status if self.state.status in STAGES else
                        self.state.artifacts.get("failed_stage", self.state.status))
        self.state.artifacts["failed_stage"] = failed_stage
        if isinstance(exc, PipelineCancelled) or (self.cancel_requested and isinstance(exc, PipelineError)):
            self.state.status = "CANCELLED"
            self.state.error = str(exc)
            self.save_state()
            self.emitter.stage("failed", stage=failed_stage, detail=str(exc))
            self.emitter.log("任务已取消", level="warning")
            return
        self.logger.exception("任务失败: %s", exc)
        self.state.status = "FAILED"
        self.state.error = str(exc)
        self.save_state()
        self.emitter.stage("failed", stage=failed_stage, detail=str(exc))
        self.emitter.log("任务失败: %s" % exc, level="error")
