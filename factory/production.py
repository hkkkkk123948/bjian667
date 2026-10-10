"""生产阶段（TTS/时间轴/渲染/复检/质检/草稿导出），从 pipeline 拆出。

所有函数接收 pipeline 实例，负责读写其 clips / state / config。
"""

from pathlib import Path

from .log import get_logger
from .render import build_timeline, export_clip, final_qc as final_qc  # Compatibility export.
from .render.qc import semantic_final_check
from .matching import quality_gate as quality_gate  # Compatibility export for older callers.
from .utils import ensure_dir, read_json, slugify, write_json

RENDER_POLICY = 'continuous_context_v4'


def reconcile_production_queue(pipeline, queue):
    from .production_queue import reconcile_production_queue as reconcile
    return reconcile(pipeline, queue, RENDER_POLICY)


def resume_export_repairs(pipeline):
    """Migrate failed old exports without rerunning approved narration."""
    if not pipeline.completed('RENDERING'):
        return
    import json
    from .utils import run_cmd
    render_again,review_again=False,False
    for clip in pipeline.clips:
        if (clip.scores or {}).get('render_policy')==RENDER_POLICY:
            continue
        if (clip.scores or {}).get('render_policy'):
            render_again = True
        if not clip.path:
            continue
        path=Path(clip.path)
        if not path.is_file():
            render_again=True
            continue
        out,_,_=run_cmd(['ffprobe','-v','error','-select_streams','v:0','-show_entries','stream=duration','-of','json',str(path)],timeout=60)
        streams=json.loads(out).get('streams',[])
        video=float(streams[0].get('duration',0)) if streams else 0
        voice=float((clip.scores or {}).get('tts_duration',clip.duration))
        black=(clip.qc or {}).get('media_audit',{}).get('black_periods')
        if abs(video-voice)>.16 or black:
            render_again=True
        timeline=read_json(path.parent/'timeline.json') or {}
        needed=2*len(timeline.get('sentence_timings',[]))
        if clip.qc and needed and len(list((path.parent/'qc_frames').glob('frame_*.jpg')))<needed:
            review_again=True
    if render_again:
        reset={'MATCHING','TIMELINE_ALIGNED','RENDERING','FINAL_MATCH_CHECK','QUALITY_CHECK','DRAFT_EXPORT','COMPLETED'}
        pipeline.emitter.log('将应用新版连续剪辑规则，复用完整配音重新生成画面','warning')
    elif review_again:
        reset={'FINAL_MATCH_CHECK','QUALITY_CHECK','DRAFT_EXPORT','COMPLETED'}
        pipeline.emitter.log('旧质检抽帧不足，复用现有音视频重新检查，不重做文案或配音','warning')
    else:
        return
    pipeline.state.completed_stages=[s for s in pipeline.state.completed_stages if s not in reset]
    pipeline.save_state()


def do_tts(pipeline, queue):
    from .pipeline import PipelineError
    from .production_queue import checkpoint_voice, verified_voice

    reconcile_production_queue(pipeline, queue)
    selected = {clip.candidate_id: clip for clip in pipeline.clips}
    missing = [cand for cand in queue if cand.candidate_id in selected
               and not verified_voice(pipeline, selected[cand.candidate_id], pipeline._pick_variant(cand))]
    preflight_scripts(pipeline, missing)
    for i, cand in enumerate(queue):
        pipeline._check_cancel()
        variant = pipeline._pick_variant(cand)
        if not variant:
            continue
        clip = selected[cand.candidate_id]
        saved = verified_voice(pipeline, clip, variant)
        if saved:
            actual = float(saved.get('duration', 0))
            minimum = float(pipeline.config.param('min_duration', 15))
            maximum = float(pipeline.config.param('max_duration', 35))
            if minimum - .16 <= actual <= maximum + .16:
                pipeline.emitter.log('复用%s完整配音，不重新合成' % clip.clip_id)
                continue
            _record_duration_warning(pipeline, clip, actual, minimum, maximum)
            continue
        from .script.review_edit import review_text, edit_variant
        from .script.comment_hook import story_text
        if any(any(ch in story_text(s) for ch in "（(") for s in variant.sentences):
            variant,_ = edit_variant(pipeline,variant,review_text(variant))
            variant = pipeline._adopt_variant(cand,variant)
        from .script.comment_hook import strip_hook, apply_comment_hook
        strip_hook(variant, float(pipeline.config.tts.get("speed", 1.2)))
        apply_comment_hook(pipeline, variant)
        pipeline._save_candidates()
        text = variant.full_text
        # Production never rewrites selected copy. An audio retry reads the
        # same whole text, in both automatic and manual approval modes.
        variant.user_approved_copy = True
        variant.confirmed_copy = True
        clip_id = clip.clip_id
        clip_dir = ensure_dir(pipeline.job_dir / "clips" / clip_id)
        if pipeline.config.param("voice_mode", "continuous") == "continuous":
            from .providers.tts.continuous import synth_continuous
            result = synth_continuous(pipeline.tts, variant, clip_dir)
        elif str(pipeline.config.param("sentence_tts", "off")).lower() == "on":
            from .providers.tts.sentences import synth_sentences
            result = synth_sentences(pipeline.tts, variant, clip_dir)
        else:
            result = pipeline.tts.synth(text, clip_dir / "voice.mp3")
        if result.get("provider") == "mock" and not pipeline.router.is_mock("script_generation"):
            raise PipelineError("真实TTS失败，拒绝导出Mock测试音")
        write_json(clip_dir / "tts.json", result)
        # Checkpoint each whole narration before requesting the next story.
        clip.gate1 = {"score": cand.final_match, "threshold": float(pipeline.config.param("min_match", 75))}
        checkpoint_voice(pipeline, clip, cand, variant, result)
        from .script.duration import record_duration
        # Duration history is a measurement, not a content acceptance gate.
        statistics = dict(result)
        statistics.pop('content_check', None)
        record_duration(pipeline.config, text, statistics)
        actual = float(result.get('duration',0))
        minimum = float(pipeline.config.param('min_duration',15))
        maximum = float(pipeline.config.param('max_duration',35))
        if actual < minimum - .16 or actual > maximum + .16:
            _record_duration_warning(pipeline, clip, actual, minimum, maximum)
        clip.scores.update(dict(variant.match_report or {}))
        pipeline.emitter.progress((i + 1) / len(queue) * 100.0, "TTS %s" % clip_id)
    if not pipeline.clips:
        raise PipelineError("TTS 阶段未生成任何配音")
    pipeline._save_clips()
    from .eval.narration_audit import release_local_models
    release_local_models()


def _record_duration_warning(pipeline, clip, actual, minimum, maximum):
    clip.scores['duration_target_warning'] = {'actual_seconds': actual, 'target_min': minimum,
                                             'target_max': maximum, 'basis': 'selected_copy_kept_intact'}
    pipeline.emitter.log('完整配音%.1f秒，参考目标%.0f–%.0f秒；仅记录片长，按整段配音匹配画面，不改词或截音' %
                         (actual, minimum, maximum), 'warning')
    pipeline._save_clips()


def preflight_scripts(pipeline, queue):
    """Check empty text and duration only; there is no copy quality gate."""
    from .pipeline import ScriptReviewRequired
    from .script.duration import estimate_duration, duration_state
    from .script.comment_hook import apply_comment_hook
    from .script.quality import compact
    problems = []
    for cand in queue:
        pipeline._check_cancel()
        variant = pipeline._pick_variant(cand)
        if not variant or not variant.sentences:
            cand.status = 'review_required'
            cand.reject_reason = '没有可朗读文案，请先填写文案'
            problems.append(cand.reject_reason)
            continue
        apply_comment_hook(pipeline, variant)
        if not compact(variant.full_text):
            cand.status = 'review_required'
            cand.reject_reason = '文案为空或只有标点，不能提交配音'
            problems.append(cand.reject_reason)
            continue
        estimate = estimate_duration(pipeline.config, variant.full_text)
        state = duration_state(estimate, float(pipeline.config.param('min_duration',15)),
                               float(pipeline.config.param('max_duration',35)))
        if state == 'too_long':
            pipeline.emitter.log('文案预计整段配音%.1f秒，可能超过参考片长；保留原稿，按实测整音匹配画面' %
                                 estimate.seconds, 'warning')
        elif state in ('too_short','near_short'):
            pipeline.emitter.log('文案%s预计配音%.1f秒，可能短于目标；保留已确认稿，未自动补词' % (cand.candidate_id,estimate.seconds),'warning')
    pipeline._save_candidates()
    if problems:
        raise ScriptReviewRequired('配音准备需处理：' + '；'.join(problems))


def rerender_clip(pipeline, clip_id, regions, method="blur"):
    """按用户框选的区域重渲染单条成片（去水印/字幕）。"""
    from pathlib import Path

    if not pipeline.clips:
        pipeline._load_clips()
    clip = next((c for c in pipeline.clips if c.clip_id == clip_id), None)
    if not clip:
        return None
    pipeline._load_index()
    if regions and method:
        clip.scores = clip.scores or {}
        clip.scores["mask_regions"] = list(regions or [])
        clip.scores["mask_method"] = method
    clip_dir = Path(pipeline.job_dir) / "clips" / clip_id
    # 重建时间轴（含证据片段），保证方向/时长/留白按当前设置
    cand = pipeline._by_id(clip.candidate_id)
    variant = pipeline._pick_variant(cand)
    from .script.comment_hook import apply_comment_hook
    apply_comment_hook(pipeline, variant)
    from .story_library import refresh_evidence_windows
    refresh_evidence_windows(pipeline.state.artifacts["video_path"], variant)
    pipeline._save_candidates()
    tts_data = read_json(clip_dir / "tts.json") or {}
    if pipeline.config.param("voice_mode", "continuous") == "continuous":
        from .providers.tts.continuous import synth_continuous
        variant.user_approved_copy = True
        variant.confirmed_copy = True
        tts_data = synth_continuous(pipeline.tts, variant, clip_dir)
        write_json(clip_dir / "tts.json", tts_data)
        clip.scores["tts_path"] = tts_data["audio_path"]
        clip.scores["tts_duration"] = tts_data["duration"]
        clip.script_text = variant.full_text
    timeline = build_timeline(cand, variant, tts_data, pipeline.config.params,
                              video_duration=pipeline.index.video.duration if pipeline.index else 0.0,
                              index=pipeline.index, router=pipeline.router)
    write_json(clip_dir / "timeline.json", timeline)
    rerender_export(pipeline, clip, clip_dir, timeline)
    pipeline._save_clips()
    return clip


def rerender_export(pipeline, clip, clip_dir, timeline):
    """按时间轴（含证据片段）对单条 clip 执行导出渲染。

    供 rerender_clip 与落地化重写共用，保证画幅、去黑边、去水印、焦点跟踪一致。
    """
    from pathlib import Path

    from .render.clean import detect_black_bars
    from .render.reframing import estimate_focus, focus_expression, resolve_output_size
    from .utils import media_duration, media_size

    video_path = Path(pipeline.state.artifacts["video_path"])
    src_w, src_h = 0, 0
    try:
        src_w, src_h = media_size(video_path)
    except Exception:
        pass
    if src_w and src_h:
        out_w, out_h = resolve_output_size(src_w, src_h, pipeline.config.params)
        pipeline.config.params["target_width"], pipeline.config.params["target_height"] = out_w, out_h
        pipeline.logger.info("画幅: %dx%d（源 %dx%d，跟随源片比例）", out_w, out_h, src_w, src_h)
    clip_dir = Path(clip_dir)
    start = float(timeline.get("source_start", clip.source_start))
    span = float(timeline.get("source_duration", clip.duration))
    focus = estimate_focus(video_path, start, start + span, pipeline.config.params)
    expr = focus_expression(video_path, start, start + span, pipeline.config.params)
    black_box = pipeline.state.artifacts.get("black_box")
    if black_box:
        black_box = tuple(black_box)
    else:
        black_box = detect_black_bars(video_path, pipeline.config.params)
    out_path = clip_dir / (clip.clip_id + ".mp4")
    voice_path = (read_json(clip_dir / "tts.json") or {}).get("audio_path") or clip.scores.get("tts_path")
    export_clip(video_path, voice_path, None, timeline,
                pipeline.config.params, out_path, workdir=clip_dir,
                focus=focus, focus_expr=expr, black_box=black_box,
                src_size=(src_w, src_h),
                regions=(clip.scores or {}).get("mask_regions"),
                region_method=(clip.scores or {}).get("mask_method"))
    clip.path = str(out_path)
    try:
        clip.duration = media_duration(out_path)
    except Exception:
        pass
    clip.status = "completed"
    clip.scores["timeline"] = timeline
    # A rerender invalidates the previous output's audit and model judgement.
    clip.qc = {}
    clip.gate2 = {}
    return clip


def do_timeline(pipeline, queue):
    from .production_queue import timeline_ready, picture_key
    if not pipeline.clips:
        pipeline._load_clips()
    pipeline._load_index()
    for clip in pipeline.clips:
        pipeline._check_cancel()
        cand = pipeline._by_id(clip.candidate_id)
        variant = pipeline._pick_variant(cand)
        if timeline_ready(pipeline, clip, cand, variant):
            continue
        tts_path = pipeline.job_dir / 'clips' / clip.clip_id / 'tts.json'
        tts_data = read_json(tts_path) or {}
        try:
            tts_data = upgrade_narration_alignment(pipeline, variant, tts_data, tts_path)
            from .matching.initial_plan import build_initial_timeline
            catalog = _initial_catalog(pipeline, cand)
            timeline = build_initial_timeline(pipeline, cand, variant, tts_data, catalog)
        except ValueError as exc:
            from .pipeline import PipelineError
            raise PipelineError('画面时间轴匹配失败，已保留原文案和完整配音：' + str(exc)) from exc
        write_json(pipeline.job_dir / "clips" / clip.clip_id / "timeline.json", timeline)
        clip.scores["timeline"] = timeline
        clip.scores['picture_key'] = picture_key(pipeline, clip, cand, variant)
        clip.status, clip.qc, clip.gate2 = 'timeline_done', {}, {}
        pipeline._save_candidates()
        pipeline._save_clips()
    pipeline._save_clips()


def _initial_catalog(pipeline, candidate):
    from .mining.shot_evidence import get_shot_catalog, build_and_observe
    catalog = get_shot_catalog(pipeline, candidate)
    if not catalog:
        folder = pipeline.job_dir / 'analysis' / 'director' / ('story_%02d' % int(candidate.candidate_id.removeprefix('C')))
        pipeline.emitter.log('建立真实切镜头目录，按实际人物与动作首次分配画面')
        catalog = build_and_observe(pipeline, Path(pipeline.state.artifacts['video_path']),
                                    candidate.source_start, candidate.source_end, folder)
    return catalog


def upgrade_narration_alignment(pipeline, variant, result, metadata_path):
    """Upgrade old timing metadata using the existing complete audio only."""
    if pipeline.config.param('voice_mode','continuous') != 'continuous':
        return result
    if str(result.get('provider','')).startswith(('mock','test_')):
        return result
    from .providers.tts.continuous import align_continuous_result
    from .utils import media_duration
    from .eval.narration_audit import audit_narration, release_local_models
    audio = Path(result.get('audio_path',''))
    if not audio.is_file():
        raise RuntimeError('完整配音文件不存在，无法对齐画面')
    if result.get('text') and result['text'] != ''.join(s.text for s in variant.sentences):
        raise RuntimeError('已保存配音与当前文案不一致，请先确认修改后的文案')
    duration = media_duration(audio)
    aligned = align_continuous_result(variant.sentences,result,duration)
    if aligned['timing_source'] == 'estimated' and pipeline.config.param('narration_audit','local') == 'local':
        pipeline.emitter.log('复用完整配音进行本地文字时间对齐，不重新合成')
        try:
            report = audit_narration(audio, ''.join(s.text for s in variant.sentences),
                                     Path(metadata_path).parent/'narration_alignment.audit.json',
                                     model_name=pipeline.config.param('narration_audit_model','small'),
                                     require_word_timings=True)
            if report.get('status') != 'pass':
                # Recognition supplies timing hints, never an audio verdict.
                report = dict(report, comparison_status=report.get('status'), status='alignment_only')
                pipeline.emitter.log('本地识别与文字有差异，仅用于时间对齐，保留整段配音继续匹配', 'warning')
            aligned = align_continuous_result(variant.sentences,result,duration,report)
            aligned['content_check'] = report
        except Exception as exc:
            if getattr(pipeline, 'cancel_requested', False):
                raise
            pipeline.emitter.log('本地时间对齐暂不可用，保留完整配音并使用估计时间：%s' % exc, 'warning')
            aligned['alignment_report'] = dict(aligned.get('alignment_report') or {}, confidence='low',
                                               local_alignment_unavailable=True)
        finally:
            release_local_models()
    write_json(metadata_path, aligned)
    source = aligned.get('timing_source','estimated')
    pipeline.emitter.log('配音时间依据：%s；置信度：%s' % (source,aligned.get('alignment_report',{}).get('confidence','low')))
    return aligned


def do_render(pipeline, queue):
    from .production_queue import render_ready, render_key, file_digest
    if not pipeline.clips:
        pipeline._load_clips()
    video_path = Path(pipeline.state.artifacts["video_path"])
    src_w, src_h = 0, 0
    try:
        from .utils import media_size
        src_w, src_h = media_size(video_path)
    except Exception:
        pass
    from .render.reframing import estimate_focus, focus_expression, resolve_output_size
    out_w, out_h = resolve_output_size(src_w, src_h, pipeline.config.params)
    pipeline.config.params["target_width"], pipeline.config.params["target_height"] = out_w, out_h
    pipeline.state.artifacts["source_size"] = [src_w, src_h]
    pipeline.state.artifacts["output_size"] = [out_w, out_h]
    pipeline.logger.info("画幅自适应: 源 %dx%d → 输出 %dx%d", src_w, src_h, out_w, out_h)
    from .render.clean import clean_preview_filter, detect_black_bars
    black_box = detect_black_bars(video_path, pipeline.config.params)
    if black_box:
        pipeline.logger.info("检测到黑边，裁剪框: %s", black_box)
    pipeline.state.artifacts["black_box"] = list(black_box) if black_box else None
    pipeline.emitter.log(clean_preview_filter(pipeline.config.params, black_box))
    for i, clip in enumerate(pipeline.clips):
        pipeline._check_cancel()
        if render_ready(pipeline, clip, RENDER_POLICY):
            continue
        clip_dir = pipeline.job_dir / "clips" / clip.clip_id
        timeline = read_json(clip_dir / "timeline.json")
        voice = timeline.get("voice_path") or (clip.scores.get("tts_path"))
        start = float(timeline.get("source_start", 0.0))
        span = float(timeline.get("source_duration", 0.0))
        focus = estimate_focus(video_path, start, start + span, pipeline.config.params)
        expr = focus_expression(video_path, start, start + span, pipeline.config.params)
        timeline["crop_focus"] = round(focus, 4)
        write_json(clip_dir / "timeline.json", timeline)
        out_path = clip_dir / (clip.clip_id + ".mp4")
        export_clip(video_path, voice, None, timeline, pipeline.config.params, out_path,
                    workdir=clip_dir, focus=focus, focus_expr=expr, black_box=black_box,
                    src_size=(src_w, src_h),
                    regions=(clip.scores or {}).get("mask_regions"),
                    region_method=(clip.scores or {}).get("mask_method"))
        clip.path = str(out_path)
        clip.subtitle_path = ""
        from .utils import media_duration
        clip.duration = media_duration(out_path)
        clip.status = "rendered"
        clip.scores['render_policy'] = RENDER_POLICY
        clip.scores['timeline'] = read_json(clip_dir/'timeline.json') or timeline
        clip.scores['render_key'] = render_key(pipeline, clip, RENDER_POLICY)
        clip.scores['render_media_digest'] = file_digest(clip.path)
        clip.qc, clip.gate2 = {}, {}
        pipeline._save_clips()
        pipeline.emitter.progress((i + 1) / len(pipeline.clips) * 100.0, "渲染 %s" % clip.clip_id)


def do_final_check(pipeline, queue):
    from .production_queue import final_check_ready, quality_key, REVIEW_POLICY
    if not pipeline.clips:
        pipeline._load_clips()
    for clip in pipeline.clips:
        cand = pipeline._by_id(clip.candidate_id)
        variant = pipeline._pick_variant(cand)
        if final_check_ready(pipeline, clip, cand, variant):
            continue
        # Kept only to resume older task state machines. This is not another
        # review: export failures remain errors, and QC performs one picture review.
        clip.gate2 = {"status": "skipped", "basis": "single_visual_review_only"}
        clip.scores['final_check_key'] = quality_key(pipeline, clip, cand, variant)
        clip.scores['quality_review_policy'] = REVIEW_POLICY
        pipeline._save_clips()
    pipeline._save_clips()


def do_qc(pipeline, queue):
    """The single user-facing review: pictures against the selected narration."""
    from .production_queue import quality_ready, quality_key, REVIEW_POLICY
    from .pipeline import PipelineError
    if not pipeline.clips:
        pipeline._load_clips()
    passed = []
    source_size = pipeline.state.artifacts.get("source_size")
    if source_size and all(source_size):
        from .render.reframing import resolve_output_size
        width, height = resolve_output_size(*source_size, pipeline.config.params)
        pipeline.config.params.update(target_width=width, target_height=height)
    for clip in pipeline.clips:
        pipeline._check_cancel()
        candidate = pipeline._by_id(clip.candidate_id)
        variant = pipeline._pick_variant(candidate)
        if not clip.path or not Path(clip.path).is_file() or Path(clip.path).stat().st_size == 0:
            raise PipelineError('输出视频不存在或为空，无法审核画面；请先完成真实导出')
        if hasattr(pipeline, 'job_dir') and quality_ready(pipeline, clip, candidate, variant):
            if clip.qc.get('status') == 'pass':
                clip.status = 'completed'
                passed.append(clip)
            else:
                clip.status = 'qc_failed'
            continue
        from .script.comment_hook import story_text
        narrative = ''.join(story_text(sentence) for sentence in variant.sentences)
        qc_params = dict(pipeline.config.params, user_approved_copy=True, confirmed_copy=True)
        semantic = semantic_final_check(clip, narrative, '', getattr(pipeline, 'router', None),
                                        params=qc_params, visuals='')
        if not semantic:
            semantic = {'status': 'review', 'score': None, 'basis': 'visual_model_unavailable',
                        'reason': '画面审核模型未配置或不可用，尚未确认匹配结果'}
        status = semantic.get('status')
        if status not in ('pass', 'fail', 'review'):
            status = 'review'
        report = {'status': status, 'issues': [], 'semantic_check': semantic,
                  'visual_match_check': semantic, 'review_scope': 'picture_matching_only'}
        if status != 'pass':
            report['issues'].append('画面匹配待处理：' + semantic.get('reason', '请查看对应句子的画面匹配问题'))
        clip.qc = report
        if hasattr(pipeline, 'job_dir'):
            clip.scores['quality_key'] = quality_key(pipeline, clip, candidate, variant)
            clip.scores['quality_review_policy'] = REVIEW_POLICY
        if status == 'pass':
            clip.status = 'completed'
            passed.append(clip)
        else:
            clip.status = 'qc_failed'
        pipeline._save_clips()
    pipeline._save_clips()
    if not passed:
        get_logger().warning('本次画面匹配审核尚未通过，请重新匹配对应镜头')
    return bool(passed)


def do_draft_export(pipeline, queue):
    if not pipeline.clips:
        pipeline._load_clips()
    from .export.capcut import detect_draft_root, export_draft
    root = detect_draft_root(pipeline.config.param("capcut_draft_dir", ""))
    pipeline.state.artifacts["draft_root"] = root
    pipeline.save_state()
    if not root:
        pipeline.emitter.log("未检测到剪映草稿目录，跳过草稿导出"
                             "（可在系统配置中指定 capcut_draft_dir）", "warning")
        return
    width = int(pipeline.config.param("target_width", 1080))
    height = int(pipeline.config.param("target_height", 1920))
    exported = 0
    for clip in pipeline.clips:
        pipeline._check_cancel()
        if clip.status != "completed":
            continue
        if clip.draft_path and (Path(clip.draft_path) / 'draft_content.json').is_file():
            exported += 1
            continue
        cand = pipeline._by_id(clip.candidate_id)
        title = cand.title or "成片 %s" % clip.clip_id
        folder_name = "%s_%s" % (clip.clip_id, slugify(title, 32))
        try:
            draft_dir = export_draft(
                clip.path, root, title, clip.duration or 0.0,
                width, height, folder_name=folder_name,
                source_start=0.0, source_duration=clip.duration or 0.0,
                voice_path=clip.scores.get("tts_path"),
                orig_audio_path=str(Path(clip.path).parent / "soundbed.wav") if (Path(clip.path).parent / "soundbed.wav").exists() else None)
            clip.draft_path = str(draft_dir)
            exported += 1
        except Exception as exc:
            pipeline.logger.warning("剪映草稿导出失败 %s: %s", clip.clip_id, exc)
            clip.draft_path = ""
            pipeline.emitter.log("剪映草稿导出失败 %s: %s" % (clip.clip_id, exc), "warning")
    pipeline._save_clips()
    done = sum(1 for c in pipeline.clips if c.status == "completed")
    pipeline.emitter.log("剪映草稿导出: %d/%d 部成片" % (exported, done))
