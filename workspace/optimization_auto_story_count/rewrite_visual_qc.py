from pathlib import Path

path = Path(__file__).resolve().parents[2] / "factory" / "production.py"
source = path.read_text(encoding="utf-8")
start = source.index("def do_qc(pipeline, queue):")
end = source.index("def do_draft_export(pipeline, queue):", start)
replacement = '''def do_qc(pipeline, queue):
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


'''
path.write_text(source[:start] + replacement + source[end:], encoding="utf-8", newline="\n")
