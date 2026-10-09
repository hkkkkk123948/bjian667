import argparse
import sys
from pathlib import Path

from factory.config import Config, ROLES, ROLE_LABELS, LLM_TYPES, TTS_TYPES
from factory.log import EventEmitter
from factory.pipeline import ProductionPipeline
from factory.story_count import clip_limit
from factory.utils import ensure_utf8_console, fmt_clock, open_path


def make_emitter(verbose=True):
    def sink(event):
        kind = event.get("kind")
        if kind == "log":
            if verbose:
                print("[%s] %s" % (event.get("level", "info"), event.get("message", "")), flush=True)
        elif kind == "stage":
            status = event.get("status", "")
            if status in ("start", "done"):
                mark = "→" if status == "start" else "✓"
                print("%s %s %s" % (mark, event.get("stage", ""), event.get("detail", "")), flush=True)
            elif status in ("failed", "blocked"):
                print("✗ %s %s" % (event.get("stage", ""), event.get("detail", "")), flush=True)
        elif kind == "progress":
            print("  [%3.0f%%] %s" % (event.get("value", 0), event.get("detail", "")), flush=True)
    return EventEmitter(sink)


def load_config(path=None):
    candidate = Path(path) if path else Path("workspace/config.json")
    if candidate.exists():
        return Config.load(candidate)
    return Config(path=candidate)


def cmd_run(args):
    config = load_config(args.config)
    overrides = {}
    if args.mode:
        overrides['review_mode'] = args.mode
    if args.max is not None:
        overrides['max_clips'] = clip_limit({'max_clips':args.max})
    emitter = make_emitter(not args.quiet)
    pipe = ProductionPipeline(config, args.url, references=args.refs or [],
                              emitter=emitter, job_id=args.resume,params_override=overrides or None)
    config = pipe.config
    state = pipe.run()
    if state.status == "FAILED":
        print("任务失败: %s" % state.error)
        return 1
    queue = pipe.review_queue()
    if config.param("review_mode") == "auto" or not queue:
        return 0 if state.status == "COMPLETED" else 2
    return interactive_review(pipe, queue)


def interactive_review(pipe, queue):
    print("\n=== 待人工审核 (%d) ===" % len(queue))
    for i, cand in enumerate(queue, 1):
        print("\n[%d] %s  得分 %.1f  故事分 %d  %s~%s" % (
            i, cand.candidate_id, cand.final_match, cand.story_score,
            fmt_clock(cand.source_start), fmt_clock(cand.source_end)))
        print("    %s" % cand.story_summary[:80])
        print("    文案: %s" % pipe.selected_variant_text(cand)[:100])
    print("\n命令: y=通过  n=淘汰  e=编辑后通过  s=跳过  done=结束审核并生产")
    approved = []
    for cand in queue:
        try:
            cmd = input("%s [y/n/e/s] " % cand.candidate_id).strip().lower()
        except EOFError:
            cmd = "s"
        if cmd == "done":
            break
        if cmd == "y":
            approved.append(cand.candidate_id)
            pipe.apply_review(cand.candidate_id, "approve")
        elif cmd == "n":
            pipe.apply_review(cand.candidate_id, "reject")
            print("  已淘汰 %s" % cand.candidate_id)
        elif cmd == "e":
            print("输入新文案（单行）:")
            text = input("> ")
            if text.strip():
                pipe.apply_review(cand.candidate_id, "approve", new_text=text)
                approved.append(cand.candidate_id)
        else:
            if cand.candidate_id in approved:
                approved.remove(cand.candidate_id)
    state = pipe.produce()
    if state.status == "COMPLETED":
        print("全部完成，成片见 %s" % (pipe.job_dir / "clips"))
        return 0
    print("生产结束状态: %s %s" % (state.status, state.error))
    return 2


def cmd_list(args):
    config = load_config(args.config)
    jobs_dir = config.workspace_path() / "jobs"
    if not jobs_dir.exists():
        print("暂无任务")
        return 0
    rows = []
    for job in sorted(jobs_dir.iterdir()):
        data = job / "job.json"
        if not data.exists():
            continue
        import json
        info = json.loads(data.read_text(encoding="utf-8"))
        rows.append((info.get("job_id", job.name), info.get("status", ""),
                     info.get("url", ""), info.get("updated_at", "")))
    if not rows:
        print("暂无任务")
        return 0
    print("%-24s %-18s %-10s %s" % ("JOB", "STATUS", "UPDATED", "SOURCE"))
    for job_id, status, url, updated in rows:
        print("%-24s %-18s %-10s %s" % (job_id, status, updated[5:16] if updated else "", url))
    return 0


def cmd_report(args):
    config = load_config(args.config)
    job_dir = config.workspace_path() / "jobs" / args.job_id
    path = job_dir / "reports" / "production_report.json"
    if not path.exists():
        print("报告不存在: %s" % path)
        return 1
    import json
    report = json.loads(path.read_text(encoding="utf-8"))
    print("来源: %s" % report.get("source_video"))
    print("候选 %s / 入选 %s / 淘汰 %s" % (
        report.get("candidate_count"), report.get("selected_count"), report.get("rejected_count")))
    for clip in report.get("clips", []):
        print("  %s  %s  故事 %s  匹配 %.1f  闸门2 %s  QC %s" % (
            clip["clip_id"], clip["source"], clip["story_score"],
            clip.get("final_match", 0), clip["gate2"]["status"], clip["qc"]))
        print("      %s" % clip["path"])
    for rej in report.get("rejected", []):
        print("  淘汰 %s [%s] %s" % (rej["candidate_id"], rej["status"], rej["reason"]))
    if args.open:
        open_path(path)
    return 0


def cmd_config(args):
    config = load_config(args.config)
    if args.set:
        for pair in args.set:
            key, _, value = pair.partition("=")
            if "." in key:
                group, _, field = key.partition(".")
                if group == "params":
                    current = config.params.get(field)
                    if isinstance(current, bool):
                        config.params[field] = value.lower() in ("1", "true", "yes")
                    elif isinstance(current, (int, float)):
                        config.params[field] = type(current)(float(value))
                    else:
                        config.params[field] = value
                elif group == "providers":
                    role, _, field2 = field.partition(".")
                    conf = dict(config.provider(role))
                    conf[field2] = value
                    config.set_provider(role, conf)
                elif group == "tts":
                    tts = dict(config.tts)
                    tts[field] = value
                    config.data["tts"] = tts
                else:
                    config.data[key] = value
            else:
                config.data[key] = value
        path = config.save(Path(args.config) if args.config else None)
        print("已保存: %s" % path)
        return 0
    print("== providers ==")
    for role in ROLES:
        conf = config.provider(role)
        print("  %-18s type=%-12s model=%s" % (ROLE_LABELS[role], conf.get("type"), conf.get("model")))
    print("== tts ==")
    print("  type=%s voice=%s" % (config.tts.get("type"), config.tts.get("voice")))
    print("== params ==")
    for key, value in sorted(config.params.items()):
        print("  %s = %s" % (key, value))
    print("\n可用: types LLM=%s TTS=%s" % (",".join(LLM_TYPES), ",".join(TTS_TYPES)))
    print("设置: python cli.py config --set params.min_match=72 --set providers.story_analysis.type=gemini")
    return 0


def cmd_gui(args):
    from factory.gui import launch_gui
    launch_gui(config_path=args.config)
    return 0


def cmd_serve(args):
    from factory.api import serve
    serve(config_path=args.config, host=args.host, port=args.port)
    return 0


def _judge_provider(config):
    from factory.providers.openai_compat import OpenAICompatProvider

    conf = dict(config.data.get("judge") or {})
    base = config.provider("story_analysis")
    if not conf.get("api_key"):
        conf["api_key"] = base.get("api_key", "")
    if not conf.get("base_url"):
        conf["base_url"] = base.get("base_url", "")
    conf.setdefault("type", "openai")
    return OpenAICompatProvider(conf)


def cmd_judge(args):
    import json

    from factory.eval import judge_video, timeline_to_srt

    config = load_config(args.config)
    job_dir = config.workspace_path() / "jobs" / args.job_id
    clips_path = job_dir / "clips" / "clips.json"
    if not clips_path.exists():
        print("找不到成片: %s" % clips_path)
        return 1
    clips = json.loads(clips_path.read_text(encoding="utf-8"))
    clip = next((c for c in clips if c.get("path") and Path(c["path"]).exists()), None)
    if not clip:
        print("没有可评估的成片")
        return 1
    timeline = json.loads((job_dir / "clips" / clip["clip_id"] / "timeline.json").read_text(encoding="utf-8"))
    our_srt = timeline_to_srt(timeline, job_dir / "eval" / (clip["clip_id"] + ".srt"))
    clip_dir = job_dir / "clips" / clip["clip_id"]
    our_audio = (clip.get("scores") or {}).get("tts_path")
    if not our_audio or not Path(our_audio).exists():
        for name in ("voice.wav", "voice.mp3"):
            if (clip_dir / name).exists():
                our_audio = str(clip_dir / name)
                break
    ref_video = args.ref_video or r"D:\惊悚短片剪辑\测试素材文件夹\风格参考视频.mp4"
    ref_text = ""
    ref_path = Path(args.ref_text or r"D:\惊悚短片剪辑\测试素材文件夹\示例参考文案.txt")
    if ref_path.exists():
        ref_text = ref_path.read_text(encoding="utf-8", errors="replace")
    provider = _judge_provider(config)
    weights = config.data.get("judge_weights") or None
    print("评委模型: %s" % provider.model)
    expect_size = None
    try:
        meta = json.loads((job_dir / "source" / "meta.json").read_text(encoding="utf-8"))
        mw = (meta.get("meta") or {}).get("width") or 0
        mh = (meta.get("meta") or {}).get("height") or 0
        expect_size = (mw, mh)
    except Exception:
        pass
    result = judge_video(provider, clip["path"], our_srt, ref_video, ref_text, job_dir / "eval",
                         our_audio=our_audio, weights=weights, timeline=timeline,
                         expect_size=expect_size)
    data = result.get("data") or {}
    print("\n=== 第三方评审 (%s) ===" % provider.model)
    issues = result.get("review_issues") or []
    if issues:
        print("✗ 机器审计未通过（=硬性不通过）:")
        for issue in issues:
            print("   - " + issue)
    if isinstance(data, dict) and data.get("ours"):
        print("我方:", json.dumps(data.get("ours"), ensure_ascii=False))
        print("参考:", json.dumps(data.get("reference"), ensure_ascii=False))
        print("加权总分: 我方 %s  vs  参考 %s" % (result.get("ours_overall"), result.get("reference_overall")))
        print("我方更优:", result.get("ours_better"))
        if issues:
            print("结论: 因机器审计不过 → 评审不通过")
        else:
            print("结论:", "评审通过（我方更优）" if result.get("ours_better") else "评审不通过（我方未优于参考）")
        for s in data.get("suggestions") or []:
            print("  建议:", s)
    else:
        print(result.get("raw", "")[:1500])
    return 0


def cmd_rerender(args):
    from factory.pipeline import ProductionPipeline
    from factory.utils import read_json

    config = load_config(args.config)
    job_dir = config.workspace_path() / "jobs" / args.job_id
    state = read_json(job_dir / "job.json") or {}
    pipe = ProductionPipeline(config, state.get("url", ""), references=state.get("references") or [],
                              job_id=args.job_id)
    pipe._load_clips()
    if not pipe.clips:
        print("该任务没有成片")
        return 1
    clip_id = args.clip or pipe.clips[0].clip_id
    clip = pipe.rerender_clip(clip_id, [], "blur")
    print("重渲染完成:", clip.path if clip else "失败")
    return 0


def cmd_ground(args):
    """落地化：逐句把解说改写成严格贴合其画面，再重新配音渲染。"""
    from factory.matching.ground import ground_clip
    from factory.utils import read_json

    config = load_config(args.config)
    job_dir = config.workspace_path() / "jobs" / args.job_id
    state = read_json(job_dir / "job.json") or {}
    pipe = ProductionPipeline(config, state.get("url", ""), references=state.get("references") or [],
                              job_id=args.job_id)
    pipe._load_index()
    pipe._load_clips()
    if not pipe.clips:
        print("该任务没有成片")
        return 1
    targets = [c for c in pipe.clips if not args.clip or c.clip_id == args.clip]
    for clip in targets:
        ground_clip(pipe, clip)
        print("落地化完成:", clip.path)
    return 0


def cmd_render_plan(args):
    from factory.planned import render_plan
    pipe = render_plan(load_config(args.config), args.plan, emitter=make_emitter(), job_id=args.resume or None)
    for clip in pipe.clips:
        print("成片: %s (%.2fs, QC=%s)" % (clip.path, clip.duration, clip.qc.get("status")), flush=True)
    return 0 if pipe.state.status == "COMPLETED" else 2


def cmd_audit(args):
    import json
    from factory.eval.media_audit import audit_media
    report = audit_media(args.video)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "pass" else 2


def build_parser():
    parser = argparse.ArgumentParser(prog="cli.py", description="短片工厂 · 长视频智能重构为惊悚短片生产线")
    parser.add_argument("--config", default="", help="配置文件路径 (默认 workspace/config.json)")
    sub = parser.add_subparsers(dest="command")

    p_audit = sub.add_parser("audit", help="审计实际成片：完整解码、响度、黑场和源片重复区间")
    p_audit.add_argument("video")
    p_audit.set_defaults(func=cmd_audit)

    p_plan = sub.add_parser("render-plan", help="从可审阅的故事分镜JSON制作成片（无需编剧API）")
    p_plan.add_argument("plan", help="含source、title、sentences和逐句证据时间的JSON")
    p_plan.add_argument("--resume", default="", help="重用已有任务及逐句配音缓存")
    p_plan.set_defaults(func=cmd_render_plan)

    p_run = sub.add_parser("run", help="运行全链路生产")
    p_run.add_argument("url", help="YouTube 链接或本地视频路径")
    p_run.add_argument("--refs", nargs="*", default=[], help="参考文案文件 (txt/md/json/docx)")
    p_run.add_argument("--mode", choices=["auto", "semi"], default="", help="auto 自动 / semi 人工审核")
    p_run.add_argument("--max", type=lambda value:clip_limit({'max_clips':value}), default=None,
                       help="0=自动制作全部合格故事；正整数=成片上限；省略沿用配置")
    p_run.add_argument("--resume", default="", help="续跑已有 job id")
    p_run.add_argument("--quiet", action="store_true", help="只输出关键事件")
    p_run.set_defaults(func=cmd_run)

    p_list = sub.add_parser("list", help="列出历史任务")
    p_list.set_defaults(func=cmd_list)

    p_report = sub.add_parser("report", help="查看任务生产报告")
    p_report.add_argument("job_id")
    p_report.add_argument("--open", action="store_true", help="用系统查看器打开")
    p_report.set_defaults(func=cmd_report)

    p_config = sub.add_parser("config", help="查看/修改配置")
    p_config.add_argument("--set", nargs="*", default=[], help="key=value，如 params.min_match=72")
    p_config.set_defaults(func=cmd_config)

    p_gui = sub.add_parser("gui", help="启动桌面窗口界面")
    p_gui.set_defaults(func=cmd_gui)

    p_serve = sub.add_parser("serve", help="启动本地 REST API 服务")
    p_serve.add_argument("--host", default="127.0.0.1", help="监听地址")
    p_serve.add_argument("--port", type=int, default=8765, help="监听端口")
    p_serve.set_defaults(func=cmd_serve)

    p_judge = sub.add_parser("judge", help="第三方(评委模型)对比评估成片")
    p_judge.add_argument("job_id", help="任务 id")
    p_judge.add_argument("--ref-video", default="", help="参考成片路径")
    p_judge.add_argument("--ref-text", default="", help="参考文案路径")
    p_judge.set_defaults(func=cmd_judge)

    p_re = sub.add_parser("rerender", help="按当前设置重渲染某任务的成片")
    p_re.add_argument("job_id", help="任务 id")
    p_re.add_argument("--clip", default="", help="成片 id，默认第一条")
    p_re.set_defaults(func=cmd_rerender)

    p_gd = sub.add_parser("ground", help="落地化：解说逐句贴合画面并重新配音渲染")
    p_gd.add_argument("job_id", help="任务 id")
    p_gd.add_argument("--clip", default="", help="成片 id，默认全部")
    p_gd.set_defaults(func=cmd_ground)
    return parser


def main(argv=None):
    ensure_utf8_console()
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "command", None):
        parser.print_help()
        return 0
    return args.func(args) or 0


if __name__ == "__main__":
    sys.exit(main())
