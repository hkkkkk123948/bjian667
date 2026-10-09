# Summary

## Objective
- 用软件把 `D:\智创下载器\youtobe下载` 的三个原片各做成一条解说惊悚短片（共 3 条）；全程由软件实现，不对视频做创作性改动，只优化软件本身。
- 用 T8 中转的 `gemini-3.1-flash`（`gemini-3.1-flash-lite-preview`）作独立评委，成片画面抽帧+音频（`input_audio`，不支持则回退 SRT）与参考片对比打分；评委判“我方更优”即达标。同时尽量节省 T8 token。

## Important Details
- 分析/生成模型：6 角色 `gpt-5.6-sol`（T8，key `sk-86iOVkl1btTKGSbKJOPo7o3FIrvTbBqLAZUG1Fp1ADJzHTxg`）；评委 `gemini-3.1-flash-lite-preview`。
- TTS：百炼 `qwen3-tts-flash`（base `https://dashscope.aliyuncs.com`，key `sk-ws-H.PEMIIXY.Qvi7.MEUCIQC4Jpu602xe_82k5dMzalHdP0Y6wqUhWxLYHWcDLGlTcwIgeSCmfa_QVQXsNcqLaFC1CZOO45Onye_sqcLDhIJRkcg`）；已克隆音色 N1；cosyvoice-v2 克隆接口。
- 内容 creative（跟随画面原创故事 + 参考示例风格）；无字幕（用户剪映手加）；aspect_ratio=auto；工作区 min/max_duration=30/50。
- 省 token：max_clips=1、candidate_pool=3、analyze_max_candidates=2、max_rewrite_attempts=1、rewrite_margin=8、llm_cache=on、visual_frames_per_scene=1。
- 运行：`$env:PYTHONIOENCODING="utf-8"; $env:PYTHONPATH="D:\惊悚短片剪辑"; python -X utf8 cli.py <cmd>`，workdir=D:\惊悚短片剪辑；改配置重启 GUI。
- 三源片对应 Job1 sChF5ilNHaM / Job2 F35JDpVZ27Y / Job3 wqQZiP2lDOY；参考片 `测试素材文件夹\风格参考视频.mp4`、参考文案 `示例参考文案.txt`。

## What Was Done
1. 借鉴“解说猫”项目（D:\claude code\jieshuo，本体为 .pyd，通过字符串恢复其思路）：
   - 其音画匹配 = Chinese-CLIP(ViT-B/16,512d) 编码句子与场景关键帧 → 余弦相似度矩阵 → 贪心分配（每句挑相似度最高的未用场景，避免相邻重复）→ 按 TTS 时长适配生成时间线；另有 Gemini 语义匹配作为备选，并输出匹配质量报告（平均/最低/低相似句比例）。
   - 落地到本仓库：`factory/matching/scene_align.py` LLM 打分矩阵（gpt-5.6-sol）→ 贪心唯一分配 + 相邻惩罚 + 时间倒退惩罚；`build_timeline` 优先用匹配场景做“证据剪辑”，并加**覆盖度门槛**（≥2 个唯一场景且 ≥60% 句子命中才启用，否则回退写稿证据时间）。
2. 证据驱动剪辑 + 零留白（timeline `_evidence_segments` + exporter `_export_evidence`，总时长=旁白时长）。
3. 第三方评审功能 `cli.py judge <job>`（抽帧+input_audio+SRT+加权权重 match/visual/story/copy/pacing/originality）与 `cli.py rerender` 重渲染。

## Results（评委对比参考片，均为我方更优）
- Job1 `20261007_182913_efb59b`：加权 87.3 vs 72.5，match 90，pacing 90（重渲染多轮：证据剪辑+零留白迭代到最优；试过“无条件场景匹配”会倒退，覆盖度门槛后自动回退保优）。
- Job2 `20261007_184705_863d49`：80.8 vs 63.0，match 85（旧渲染 90.0 vs 68.0，重渲后为新分）。
- Job3 `20261007_200203_34692b`：84.2 vs 60.0，match 90。
- 评委遗留建议：增加环境音/音效（脚步、心跳、风声）与 BGM 以加深沉浸感；参考片被评问题（文案生硬、逻辑破碎、音画脱节）恰好说明我方文案与匹配占优。

## Next Step（建议）
- 音效/BGM：参考 jieshuo 的 OST 选曲思路，为成片加入环境音与低音 BGM（保持音量低于旁白），有望再提 pacing/visual 分。
- 如需更多条成片，直接 `cli.py run <新源片> --refs ... --mode auto --max 1`，完成后 `cli.py judge <job>`。