STAGES = [
    "CREATED",
    "URL_ANALYZING",
    "DOWNLOADING",
    "MEDIA_INDEXING",
    "VISUAL_ANALYSIS",
    "STORY_MINING",
    "CANDIDATES_GENERATED",
    "REFERENCE_ANALYZED",
    "SCRIPT_GENERATED",
    "CANDIDATE_FILTERED",
    "TTS_GENERATED",
    "MATCHING",
    "TIMELINE_ALIGNED",
    "RENDERING",
    "FINAL_MATCH_CHECK",
    "QUALITY_CHECK",
    "DRAFT_EXPORT",
    "COMPLETED",
]

STAGE_LABELS = {
    "CREATED": "任务创建",
    "URL_ANALYZING": "视频信息解析",
    "DOWNLOADING": "下载视频",
    "MEDIA_INDEXING": "建立媒体索引",
    "VISUAL_ANALYSIS": "原片视觉分析",
    "STORY_MINING": "故事挖掘",
    "CANDIDATES_GENERATED": "候选故事生成",
    "REFERENCE_ANALYZED": "参考文案分析",
    "SCRIPT_GENERATED": "解说脚本生成",
    "CANDIDATE_FILTERED": "文案确认",
    "TTS_GENERATED": "整段配音",
    "MATCHING": "配音与画面匹配",
    "TIMELINE_ALIGNED": "时间轴对齐",
    "RENDERING": "画面重构与渲染",
    "FINAL_MATCH_CHECK": "保存制作结果",
    "QUALITY_CHECK": "画面匹配审核",
    "DRAFT_EXPORT": "剪映草稿导出",
    "COMPLETED": "完成",
}

TERMINAL = {"COMPLETED", "FAILED", "REVIEW_REQUIRED", "CANCELLED"}


def stage_index(status):
    if status in STAGES:
        return STAGES.index(status)
    return -1


def next_stage(status):
    idx = stage_index(status)
    if idx < 0 or idx + 1 >= len(STAGES):
        return status
    return STAGES[idx + 1]


def stage_label(status):
    return STAGE_LABELS.get(status, status)


def can_resume(status):
    return status not in TERMINAL


def stage_progress(status):
    idx = stage_index(status)
    if idx < 0:
        return 0.0
    return round(idx / (len(STAGES) - 1) * 100.0, 1)
