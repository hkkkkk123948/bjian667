"""提示词/模型版本号（doc §91）：修改提示词时递增，用于缓存失效与效果对比。"""

PROMPT_VERSIONS = {
    "story_discovery": "v2_visual_story",
    "script_generation": "v5_grounded_duration",
    "match_judge": "v1",
    "qc_semantic": "v7_approved_script_visual_alignment",
    "reference_style": "v1",
    "visual_analysis": "v1",
    "ocr": "v1",
}

SCHEMA_VERSION = "1.4"


def prompt_salt():
    return "|".join("%s=%s" % (k, v) for k, v in sorted(PROMPT_VERSIONS.items()))


def describe():
    return {"prompt_versions": dict(PROMPT_VERSIONS), "schema_version": SCHEMA_VERSION}
