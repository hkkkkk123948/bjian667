from dataclasses import asdict, dataclass, field, is_dataclass


def to_dict(obj):
    if obj is None:
        return None
    if is_dataclass(obj):
        return {k: to_dict(v) for k, v in asdict(obj).items()}
    if isinstance(obj, dict):
        return {k: to_dict(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_dict(v) for v in obj]
    return obj


def _convert_value(value, annotation):
    if value is None:
        return None
    try:
        if is_dataclass(annotation) and isinstance(value, dict):
            return from_dict(annotation, value)
    except TypeError:
        return value
    return value


def from_dict(cls, data):
    if data is None:
        return None
    if not is_dataclass(cls):
        return data
    if isinstance(data, cls):
        return data
    hints = getattr(cls, "__annotations__", {})
    kwargs = {}
    for key, value in (data or {}).items():
        if key not in hints:
            continue
        annotation = hints[key]
        if isinstance(annotation, type) and is_dataclass(annotation):
            kwargs[key] = _convert_value(value, annotation)
        else:
            kwargs[key] = value
    return cls(**kwargs)


@dataclass
class VideoMeta:
    video_id: str = ""
    title: str = ""
    channel: str = ""
    uploader: str = ""
    duration: float = 0.0
    width: int = 0
    height: int = 0
    fps: float = 0.0
    path: str = ""
    url: str = ""
    extractor: str = ""
    subtitle_path: str = ""
    language: str = ""
    description: str = ""


@dataclass
class TranscriptSegment:
    start: float = 0.0
    end: float = 0.0
    text: str = ""
    source: str = "youtube"
    speaker: str = ""


@dataclass
class SceneSegment:
    scene_id: int = 0
    start: float = 0.0
    end: float = 0.0
    transcript: str = ""
    visual: str = ""
    ocr: str = ""
    keyframes: list = field(default_factory=list)
    characters: list = field(default_factory=list)
    scene_type: str = ""
    rms: float = 0.0
    speech_ratio: float = 0.0
    brightness: float = -1.0
    transition_count: int = 0
    tags: list = field(default_factory=list)


@dataclass
class AudioEvent:
    kind: str = ""
    start: float = 0.0
    end: float = 0.0
    value: float = 0.0


@dataclass
class MediaIndex:
    video: VideoMeta = None
    scenes: list = field(default_factory=list)
    transcript: list = field(default_factory=list)
    audio_events: list = field(default_factory=list)
    frames_dir: str = ""
    built_at: str = ""
    transcript_source: str = "none"
    transcript_quality: dict = field(default_factory=dict)

    def scene_at(self, t):
        for scene in self.scenes:
            if scene.start <= t <= scene.end:
                return scene
        return None

    def transcript_in(self, start, end):
        out = []
        for seg in self.transcript:
            if seg.end > start and seg.start < end:
                out.append(seg)
        return out

    def text_in(self, start, end):
        return normalize_join([s.text for s in self.transcript_in(start, end)])


def normalize_join(parts):
    text = " ".join(p.strip() for p in parts if p and p.strip())
    return re_sub_space(text)


def re_sub_space(text):
    import re as _re
    return _re.sub(r"\s+", " ", text or "").strip()


@dataclass
class Evidence:
    start: float = 0.0
    end: float = 0.0
    scene_ids: list = field(default_factory=list)
    quote: str = ""


@dataclass
class StoryCandidate:
    candidate_id: str = ""
    source_start: float = 0.0
    source_end: float = 0.0
    story_summary: str = ""
    title: str = ""
    hook: str = ""
    characters: list = field(default_factory=list)
    conflict: str = ""
    turning_point: str = ""
    payoff: str = ""
    hook_visual: dict = field(default_factory=dict)
    best_visual_moments: list = field(default_factory=list)
    tags: list = field(default_factory=list)
    evidence: list = field(default_factory=list)
    story_completeness: int = 60
    visual_quality: int = 60
    short_video_potential: int = 60
    story_score: int = 0
    diversity_similarity: float = 0.0
    status: str = "generated"
    match_report: dict = None
    scripts: list = field(default_factory=list)
    selected_script: str = ""
    reject_reason: str = ""
    final_match: float = 0.0

    @property
    def duration(self):
        return max(0.0, self.source_end - self.source_start)


@dataclass
class StyleProfile:
    name: str = "default"
    sample_count: int = 0
    hook_length: str = "1.5-3.0s"
    hook_seconds: float = 2.2
    sentence_length: str = "short"
    avg_sentence_chars: float = 0.0
    short_sentence_ratio: float = 0.0
    narration: str = "third_person"
    pacing: str = "fast"
    chars_per_second: float = 4.4
    emotion: dict = field(default_factory=dict)
    structure: list = field(default_factory=list)
    ending: str = "open_loop"
    colloquial_ratio: float = 0.0
    pause_density: float = 0.0
    repetition_patterns: list = field(default_factory=list)
    lexical_fingerprint: list = field(default_factory=list)
    raw_metrics: dict = field(default_factory=dict)
    notes: str = ""


@dataclass
class ScriptSentence:
    index: int = 0
    text: str = ""
    source_evidence: Evidence = None
    estimated_seconds: float = 0.0
    actual_seconds: float = 0.0
    start_offset: float = 0.0
    role: str = ""
    narrative_text: str = ""
    comment_hook: str = ""
    comment_hook_mode: str = ""
    comment_hooks: list = field(default_factory=list)


@dataclass
class ScriptVariant:
    variant_id: str = ""
    candidate_id: str = ""
    flavor: str = "suspense"
    sentences: list = field(default_factory=list)
    full_text: str = ""
    estimated_seconds: float = 0.0
    actual_seconds: float = 0.0
    match_report: dict = None
    gate1_pass: bool = False
    attempts: int = 0
    status: str = "generated"
    scene_matches: list = field(default_factory=list)
    review_comment_hook: dict = None
    review_comment_hooks: list = None
    comment_visual_mode: str = ""

    @property
    def text(self):
        return self.full_text or " ".join(s.text for s in self.sentences)


@dataclass
class MatchReport:
    final_score: float = 0.0
    visual_alignment: float = 0.0
    narrative_alignment: float = 0.0
    reference_style_match: float = 0.0
    temporal_alignment: float = 0.0
    information_coverage: float = 0.0
    audio_tts_fit: float = 0.0
    story_completeness: float = 0.0
    fact_alignment: float = 0.0
    details: dict = field(default_factory=dict)

    WEIGHTS = {
        "visual_alignment": 0.25,
        "temporal_alignment": 0.20,
        "fact_alignment": 0.20,
        "narrative_alignment": 0.15,
        "information_coverage": 0.10,
        "reference_style_match": 0.10,
        "audio_tts_fit": 0.0,
        "story_completeness": 0.0,
    }

    HARD_GATES = {
        "visual_alignment": 75.0,
        "fact_alignment": 90.0,
        "temporal_alignment": 80.0,
        "narrative_alignment": 80.0,
    }

    CREATIVE_HARD_GATES = {
        "visual_alignment": 70.0,
        "fact_alignment": 70.0,
        "narrative_alignment": 60.0,
        "temporal_alignment": 55.0,
    }

    def compute(self):
        total = 0.0
        for key, weight in self.WEIGHTS.items():
            total += weight * float(getattr(self, key, 0.0))
        self.final_score = round(total, 2)
        return self.final_score

    def as_dict(self):
        data = to_dict(self)
        data["weights"] = dict(self.WEIGHTS)
        return data


@dataclass
class TTSResult:
    audio_path: str = ""
    duration: float = 0.0
    word_timings: list = field(default_factory=list)
    voice: str = ""
    provider: str = "mock"
    sample_rate: int = 44100


@dataclass
class ClipResult:
    clip_id: str = ""
    candidate_id: str = ""
    variant_id: str = ""
    path: str = ""
    subtitle_path: str = ""
    draft_path: str = ""
    duration: float = 0.0
    width: int = 0
    height: int = 0
    scores: dict = field(default_factory=dict)
    gate1: dict = field(default_factory=dict)
    gate2: dict = field(default_factory=dict)
    qc: dict = field(default_factory=dict)
    status: str = "pending"
    script_text: str = ""
    source_start: float = 0.0
    source_end: float = 0.0


@dataclass
class JobState:
    job_id: str = ""
    url: str = ""
    status: str = "CREATED"
    stage_index: int = 0
    created_at: str = ""
    updated_at: str = ""
    workspace: str = ""
    error: str = ""
    resume_from: str = ""
    params: dict = field(default_factory=dict)
    artifacts: dict = field(default_factory=dict)
    events: list = field(default_factory=list)
    references: list = field(default_factory=list)
    completed_stages: list = field(default_factory=list)


def evidence_from_dict(data):
    if data is None:
        return None
    if isinstance(data, Evidence):
        return data
    if not isinstance(data, dict):
        return None
    return Evidence(
        start=float(data.get("start", 0.0) or 0.0),
        end=float(data.get("end", 0.0) or 0.0),
        scene_ids=list(data.get("scene_ids") or []),
        quote=str(data.get("quote", "")),
    )


def sentence_from_dict(data):
    if isinstance(data, ScriptSentence):
        return data
    return ScriptSentence(
        index=int(data.get("index", 0) or 0),
        text=str(data.get("text", "")),
        source_evidence=evidence_from_dict(data.get("source_evidence")),
        estimated_seconds=float(data.get("estimated_seconds", 0.0) or 0.0),
        actual_seconds=float(data.get("actual_seconds", 0.0) or 0.0),
        start_offset=float(data.get("start_offset", 0.0) or 0.0),
        role=str(data.get("role", "")),
        narrative_text=str(data.get("narrative_text", "")),
        comment_hook=str(data.get("comment_hook", "")),
        comment_hook_mode=str(data.get("comment_hook_mode", "")),
        comment_hooks=list(data.get("comment_hooks") or []),
    )


def variant_from_dict(data):
    if isinstance(data, ScriptVariant):
        return data
    return ScriptVariant(
        variant_id=str(data.get("variant_id", "")),
        candidate_id=str(data.get("candidate_id", "")),
        flavor=str(data.get("flavor", "suspense")),
        sentences=[sentence_from_dict(s) for s in data.get("sentences") or []],
        full_text=str(data.get("full_text", "")),
        estimated_seconds=float(data.get("estimated_seconds", 0.0) or 0.0),
        actual_seconds=float(data.get("actual_seconds", 0.0) or 0.0),
        match_report=data.get("match_report"),
        gate1_pass=bool(data.get("gate1_pass", False)),
        attempts=int(data.get("attempts", 0) or 0),
        status=str(data.get("status", "generated")),
        scene_matches=list(data.get("scene_matches") or []),
        review_comment_hook=data.get("review_comment_hook"),
        review_comment_hooks=data.get("review_comment_hooks"),
        comment_visual_mode=str(data.get("comment_visual_mode", "")),
    )


def candidate_from_dict(data):
    if isinstance(data, StoryCandidate):
        return data
    ev = [evidence_from_dict(x) for x in data.get("evidence") or []]
    return StoryCandidate(
        candidate_id=str(data.get("candidate_id", "")),
        source_start=float(data.get("source_start", 0.0) or 0.0),
        source_end=float(data.get("source_end", 0.0) or 0.0),
        story_summary=str(data.get("story_summary", "")),
        title=str(data.get("title", "")),
        hook=str(data.get("hook", "")),
        characters=list(data.get("characters") or []),
        conflict=str(data.get("conflict", "")),
        turning_point=str(data.get("turning_point", "")),
        payoff=str(data.get("payoff", "")),
        hook_visual=dict(data.get("hook_visual") or {}),
        best_visual_moments=list(data.get("best_visual_moments") or []),
        tags=list(data.get("tags") or []),
        evidence=ev,
        story_completeness=int(data.get("story_completeness", 60) or 60),
        visual_quality=int(data.get("visual_quality", 60) or 60),
        short_video_potential=int(data.get("short_video_potential", 60) or 60),
        story_score=int(data.get("story_score", 0) or 0),
        diversity_similarity=float(data.get("diversity_similarity", 0.0) or 0.0),
        status=str(data.get("status", "generated")),
        match_report=data.get("match_report"),
        scripts=[variant_from_dict(s) for s in data.get("scripts") or []],
        selected_script=str(data.get("selected_script", "")),
        reject_reason=str(data.get("reject_reason", "")),
        final_match=float(data.get("final_match", 0.0) or 0.0),
    )
