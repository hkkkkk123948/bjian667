from factory.index.builder import _from_cache, _full_transcribe
from factory.index.cache import index_cache_key, load_cached, save_cached
from factory.index.transcript import _parse_ass
from factory.models import MediaIndex, SceneSegment, TranscriptSegment, VideoMeta
from factory.pipeline import _dedupe_segments, _weak_sentence_indices
from factory.providers.base import LLMProvider, configure_llm_cache, reset_usage, usage_summary
from factory.utils import audio_fit_score, split_sentences


def test_split_sentences_mixed():
    assert len(split_sentences("他推开了门。里面一片漆黑！你敢进去吗？")) == 3
    assert len(split_sentences("He opened the door. It was dark! Are you serious?")) == 3
    assert len(split_sentences("Mr. Smith went home.")) == 1
    assert split_sentences("") == []


def test_audio_fit_score_bands():
    assert audio_fit_score(1.0) == 95.0
    assert audio_fit_score(1.2) == 80.0
    assert audio_fit_score(1.4) == 60.0
    assert audio_fit_score(2.0) == 40.0


def test_index_cache_key_stable_and_param_sensitive(tmp_path):
    video = tmp_path / "v.mp4"
    video.write_bytes(b"abcdef")
    params = {"scene_threshold": 0.35, "language": "zh"}
    assert index_cache_key(video, params) == index_cache_key(video, params)
    assert index_cache_key(video, params) != index_cache_key(video, {**params, "language": "en"})


def test_full_transcribe_decision():
    assert _full_transcribe({"transcribe_full": "on"}, 99999) is True
    assert _full_transcribe({"transcribe_full": "off"}, 10) is False
    assert _full_transcribe({"transcribe_full": "auto", "window_transcribe_min": 1200}, 600) is True
    assert _full_transcribe({"transcribe_full": "auto", "window_transcribe_min": 1200}, 3600) is False


def test_parse_ass(tmp_path):
    path = tmp_path / "a.ass"
    path.write_text(
        "[Events]\n"
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
        "Dialogue: 0,0:00:01.00,0:00:03.50,Default,,0,0,0,,你好\\N世界\n",
        encoding="utf-8",
    )
    segs = _parse_ass(path)
    assert len(segs) == 1
    assert segs[0].start == 1.0
    assert segs[0].end == 3.5
    assert "你好" in segs[0].text and "世界" in segs[0].text


def test_weak_sentence_indices():
    sentences = [object()] * 5
    details = {"visual_sentence_scores": [90, 40, 80, 30, 70],
               "fact_sentence_scores": [95, 50, 85, 20, 88]}
    assert _weak_sentence_indices(sentences, details, limit=3) == [4, 2, 5]


def test_dedupe_segments():
    a = TranscriptSegment(start=1.0, end=2.0, text="hi")
    b = TranscriptSegment(start=1.0, end=2.0, text="hi")
    c = TranscriptSegment(start=3.0, end=4.0, text="yo")
    assert len(_dedupe_segments([a, b, c])) == 2


def test_provider_block_detection():
    from factory.providers.gemini import _is_blocked as gemini_blocked
    from factory.providers.openai_compat import _is_blocked as openai_blocked

    assert gemini_blocked("request blocked: PROHIBITED_CONTENT")
    assert not gemini_blocked("normal error")
    assert openai_blocked("content_filter triggered")
    assert not openai_blocked("rate limited")


class _StubProvider(LLMProvider):
    def __init__(self, conf):
        super().__init__(conf)
        self.calls = 0

    def _complete(self, prompt, system=None, images=None, audios=None, json_mode=False, timeout=None):
        self.calls += 1
        return '{"ok": 1}'


def test_index_cache_roundtrip(tmp_path):
    index = MediaIndex(video=VideoMeta(title="orig", duration=60.0), transcript_source="whisper")
    index.scenes = [SceneSegment(scene_id=1, start=0.0, end=5.0, visual="画面描述")]
    index.transcript = [TranscriptSegment(start=0.0, end=3.0, text="你好")]
    save_cached(tmp_path / "idx", "key1", index)
    data = load_cached(tmp_path / "idx", "key1")
    assert data
    meta = VideoMeta(title="new", duration=60.0)
    restored = _from_cache(data, meta, tmp_path / "v.mp4", tmp_path / "job")
    assert restored is not None
    assert restored.video.title == "new"
    assert restored.scenes[0].visual == "画面描述"
    assert restored.transcript[0].text == "你好"


def test_llm_response_cache(tmp_path):
    configure_llm_cache(tmp_path / "llm", True)
    reset_usage()
    try:
        provider = _StubProvider({"type": "openai", "model": "m1", "temperature": 0.7})
        first = provider.complete_json("hello world")
        second = provider.complete_json("hello world")
        assert first == second == {"ok": 1}
        assert provider.calls == 1
        assert usage_summary()["cache_hits"] >= 1
    finally:
        configure_llm_cache(None, False)


def test_versioning_stable():
    from factory.versioning import describe, prompt_salt

    assert prompt_salt() == prompt_salt()
    assert "prompt_versions" in describe()


def test_capcut_rich_draft(tmp_path):
    import json

    from factory.export.capcut import export_draft

    src = tmp_path / "source.mp4"
    src.write_bytes(b"video")
    voice = tmp_path / "voice.mp3"
    voice.write_bytes(b"audio")
    draft_dir = export_draft(src, tmp_path / "drafts", "片", 10.0, 1080, 1920,
                             source_start=5.0, source_duration=10.0,
                             voice_path=voice, orig_audio_path=src)
    content = json.loads((draft_dir / "draft_content.json").read_text(encoding="utf-8"))
    assert content["materials"]["videos"][0]["mute"] is True
    assert len(content["materials"]["audios"]) == 2
    assert [t["type"] for t in content["all_tracks"]] == ["video", "audio", "audio"]
    seg = content["all_tracks"][0]["segments"][0]
    assert seg["source_timerange"]["start"] == 5000


def test_api_list_jobs(tmp_path):
    from factory.api import _Server
    from factory.config import Config

    workspace = tmp_path / "ws"
    job = workspace / "jobs" / "job1"
    job.mkdir(parents=True)
    (job / "job.json").write_text('{"job_id": "job1", "status": "COMPLETED"}', encoding="utf-8")
    server = _Server(Config({"workspace": str(workspace)}))
    rows = server.list_jobs()
    assert rows and rows[0]["job_id"] == "job1"


def test_compress_variant_mock_returns_none(tmp_path):
    from factory.config import Config
    from factory.models import ScriptSentence, ScriptVariant, StoryCandidate
    from factory.pipeline import ProductionPipeline
    from factory.script.repair import compress_variant

    pipe = ProductionPipeline(Config({"workspace": str(tmp_path / "ws")}), "local.mp4")
    cand = StoryCandidate(candidate_id="C1", source_start=0.0, source_end=30.0)
    variant = ScriptVariant(variant_id="S1", sentences=[ScriptSentence(index=1, text="测试")],
                            estimated_seconds=40.0)
    assert compress_variant(pipe, cand, variant, 30.0) is None


def test_rebuild_disabled(tmp_path):
    from factory.mining.rebuild import rebuild_candidate

    class _P:
        pass

    assert rebuild_candidate(_P(), None, None, None, {"story_rebuild": "off"}, None) is False


def test_subtitle_clean_graph():
    from factory.render.clean import build_video_graph

    base = "scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920:0:0,setsar=1"
    assert build_video_graph({"remove_subtitles": "off"}, base) == "[0:v]%s[vout]" % base
    crop = build_video_graph({"remove_subtitles": "crop", "subtitle_band": 0.2}, base)
    assert "crop=iw:ih*0.8000:0:0" in crop and crop.endswith("[vout]")
    assert "delogo=" in build_video_graph({"remove_subtitles": "delogo", "subtitle_band": 0.16}, base,
                                          src_size=(1920, 1080))
    blur = build_video_graph({"remove_subtitles": "blur", "subtitle_band": 0.16}, base)
    assert "split=2" in blur and "overlay" in blur
    bb = build_video_graph({"remove_subtitles": "off"}, base, black_box=(1920, 1040, 0, 20))
    assert bb.startswith("[0:v]crop=1920:1040:0:20,")


def test_region_graph():
    from factory.render.clean import build_video_graph

    base = "scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920:0:0,setsar=1"
    regions = [{"x": 0.1, "y": 0.7, "w": 0.8, "h": 0.15}]
    cover = build_video_graph({"remove_subtitles": "off"}, base, src_size=(1920, 1080),
                              regions=regions, region_method="cover")
    assert "drawbox=" in cover
    blur = build_video_graph({"remove_subtitles": "off"}, base, src_size=(1920, 1080),
                             regions=regions, region_method="blur")
    assert "split=2" in blur and "overlay=" in blur
    mosaic = build_video_graph({"remove_subtitles": "off"}, base, src_size=(1920, 1080),
                               regions=regions, region_method="mosaic")
    assert "scale=iw/16:ih/16" in mosaic


def test_bailian_voice_resolution():
    from factory.providers.tts.bailian import BailianTTS

    provider = BailianTTS({
        "model": "qwen3-tts-flash",
        "voices": [{"name": "我的音色", "voice_id": "cosyvoice-v2-myvoice-abc", "model": "cosyvoice-v2"}],
    })
    assert provider._resolve_voice("我的音色") == "cosyvoice-v2-myvoice-abc"
    assert provider._resolve_voice("Cherry") == "Cherry"
    assert provider._voice_model("cosyvoice-v2-myvoice-abc", "qwen3-tts-flash") == "cosyvoice-v2"
    assert provider._voice_model("Cherry", "qwen3-tts-flash") == "qwen3-tts-flash"

