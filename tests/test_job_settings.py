import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

from factory.config import Config
from factory.models import ClipResult, to_dict
from factory.pipeline import ProductionPipeline
from factory.utils import read_json, write_json


def config(tmp_path, **changes):
    return Config({"workspace": str(tmp_path / "ws"), "params": {"review_mode": "semi",
                  "min_duration": 35, "max_duration": 50, "aspect_ratio": "auto"},
                  "tts": {"type": "bailian", "voice": "N1", "model": "cosyvoice-v2", "speed": 1.2,
                          "api_key": "fixture-original-key", "base_url": "https://original.example",
                          "voices": [{"name": "N1", "voice_id": "cosyvoice-v2-original", "model": "cosyvoice-v2",
                                      "unused_secret": "must not snapshot"},
                                     {"name": "unused", "voice_id": "cosyvoice-v2-unused", "model": "cosyvoice-v2"}],
                          **changes}})


def test_pipeline_configuration_is_independent_of_global_and_other_workers(tmp_path):
    global_config = config(tmp_path)
    a = ProductionPipeline(global_config, "first.mp4")
    b = ProductionPipeline(global_config, "second.mp4")
    a.config.params["target_width"] = 640
    a.config.data["tts"]["voice"] = "different"
    a.config.data["tts"]["voices"][0]["name"] = "different"
    assert b.config.param("target_width") == global_config.param("target_width")
    assert b.config.tts["voice"] == global_config.tts["voice"] == "N1"
    assert b.config.tts["voices"][0]["name"] == global_config.tts["voices"][0]["name"] == "N1"
    assert a.router.config is a.config and a.tts.config is a.config


def test_saved_params_restore_but_new_defaults_and_explicit_overrides_survive(tmp_path):
    original = config(tmp_path)
    p = ProductionPipeline(original, "source.mp4")
    p.config.params.update(target_width=1920, target_height=1080)
    p.save_state()
    current = config(tmp_path)
    current.params.update(min_duration=15, max_duration=30, aspect_ratio="vertical", new_motion_default="on")
    resumed = ProductionPipeline(current, "source.mp4", job_id=p.job_id,
                                 params_override={"max_duration": 55})
    assert resumed.config.param("min_duration") == 35
    assert resumed.config.param("max_duration") == 55
    assert resumed.config.param("aspect_ratio") == "auto"
    assert resumed.config.param("new_motion_default") == "on"
    assert resumed.config.param("target_width") == 1920
    resumed.save_state()
    state = read_json(resumed.job_dir / "job.json")
    assert state["params"]["max_duration"] == 55
    assert state["params"]["min_duration"] == 35
    assert current.param("min_duration") == 15 and current.param("aspect_ratio") == "vertical"
    current.params.update(min_duration=5, max_duration=10)
    resumed.save_state()
    assert read_json(resumed.job_dir / "job.json")["params"]["min_duration"] == 35


def test_restore_tts_choices_and_clone_mapping_with_current_credentials(tmp_path):
    p = ProductionPipeline(config(tmp_path), "source.mp4")
    p.save_state()
    current = config(tmp_path, voice="N2", speed=0.8, api_key="fixture-new-key", base_url="https://new.example",
                     cosyvoice_model="cosyvoice-v3", voices=[{"name": "N1", "voice_id": "cosyvoice-v2-replaced",
                     "model": "cosyvoice-v3"}, {"name": "N2", "voice_id": "cosyvoice-v2-second", "model": "cosyvoice-v2"}])
    resumed = ProductionPipeline(current, "source.mp4", job_id=p.job_id)
    assert resumed.config.tts["voice"] == "N1" and resumed.config.tts["speed"] == 1.2
    assert resumed.config.tts["api_key"] == "fixture-new-key"
    assert resumed.config.tts["base_url"] == "https://new.example"
    assert not resumed.config.tts["cosyvoice_model"]
    mapping = [v for v in resumed.config.tts["voices"] if v["name"] == "N1"]
    assert mapping == [{"name": "N1", "voice_id": "cosyvoice-v2-original", "model": "cosyvoice-v2"}]
    assert current.tts["voice"] == "N2" and current.tts["speed"] == 0.8
    snapshot = read_json(p.job_dir / "job.json")["artifacts"]["tts_settings"]
    assert "api_key" not in snapshot and "base_url" not in snapshot
    assert snapshot["voices"] == mapping
    assert "must not snapshot" not in str(snapshot)


def test_legacy_native_audio_recovers_actual_voice_and_speed(tmp_path):
    current = config(tmp_path, voice="N2", speed=0.8)
    p = ProductionPipeline(current, "source.mp4")
    p.save_state()
    state = read_json(p.job_dir / "job.json")
    state["artifacts"].pop("tts_settings")
    write_json(p.job_dir / "job.json", state)
    write_json(p.job_dir / "clips/clip_001/tts.json", {"provider": "bailian-cosyvoice",
               "voice": "cosyvoice-v2-real-original", "playback_speed": 1.2, "duration": 26.4})
    resumed = ProductionPipeline(current, "source.mp4", job_id=p.job_id)
    assert resumed.config.tts["voice"] == "cosyvoice-v2-real-original"
    assert resumed.config.tts["speed"] == 1.2
    assert resumed.config.tts["cosyvoice_model"] == "cosyvoice-v2"
    assert current.tts["voice"] == "N2"


@pytest.mark.parametrize("metadata", [
    {"provider": "unknown", "voice": "voice", "playback_speed": 1.2},
    {"provider": "bailian-cosyvoice", "voice": "cosyvoice-v3-other", "playback_speed": 1.2},
    {"provider": "bailian-cosyvoice", "voice": "cosyvoice-v2-other", "playback_speed": float("nan")},
    {"provider": "bailian-cosyvoice", "voice": "cosyvoice-v2-other"},
])
def test_unknown_legacy_identity_keeps_current_settings(tmp_path, metadata):
    current = config(tmp_path)
    p = ProductionPipeline(current, "source.mp4")
    p.save_state()
    state = read_json(p.job_dir / "job.json")
    state["artifacts"].pop("tts_settings")
    write_json(p.job_dir / "job.json", state)
    write_json(p.job_dir / "clips/clip_001/tts.json", metadata)
    resumed = ProductionPipeline(current, "source.mp4", job_id=p.job_id)
    assert resumed.config.tts["voice"] == "N1" and resumed.config.tts["speed"] == 1.2


def test_resuming_without_reference_argument_retains_saved_reference(tmp_path):
    p = ProductionPipeline(config(tmp_path), "source.mp4", references=["原来的参考文案"])
    p.save_state()
    resumed = ProductionPipeline(config(tmp_path), "", job_id=p.job_id)
    assert resumed.references == ["原来的参考文案"]
    assert resumed.url == "source.mp4"
    resumed.save_state()
    assert read_json(resumed.job_dir / "job.json")["references"] == ["原来的参考文案"]


def test_gui_loading_finished_job_populates_output_without_running_production(tmp_path):
    pytest.importorskip("PySide6")
    from factory.gui.qt import create_app
    conf = config(tmp_path)
    config_path = tmp_path / "config.json"
    conf.save(config_path)
    p = ProductionPipeline(conf, "source.mp4")
    p.state.status = "COMPLETED"
    p.save_state()
    write_json(p.job_dir / "clips/clips.json", [to_dict(ClipResult(clip_id="clip_001", candidate_id="C001",
        path=str(tmp_path / "missing-preview.mp4"), status="completed", duration=38.5))])
    app, window = create_app(config_path=str(config_path))
    try:
        window.resume_job(p.job_id)
        app.processEvents()
        assert not window.running
        assert len(window.pipeline.clips) == 1
        assert window.pages["output"].tree.topLevelItemCount() == 1
        assert "通过 1" in window.pages["output"].summary.text()
    finally:
        window.close()


def test_failure_events_preserve_real_stage_while_job_state_is_failed(tmp_path):
    p = ProductionPipeline(config(tmp_path), "source.mp4")
    p.state.status = "TTS_GENERATED"
    p._fail(RuntimeError("fixture synthesis failure"))
    assert p.state.status == "FAILED"
    assert p.state.artifacts["failed_stage"] == "TTS_GENERATED"
    stages = [event for event in p.emitter.history if event["kind"] == "stage"]
    assert stages[-1]["stage"] == "TTS_GENERATED" and stages[-1]["status"] == "failed"
    assert read_json(p.job_dir / "job.json")["artifacts"]["failed_stage"] == "TTS_GENERATED"


def test_cancel_events_preserve_real_stage_without_touching_other_processes(tmp_path):
    from factory.pipeline import PipelineCancelled
    p = ProductionPipeline(config(tmp_path), "source.mp4")
    p.state.status = "RENDERING"
    p.cancel_requested = True
    p._fail(PipelineCancelled("fixture cancellation"))
    assert p.state.status == "CANCELLED" and p.state.artifacts["failed_stage"] == "RENDERING"
    stages = [event for event in p.emitter.history if event["kind"] == "stage"]
    assert stages[-1]["stage"] == "RENDERING"
