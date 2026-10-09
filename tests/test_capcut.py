import json

from factory.export.capcut import detect_draft_root, export_draft


def test_export_draft_structure(tmp_path):
    video = tmp_path / "ws" / "clip.mp4"
    video.parent.mkdir(parents=True)
    video.write_bytes(b"fake-video")
    root = tmp_path / "drafts"
    draft_dir = export_draft(video, root, "测试 成片", 12.6, 1080, 1920,
                             folder_name="C001_test")
    content = json.loads((draft_dir / "draft_content.json").read_text(encoding="utf-8"))
    meta = json.loads((draft_dir / "draft_meta_info.json").read_text(encoding="utf-8"))

    assert draft_dir.name == "C001_test"
    assert meta["tm_draft_name"] == "测试 成片"
    assert meta["tm_draft_version"]
    videos = content["materials"]["videos"]
    assert len(videos) == 1
    assert videos[0]["path"].endswith("clip.mp4")
    assert videos[0]["duration"] == 12.6
    assert videos[0]["width"] == 1080 and videos[0]["height"] == 1920
    track = content["all_tracks"][0]
    assert track["type"] == "video"
    seg = track["segments"][0]
    assert seg["material_id"] == videos[0]["id"]
    assert seg["target_timerange"]["duration"] == 12600
    assert videos[0]['mute'] is False and videos[0]['audio_volume'] == 1


def test_draft_mutes_embedded_audio_only_when_replacement_track_exists(tmp_path):
    video = tmp_path/'clip.mp4'
    voice = tmp_path/'voice.wav'
    video.write_bytes(b'video with embedded narration')
    voice.write_bytes(b'intact separate narration')
    folder = export_draft(video,tmp_path/'drafts','test',8,1920,1080,voice_path=voice)
    content = json.loads((folder/'draft_content.json').read_text(encoding='utf-8'))
    assert content['materials']['videos'][0]['mute'] is True
    assert content['materials']['videos'][0]['audio_volume'] == 0
    assert len(content['materials']['audios']) == 1
    assert content['all_tracks'][1]['segments'][0]['speed'] == 1


def test_export_draft_uses_existing_template(tmp_path):
    root = tmp_path / "drafts"
    existing = root / "old_draft"
    existing.mkdir(parents=True)
    old = {"version": "v1_2_3", "all_tracks": [], "materials": {"videos": []},
           "canvas_config": {"canvas_size": {"width": 1280, "height": 720}}}
    (existing / "draft_content.json").write_text(json.dumps(old, ensure_ascii=False), encoding="utf-8")
    (existing / "draft_meta_info.json").write_text(
        json.dumps({"tm_draft_name": "old", "tm_draft_version": "10.5.0"}, ensure_ascii=False),
        encoding="utf-8")

    video = tmp_path / "ws" / "clip.mp4"
    video.parent.mkdir(parents=True)
    video.write_bytes(b"fake-video")
    draft_dir = export_draft(video, root, "成片", 8.0, 1080, 1920)
    content = json.loads((draft_dir / "draft_content.json").read_text(encoding="utf-8"))
    meta = json.loads((draft_dir / "draft_meta_info.json").read_text(encoding="utf-8"))
    assert content["version"] == "v1_2_3"
    assert meta["tm_draft_version"] == "10.5.0"
    assert len(content["materials"]["videos"]) == 1


def test_detect_draft_root_override_wins(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "local"))
    custom = tmp_path / "my_draft_root"
    custom.mkdir()
    assert detect_draft_root(str(custom)) == str(custom)
    assert detect_draft_root() == ""
    absent = tmp_path / "nope"
    assert detect_draft_root(str(absent)) == ""


def test_detect_draft_root_candidates(tmp_path, monkeypatch):
    local = tmp_path / "local"
    candidate = local / "JianyingPro/User Data/Projects/com.lveditor.draft"
    candidate.mkdir(parents=True)
    monkeypatch.setenv("LOCALAPPDATA", str(local))
    assert detect_draft_root() == str(candidate)
