import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from factory.config import Config
from factory.pipeline import ProductionPipeline
from factory.render.exporter import export_clip
from factory.eval.media_audit import audit_media
from factory.render.reframing import resolve_output_size
from factory.utils import read_json,write_json,ensure_utf8_console

ensure_utf8_console()
root=Path(__file__).resolve().parent
results=read_json(root/'preview_repairs.json') or []
for number,jid in enumerate(('20261009_003907_506bf7','20261009_001504_2cfe45'),1):
    if '--second-only' in sys.argv and number != 2:
        continue
    config=Config.load('workspace/config.json')
    p=ProductionPipeline(config,'',job_id=jid)
    p._load_index();p._load_clips()
    clip=p.clips[0]
    old=Path(clip.path).parent
    tts=read_json(old/'tts.json')
    timeline=read_json(old/'timeline.json')
    width,height=p.state.artifacts.get('source_size') or (p.index.video.width,p.index.video.height)
    config.params.update(zip(('target_width','target_height'),resolve_output_size(width,height,config.params)))
    out=root/('repaired_%s'%jid);out.mkdir(exist_ok=True)
    target=out/'视频轨修复预览.mp4'
    write_json(out/'timeline.json',timeline)
    export_clip(p.state.artifacts['video_path'],tts['audio_path'],None,timeline,config.params,target,
                black_box=tuple(p.state.artifacts['black_box']) if p.state.artifacts.get('black_box') else None,
                src_size=(width,height))
    audit=audit_media(target,timeline)
    item={'job_id':jid,'path':str(target.resolve()),'audit':audit,'note':'仅修复导出视频轨，保留原文案、完整原配音；时长要求和真实语义仍需另审'}
    results=[r for r in results if r['job_id'] != jid]+[item];write_json(root/'preview_repairs.json',results)
    print('REPAIRED',jid,audit['video_duration'],audit['audio_duration'],audit['status'],flush=True)
