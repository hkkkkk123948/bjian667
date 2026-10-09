from pathlib import Path
import hashlib
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from factory.config import Config
from factory.pipeline import ProductionPipeline
from factory.render.exporter import export_clip
from factory.utils import read_json,write_json,media_duration,run_cmd,ensure_utf8_console
ensure_utf8_console()
root=Path(__file__).resolve().parent
p=ProductionPipeline(Config.load('workspace/config.json'),'',job_id='20261009_182042_c167f6')
p._load_index()
old=p.job_dir/'clips'/'clip_001'
timeline=read_json(old/'timeline.json')
voice=Path(timeline['voice_path'])
digest=hashlib.sha256(voice.read_bytes()).hexdigest()
params=dict(p.config.params,prefer_nvenc='off')
size=tuple(p.state.artifacts.get('source_size') or (1920,1080))
params.update(target_width=size[0],target_height=size[1])
target=root/'内存修复_原整段配音_画面待审.mp4'
write_json(root/'timeline.json',timeline)
print('RENDER_START',flush=True)
export_clip(p.state.artifacts['video_path'],voice,None,timeline,params,target,src_size=size,
            black_box=tuple(p.state.artifacts['black_box']) if p.state.artifacts.get('black_box') else None)
assert hashlib.sha256(voice.read_bytes()).hexdigest()==digest
out,_,_=run_cmd(['ffprobe','-v','error','-show_entries','stream=codec_type,width,height,duration,nb_frames','-of','json',str(target)])
import json
record={'file':str(target),'streams':json.loads(out)['streams'],'duration':media_duration(target),
        'voice_unchanged':True,'render_budget':read_json(root/'render_resources.json'),'visual_review':'pending'}
write_json(root/'render_result.json',record)
print('RENDER_DONE',record,flush=True)
