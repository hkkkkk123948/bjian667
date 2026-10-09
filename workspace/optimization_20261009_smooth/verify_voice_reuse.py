from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from factory.config import Config
from factory.pipeline import ProductionPipeline
from factory.providers.tts.continuous import synthesis_identity,_cache_key,_recover_result
from factory.utils import read_json,write_json

records=[]
for jid in ('20261009_003907_506bf7','20261009_001504_2cfe45','20261008_230044_95e6cd'):
    p=ProductionPipeline(Config.load('workspace/config.json'),'',job_id=jid)
    p._load_clips()
    folder=p.job_dir/'clips'/p.clips[0].clip_id
    old=read_json(folder/'tts.json')
    identity=synthesis_identity(p.config.tts)
    text=old['text']
    recovered,_=_recover_result(folder,p.config.workspace_path()/'.cache'/'tts_continuous',
                                _cache_key(text,identity),text,identity,p.config.tts,p.config.tts['speed'])
    records.append({'job_id':jid,'existing_full_audio_recovered':bool(recovered),'tts_called':False})
write_json(Path(__file__).parent/'voice_reuse.json',records)
print(records)
