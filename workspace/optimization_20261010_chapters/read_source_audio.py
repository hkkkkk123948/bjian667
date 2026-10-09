"""Read the source once for story context, never judge or alter narration."""
from pathlib import Path
import json
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from factory.config import Config
from factory.pipeline import ProductionPipeline
from factory.utils import ensure_utf8_console,run_cmd,ffmpeg_bin,write_json
from factory.mining.source_context import cache_source_context

ensure_utf8_console()
root=Path(__file__).resolve().parent
p=ProductionPipeline(Config.load('workspace/config.json'),'',job_id='20261010_002310_e75b17')
audio=root/'source_audio.wav'
run_cmd([ffmpeg_bin(),'-v','error','-y','-threads','1','-i',p.state.artifacts['video_path'],
         '-vn','-ac','1','-ar','16000',str(audio)],timeout=180)
print('LOCAL_SOURCE_ASR_START',flush=True)
import torch
import whisper
torch.set_num_threads(2)
model=whisper.load_model('tiny',device='cpu',download_root=str(Path.home()/'.cache/whisper'))
result=model.transcribe(str(audio),language='en',fp16=False,temperature=0,
                        condition_on_previous_text=False,word_timestamps=False)
write_json(root/'source_transcript_raw.json',result)
cache_source_context(p,result.get('segments',[]),language='en',origin='local_asr_tiny')
print('LOCAL_SOURCE_ASR_DONE',len(result.get('segments',[])),flush=True)
