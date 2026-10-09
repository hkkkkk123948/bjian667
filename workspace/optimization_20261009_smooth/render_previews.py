import hashlib
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from factory.config import Config
from factory.pipeline import ProductionPipeline
from factory.providers.tts.continuous import align_continuous_result
from factory.render.timeline import build_timeline
from factory.render.exporter import export_clip
from factory.render.reframing import resolve_output_size
from factory.eval.media_audit import audit_media
from factory.utils import read_json, write_json, ensure_utf8_console, media_size, media_duration

ensure_utf8_console()
root = Path(__file__).resolve().parent
delivery = root.parents[1]/'交付成片'/'2026-10-09_流畅剪辑优化预览'
delivery.mkdir(parents=True,exist_ok=True)
records = []
jobs = [('20261009_003907_506bf7','boy','01_雨夜恶犬'),
        ('20261009_001504_2cfe45','girl','02_林地女孩'),
        ('20261008_230044_95e6cd',None,'03_后巷怪人')]
for jid,audit_name,name in jobs:
    config = Config.load('workspace/config.json')
    p = ProductionPipeline(config,'',job_id=jid)
    config = p.config
    config.params.update(p.state.params)
    p._load_candidates();p._load_index();p._load_clips()
    clip = p.clips[0]
    cand = p._by_id(clip.candidate_id)
    variant = next((v for v in cand.scripts if v.variant_id == clip.variant_id),None) or p._pick_variant(cand)
    original = p.job_dir/'clips'/clip.clip_id
    tts = read_json(original/'tts.json')
    voice = Path(tts['audio_path'])
    before_voice = hashlib.sha256(voice.read_bytes()).hexdigest()
    before_copy = hashlib.sha256(variant.full_text.encode()).hexdigest()
    if tts.get('text') and tts['text'] != ''.join(s.text for s in variant.sentences):
        raise RuntimeError('Existing narration differs from saved script '+jid)
    audit = read_json(root.parent/'optimization_20261009_speech_alignment'/(audit_name+'.audit.json')) if audit_name else None
    aligned = align_continuous_result(variant.sentences,tts,media_duration(voice),audit)
    width,height = media_size(p.state.artifacts['video_path'])
    config.params.update(zip(('target_width','target_height'),resolve_output_size(width,height,config.params)))
    timeline = build_timeline(cand,variant,aligned,config.params,index=p.index,router=None,
                              video_duration=p.index.video.duration if p.index else 0)
    folder = root/jid
    folder.mkdir(exist_ok=True)
    target = folder/(name+'_流畅剪辑_无字幕预览.mp4')
    write_json(folder/'timeline.json',timeline)
    write_json(folder/'tts_alignment.json',aligned)
    print('RENDER_START',jid,'beats',len(variant.sentences),'segments',len(timeline['segments']),flush=True)
    export_clip(p.state.artifacts['video_path'],voice,None,timeline,config.params,target,
                src_size=(width,height),
                black_box=tuple(p.state.artifacts['black_box']) if p.state.artifacts.get('black_box') else None)
    result = audit_media(target,timeline)
    assert hashlib.sha256(voice.read_bytes()).hexdigest() == before_voice
    assert hashlib.sha256(variant.full_text.encode()).hexdigest() == before_copy
    import shutil
    copied = delivery/target.name
    shutil.copy2(target,copied)
    decisions = read_json(folder/'edit_decisions.json')
    records.append({'job_id':jid,'preview':str(copied),'audit':result,'edit_decisions':decisions,
                    'voice_sha256_unchanged':True,'script_sha256_unchanged':True,
                    'alignment':aligned.get('alignment_report'),
                    'content_note':'保留原审核稿与配音；第一条旧稿低于35秒，第二条原语义问题未擅改。'})
    write_json(root/'results.json',records)
    write_json(delivery/'验证记录.json',records)
    print('RENDER_DONE',jid,result['status'],result['video_duration'],result['audio_duration'],
          'shots',decisions['rendered_shots'],'joins',decisions['context_joins'],flush=True)
