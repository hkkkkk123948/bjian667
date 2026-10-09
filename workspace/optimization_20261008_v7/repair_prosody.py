import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from factory.config import Config
from factory.pipeline import ProductionPipeline
from factory.models import to_dict, candidate_from_dict
from factory.utils import read_json, write_json, ensure_utf8_console
from cli import make_emitter

ensure_utf8_console()
root=Path(__file__).resolve().parent
config=Config.load(root/'preview_config.json')
old=ProductionPipeline(config,'',job_id='20261008_223129_3e2124')
old._load_candidates();old._load_index()
p=ProductionPipeline(config,old.state.url,references=old.references,emitter=make_emitter())
write_json(root/'preview_job.json',{'job_id':p.job_id})
p._stage('URL_ANALYZING');p._do_ingest();p._done('URL_ANALYZING')
p.index=old.index
p.index.video.path=p.state.artifacts['video_path']
write_json(p.job_dir/'index/video_index.json',to_dict(p.index))
p.candidates=[candidate_from_dict(to_dict(old.candidates[0]))]
candidate=p.candidates[0]
p.state.artifacts['director_local_gate']=True
p._apply_script_edit(candidate,(root/'03_文案待审.txt').read_text(encoding='utf-8'))
candidate.status='approved'
p._save_candidates()
for stage in ['MEDIA_INDEXING','SCRIPT_GENERATED','CANDIDATE_FILTERED']:
    p._done(stage)
from factory.script.review_edit import review_text
(root/'03_标注版文案.txt').write_text(review_text(p._pick_variant(candidate)),encoding='utf-8')
state=p.produce()
print('RESULT',p.job_id,state.status,flush=True)
write_json(root/'preview_result.json',{'job_id':p.job_id,'status':state.status,
    'clips':[to_dict(c) for c in p.clips]})
raise SystemExit(0 if state.status=='COMPLETED' else 2)
