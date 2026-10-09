"""Exercise the real draft/rewrite workflow in an isolated validation job."""
import copy
import hashlib
from pathlib import Path
import shutil
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from factory.config import Config
from factory.pipeline import ProductionPipeline
from factory.providers.base import reset_usage
from factory.utils import read_json,write_json,ensure_utf8_console
from factory.script.duration import estimate_duration
from factory.script.review_edit import review_text

ensure_utf8_console()
config = Config.load('workspace/config.json')
old = ProductionPipeline(config,'',job_id='20261009_001504_2cfe45')
old._load_candidates();old._load_index()
candidate = old._by_id('C002')
before = hashlib.sha256((old.job_dir/'analysis'/'story_candidates.json').read_bytes()).hexdigest()
p = ProductionPipeline(config,old.url,references=old.references,
                       job_id='20261009_软件优化验证_文案待审',
                       params_override={'review_mode':'semi','min_duration':35,'max_duration':50,'max_clips':1})
p.state.artifacts = copy.deepcopy(old.state.artifacts)
p.state.artifacts['validation_job'] = True
p.state.completed_stages = [s for s in old.state.completed_stages if s not in
                           {'TTS_GENERATED','MATCHING','TIMELINE_ALIGNED','RENDERING','FINAL_MATCH_CHECK',
                            'QUALITY_CHECK','DRAFT_EXPORT','COMPLETED'}]
p.index = copy.deepcopy(old.index)
p.candidates = [copy.deepcopy(candidate)]
p._save_candidates();p.save_state()
source = old.job_dir/'analysis'/'director'/'story_02'/'detail'
target = p.job_dir/'analysis'/'director'/'story_02'/'detail'
if source.exists() and not target.exists():
    shutil.copytree(source,target)
reset_usage()
print('REAL_DRAFT_START',flush=True)
p.apply_review('C002','regenerate')
v = p._pick_variant(p.candidates[0])
estimate = estimate_duration(p.config,v.full_text)
record = {'job_id':p.job_id,'status':p.candidates[0].status,'estimated_seconds':estimate.seconds,
          'estimated_range':[estimate.low_seconds,estimate.high_seconds],
          'sample_count':estimate.sample_count,'story_grounding':v.match_report.get('story_grounding'),
          'script':review_text(v),'original_job_untouched':
              hashlib.sha256((old.job_dir/'analysis'/'story_candidates.json').read_bytes()).hexdigest()==before,
          'tts_called':False,'picture_matching_called':False}
write_json(Path(__file__).parent/'real_draft_validation.json',record)
(Path(__file__).parent/'新文案_待审核.txt').write_text(review_text(v),encoding='utf-8')
print('REAL_DRAFT_DONE',record['status'],record['estimated_seconds'],record['story_grounding'],flush=True)
