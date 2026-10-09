import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from factory.config import Config
from factory.pipeline import ProductionPipeline
from factory.utils import ensure_utf8_console, write_json
from factory.models import to_dict
from cli import make_emitter

ensure_utf8_console()
root = Path(__file__).resolve().parent
config = Config.load(root / 'formal_config.json')
p = ProductionPipeline(config, '', job_id='20261008_230044_95e6cd', emitter=make_emitter())
p.url = p.state.url
p._load_candidates()
p._load_index()
p._load_clips()
text = Path('交付成片/2026-10-08_第三条断句修复与文案审核/03_文案可修改.txt').read_text(encoding='utf-8-sig')
p.apply_review(p.candidates[0].candidate_id, 'approve', new_text=text)
p.state.artifacts['user_script_approval'] = {'source': '用户在聊天中确认当前标注文案', 'date': '2026-10-08'}
p.state.artifacts['qc_provider_model'] = config.provider('qc')['model']
p.state.completed_stages = [s for s in p.state.completed_stages if s not in {'QUALITY_CHECK','DRAFT_EXPORT','COMPLETED'}]
p.state.error = ''
p.save_state()
state = p.produce()
write_json(root / 'formal_result.json', {'job_id':p.job_id,'status':state.status,'clips':[to_dict(c) for c in p.clips]})
print('FORMAL_RESULT',p.job_id,state.status,flush=True)
raise SystemExit(0 if state.status=='COMPLETED' else 2)
