import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from factory.config import Config
from factory.pipeline import ProductionPipeline
from factory.utils import ensure_utf8_console, write_json
from cli import make_emitter

ensure_utf8_console()
root = Path(__file__).resolve().parent
records = []
for number, tag in [(3, 'wqQZiP2lDOY'), (1, 'F35JDpVZ27Y'), (2, 'sChF5ilNHaM')]:
    source = next(Path('测试素材文件夹/原片').glob('*' + tag + '*'))
    config = Config.load(root / 'run_config.json')
    pipe = ProductionPipeline(config, str(source.resolve()), references=[str(root / 'reference.txt')], emitter=make_emitter())
    records.append({'number': number, 'job_id': pipe.job_id, 'source': str(source.resolve()), 'status': 'running'})
    write_json(root / 'batch.json', records)
    print('START', number, pipe.job_id, flush=True)
    state = pipe.run()
    records[-1].update(status=state.status, error=state.error,
                       clips=[{'path': c.path, 'status': c.status, 'qc': c.qc, 'text': c.script_text} for c in pipe.clips])
    write_json(root / 'batch.json', records)
    print('RESULT', number, pipe.job_id, state.status, state.error, flush=True)
    if state.status != 'COMPLETED':
        raise SystemExit(2)
print('ALL_COMPLETED', json.dumps([{'number': r['number'], 'job_id': r['job_id']} for r in records]), flush=True)
