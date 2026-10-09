"""Real app pipeline validation; no handwritten script or audio edits."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from factory.config import Config
from factory.pipeline import ProductionPipeline
from factory.log import EventEmitter
from factory.mining.story_discovery import discover
from factory.mining.visual_story import storyboard, _cached_call
from factory.mining.source_context import ensure_source_context
from factory.utils import read_json, write_json, ensure_utf8_console, media_duration

ensure_utf8_console()
config = Config.load('workspace/config.json')
source_job = read_json('workspace/jobs/20261010_002310_e75b17/job.json')
source = Path(source_job['artifacts']['video_path'])
p = ProductionPipeline(config, str(source), emitter=EventEmitter(lambda e:
    print(e.get('message') or e.get('detail') or e.get('stage') or '', flush=True)
    if e.get('kind') in ('log', 'stage') else None),
    job_id='software_validation_20261010', params_override={'max_clips': 0, 'review_mode': 'semi'})
p.state.artifacts.update(video_path=str(source), meta=source_job['artifacts'].get('meta', {}),
                         validation_only=True, original_task_preserved=source_job['job_id'])
p.state.completed_stages = ['DOWNLOADING', 'URL_ANALYZING']
p.save_state()
duration = media_duration(source)
ensure_source_context(p, source, duration)
root = p.job_dir/'analysis'/'director'
selection = discover(p, source, duration, root, p.router.get('video_analysis'), storyboard, _cached_call)
write_json(p.job_dir/'analysis'/'validation_chapters.json', selection)
print('AUTO_CHAPTERS', len(selection['candidates']), flush=True)
# Verify manual upper limit using the same discovered complete catalogue.
p.config.params['max_clips'] = 1
p.state.params['max_clips'] = 1
p.save_state()
p.analyze()
print('SOFTWARE_DRAFTS', [(c.candidate_id, c.title, c.source_start, c.source_end,
                          p.selected_variant_text(c)) for c in p.candidates], flush=True)
print('STATE', p.state.status, 'ERROR', p.state.error, flush=True)
