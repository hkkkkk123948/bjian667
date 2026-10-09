from types import SimpleNamespace

import pytest

from factory.models import ClipResult
from factory.utils import write_json


def prepare(tmp_path, monkeypatch, verdict='pass', export_error=False):
    from factory import planned
    source = tmp_path/'source.mp4'
    source.write_bytes(b'source fixture')
    plan_path = tmp_path/'plan.json'
    write_json(plan_path,{'source':str(source),'sentences':[
        {'text':'快跑！','evidence_start':1,'evidence_end':3},
        {'text':'快跑！','evidence_start':1,'evidence_end':3}]})
    events = []

    class Pipeline:
        def __init__(self, config, *args, **kwargs):
            self.config = config
            self.job_dir = tmp_path/'job'
            self.state = SimpleNamespace(artifacts={'meta':{}},status='CREATED')
            self.clips = []

        def completed(self, name):
            return False

        def _stage(self, name):
            events.append('stage:'+name)

        def _done(self, name):
            events.append('done:'+name)

        def _do_ingest(self):
            events.append('ingest')

        def _save_candidates(self):
            pass

        def _do_tts(self, queue):
            events.append('tts')
            self.clips = [ClipResult(clip_id='clip001',script_text=queue[0].scripts[0].full_text,duration=99)]

        def _do_timeline(self, queue):
            events.append('timeline')
            write_json(self.job_dir/'clips/timeline.json',{'narration_duration':99,'segments':[],
                        'sentence_timings':[{'index':1,'start':0,'end':99,'text':'快跑！'}]})

        def _do_render(self, queue):
            events.append('render')
            target = self.job_dir/'clips/out.mp4'
            target.parent.mkdir(exist_ok=True,parents=True)
            target.write_bytes(b'export fixture')
            self.clips[0].path = str(target)

        def _do_qc(self, queue):
            events.append('picture_review')
            self.clips[0].qc = {'status':verdict,'review_scope':'picture_matching_only'}
            self.clips[0].status = 'completed' if verdict=='pass' else 'qc_failed'

        def _save_clips(self):
            pass

        def save_state(self):
            pass

        def _write_report(self):
            pass

        def _fail(self, error):
            events.append('error:'+str(error))

    monkeypatch.setattr(planned,'ProductionPipeline',Pipeline)
    monkeypatch.setattr(planned,'media_duration',lambda path:10)
    monkeypatch.setattr(planned,'final_qc',lambda *args,**kwargs:{
        'status':'fail' if export_error else 'pass','issues':['导出文件无法读取'] if export_error else []})
    return planned,plan_path,events


@pytest.mark.parametrize('verdict',['pass','fail'])
def test_authored_plan_uses_one_picture_review_and_no_script_duration_or_structure_gate(tmp_path,monkeypatch,verdict):
    planned,path,events = prepare(tmp_path,monkeypatch,verdict)
    pipeline = planned.render_plan({'workspace':str(tmp_path/'ws'),'params':{'min_duration':35,'max_duration':50}},path)
    assert events.count('tts') == events.count('render') == events.count('picture_review') == 1
    assert pipeline.clips[0].duration == 99
    assert pipeline.clips[0].qc['review_scope'] == 'picture_matching_only'
    assert pipeline.state.status == ('COMPLETED' if verdict=='pass' else 'REVIEW_REQUIRED')
    assert pipeline.clips[0].scores['timeline_diagnostics']['gate_enabled'] is False


def test_authored_plan_preserves_real_export_errors(tmp_path,monkeypatch):
    planned,path,events = prepare(tmp_path,monkeypatch,export_error=True)
    with pytest.raises(RuntimeError,match='真实导出失败'):
        planned.render_plan({'workspace':str(tmp_path/'ws')},path)
    assert events.count('tts') == events.count('render') == 1
    assert 'picture_review' not in events
    assert any(event.startswith('error:') for event in events)


def test_invalid_source_range_remains_a_real_input_error(tmp_path,monkeypatch):
    planned,path,events = prepare(tmp_path,monkeypatch)
    write_json(path,{'source':str(tmp_path/'source.mp4'),'sentences':[
        {'text':'快跑！','evidence_start':1,'evidence_end':11}]})
    with pytest.raises(ValueError,match='超出源片'):
        planned.render_plan({'workspace':str(tmp_path/'ws')},path)
    assert events == []
