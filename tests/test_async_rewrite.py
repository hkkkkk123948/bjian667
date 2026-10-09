import pytest

from factory.utils import read_json
from test_review_edit import fixture


@pytest.mark.parametrize('fail',[False,True])
def test_rewrite_worker_keeps_ui_responsive_and_preserves_draft_on_failure(tmp_path,monkeypatch,fail):
    pytest.importorskip('PySide6')
    from factory.gui.qt import create_app
    p,c,v = fixture(tmp_path)
    p.config.params['review_mode'] = 'semi'
    path = tmp_path/'config.json'
    p.config.save(path)
    p.state.status = 'CANDIDATE_FILTERED'
    p._save_candidates()
    p.save_state()
    app,window = create_app(config_path=str(path))
    try:
        window.resume_job(p.job_id)
        pipeline = window.pipeline
        page = window.pages['review']
        draft = '用户还没保存的文案（独立评论话术）'
        page._drafts[c.candidate_id] = draft
        calls,workers = [],[]
        def rewrite(cid,action):
            calls.append((cid,action))
            if fail:
                raise ValueError('provider unavailable')
        def spawn(fn):
            window.running = True
            workers.append(fn)
        monkeypatch.setattr(pipeline,'apply_review',rewrite)
        monkeypatch.setattr(window,'_spawn',spawn)
        pipeline.cancel_requested = True
        window.regenerate_script(c.candidate_id)
        app.processEvents()
        assert calls == [] and len(workers) == 1
        assert pipeline.cancel_requested is False
        assert page.script.isReadOnly()
        if fail:
            with pytest.raises(ValueError):
                workers[0]()
        else:
            workers[0]()
            window._pump()
        window._dispatch({'kind':'worker_done'})
        app.processEvents()
        assert not window.running and not page.script.isReadOnly()
        if fail:
            assert page._drafts[c.candidate_id] == draft
        else:
            assert c.candidate_id not in page._drafts
            assert read_json(pipeline.job_dir/'analysis'/'ui_draft_history.json')[-1]['text'] == draft
    finally:
        window.close()
