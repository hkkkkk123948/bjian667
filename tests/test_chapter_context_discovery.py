from types import SimpleNamespace

from factory.config import Config
from factory.models import StoryCandidate
from factory.utils import write_json
from factory.mining.chapter_hints import cache_verified_cards, load_chapter_hints, normalize_cards
from factory.mining.source_context import cache_source_context, source_context
from factory.mining.story_discovery import discover, normalize_global


def stub(tmp_path, candidates=None):
    video = tmp_path / 'source.mp4'
    video.write_bytes(b'exact same source footage')
    config = Config({'workspace': str(tmp_path/'ws'), 'params': {'max_clips': 0}})
    p = SimpleNamespace(config=config, job_dir=tmp_path/'job', candidates=candidates or [],
                        state=SimpleNamespace(artifacts={'video_path': str(video)}),
                        emitter=SimpleNamespace(log=lambda *args, **kw: None),
                        _check_cancel=lambda: None, save_state=lambda: None)
    p._load_candidates = lambda: p.candidates
    return p, video


def cards():
    return [{'chapter_id': 'CH01', 'title': 'Actual chapter one', 'frame_ids': ['REAL_A'],
             'source_start': 4, 'source_end': 159},
            {'chapter_id': 'CH02', 'title': 'Actual chapter two', 'frame_ids': ['REAL_B'],
             'source_start': 168, 'source_end': 442},
            {'chapter_id': 'CH03', 'title': 'Actual chapter three', 'frame_ids': ['REAL_C'],
             'source_start': 448, 'source_end': 612}]


def test_verified_actual_chapter_cards_produce_three_full_chapters_without_splitting_context(tmp_path):
    p, video = stub(tmp_path)
    cache_verified_cards(p, cards())
    def no_model(*args, **kwargs):
        raise AssertionError('已核实章节卡不应再次收费看图/编目录')
    result = discover(p, video, 619, p.job_dir/'analysis/director', SimpleNamespace(), no_model, no_model)
    assert len(result['candidates']) == 3
    assert [(row['source_start'], row['source_end']) for row in result['candidates']] == [(4,159),(168,442),(448,612)]
    assert result['discovery']['complete']


def test_old_eight_proposals_keep_only_three_saved_drafts_and_never_resume_wrong_four_to_eight(tmp_path):
    saved = [StoryCandidate(candidate_id='C%03d' % (i+1), title='保存稿%d' % i,
                            source_start=start, source_end=end) for i,(start,end) in enumerate([(5,130),(120,153),(164,255)])]
    p, video = stub(tmp_path, saved)
    cache_verified_cards(p, cards())
    root = p.job_dir/'analysis/director'
    write_json(root/'selection.json', {'policy':'old', 'mode':'auto', 'candidates':[
        {'title':'旧提案%d' % i,'source_start':i*60, 'source_end':i*60+50,'story_summary':'旧事件'} for i in range(8)]})
    def no_model(*args, **kwargs):
        raise AssertionError('不能继续旧拆分提案4–8收费')
    first = discover(p, video, 619, root, SimpleNamespace(), no_model, no_model)
    second = discover(p, video, 619, root, SimpleNamespace(), no_model, no_model)
    for result in (first,second):
        assert len(result['candidates']) == 3 and len(result['corrected_candidates']) == 3
        assert result['legacy_saved_drafts'] and result['discovery']['requires_new_catalogue_task']
        assert not result['discovery']['complete']
    assert [item.title for item in p.candidates] == ['保存稿0','保存稿1','保存稿2']


def test_nonoverlapping_fragments_of_same_protagonist_chain_can_merge_across_tv_or_map_context():
    parts = [{'id':'A', 'title':'遇险', 'source_start':10,'source_end':50,'story_summary':'男孩遇险',
              'opening_seen':True,'outcome_seen':False},
             {'id':'B', 'title':'求助', 'source_start':80,'source_end':120,'story_summary':'同一男孩求助',
              'opening_seen':False,'outcome_seen':True}]
    data = {'stories':[{'title':'一篇完整故事','story_summary':'遇险到求助','continuity_reason':'同一主角同一危险链，中间电视为铺垫',
                        'observation_ids':['A','B']}], 'unresolved':[]}
    stories, uncertain = normalize_global(data,parts)
    assert len(stories) == 1 and uncertain == []
    assert (stories[0]['source_start'],stories[0]['source_end']) == (10,120)


def test_source_narration_context_is_cached_shared_and_never_promoted_to_visible_action(tmp_path):
    p, video = stub(tmp_path)
    path = cache_source_context(p,[{'start':1,'end':6,'text':'My friend called me.'},
                                  {'start':50,'end':55,'text':'Later we went home.'}])
    first = source_context(p,0,10)
    assert first['available'] and first['segments'][0]['text'] == 'My friend called me.'
    assert first['basis'] == 'original_narration_context_not_visual_evidence'
    assert len(source_context(p,45,60)['segments']) == 1
    p.state.artifacts.pop('source_transcript_path')
    p.job_dir = tmp_path/'second_job'
    assert source_context(p,0,10)['available']
    assert path.endswith('transcript.json')


def test_card_reader_binds_actual_frame_times_and_excludes_map_without_guessing_seconds():
    frames = [{'frame_id':'TITLE_A','time':2},{'frame_id':'MAP','time':155},{'frame_id':'TITLE_B','time':164}]
    response = {'chapter_cards':[{'frame_ids':['TITLE_A'],'title':'First actual story'},
                                {'frame_ids':['TITLE_B'],'title':'Second actual story'}],
                'ignored':[{'frame_id':'MAP','reason':'同章路线地图，不是新章节'}]}
    result = normalize_cards(response,frames,619)
    assert len(result) == 2
    assert result[0]['source_start'] == 3 and result[0]['source_end'] == 163
    assert result[1]['source_start'] == 165


def test_pure_card_read_interface_uses_content_not_filename_number(tmp_path):
    p, video = stub(tmp_path)
    cache_verified_cards(p,cards())
    assert len(load_chapter_hints(p,video,619,p.job_dir)) == 3


def test_bounded_context_keeps_actual_ending_instead_of_only_prefix(tmp_path):
    p, video = stub(tmp_path)
    rows = [{'start':i*5,'end':i*5+4,'text':('setup_%d_' % i)*30} for i in range(30)]
    rows[-1]['text'] = 'Police arrested the intruder. Becky did not eat the food.'
    cache_source_context(p,rows)
    result = source_context(p,0,160,max_chars=1000)
    assert result['truncated'] and result['omitted_middle']
    assert result['sampling'] == 'opening_middle_ending'
    assert 'Police arrested' in result['segments'][-1]['text']
    assert sum(len(item['text']) for item in result['segments']) <= 1000


def test_unrelated_subtitle_in_same_folder_cannot_be_used_as_this_source_context(tmp_path):
    from factory.mining.source_context import ensure_source_context
    p, video = stub(tmp_path)
    p.config.params['source_context_auto'] = 'off'
    (tmp_path/'another_movie.srt').write_text('1\n00:00:01,000 --> 00:00:05,000\nUnrelated story that must not be used.\n',encoding='utf-8')
    result = ensure_source_context(p,video,100)
    assert not result['available'] and not source_context(p,0,100)['available']


def test_same_stem_subtitle_is_automatically_cached_without_cpu_or_cloud(tmp_path):
    from factory.mining.source_context import ensure_source_context
    p, video = stub(tmp_path)
    video.with_suffix('.srt').write_text('1\n00:00:01,000 --> 00:00:05,000\nThis is the original source narration.\n',encoding='utf-8')
    result = ensure_source_context(p,video,100)
    assert result['available']
    context = source_context(p,0,100)
    assert context['origin'] == 'original_subtitles'
    assert context['segments'][0]['text'] == 'This is the original source narration.'


def test_local_asr_uses_explicit_existing_checkpoint_two_thread_worker_and_sha_reuse(tmp_path,monkeypatch):
    import factory.mining.source_context as module
    from pathlib import Path
    p, video = stub(tmp_path)
    fake_home = tmp_path/'home'
    checkpoint=fake_home/'.cache/whisper/tiny.pt'
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b'local model fixture')
    monkeypatch.setattr(Path,'home',classmethod(lambda cls:fake_home))
    calls=[]
    def run(args,**kwargs):
        calls.append(args)
        if '--output' in args:
            output=Path(args[args.index('--output')+1])
            write_json(output,{'language':'en','segments':[{'start':0,'end':10,'text':'Original local English narration.'}]})
        else:
            Path(args[-1]).write_bytes(b'local wav fixture')
        return '', '', 0
    monkeypatch.setattr(module,'run_cmd',run)
    assert module.ensure_source_context(p,video,100)['available']
    assert len(calls)==2 and calls[0][calls[0].index('-threads')+1]=='2'
    assert '--checkpoint' in calls[1] and str(checkpoint) in calls[1]
    assert 'factory.mining.source_context_worker' in calls[1]
    p.job_dir=tmp_path/'new_same_source_job'
    p.state.artifacts.pop('source_transcript_path',None)
    assert module.ensure_source_context(p,video,100)['available'] and len(calls)==2


def test_candidate_facts_does_not_trust_old_guessed_collage_table(tmp_path,monkeypatch):
    from factory.mining.visual_story import _candidate_facts
    import factory.mining.shot_evidence as module
    p, video=stub(tmp_path)
    candidate=StoryCandidate(candidate_id='C001',source_start=0,source_end=100)
    write_json(p.job_dir/'analysis/director/story_01/visual_facts.json',{'facts':[
        {'id':'F%d'%i,'start':i*10,'end':i*10+5,'visible':'old guessed picture'} for i in range(6)]})
    monkeypatch.setattr(module,'get_shot_catalog',lambda *args:[])
    facts,_=_candidate_facts(p,candidate,extract=False)
    assert facts==[]
