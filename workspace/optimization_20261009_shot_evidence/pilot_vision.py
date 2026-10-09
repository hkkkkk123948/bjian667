import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from factory.config import Config
from factory.providers.manager import LLMRouter
from factory.utils import ensure_dir, run_cmd, ffmpeg_bin, write_json

root = ensure_dir(Path('workspace/optimization_20261009_shot_evidence'))
video = Path('workspace/jobs/20261009_182042_c167f6/source/02.mp4')
times = [302.5, 375.5, 421.5, 442.5]
images = []
for index, time in enumerate(times, 1):
    path = root / ('pilot_%d.jpg' % index)
    if not path.exists():
        run_cmd([ffmpeg_bin(), '-v', 'error', '-y', '-threads', '1', '-ss', str(time), '-i', str(video),
                 '-frames:v', '1', '-vf', 'scale=480:-2', '-filter_threads', '1', str(path)], timeout=90)
    images.append(str(path))
write_json(root / 'pilot_frames.json', [{'image': i+1, 'time': time, 'path': images[i]} for i, time in enumerate(times)])

prompt = ('只描述这4张独立真实动画帧，每图编号1至4，不编故事，不推测之前或之后。'
          '逐图说明可见人物数量、穿着/外貌、具体位置、姿态或动作、动物/物件；看不清就说不确定。'
          '区分谁在床上谁在床旁，站立、坐着、躺着；没有实际看见就不写帽子、警车、打斗或追逐。'
          '不同图人物是否同一人不确定时不要强行合并。只返回JSON '
          '{"frames":[{"image":1,"visible":"80字内实际可见内容"}],"uncertain":["不确定之处"]}。')
config = Config.load(Path('workspace/config.json'))
router = LLMRouter(config)

def call(role):
    provider = router.get(role)
    provider.max_retry = 1
    provider.conf['max_tokens'] = 900
    result = provider.complete_json(prompt, images=images, json_mode=True, default=None)
    return {'role': role, 'model': provider.model, 'response': result, 'usage': provider.last_usage}

with ThreadPoolExecutor(max_workers=2) as pool:
    results = list(pool.map(call, ['video_analysis', 'qc']))
write_json(root / 'pilot_results.json', {'frames': [{'image': i+1, 'time': time} for i, time in enumerate(times)], 'results': results})
print(json.dumps(results, ensure_ascii=False, indent=2))
