import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from factory.config import Config
from factory.providers.openai_compat import OpenAICompatProvider
from factory.utils import write_json,ensure_utf8_console
ensure_utf8_console()
config=Config.load('workspace/config.json')
conf=dict(config.provider('qc'),model='qwen/qwen3.6-plus',max_tokens=2200,max_retry=1,
          enable_thinking=False,temperature=.1,timeout=120)
conf.pop('reasoning_effort',None);conf.pop('thinking',None)
p=OpenAICompatProvider(conf)
images=[str(Path('workspace/optimization_20261009_shot_evidence')/('pilot_%d.jpg'%i)) for i in range(1,5)]
prompt=('依次观察四张真实动画图片，逐图描述实际可见的人物、动物、动作和所在位置。'
        '不讲故事、不推断没看到的动作。没有明确证据就写不确定。每图描述不超过60字。'
        '另外逐图回答图中是否确实能看到帽子、车辆；不要把衣服颜色当成帽子。'
        '只返回JSON {"images":[{"image_number":1,"visible":"描述","hat":"有/无/不确定","vehicle":"有/无/不确定"}]}。')
result=p.complete_json(prompt,images=images,json_mode=True,default=None)
write_json('workspace/optimization_20261010_chapters/qwen_vision_probe.json',{'model':p.model,'result':result,'usage':p.last_usage})
print(p.model,result,p.last_usage,flush=True)
