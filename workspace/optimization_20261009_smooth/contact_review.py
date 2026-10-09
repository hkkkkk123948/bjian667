from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from PIL import Image,ImageDraw,ImageFont
from factory.config import Config
from factory.pipeline import ProductionPipeline
from factory.utils import run_cmd,ffmpeg_bin

root=Path(__file__).resolve().parent
p=ProductionPipeline(Config.load('workspace/config.json'),'',job_id='20261009_软件优化验证_文案待审')
p._load_candidates()
v=p._pick_variant(p.candidates[0])
font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',17)
canvas=Image.new('RGB',(1024,4*210),'#161d24')
draw=ImageDraw.Draw(canvas)
for i,s in enumerate(v.sentences):
    ev=s.source_evidence
    time=(ev.start+ev.end)/2
    target=root/('draft_frame_%02d.jpg'%i)
    run_cmd([ffmpeg_bin(),'-v','error','-y','-threads','1','-ss',str(time),'-i',p.state.artifacts['video_path'],
             '-frames:v','1','-filter_threads','1','-vf','scale=480:-2',str(target)])
    image=Image.open(target).convert('RGB')
    image.thumbnail((480,168))
    x=(i%2)*512;y=(i//2)*210
    canvas.paste(image,(x,y))
    draw.text((x+8,y+173),'第%d句 · 源片 %.2f–%.2f秒'%(i+1,ev.start,ev.end),font=font,fill='white')
canvas.save(root/'new_draft_frames.jpg')
