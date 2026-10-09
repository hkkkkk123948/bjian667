from pathlib import Path
import json
from PIL import Image,ImageDraw,ImageFont
from factory.utils import ffmpeg_bin,run_cmd
root=Path('workspace/optimization_20261009_single_review')
job=Path('workspace/jobs/20261009_182042_c167f6/clips/clip_001')
tl=json.loads((job/'timeline.json').read_text(encoding='utf-8'))
video=job/'clip_001.mp4'
if not video.exists():video=root/'内存修复_原整段配音_画面待审.mp4'
font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',16)
sheet=Image.new('RGB',(1000,8*230),'#111923')
draw=ImageDraw.Draw(sheet)
for i,row in enumerate(tl['sentence_timings']):
    for n,f in enumerate((.25,.75)):
        time=row['start']+(row['end']-row['start'])*f
        image=root/('actual_%d_%d.jpg'%(i,n))
        run_cmd([ffmpeg_bin(),'-v','error','-y','-threads','1','-ss',str(time),'-i',str(video),
                 '-frames:v','1','-filter_threads','1','-vf','scale=480:-2',str(image)])
        im=Image.open(image);im.thumbnail((480,180))
        sheet.paste(im,(n*500,i*230))
        draw.text((n*500+6,i*230+183),'第%d句 · 成片 %.2f秒'%(i+1,time),font=font,fill='white')
    text=row.get('narrative_text') or row['text']
    draw.text((6,i*230+206),text[:60],font=font,fill='#a5e7ef')
sheet.save(root/'actual_sentence_frames.jpg')
