"""Denser review sheets: captures 2fps, overlapping forward/elevated camera views."""
import json,sys
from pathlib import Path
from PIL import Image,ImageDraw
folder=Path(sys.argv[1]);out=folder/'dense-review';out.mkdir(exist_ok=True)
rows=[]
for line in (folder/'coverage.jsonl').read_text().splitlines():
 try:rows.append(json.loads(line))
 except ValueError:pass
last=rows[-1]['shot']+(1 if (folder/'videos').exists() else 0)
for shot in sorted(set(r['shot'] for r in rows if 0<r['shot']<last)):
 frames=sorted((folder/'inspection').glob(f'shot-{shot:02}-frame-*.png'));rr=[r for r in rows if r['shot']==shot]
 selected=[];t=-1
 for i,r in enumerate(rr):
  if r['wallTime']-t>=.5:selected.append((i,r));t=r['wallTime']
 for page in range((len(selected)+15)//16):
  target=out/f'{shot:02}-{page:02}.jpg'
  if target.exists():continue
  im=Image.new('RGB',(1280,800),'#152229');draw=ImageDraw.Draw(im)
  for n,(i,r) in enumerate(selected[page*16:(page+1)*16]):
   pic=Image.open(frames[min(i,len(frames)-1)]);pic.thumbnail((320,180));x=n%4*320;y=n//4*200;im.paste(pic,(x,y));draw.text((x+4,y+182),f'{r["label"]} / {r["station"]:.0f}m',fill='white')
  im.save(target)
print(out)
