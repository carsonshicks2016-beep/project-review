"""Review multiple existing native camera modes together, retaining their original videos."""
import json,sys
from pathlib import Path
from PIL import Image,ImageDraw
folder=Path(sys.argv[1]);out=folder/'camera-review';out.mkdir(exist_ok=True)
rows=[]
for line in (folder/'coverage.jsonl').read_text().splitlines():
 try:rows.append(json.loads(line))
 except ValueError:pass
plan=json.loads((folder/'inspection-plan.json').read_text())['shots'];groups={};last=rows[-1]['shot']+(1 if (folder/'videos').exists() else 0)
for i,s in enumerate(plan):
 if i==0 or i>=last:continue
 key=s['label'].rsplit('-',1)[0];groups.setdefault(key,[]).append((i,s))
for key,items in groups.items():
 expected=6 if 'cameras-' in folder.name else 3
 if len(items)!=expected:continue
 target=out/(key+'.jpg')
 if target.exists():continue
 im=Image.new('RGB',(1280,expected*210),'#152229');draw=ImageDraw.Draw(im)
 for row,(i,s) in enumerate(items):
  frames=sorted((folder/'inspection').glob(f'shot-{i:02}-frame-*.png'));rr=[r for r in rows if r['shot']==i]
  for col,fraction in enumerate((.08,.35,.65,.92)):
   n=min(len(frames)-1,int(fraction*len(frames)));pic=Image.open(frames[n]);pic.thumbnail((320,180));x=col*320;y=row*210;im.paste(pic,(x,y));draw.text((x+4,y+184),f'{s["camera"]} / {rr[min(n,len(rr)-1)]["station"]:.0f}m',fill='white')
 im.save(target)
print(out)
