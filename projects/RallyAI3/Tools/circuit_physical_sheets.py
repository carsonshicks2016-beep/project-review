"""Contact sheets of completed native vehicle probes; original frames remain intact."""
import json,sys
from pathlib import Path
from PIL import Image,ImageDraw
root=Path('.rally/circuit-review');prefix=sys.argv[1];out=root/(prefix+'-physical-review');out.mkdir(exist_ok=True)
for folder in sorted(root.glob(prefix+'-probe-*')):
 summary=folder/'probe-summary.json'
 if not summary.exists():continue
 data=json.loads(summary.read_text());frames=sorted(folder.glob('drive-*.png'))
 if not frames:continue
 target=out/(folder.name.removeprefix(prefix+'-probe-')+'.jpg')
 if target.exists():continue
 im=Image.new('RGB',(1280,760),'#152229');draw=ImageDraw.Draw(im)
 for i,f in enumerate(frames[:6]):
  pic=Image.open(f);pic.thumbnail((426,240));x=(i%3)*426;y=(i//3)*270;im.paste(pic,(x,y));draw.text((x+4,y+242),f.name,fill='white')
 draw.text((8,550),folder.name,fill='white');draw.text((8,574),f'Contact samples: {data["contacts"]}',fill='white');draw.text((8,598),f'Expected: {data["expectedRegions"]}; wrong surface samples: {data["wrongSurfaceSamples"]}',fill='white')
 if data['records']:draw.text((8,622),f'Bounded outcome: {data["records"][-1]["outcome"]}; max speed {data["maxSpeed"]:.2f} m/s',fill='white')
 draw.text((8,646),'Real vehicle/suspension/tyres; diagnostic only, no valid ranked time.',fill='white');im.save(target)
print(out)
