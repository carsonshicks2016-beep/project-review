import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from flygarden.body import Body
from flygarden.world import default_arena
import imageio.v3 as iio
root=Path(__file__).resolve().parents[1];a=default_arena(6);b=Body(blocks=a['blocks'],spawn=(0,0),foods=a['foods'],predator=a['predator'])
frames=b.eye_frames();readouts=b.retinal_readouts();iio.imwrite(root/'reports/left-eye.png',frames[0]);iio.imwrite(root/'reports/right-eye.png',frames[1]);r=dict(status='completed',frame_shape=list(frames.shape),readout_shape=list(readouts.shape),mean=float(readouts.mean()),mapping_to_brain='not validated; diagnostic retinal sampling only');(root/'reports/retina-diagnostic.json').write_text(json.dumps(r,indent=2));print(json.dumps(r));b.close()
