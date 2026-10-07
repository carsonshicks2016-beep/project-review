import sys,pickle
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from flygarden.body import Body
from flygarden.world import default_arena
import numpy as np
arena=default_arena();b=Body(blocks=arena['blocks'],spawn=arena['spawn']);b.advance(2,[.65,.65]);s=pickle.loads(pickle.dumps(b.snapshot()));print('saved',flush=True);a=b.advance(.1,[.7,.6]);print(a['position'],flush=True);b.close();b=Body(blocks=arena['blocks'],spawn=arena['spawn']);b.restore(s);print('restored',flush=True);c=b.advance(.1,[.7,.6]);print(c['position'],np.max(np.abs(np.array(a['position'])-c['position'])),flush=True)
