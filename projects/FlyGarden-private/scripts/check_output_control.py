import sys,json
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from flygarden.brain import FullBrain,ROOT
from flygarden.body import Body
brain=FullBrain(seed=8);brain.mark_initial();results=[]
for drive in ((1.,0.),(0.,1.)):
 brain.reset_transient();brain.rng=np.random.default_rng(8);body=Body(seed=8);motors=[]
 for i in range(20):
  motor=brain.advance(.1,p9=drive);obs=body.advance(.1,motor);motors.append(motor.tolist())
 results.append({'drive':drive,'motor':motors[-1],'position':obs['position'],'heading':obs['heading'],'rates':brain.last_rates});body.close()
r={'status':'completed','trials':results,'motor_changes':not np.allclose(results[0]['motor'],results[1]['motor']),'movement_changes':not np.allclose(results[0]['position'],results[1]['position']),'scope':'Causal P9 stimulation test with full network and supplied gait; does not validate navigation or learning'};r['passed']=r['motor_changes'] and r['movement_changes'];(ROOT/'reports/output-control.json').write_text(json.dumps(r,indent=2));print(r);assert r['passed']
