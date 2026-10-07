"""Experimental measured-rate encoder. Not connected to the production controller."""
import math
import numpy as np
class OdorReference:
 def __init__(self,artifact,neuron_ids):
  self.artifact=artifact;self.entries=[];seen=set()
  if len(neuron_ids)!=artifact['modeled_neurons']:raise ValueError('Neuron ordering size mismatch')
  for unit in artifact['units']:
   for neuron in unit['neurons']:
    i=neuron['index']
    if str(neuron_ids[i])!=neuron['root_id'] or i in seen:raise ValueError('Invalid or duplicate exact root mapping')
    seen.add(i);self.entries.append((unit,neuron))
  self.targets=np.array([n['index'] for _,n in self.entries],dtype=np.int32)
 def rates(self,chemical=None,left=0.,right=0.):
  if any(not math.isfinite(v) or not 0<=v<=1 for v in (left,right)):raise ValueError('Exposure must be finite in [0,1]')
  values=[]
  for unit,neuron in self.entries:
   exposure=left if neuron['side']=='left' else right
   base=unit['baseline_hz']
   if chemical is None:
    if exposure:raise ValueError('Exposure requires a chemical')
    rate=base
   else:
    if chemical not in unit['responses']:raise ValueError('Missing measured response: '+chemical)
    rate=base+exposure*unit['responses'][chemical]['delta_hz']
   values.append(max(0,rate))
  return np.array(values)
