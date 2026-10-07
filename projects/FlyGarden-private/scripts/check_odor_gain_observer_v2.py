"""Fresh unprobed candidate versus retained full-network probed diagnostic."""
import sys,json,fcntl,time,resource
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from flygarden.continuous_candidate import ContinuousCandidate,file_sha
from flygarden.odor_gain_candidate import OdorGainCandidate
from flygarden.recording import atomic_json
OUT=ROOT/'reports/brain-integration/recovery/odor-gain-calibration-v2'
with (ROOT/'.runtime/experiment.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 protocol=json.loads((OUT/'diagnostic-protocol.json').read_text());assert all(file_sha(ROOT/n)==h for n,h in protocol['sources'].items())
 folder=OUT/'diagnostics/a_both-support65-candidate-gain1-9901';m=json.loads((folder/'manifest.json').read_text());assert m['status']=='complete';rows=json.loads((folder/'rows.json').read_text());cand=ContinuousCandidate(protocol['mapping'],9901,OdorGainCandidate(65,1.));started=time.monotonic()
 for k,(chunk,row) in enumerate(zip(m['chunks'],rows)):
  odors=np.ones((2,2));odors[:,1]=0
  if not 12<=k<32:odors[:]=0
  command=cand.advance(.025,odors,[0,0])
  with np.load(folder/chunk['file'],allow_pickle=False) as z:
   assert np.array_equal(cand.last_spikes[0],z['spike_i']) and np.array_equal(cand.last_spikes[1],z['spike_t'])
   assert np.array_equal(cand.last['spike_counts'],z['counts'])
   assert np.array_equal(cand.last['external_indices'],z['external_i']) and np.array_equal(cand.last['external_times'],z['external_t'])
  assert cand.last['population_hz']==row['population_hz'] and np.array_equal(command,row['candidate_motor'])
 atomic_json(OUT/'observer-parity.json',{'status':'passed','windows':60,'all_neuron_spike_ids_times_counts_exact':True,'inputs_populations_and_commands_exact':True,'probed_manifest_sha256':file_sha(folder/'manifest.json'),'checker_sha256':file_sha(Path(__file__)),'wall_seconds':time.monotonic()-started,'scope':'One fresh full-network candidate replay without observers matches its source-pinned probed trial; integration parity, not behavioral success.'})
 print('Unprobed/probed candidate matches all60 full-network windows',flush=True)
