"""Resume the unchanged registered protocol with the user's power permission.

Only the parent launcher's AC precondition is overridden. Child workers, model
sources, seeds, controls, dynamics, storage reserve and worker lock are unchanged.
"""
import sys,time,uuid
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import evaluate_odor_candidate as experiment
from flygarden.recording import atomic_json
from flygarden.continuous_candidate import file_sha
from flygarden.power import on_ac_power
receipt=experiment.OUT/('power-authorization-'+uuid.uuid4().hex+'.json')
atomic_json(receipt,{'time':time.time(),'user_instruction':'you are free to use maximal compute power at all times',
 'override':'AC launch precondition only','ac_at_resume':on_ac_power(),
 'launcher_sha256':file_sha(Path(__file__)),'registered_evaluator_sha256':file_sha(Path(experiment.__file__)),
 'status':'starting'})
experiment.on_ac_power=lambda:True
experiment.run()
atomic_json(receipt,{'time':time.time(),'status':'completed','override':'User-authorized AC launch precondition only',
 'launcher_sha256':file_sha(Path(__file__)),'registered_evaluator_sha256':file_sha(Path(experiment.__file__))})
