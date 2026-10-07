import numpy as np
import pytest
from flygarden.controller_contract import attribute_motor, controller_contract

def test_hybrid_discloses_support_and_all_behavioral_claims_remain_unvalidated():
    hybrid=controller_contract('full',True)
    assert hybrid['kind']=='engineered_hybrid' and hybrid['learning_active']
    assert hybrid['tonic_descending_stimulation']==[.65,.65]
    assert hybrid['supplied_escape'] and not hybrid['validated_neural_navigation']
    assert not hybrid['experimental_causal_scheduler_integrated']
    assert 'flygarden/engine.py' in hybrid['sources']
    baseline=controller_contract('baseline',True)
    assert not baseline['brain_enabled'] and not baseline['learning_active']
    assert baseline['tonic_descending_stimulation']==[]

def test_attribution_preserves_old_commands_and_exposes_escape_source():
    for threat in (0,.25,.25001,1):
        for bearing in (-1,0,1):
            sensory={'threat':threat,'threat_bearing':bearing}
            raw=[.08,.13]
            decision=attribute_motor('full',sensory,raw)
            turn=.6 if bearing>0 else -.6
            expected=np.clip([1+turn,1-turn] if threat>.25 else raw,0,1.2)
            assert np.array_equal(decision['applied_motor'],expected)
            assert decision['proposed_motor']==raw
            assert decision['escape_override_active']==(threat>.25)
            assert attribute_motor('baseline',sensory,raw)['applied_motor']==raw
    with pytest.raises(ValueError):attribute_motor('full',{},[float('nan'),0])
    with pytest.raises(ValueError):controller_contract('unvalidated_new_mode')

def test_live_recordings_pin_contract_and_baseline_cannot_inherit_brain_learning(tmp_path,monkeypatch):
    from types import SimpleNamespace
    from flygarden.engine import Engine
    from flygarden.storage import Store
    monkeypatch.setenv('FLYGARDEN_MEDIA_DIR',str(tmp_path/'media'))
    engine=Engine(shared={});engine.store=Store(tmp_path/'checkpoints')
    engine.brain=SimpleNamespace(ids=np.array([11,22]),plasticity=SimpleNamespace(enabled=True))
    engine.geometry_cache={}
    engine.new_run()
    assert engine.recorder.manifest['controller']['kind']=='engineered_hybrid'
    assert engine.recorder.manifest['learning']
    engine.mode='baseline';engine.new_run()
    assert not engine.recorder.manifest['learning']
    assert not engine.recorder.manifest['controller']['brain_enabled']
    with pytest.raises(ValueError,match='hybrid'):engine.command('learning',{'enabled':True})
