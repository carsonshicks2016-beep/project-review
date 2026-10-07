import json
from pathlib import Path
import numpy as np
import pytest
from flygarden.odor_encoding import OdorReference
from scripts.diagnose_reference_odor import rates_for
ROOT=Path(__file__).resolve().parents[1]
@pytest.fixture
def encoder():
 import pandas as pd
 a=json.loads((ROOT/'reports/brain-integration/stage5-20261006/odor-reference.json').read_text())
 ids=pd.read_csv(ROOT/'vendor/fly-brain/data/2025_Completeness_783.csv',index_col=0).index.to_numpy(dtype=np.int64)
 return OdorReference(a,ids)
def index(e,unit,side):return next(i for i,(u,n) in enumerate(e.entries) if u['unit']==unit and n['side']==side)
def test_measured_units_baseline_inhibition_and_uncapped_rates(encoder):
 e=encoder;i=index(e,'Or42b','left');j=index(e,'Or22a','left');k=index(e,'Or49b','left')
 assert e.rates()[i]==9 and e.rates()[j]==4
 r=e.rates('ethyl acetate',1,0)
 assert r[i]==235 and r[j]==pytest.approx(30.308) and r[k]==pytest.approx(4.8)
 assert r[index(e,'Or42b','right')]==9
 assert e.rates('oil',1,0)[i]==74
 assert e.rates('ethyl acetate',.5,0)[i]==122
def test_invalid_and_missing_data_rejected(encoder):
 for x in (-.1,1.1,float('nan'),float('inf')):
  with pytest.raises(ValueError):encoder.rates('ethyl acetate',x,0)
 with pytest.raises(ValueError):encoder.rates('invented odor',1,0)
 with pytest.raises(ValueError):encoder.rates(None,1,0)
 u,_=encoder.entries[0];old=u['responses'].pop('oil')
 try:
  with pytest.raises(ValueError):encoder.rates('oil',1,0)
 finally:u['responses']['oil']=old
def test_schedule_returns_baseline_and_switches_side(encoder):
 e=encoder
 assert np.array_equal(rates_for('acetate_left',1.01,e),e.rates())
 assert np.array_equal(rates_for('acetate_left',.51,e),e.rates('ethyl acetate',1,0))
 assert np.array_equal(rates_for('acetate_switch',1.51,e),e.rates('ethyl acetate',0,1))
 assert not rates_for('quiet',.51,e).any()
def test_exact_mapping_required(encoder):
 ids=['incorrect']*encoder.artifact['modeled_neurons']
 with pytest.raises(ValueError):OdorReference(encoder.artifact,ids)
