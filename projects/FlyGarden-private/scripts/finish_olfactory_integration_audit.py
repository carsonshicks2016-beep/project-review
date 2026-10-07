"""Publish evidence and retain unmet integration gates."""
import sys, json, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from flygarden.recording import atomic_json
from flygarden.continuous_candidate import file_sha
OUT=ROOT/'reports/brain-integration/recovery/olfactory-spiking-mechanism-v1'
m=json.loads((OUT/'mapping.json').read_text())
v=json.loads((OUT/'event-validation.json').read_text())
b=json.loads((OUT/'brian-validation.json').read_text())
assert v['status']=='event_arithmetic_and_continuation_passed' and b['status']=='passed'
rows=[]
for item,loops in zip(m['summary'],m['structural_return_paths']):
    a,other,non=item['incoming']
    rows.append(f"| {item['glomerulus']} | {item['input_neurons']} | {item['projection_neurons']} | {a['aggregated_records']} / {a['anatomical_synapses']} | {non['aggregated_records']} / {non['anatomical_synapses']} | {loops['two_hop_intermediates']} |")
(OUT/'RESULTS.md').write_text('''# Olfactory release integration audit

Completed an isolated event-release implementation, native Brian2 validation,
and exact-root anatomical mapping. No full-brain or application controller was changed.

## Mapping

| Channel | ORNs | PNs | Cognate afferent records / anatomical synapses | Non-ORN incoming records / anatomical synapses | Two-hop PN return intermediates |
| --- | ---: | ---: | ---: | ---: | ---: |
'''+ '\n'.join(rows)+f'''

The 351 exact cognate ORN–PN records represent 10,426 anatomical synapses;
they are not 351 individual biological synapses. Root IDs, positional source-row
identifiers, signs and all selected edges are retained in mapping.json. Additional
other-ORN inputs are included in the mapping summary. Two DM1 incoming non-ORN
records lack cell-type annotations. Return intermediates include paths between
members of a PN population; their existence does not prove a same-cell cycle,
positive feedback, activity, or the cause of persistent firing.

## Event checks

The event interpretation decays facilitation and replenishes available resources
between events, increases release probability at an accepted event, releases
u*x, and depletes x by the same amount. It uses generic Liu et al. 2021 parameters
U=.24, tauD=.1 s and tauF=.05 s. These are not established DM1/DM2 estimates.

- Independent event-by-event ODE calculation: maximum error {v['independent_ode_max_error']:.3g}.
- Native Brian2 event-driven synapse versus analytic reference: maximum cumulative-release error {b['max_cumulative_release_error']:.3g}, across 300 events on a 100 μs clock.
- JSON transient-state/owned-RNG continuation and native Brian2 store/restore: exact for their respective isolated tests.
- Zero accepted events produce zero release; resources recover during silence.
- Invalid event times/probabilities and incompatible checkpoint identity are rejected; two focused tests pass.

Presynaptic inhibition is represented only by an **externally supplied event
acceptance probability** in the analytic reference. It is an engineered thinning
interpretation of effective rate pR. It has not been wired to anatomical LNs;
the Brian2 comparison deliberately tests STP alone (p=1).

An ensemble of 4,096 independent Poisson event synapses at each of 5, 20, 50 and
120 Hz differs from the published product-of-means steady expression by about
0.9–4.5% in these finite measurements. This is a descriptive discrepancy, not
an equivalence pass or a precision estimate. The event process has correlations
and noise absent from the mean-field closure. No parameters were tuned to remove
that gap. This package does not reproduce PN spike responses or paper Figures4/5;
the preceding mean-field reference package covers its specified figure checks.

## Integration decision

The existing aggregated connectome and annotations identify cognate afferent
neurons but do not establish bouton locations, presynaptic receptor effects,
LN-to-terminal efficacy, or a DM1/DM2-specific physiological parameter set.
Whole-neuron negative graph weights cannot be relabeled as terminal inhibition.
The published conductance-to-rate coefficients also are not LIF voltage weights.

Earlier full-network pulse trials showed quiet stimulated DM1 ORNs while PNs and
descending neurons continued firing. Odor-afferent adaptation could change the
onset, but cannot be assumed to cure downstream persistence. Keep this diagnosis
visible and keep navigation and learning gates failed/pending.

Next: preregister an explicitly engineered **STP-only afferent model hypothesis**,
with exact selected-edge identity, fixed first-release scaling convention,
unchanged non-selected edges/decoder, persistent per-edge state and delay queues,
and separate full checkpoint tests. Compare original versus STP-only pulse/no-odor
controls before any physical navigation evaluation. A failed recovery is a result,
not a reason to silently extend plasticity to other edges. Full published PI
integration requires a separately supported terminal/LN mapping and parameter
convention; those biological claims remain blocked by missing evidence.

## Source

[Liu et al. 2021, official article](https://www.frontiersin.org/journals/computational-neuroscience/articles/10.3389/fncom.2021.730431/full).
The cached, hashed article and prior selected mean-field reproduction are in
../olfactory-dynamics-reference-v1/. This package is an independent implementation,
not the author's simulator. Brian2 uses NumPy code generation for this bounded
one-synapse validation. No reduction has been substituted into the application.
''')
atomic_json(OUT/'acceptance.json',{'status':'focused_audit_complete',
    'exact_afferent_mapping':True,'independent_event_arithmetic':True,
    'native_brian_event_arithmetic':True,'isolated_checkpoint_continuation':True,
    'presynaptic_inhibition_anatomical_mapping':False,'dm1_dm2_parameter_validation':False,
    'full_network_integrated':False,'navigation_validated':False,'learning_validated':False})
receipt=json.loads((OUT/'receipt.json').read_text())
receipt['final_source_hashes']={str(p.relative_to(ROOT)):file_sha(p) for p in [ROOT/'scripts/audit_olfactory_integration.py',ROOT/'scripts/validate_olfactory_brian_events.py',ROOT/'scripts/finish_olfactory_integration_audit.py',ROOT/'flygarden/olfactory_release_reference.py',ROOT/'tests/test_olfactory_release_reference.py']}
receipt['artifacts']={p.name:file_sha(p) for p in OUT.iterdir() if p.is_file() and p.name!='receipt.json'}
for name,original in receipt['source_hashes'].items():
    if name in ('flygarden/brain.py','flygarden/continuous_candidate.py'):
        assert file_sha(ROOT/name)==original
receipt['production_sources_unchanged']=True
atomic_json(OUT/'receipt.json',receipt)
progress_path=ROOT/'reports/brain-integration/recovery/progress.json'
progress=json.loads(progress_path.read_text()); phase=progress['phases']['4']
phase['status']='exact_afferent_mapping_and_isolated_spiking_release_verified_full_network_hypothesis_pending'
for filename in ('RESULTS.md','acceptance.json','mapping.json','event-validation.json','brian-validation.json'):
    evidence=str((OUT/filename).relative_to(ROOT))
    if evidence not in phase['evidence']:phase['evidence'].append(evidence)
progress['next_action']='Preregister an explicit STP-only exact-afferent full-network hypothesis; preserve original model and test persistent edge state/delays before pulse recovery comparisons. Full PI physiology mapping remains unsupported.'
progress['next']=progress['next_action'];progress['updated']=time.time()
atomic_json(progress_path,progress)
ledger_path=ROOT/'reports/brain-integration/acceptance-ledger.json'
ledger=json.loads(ledger_path.read_text())
ledger['updated']=time.time()
for item in ledger['requirements']:
    if item['step']==8:
        item['status']='isolated_spiking_release_and_exact_afferent_mapping_verified_full_network_hypothesis_pending'
        evidence=str((OUT/'RESULTS.md').relative_to(ROOT/'reports/brain-integration'))
        if evidence not in item['evidence']:item['evidence'].append(evidence)
        item['completion_proven']=False
atomic_json(ledger_path,ledger)
print('Focused integration audit published; full-network gates remain unpassed.')
