"""Exact afferent mapping and independent event-reference validation.

No native full-brain worker, no production controller mutation.
"""
import sys, json, time, hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from flygarden.olfactory_release_reference import ReleaseReference
from flygarden.recording import atomic_json
from flygarden.continuous_candidate import file_sha

OUT = ROOT / 'reports/brain-integration/recovery/olfactory-spiking-mechanism-v1'
OUT.mkdir(exist_ok=True)


def mapping():
    ids = pd.read_csv(ROOT/'vendor/fly-brain/data/2025_Completeness_783.csv', index_col=0).index.to_numpy()
    ann = pd.read_csv(ROOT/'data/annotations.tsv', sep='\t', low_memory=False).fillna('')
    assert not ann.root_id.duplicated().any()
    ann = ann.set_index('root_id').reindex(ids).fillna('')
    con = pd.read_parquet(ROOT/'vendor/fly-brain/data/2025_Connectivity_783.parquet')
    pre = con.Presynaptic_Index.to_numpy(dtype=np.int64)
    post = con.Postsynaptic_Index.to_numpy(dtype=np.int64)
    types = ann.cell_type.to_numpy()
    is_orn = ann.cell_type.str.startswith('ORN_').to_numpy()
    populations, summary, exact, loops = {}, [], [], []
    for glom in ('DM1', 'DM2'):
        orn = np.flatnonzero(types == 'ORN_'+glom)
        pn = np.flatnonzero(types == glom+'_lPN')
        for name, selected in [('ORN_'+glom, orn), (glom+'_lPN', pn)]:
            populations[name] = [{'index':int(i), 'root_id':str(ids[i]),
                                  'side':str(ann.iloc[i]['side'])} for i in selected]
        incoming = np.isin(post, pn)
        afferent = incoming & np.isin(pre, orn)
        # Positional row IDs, never the duplicated parquet labels.
        for row in np.flatnonzero(afferent):
            r = con.iloc[row]
            exact.append({'row_position':int(row), 'glomerulus':glom,
                          'pre_root_id':str(ids[pre[row]]), 'post_root_id':str(ids[post[row]]),
                          'pre_index':int(pre[row]), 'post_index':int(post[row]),
                          'anatomical_synapses':int(r['Connectivity']),
                          'signed_synapses':float(r['Excitatory x Connectivity'])})
        groups = []
        for name, mask in [('cognate_orn', afferent), ('other_orn', incoming & ~afferent & is_orn[pre]),
                           ('non_orn', incoming & ~is_orn[pre])]:
            e = con.iloc[np.flatnonzero(mask)]
            groups.append({'source':name, 'aggregated_records':len(e),
                           'anatomical_synapses':int(e.Connectivity.sum()),
                           'unique_presynaptic_neurons':int(e.Presynaptic_Index.nunique()),
                           'positive_records':int((e['Excitatory x Connectivity'] > 0).sum()),
                           'negative_records':int((e['Excitatory x Connectivity'] < 0).sum()),
                           'missing_pre_cell_type_records':int(np.sum(types[pre[mask]] == ''))})
        summary.append({'glomerulus':glom, 'input_neurons':len(orn), 'projection_neurons':len(pn), 'incoming':groups})
        outgoing = np.isin(pre, pn)
        intermediates = np.intersect1d(post[outgoing], pre[incoming])
        outgoing_rows = np.flatnonzero(outgoing)
        incoming_rows = np.flatnonzero(incoming)
        paths = []
        for i in intermediates:
            out_rows = outgoing_rows[post[outgoing_rows] == i]
            in_rows = incoming_rows[pre[incoming_rows] == i]
            # Descriptive structural loops only; no firing/gating assumption.
            paths.append({'intermediate_root_id':str(ids[i]), 'cell_type':str(types[i]),
                          'side':str(ann.iloc[i]['side']),
                          'pn_to_intermediate_row_positions':out_rows.tolist(),
                          'intermediate_to_pn_row_positions':in_rows.tolist(),
                          'out_anatomical_synapses':int(con.iloc[out_rows].Connectivity.sum()),
                          'return_anatomical_synapses':int(con.iloc[in_rows].Connectivity.sum()),
                          'out_signed_synapses':float(con.iloc[out_rows]['Excitatory x Connectivity'].sum()),
                          'return_signed_synapses':float(con.iloc[in_rows]['Excitatory x Connectivity'].sum())})
        loops.append({'glomerulus':glom,'two_hop_intermediates':len(paths),'paths':paths,
                      'interpretation':'PN-population return paths, not necessarily same-neuron cycles or causal persistence.'})
    result = {'status':'exact_root_mapping_complete', 'populations':populations, 'summary':summary,
              'afferent_edges':exact, 'afferent_records':len(exact), 'structural_return_paths':loops,
              'limitations':['Aggregated neuron-pair graph has no bouton locations or presynaptic receptor identities.',
                             'Negative whole-neuron inputs do not identify LN-mediated presynaptic inhibition.',
                             'Generic published parameters are not validated DM1/DM2 parameter estimates.',
                             'Incoming non-ORN connections are structural paths, not proven causes of persistence.'],
              'sources':{str(p.relative_to(ROOT)):file_sha(p) for p in [ROOT/'data/annotations.tsv', ROOT/'vendor/fly-brain/data/2025_Completeness_783.csv', ROOT/'vendor/fly-brain/data/2025_Connectivity_783.parquet']}}
    atomic_json(OUT/'mapping.json', result)
    return result


def validate():
    # Independent continuous-state integration between accepted events.
    rng = np.random.default_rng(11001)
    times = np.cumsum(rng.exponential(1/50, 300))
    s = ReleaseReference(); y = np.array([1., 0.]); previous = 0.; worst = 0.
    releases = []
    for t in times:
        y = solve_ivp(lambda _, z: [(1-z[0])/.1, -z[1]/.05], [previous, t], y,
                      method='DOP853', rtol=1e-11, atol=1e-13).y[:, -1]
        y[1] = .24 + .76*y[1]
        released = y[0]*y[1]; y[0] -= released
        actual = s.event(float(t)); releases.append(actual)
        worst = max(worst, abs(released-actual), float(np.max(np.abs(y-[s.x,s.u]))))
        previous = t
    assert worst < 1e-9
    # Seed and transient/RNG checkpoint continuation across a non-event boundary.
    rng = np.random.default_rng(11002); s = ReleaseReference()
    for t in times[:100]: s.event(float(t), .35, rng)
    middle = float((times[99]+times[100])/2); s.advance(middle)
    atomic_json(OUT/'reference-checkpoint.json', s.checkpoint(rng))
    r = np.random.default_rng(); restored = ReleaseReference.restore(json.loads((OUT/'reference-checkpoint.json').read_text()), r)
    a = [s.event(float(t), .35, rng) for t in times[100:]]
    b = [restored.event(float(t), .35, r) for t in times[100:]]
    assert a == b and vars(s) == vars(restored) and rng.bit_generator.state == r.bit_generator.state
    suppressed = ReleaseReference()
    assert all(suppressed.event(float(t), 0) == 0 for t in times)
    assert suppressed.x == 1 and suppressed.u == 0
    before = [s.x, s.u]; s.advance(s.time+2)
    assert s.x > .999999 and s.u < 1e-12
    # Exact event mean need not match a product-of-means closure. Quantify,
    # don't fit away, that gap with predeclared rates and independent replicas.
    comparisons = []
    for rate in (5., 20., 50., 120.):
        rng = np.random.default_rng(11100+int(rate)); n = 4096; dt = .001
        x = np.ones(n); u = np.zeros(n); total = 0.
        # Counts are genuinely Poisson; conditional event times uniform in bin.
        for step in range(5000):
            counts = rng.poisson(rate*dt, n); remaining = np.zeros(n)
            for k in range(int(counts.max())):
                selected = np.flatnonzero(counts > k)
                # k-th order statistic via sorted per-replica event arrays.
                if k == 0:
                    event_times = {int(i):np.sort(rng.uniform(0,dt,int(counts[i]))) for i in selected}
                t = np.array([event_times[int(i)][k] for i in selected])
                gap = t-remaining[selected]
                x[selected] = 1-(1-x[selected])*np.exp(-gap/.1)
                u[selected] *= np.exp(-gap/.05)
                u[selected] += .24*(1-u[selected])
                rel = x[selected]*u[selected]; x[selected] -= rel
                if step >= 2000: total += float(rel.sum())
                remaining[selected] = t
            x = 1-(1-x)*np.exp(-(dt-remaining)/.1)
            u *= np.exp(-(dt-remaining)/.05)
        um = .24*.05*rate/(1+.24*.05*rate)
        up = um+.24*(1-um); xm = 1/(1+.1*up*rate)
        observed = total/n/3
        comparisons.append({'accepted_hz':rate, 'replicas':n, 'measurement_seconds':3,
                            'event_release_per_second':observed,
                            'mean_field_release_per_second':xm*up*rate,
                            'relative_difference':(observed-xm*up*rate)/(xm*up*rate)})
    result = {'status':'event_arithmetic_and_continuation_passed', 'independent_ode_max_error':worst,
              'checkpoint_continuation_exact':True, 'zero_probability_release_zero':True,
              'recovery_after_two_seconds':{'x':s.x,'u':s.u}, 'ensemble_comparisons':comparisons,
              'mean_field_equivalence_claimed':False, 'full_network_integrated':False,
              'scope':'Event release reference only, no PN spiking or locomotion validation.'}
    atomic_json(OUT/'event-validation.json', result)
    return result


if __name__ == '__main__':
    started = time.perf_counter()
    atomic_json(OUT/'protocol.json', {'version':1,'model':'liu2021-engineered-event-reference-v1',
                'source':'https://www.frontiersin.org/journals/computational-neuroscience/articles/10.3389/fncom.2021.730431/full',
                'reference_parameters':{'U':.24,'tau_d':.1,'tau_f':.05},
                'checks':['300 event independent ODE agreement <1e-9','exact JSON checkpoint/RNG continuation',
                          'zero accepted events imply zero release','resource recovery after 2 seconds',
                          'descriptive ensemble comparison; no mean-field-equivalence threshold'],
                'production_change':False})
    m = mapping(); v = validate()
    atomic_json(OUT/'receipt.json', {'wall_seconds':time.perf_counter()-started,
                'source_hashes':{str(p.relative_to(ROOT)):file_sha(p) for p in [Path(__file__),ROOT/'flygarden/olfactory_release_reference.py',ROOT/'flygarden/brain.py',ROOT/'flygarden/continuous_candidate.py']},
                'next_gate':'Presynaptic inhibition mapping and parameter applicability remain unresolved; no full-network promotion.'})
    print(json.dumps({'mapping_summary':m['summary'], 'validation':v}, indent=2))
