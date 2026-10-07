"""Read-only Stage 1 audit. Never changes the application, vendor, or checkpoints.

Reference parameters are extracted from the pinned source, not copied by hand.
Shared deterministic stimuli isolate equation parity from input RNG differences.
This is NOT a claim of equivalence to upstream PoissonInput experiments.
"""
import argparse
import ast
import hashlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from textwrap import dedent

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
REFERENCE = ROOT / 'vendor/fly-brain/code/run_brian2_cuda.py'


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def params():
    import brian2 as b
    tree = ast.parse(REFERENCE.read_text())
    node = next(n.value for n in tree.body if isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == 'default_params' for t in n.targets))
    return eval(compile(ast.Expression(node), str(REFERENCE), 'eval'),
                {'__builtins__': {}, 'mV': b.mV, 'ms': b.ms, 'Hz': b.Hz, 'dedent': dedent})


def inventory(folder):
    import numpy as np
    import pandas as pd
    import pyarrow.parquet as pq
    from flygarden.recording import space_check
    space_check(ROOT, 32 * 1024**2)
    folder.mkdir(parents=True, exist_ok=False)
    snapshot = folder / 'baseline'
    snapshot.mkdir()
    # Source/configuration copies plus inventories preserve identity without
    # duplicating gigabytes of immutable brain/checkpoint/recording artifacts.
    copied = {}
    for name in ('flygarden', 'scripts', 'static', 'tests'):
        for source in (ROOT / name).rglob('*'):
            if not source.is_file() or '__pycache__' in source.parts:
                continue
            relative = source.relative_to(ROOT)
            target = snapshot / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            copied[str(relative)] = sha(target)
    for name in ('README.md', 'RECORDING-GUIDE.md', 'requirements.lock',
                 'package.json', 'package-lock.json', 'reports/provenance.json',
                 'reports/neuron-mapping.json', 'Open Fly Garden.command',
                 'Close Fly Garden.command'):
        source = ROOT / name
        if source.exists():
            target = snapshot / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            copied[name] = sha(target)
    artifacts = {}
    for name in ('checkpoints', 'runs', 'exports', 'experiments'):
        artifacts[name] = {str(p.relative_to(ROOT)): sha(p)
                           for p in (ROOT / 'data' / name).rglob('*.json')}
    provenance = json.loads((ROOT / 'reports/provenance.json').read_text())
    integrity = {name: {'expected': expected, 'actual': sha(ROOT / name)}
                 for name, expected in provenance['hashes'].items()}
    ids = pd.read_csv(ROOT / 'vendor/fly-brain/data/2025_Completeness_783.csv', index_col=0).index.to_numpy(dtype=np.int64)
    columns = ['Presynaptic_ID', 'Postsynaptic_ID', 'Presynaptic_Index',
               'Postsynaptic_Index', 'Connectivity', 'Excitatory', 'Excitatory x Connectivity']
    rows = anatomical = bad_pre = bad_post = bad_sign = bad_count = bad_index = 0
    positive = negative = zero = 0
    for batch in pq.ParquetFile(ROOT / 'vendor/fly-brain/data/2025_Connectivity_783.parquet').iter_batches(batch_size=250_000, columns=columns):
        d = batch.to_pandas()
        pre = d.Presynaptic_Index.to_numpy()
        post = d.Postsynaptic_Index.to_numpy()
        valid = (pre >= 0) & (pre < len(ids)) & (post >= 0) & (post < len(ids))
        bad_index += int((~valid).sum())
        bad_pre += int((ids[pre[valid]] != d.Presynaptic_ID.to_numpy()[valid]).sum())
        bad_post += int((ids[post[valid]] != d.Postsynaptic_ID.to_numpy()[valid]).sum())
        weights = d['Excitatory x Connectivity'].to_numpy()
        bad_sign += int((weights != d.Excitatory.to_numpy() * d.Connectivity.to_numpy()).sum())
        bad_count += int((d.Connectivity <= 0).sum())
        rows += len(d)
        anatomical += int(d.Connectivity.sum())
        positive += int((weights > 0).sum()); negative += int((weights < 0).sum()); zero += int((weights == 0).sum())
    ann = pd.read_csv(ROOT / 'data/annotations.tsv', sep='\t', low_memory=False).fillna('')
    modeled = ann[ann.root_id.isin(ids)]
    mapping = json.loads((ROOT / 'reports/neuron-mapping.json').read_text())
    mapping_checks = {}
    for name, population in mapping['populations'].items():
        supplied = [int(n['root_id']) for n in population['neurons']]
        expected = set(int(n) for n in modeled.loc[modeled.cell_type.eq(population['cell_type']), 'root_id'])
        mapping_checks[name] = {'count': len(supplied), 'exact_annotation_set': set(supplied) == expected,
                               'unique': len(set(supplied)) == len(supplied),
                               'sides': modeled.loc[modeled.cell_type.eq(population['cell_type']), 'side'].value_counts().to_dict()}
    revision = subprocess.check_output(['git', '-C', str(ROOT / 'vendor/fly-brain'), 'rev-parse', 'HEAD'], text=True).strip()
    dirty = subprocess.check_output(['git', '-C', str(ROOT / 'vendor/fly-brain'), 'status', '--porcelain'], text=True).strip()
    checks = {'data_hashes': all(v['expected'] == v['actual'] for v in integrity.values()),
              'source_revision': revision == provenance['source_revisions']['fly-brain'],
              'vendor_clean': not dirty, 'unique_neuron_ids': len(np.unique(ids)) == len(ids),
              'connection_ids_and_indices': bad_index == bad_pre == bad_post == 0,
              'signed_counts': bad_sign == bad_count == 0,
              'unique_annotation_ids': not modeled.root_id.duplicated().any(),
              'mapping_sets': all(v['exact_annotation_set'] and v['unique'] for v in mapping_checks.values())}
    report = {'version': 1, 'checks': checks, 'passed': all(checks.values()), 'revision': revision,
              'reference_source_sha256': sha(REFERENCE), 'integrity': integrity,
              'neurons': len(ids), 'aggregated_connections': rows, 'anatomical_synapses': anatomical,
              'signed_edges': {'positive': positive, 'negative': negative, 'zero': zero},
              'invalid': {'indices': bad_index, 'pre_id': bad_pre, 'post_id': bad_post,
                          'sign_product': bad_sign, 'nonpositive_counts': bad_count},
              'annotations_present': len(modeled), 'annotations_missing': len(ids) - modeled.root_id.nunique(),
              'mapping_checks': mapping_checks, 'files': copied, 'artifact_metadata': artifacts,
              'preservation': 'Source/config copies; original checkpoint and raw recording files retained in place. Metadata inventory does not validate every artifact payload.',
              'profiles': {'pinned_reference': 'Original source untouched; extra reset assignment tested as an inert local temporary.',
                           'current_embodied_v1': 'Frozen copied application identity; engineered sensory/motor/plasticity adapters.',
                           'reference_equations_replayed_input': 'Diagnostic only; redundant reset assignment omitted, shared adapter input scheduling and refractory configuration.',
                           'reference_selected_refractory': 'Diagnostic only; redundant reset assignment omitted and shared replay, refractory disabled only on selected populations.'}}
    (folder / 'inventory.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({k: report[k] for k in ('passed', 'checks', 'neurons', 'aggregated_connections', 'anatomical_synapses', 'annotations_missing')}, indent=2), flush=True)
    if not report['passed']:
        raise RuntimeError('Inventory audit failed; inspect inventory.json')


def worker(folder, backend):
    import brian2 as b
    import numpy as np
    import pandas as pd
    from flygarden.brain import FullBrain
    from flygarden.recording import space_check
    b.prefs.codegen.target = 'cython'
    b.defaultclock.dt = .1 * b.ms
    p = params()
    print(f'Building {backend}', flush=True)
    if backend == 'production':
        brain = FullBrain(seed=0, learning=False)
        ids = brain.ids
        groups = brain.populations
        keys = brain.input_keys
        input_indices = brain.input_indices
        net, neu, monitor = brain.network, brain.neurons, brain.monitor
    else:
        ids = pd.read_csv(ROOT / 'vendor/fly-brain/data/2025_Completeness_783.csv', index_col=0).index.to_numpy(dtype=np.int64)
        ordering = {int(root): i for i, root in enumerate(ids)}
        ann = pd.read_csv(ROOT / 'data/annotations.tsv', sep='\t', low_memory=False).fillna('')
        definitions = [('odor_a', 'ORN_DM1', None), ('odor_b', 'ORN_DM2', None),
                       ('p9_left', 'DNp09', 'left'), ('p9_right', 'DNp09', 'right'),
                       ('loom_left', 'LPLC2', 'left'), ('loom_right', 'LPLC2', 'right'),
                       ('pam_gamma5', 'PAM01', None), ('ppl_gamma1', 'PPL101', None)]
        groups = {}
        for key, cell, side in definitions:
            mask = ann.root_id.isin(ids) & ann.cell_type.eq(cell)
            if side: mask &= ann.side.eq(side)
            groups[key] = np.array([ordering[int(root)] for root in ann.loc[mask, 'root_id']], dtype=np.int32)
        keys = tuple(groups)
        input_indices = np.concatenate(list(groups.values()))
        neu = b.NeuronGroup(len(ids), p['eqs'], threshold=p['eq_th'],
                           reset='v = v_rst; g = 0 * mV', refractory='rfc',
                           method='linear', namespace=p)
        neu.v = p['v_0']; neu.g = 0 * b.mV; neu.rfc = p['t_rfc']
        con = pd.read_parquet(ROOT / 'vendor/fly-brain/data/2025_Connectivity_783.parquet',
                              columns=['Presynaptic_Index', 'Postsynaptic_Index', 'Excitatory x Connectivity'])
        syn = b.Synapses(neu, neu, 'w : volt', on_pre='g += w', delay=p['t_dly'])
        syn.connect(i=con.Presynaptic_Index.to_numpy(dtype=np.int32), j=con.Postsynaptic_Index.to_numpy(dtype=np.int32))
        syn.w = con['Excitatory x Connectivity'].to_numpy() * p['w_syn']
        del con
        stimulus = b.SpikeGeneratorGroup(len(input_indices), np.array([], dtype=np.int32), np.array([]) * b.second)
        adapter = b.Synapses(stimulus, neu, on_pre='v_post += amplitude', namespace={'amplitude': p['w_syn'] * p['f_poi']})
        adapter.connect(i=np.arange(len(input_indices)), j=input_indices)
        monitor = b.SpikeMonitor(neu)
        net = b.Network(neu, syn, stimulus, adapter, monitor)
    net.store('initial')
    cases = {'quiet': [0]*8, 'odor_a': [1,0,0,0,0,0,0,0],
             'walking': [0,0,.65,.65,0,0,0,0], 'loom_left': [0,0,0,0,1,0,0,0],
             'combined': [.4,.2,.65,.65,.5,0,0,0]}
    results = []
    for seed in (1101, 1102):
        for case, values in cases.items():
            net.restore('initial')
            rates = np.concatenate([np.full(len(groups[k]), value*100.) for k, value in zip(keys, values)])
            if backend != 'production':
                neu.rfc = p['t_rfc']
                selected = input_indices if backend == 'reference_adapter' else input_indices[rates > 0]
                neu.rfc[selected] = 0 * b.ms
            rng = np.random.default_rng(seed)
            ticks, channels = np.nonzero(rng.random((1000, len(rates))) < rates * .0001)
            if backend == 'production':
                brain.rng = np.random.default_rng(seed)
                brain.previous_counts[:] = 0
                brain.plasticity.reset_transient()
                brain.motor[:] = 0
                brain.advance(.1, odor=values[:2], p9=values[2:4], loom=values[4:6])
                first_i, first_t = brain.last_spikes
                brain.advance(.1)
                second_i, second_t = brain.last_spikes
                indices = np.concatenate([first_i, second_i]); times = np.concatenate([first_t, second_t])
            else:
                stimulus.set_spikes(channels, ticks * .0001 * b.second, sorted=True)
                net.run(.1 * b.second)
                stimulus.set_spikes(np.array([], dtype=np.int32), np.array([])*b.second)
                net.run(.1 * b.second)
                indices = np.asarray(monitor.i[:], dtype=np.int32).copy()
                times = np.asarray(monitor.t[:]/b.second).copy()
            counts = np.asarray(monitor.count[:]).copy()
            filename = f'{backend}-{case}-{seed}.npz'
            space_check(folder, indices.nbytes + times.nbytes + counts.nbytes + len(ids)*16 + 1024**2)
            np.savez_compressed(folder / filename, i=indices, t=times, counts=counts,
                                input_channels=channels, input_ticks=ticks, input_indices=input_indices,
                                v=np.asarray(neu.v[:]/b.mV), g=np.asarray(neu.g[:]/b.mV))
            results.append({'case': case, 'seed': seed, 'spikes': len(indices),
                            'input_events': len(ticks), 'active_neurons': int((counts > 0).sum()),
                            'finite_state': bool(np.isfinite(neu.v[:]).all() and np.isfinite(neu.g[:]).all()),
                            'file': filename, 'sha256': sha(folder / filename)})
            print(f'{backend}: {case} seed {seed}, {len(indices)} spikes', flush=True)
    (folder / f'{backend}.json').write_text(json.dumps(results, indent=2))


def compare(folder):
    import numpy as np
    comparisons = []
    for row in json.loads((folder / 'production.json').read_text()):
        case, seed = row['case'], row['seed']
        with np.load(folder / row['file']) as actual, np.load(folder / f'reference_adapter-{case}-{seed}.npz') as ref, np.load(folder / f'reference_selected-{case}-{seed}.npz') as strict:
            checks = {key: bool(np.array_equal(actual[key], ref[key]))
                      for key in ('i', 't', 'counts', 'input_channels', 'input_ticks', 'input_indices')}
            checks.update(v=bool(np.allclose(actual['v'], ref['v'], rtol=0, atol=1e-10)),
                          g=bool(np.allclose(actual['g'], ref['g'], rtol=0, atol=1e-10)))
            comparisons.append({'case': case, 'seed': seed, 'equation_replay_checks': checks,
                                'equation_replay_passed': all(checks.values()),
                                'selected_refractory_counts_match': bool(np.array_equal(actual['counts'], strict['counts'])),
                                'neurons_with_different_counts': int((actual['counts'] != strict['counts']).sum()),
                                'production_spikes': int(actual['counts'].sum()), 'selected_refractory_spikes': int(strict['counts'].sum())})
    report = {'version': 1, 'equation_replay_passed': all(r['equation_replay_passed'] for r in comparisons),
              'cases': comparisons, 'scope': 'Entire imported network; 2 seeds, 5 stimuli, 100ms stimulus + 100ms recovery. Learning frozen. No body/controller changes.',
              'limitations': ['Redundant upstream reset assignment omitted in replay diagnostic; original source retained and separately tested.',
                              'Shared SpikeGeneratorGroup input replaces upstream PoissonInput for replay tests.',
                              'Exact equation replay is not equivalence to original stochastic input scheduling, long runs, natural sensory processing, or behavior.',
                              'Selected-refractory diagnostic isolates one adapter difference and is not a proposed production configuration.']}
    (folder / 'reference-comparison.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)
    if not report['equation_replay_passed']:
        raise RuntimeError('Equation replay failed; inspect reference-comparison.json')


def finalize(folder):
    """Publish source identities and a readable evidence/alteration ledger."""
    inventory = json.loads((folder/'inventory.json').read_text())
    replay = json.loads((folder/'reference-comparison.json').read_text())
    native = json.loads((folder/'native-reference.json').read_text())
    source_files = inventory['files']
    unchanged = {name: sha(ROOT/name) == digest for name, digest in source_files.items()
                 if name.startswith(('flygarden/', 'static/')) or name in
                 ('requirements.lock', 'package.json', 'package-lock.json', 'reports/provenance.json', 'reports/neuron-mapping.json')}
    if not all(unchanged.values()):
        raise RuntimeError('Application/configuration changed during audit')
    import pandas as pd
    mapping = json.loads((ROOT/'reports/neuron-mapping.json').read_text())
    ann = pd.read_csv(ROOT/'data/annotations.tsv', sep='\t', low_memory=False).fillna('').set_index('root_id')
    entry_checks = {name: all(ann.loc[int(entry['root_id']), 'side'] == entry['side']
                             and ann.loc[int(entry['root_id']), 'top_nt'] == entry['transmitter']
                             for entry in population['neurons'])
                    for name, population in mapping['populations'].items()}
    if not all(entry_checks.values()):
        raise RuntimeError('Mapping annotation fields differ from pinned table')
    profiles = {'version': 1, 'pinned_reference': {'revision': inventory['revision'],
                 'source': str(REFERENCE.relative_to(ROOT)), 'sha256': inventory['reference_source_sha256'],
                 'equations_and_reset': 'unchanged in native CPU execution checks'},
                'current_embodied_v1': {'snapshot': 'baseline', 'brain_sha256': source_files['flygarden/brain.py'],
                 'engine_sha256': source_files['flygarden/engine.py'], 'learning': 'frozen during audit'},
                'application_unchanged': unchanged, 'mapping_annotation_fields_match': entry_checks,
                'annotation_warning': 'Agreement verifies import fidelity, not biological annotation confidence or neurotransmitter correctness.',
                'tool_sha256': sha(Path(__file__)),
                'next_stage': 'Diagnose persistent activity and input/refractory coupling; no controller promotion from Stage 1.'}
    (folder/'profiles.json').write_text(json.dumps(profiles, indent=2))
    changed = [r for r in replay['cases'] if not r['selected_refractory_counts_match']]
    native_rows = '\n'.join(f"| {r['cell_type']} | {r['rate_hz']} | {r['spikes']:,} | {r['active_neurons']:,} |" for r in native['results'])
    changed_rows = '\n'.join(f"| {r['case']} | {r['seed']} | {r['neurons_with_different_counts']:,} | {r['production_spikes']:,} | {r['selected_refractory_spikes']:,} |" for r in changed)
    text = f'''# Fly Garden — Stage 1 reference audit

Stage 1 is complete for preservation, data/import integrity, short full-network equation replay, and native upstream CPU execution. It does not establish long-run stochastic equivalence, sensory control, navigation, or learning.

## Preserved baseline

Application sources, interface, scripts, configuration and provenance are copied into `baseline/`, with hashes in `inventory.json`. Original checkpoint, recording and export files remain in place. Their JSON metadata are inventoried; this is not a new full-state checkpoint or verification of every historical payload. The paused live application was not commanded or changed. `profiles.json` distinguishes the pinned upstream source, current embodied adapter, and diagnostic variants. No production parameter or dependency was changed.

Source revision: `{inventory['revision']}`. Whole imported network: {inventory['neurons']:,} neurons, {inventory['aggregated_connections']:,} aggregated connection records and {inventory['anatomical_synapses']:,} anatomical synapses. No thresholding or reduction.

## Data and mapping checks

All pinned data hashes match. Vendor checkout is clean. Every connection index maps back to its exact presynaptic and postsynaptic root ID. Counts are positive and signed weights equal connectivity multiplied by the released sign. Neuron IDs and modeled annotation IDs are unique. All listed input/output population IDs, sides and transmitter fields match the pinned annotation table.

The annotation table covers {inventory['annotations_present']:,} modeled neurons; {inventory['annotations_missing']} are missing. Import agreement does not resolve annotation confidence or validate transmitter predictions. Missing annotations block claims about those cells, not execution of the full network.

## Reference comparisons

Rest/reset voltage -52 mV; threshold -45 mV; membrane time 20 ms; synaptic time 5 ms; ordinary refractory period 2.2 ms; connection delay 1.8 ms; signed connection scale 0.275 mV; artificial input jump 68.75 mV. These match the pinned benchmark defaults and current implementation.

Ten comparisons (five stimuli × two seeds) used 100 ms stimulation followed by 100 ms recovery, frozen learning and the full network. Spike neuron IDs, spike times and counts matched exactly. Final voltage and synaptic state matched within 1e-10 mV. Shared realized Bernoulli input isolated equations and connectivity from RNG differences. Both quiet trials emitted zero spikes.

The original pinned CPU function also ran with its original equations, reset and per-neuron PoissonInput. Only explicit experiment settings were supplied (0.1 ms clock, seed 1101, 200 ms duration and stimulation rates). These native-input checks are independent realizations, not exact or statistical equivalence tests.

| Native reference stimulus | Input Hz | Total spikes | Active neurons |
| --- | ---: | ---: | ---: |
{native_rows}

## Alteration ledger

| Component | Difference from pinned reference | Interpretation |
| --- | --- | --- |
| Reset | Application omits `w = 0` | Earlier provenance called this undefined; in Brian2 2.10.1 it is an inert local temporary, not a demonstrated execution defect. Original and omitted-reset one-cell tests agree. |
| Input RNG | Application owns a checkpointable NumPy generator and replays Bernoulli events through SpikeGeneratorGroup/Synapses; reference uses PoissonInput | Both use a per-timestep Bernoulli distribution for one input, but RNG streams and input machinery differ. Exact replay does not establish native stochastic equivalence. |
| Refractory period | Application sets zero on every possible input neuron; reference selects stimulated neurons | Detectable downstream effect in odor cases; investigate before changing production. |
| Stimulation | Synthetic odor, looming, dopamine and walking channel encodings; values clipped to 0–100 Hz | Strong activation and zero refractory on selected neurons are upstream artificial-stimulation conventions, not automatically implementation bugs or natural physiology. Original paper helper defaults to 150 Hz; pinned benchmark defaults to 100 Hz. |
| Motor output | DNp09 left/right mean rate divided by 100 Hz, clipped and smoothed | Engineered decoder into supplied leg controller; not validated natural steering. |
| Walking drive | Constant 65 Hz descending stimulation in arena loop | Can dominate output; no natural sensory decision claim. |
| Vision | Egocentric geometric threat proxy stimulates LPLC2; retinal images inspection only | Actual retinal-to-brain and obstacle pathways remain unimplemented. |
| Olfaction | Two synthetic channels use averaged antenna measurements | Bilateral sensing is not preserved in brain input. |
| Threat response | Direct motor override at threat >0.25 | Engineered escape behavior can supersede neural output. |
| Body feedback | Contact/motion feed supplied gait controller | Ascending brain mapping remains unvalidated. |
| Learning | Selected KCg→MBON01/11 plasticity; rest of weights fixed | Engineered integration; existing learning evaluation failed. Frozen throughout this audit. |
| Persistence | Neural state retained between windows, portable input RNG checkpoints, bounded spike recording | Application extensions; not biological validation. |

## Measured refractory difference

Identical stimuli and equations, changing only which input neurons have zero refractory period:

| Stimulus | Seed | Neurons with differing counts | Application spikes | Selected-input refractory spikes |
| --- | ---: | ---: | ---: | ---: |
{changed_rows}

Other six comparisons matched counts. This isolates an adapter effect; it does not establish that refractory configuration explains the persistent bright brain or that the diagnostic convention should be promoted unchanged.

## Remaining boundaries and next step

These short tests do not establish timestep convergence, multi-second/long-run equivalence, natural physiological response, complete sensory processing, useful movement or learning. Long-run stimulus dose, recovery, internal excitation/inhibition and refractory diagnostics belong to Stage 2. Preserve the reference and calibrate only a separately versioned embodied profile if evidence warrants it.

Reproduce in a fresh evidence folder: `.venv-next/bin/python scripts/audit_brain_reference.py`. Workers run sequentially. Existing evidence folders are refused. Raw diagnostic events, source identities and logs are retained alongside this report. The audit respects the existing 2 GiB storage reserve and deletes no recordings or checkpoints.
'''
    (folder/'AUDIT.md').write_text(text)


def native_reference(folder):
    """Execute the pinned CPU function with its actual PoissonInput machinery.

    The upstream function and reset are unchanged. These are execution checks, not statistical
    equivalence tests: a different RNG cannot share exact realized stimuli.
    """
    import brian2 as b
    import pandas as pd
    import numpy as np
    from flygarden.recording import space_check
    b.prefs.codegen.target = 'cython'
    b.defaultclock.dt = .1*b.ms
    tree = ast.parse(REFERENCE.read_text())
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_run_trial_cpu')
    namespace = {name: getattr(b, name) for name in
                 ('NeuronGroup', 'Synapses', 'PoissonInput', 'SpikeMonitor', 'Network', 'mV', 'ms', 'Hz')}
    namespace.update(pd=pd, spike_io_enabled=lambda: True)
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(REFERENCE), 'exec'), namespace)
    ids = pd.read_csv(ROOT / 'vendor/fly-brain/data/2025_Completeness_783.csv', index_col=0).index.to_numpy(dtype=np.int64)
    order = {int(root): i for i, root in enumerate(ids)}
    ann = pd.read_csv(ROOT / 'data/annotations.tsv', sep='\t', low_memory=False)
    results = []
    for cell, rate in [('ORN_DM1', 100), ('DNp09', 65)]:
        selected = [order[int(root)] for root in ann.loc[ann.root_id.isin(ids) & ann.cell_type.eq(cell), 'root_id']]
        p = params(); p.update(t_run=.2*b.second, r_poi=rate*b.Hz)
        b.seed(1101)
        start = time.perf_counter()
        trains = namespace['_run_trial_cpu'](selected, [], [],
                   str(ROOT/'vendor/fly-brain/data/2025_Completeness_783.csv'),
                   str(ROOT/'vendor/fly-brain/data/2025_Connectivity_783.parquet'), p)
        indices = np.concatenate([np.full(len(t), int(i), dtype=np.int32) for i, t in trains.items()])
        times = np.concatenate([np.asarray(t/b.second) for t in trains.values()])
        ordering = np.lexsort((indices, times))
        target = folder / f'native-reference-{cell}.npz'
        space_check(folder, indices.nbytes + times.nbytes + 1024**2)
        np.savez_compressed(target, i=indices[ordering], t=times[ordering], selected=np.asarray(selected))
        results.append({'cell_type': cell, 'rate_hz': rate, 'seed': 1101, 'duration_seconds': .2,
                        'spikes': len(indices), 'active_neurons': len(trains), 'wall_seconds': time.perf_counter()-start,
                        'finite_timestamps': bool(np.isfinite(times).all()), 'file': target.name, 'sha256': sha(target)})
        print(f'Native upstream CPU: {cell}, {len(indices)} spikes', flush=True)
    (folder/'native-reference.json').write_text(json.dumps({'results': results,
        'alterations': ['Original equations/reset/stimulation unchanged; explicit clock 0.1ms, experiment rate/duration and seed 1101.',
                       'Original upstream CPU function executed using AST extraction to avoid GPU orchestration imports.'],
        'scope': 'Two full-network upstream stimulation execution checks. Not matched-RNG or statistical equivalence; no body or learning.'}, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', type=Path)
    parser.add_argument('--worker', choices=['production', 'reference_adapter', 'reference_selected', 'native_reference'])
    args = parser.parse_args()
    if args.worker:
        if args.worker == 'native_reference': native_reference(args.folder)
        else: worker(args.folder, args.worker)
        return
    folder = args.folder or ROOT / 'reports/brain-integration' / time.strftime('stage1-%Y%m%d-%H%M%S')
    inventory(folder)
    for backend in ('production', 'reference_adapter', 'reference_selected', 'native_reference'):
        with open(folder / f'{backend}.log', 'w') as log:
            subprocess.run([sys.executable, str(Path(__file__).resolve()), '--worker', backend, '--folder', str(folder)],
                           stdout=log, stderr=subprocess.STDOUT, check=True, cwd=ROOT)
        print(f'{backend} finished', flush=True)
    compare(folder)
    finalize(folder)
    print(f'Audit saved: {folder}', flush=True)


if __name__ == '__main__':
    main()
