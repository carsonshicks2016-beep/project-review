"""Publish the anatomical audit with explicit unpassed integration gates."""
from pathlib import Path
import hashlib
import json
import shutil
import time

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'reports/brain-integration/recovery'
OLD=BASE/'patchy-compartment-mapping-v1'
OUT=BASE/'patchy-compartment-mapping-v2'

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for part in iter(lambda:f.read(8*1024**2),b''):h.update(part)
    return h.hexdigest()

def main():
    syn=json.loads((OUT/'synapse-results.json').read_text())
    mirror=json.loads((OUT/'mirror-results.json').read_text())
    mapping=json.loads((OUT/'mapping-results.json').read_text())
    conservation=json.loads((OUT/'endpoint-conservation.json').read_text())
    if not conservation['directional_counts_exact'] or not all(r['exact'] for r in conservation['per_root_direction']):
        raise ValueError('All root/direction counts must be conserved')
    if not syn['every_imported_pair_exact'] or not syn['sources_reverified'] or not mapping['sources_reverified']:
        raise ValueError('Source and pair-count gates must pass')
    frozen=json.loads((OLD/'protocol.json').read_text())['source_hashes']
    unchanged={p:sha(ROOT/p)==h for p,h in frozen.items()}
    receipt=json.loads((OLD/'delivery-receipt.json').read_text())
    prior_receipt={p:sha(OLD/p)==h for p,h in receipt['artifacts_sha256'].items()}
    if not all(unchanged.values()) or not all(prior_receipt.values()):
        raise ValueError('Frozen baseline or v1 package changed')
    index=json.loads((OLD/'root-to-model-index.json').read_text())
    summaries=json.loads((OUT/'endpoint-root-summary.json').read_text())
    candidates=[]
    for neuron in index:
        rid=neuron['root_id']
        candidates.append({'root_id':rid,'model_index':neuron['model_index'],'cell_type':neuron['cell_type'],
                           'soma_side':neuron['side'],'directions':[a for a in summaries if a['root_id']==rid],
                           'electrical_compartments':None,'coupling':None,'release_parameters':None})
    (OUT/'anatomical-candidates.json').write_text(json.dumps({'version':2,'materialization':783,
        'status':'complete_modeled_site_coordinates_partial_native_atlas_candidates',
        'roots':candidates,'synapse_source':'selected-synapses.parquet','endpoint_source':'mapped-endpoints.parquet',
        'bilateral_glomerular_atlas_validated':False,'physiological_compartments_created':0,
        'controller_ready':False,'application_changed':False},indent=2)+'\n')
    storage=sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file())
    right=mirror['anchor_side_statistics']['right']
    text=f'''# Complete site coordinates; opposite-side atlas rejected

The read-only anatomical stage is complete. Every imported pair touching the 34 patchy roots is reconciled to exact released synapse IDs and distinct release/reception coordinates. A bilateral glomerular atlas and a full-network physiological adapter remain unvalidated. The application, brain/controller source, graph, original recordings, and checkpoints are unchanged.

## Exact source/site audit

The complete authors' version-783 archive contains **{syn['archive_rows_scanned']:,}** rows. Its **9,492,998,242-byte** source passes the published MD5 (`f8f1b97c9d4b0ea9b4c8b287f6b99091`) and a retained SHA256. Source identity is rechecked before and after extraction. The archive is already filtered by its authors; "unthresholded" here means no minimum connection-count cutoff, not unfiltered cleft predictions. No additional score threshold, autapse removal, count renormalization, or graph replacement is applied.

All **{syn['imported_pairs']:,}** modeled pairs and **{syn['imported_synapses']:,}** anatomical sites reconcile exactly. The prior export lacked **42,645** low-count-pair sites; the complete archive now accounts for those sites. There are **{syn['deficient_pairs']}** deficient and **{syn['excess_pairs']}** excess modeled pairs. IDs never pass through floating point. Synapse IDs are unique within the selected source subset, and raw source row positions are retained.

The archive additionally contains **{syn['unmodeled_partner_sites']:,}** sites on **{syn['unmodeled_partner_pairs']:,}** pairs absent from the imported controller graph. They remain in the selected-source recording and pair ledger, separately identified. They are not imported into the brain or used to fill missing modeled counts.

Coordinates are native nanometers, per the deposit metadata. Presynaptic release and postsynaptic reception coordinates remain distinct. An input endpoint uses its post position; an output endpoint uses its pre position. Selected-to-selected synapses contribute two directional endpoints but remain one anatomical synapse record. Source neuropil labels are retained independently of glomerular mesh labels.

## Opposite-side registration: failed

The pinned navis/flybrains method uses the FlyWire bounding-box x flip followed by its published non-rigid TPS. All **{mirror['landmarks']:,}** source landmarks interpolate with maximum residual **{mirror['landmark_max_error_nm']:.3g} nm**. All 58 transformed meshes are closed and consistently wound; self-intersections were not separately tested. The existing fixed sensory-anchor reservoir was not used to fit or tune the transform.

For **{right['samples']:,}** opposite-side anchors, **{right['unique_fraction']*100:.2f}%** are uniquely enclosed, but only **{right['correct_unique_fraction']*100:.2f}%** of those receive the correct glomerulus label. The prespecified gates were at least 70% unique enclosure and 80% correct labels among uniquely enclosed anchors. The second gate fails. DM1 also fails. Per-glomerulus results and the DM1 confusion table are preserved; no alias, local translation or relabeling is applied to hide the failure.

The published whole-brain registration's availability is not proof of glomerulus-scale accuracy for these atlas boundaries. The mirrored geometry remains a diagnostic artifact and is excluded from candidate/controller assignment.

## Accepted partial anatomy

The validated native-side atlas is applied to the complete modeled sites using their correct directional positions. It produces **{mapping['directional_endpoints']:,}** input/output endpoints: **{mapping['unique_enclosed_endpoints']:,}** unique glomerular candidates, **{mapping['outside_endpoints']:,}** outside the available atlas, and **{mapping['overlap_endpoints']:,}** overlapping/unavailable endpoints. Endpoint totals are not interchangeable with the one-per-synapse site total above. Outside does not mean outside the brain.

Exact-root/model-index candidates and per-direction counts are saved in `anatomical-candidates.json` and `endpoint-root-summary.json`. They are anatomical groupings only: electrical compartments, branch coupling, release units and receptor kinetics remain null. A mesh enclosure or partner label does not establish electrotonic isolation.

## Verification and resources

Eight focused tests pass: exact large IDs, both selected directions, real Arrow chunk boundaries, unmodeled-partner separation, distinct autapse endpoints, nonfinite rejection, native-flip ordering, and published TPS interpolation/batch parity. All **68** per-root input/output totals also match the imported graph exactly. The **{conservation['internal_selected_synapses']:,}** selected-to-selected synapses explain the extra directional endpoints. The two overlapping endpoints remain unavailable and are retained in `overlap-endpoints.csv`. The registration figure has been visually inspected. The unchanged baseline source hashes and the previous delivered v1 artifact hashes are reverified.

The streamed synapse audit took **{syn['wall_seconds']:.2f} s**, with peak RSS **{syn['peak_rss_bytes']/1024**2:.1f} MiB**. Mirror validation took **{mirror['wall_seconds']:.2f} s**; native endpoint mapping took **{mapping['wall_seconds']:.2f} s**. These are anatomical analysis timings, not full-brain throughput. The package currently occupies about **{storage/1024**3:.2f} GiB**, including retained verified download ranges and the assembled archive; no recording/checkpoint was removed. Free space is **{shutil.disk_usage(OUT).free/1024**3:.1f} GiB**, above the 2-GiB reserve. Reproduction scripts and optional dependency versions are retained.

## Next gate

Obtain independently supported opposite-side glomerular boundaries or prospectively validate a sourced local registration. Then audit exact-root branch topology and specify physiological unit/target conversions before a separate full-network adapter. The prior published DC3/LN2P_c rate fit is not a fit for these DM1/lLN2P_b roots. See `NEXT_REQUIREMENTS.md`.

This stage recovers missing anatomy. It does **not** repair neural persistence, establish useful navigation or learning, or complete Fly Garden.

## Primary sources and reproduction

- [Authors' version-783 synapse deposit](https://doi.org/10.5281/zenodo.10676866), CC BY 4.0, with exact file identity in `archive-identity.json`.
- [Official FlyWire mirroring documentation](https://fafbseg-py.readthedocs.io/en/latest/source/tutorials/flywire_mirror.html); navis revision `cb9a5915b6b3587cb81154f4f77ffc62fe12b03a`, flybrains revision `273333c8d8bf5adeebebd274e554621462e388bd`.
- [Authors' atlas package](https://github.com/natverse/hemibrainr), pinned in the frozen v1 provenance. Mirror source code/package licenses are recorded as GPL-3.0-or-later; no blanket claim about independent data redistribution rights is made.

From the project root, use `.venv-next/bin/python` with `PYTHONPATH=.`. Run `scripts/acquire_unthresholded_783.py`, `scripts/validate_anatomical_mirror.py`, `scripts/audit_unthresholded_783.py`, `scripts/map_verified_783_sites.py`, and `scripts/finish_anatomical_stage_v2.py` in that order. Acquisition reuses verified ranges. The audit worker can wait for verified archive publication automatically. These scripts do not launch a neural simulation or alter application state.
'''
    (OUT/'RESULTS.md').write_text(text)
    verification={'version':1,'baseline_sources_unchanged':unchanged,'prior_v1_artifacts_unchanged':prior_receipt}
    (OUT/'baseline-integrity.json').write_text(json.dumps(verification,indent=2)+'\n')
    files=['RESULTS.md','source-provenance.json','synapse-protocol.json','synapse-results.json','mirror-protocol.json',
           'mirror-results.json','mapping-protocol.json','mapping-results.json','pair-count-audit.csv',
           'anatomical-candidates.json','mapped-endpoints.parquet','selected-synapses.parquet',
           'archive-identity.json','endpoint-conservation.json','overlap-endpoints.csv','endpoint-root-summary.json',
           'baseline-integrity.json','tests.log','mirror-validation.png']
    delivery={'version':2,'status':'complete_site_audit_partial_anatomy_mirror_rejected','tests_passed':8,
              'registration_plot_visually_inspected':True,
              'exact_root_joins':34,'every_imported_pair_exact':True,'full_modeled_coordinate_coverage':True,
              'bilateral_atlas_coverage':False,'physiological_adapter_ready':False,'full_controller_ready':False,
              'application_changed':False,'promoted':False,'full_goal_complete':False,
              'sources_reverified':True,'storage_bytes':storage,'artifacts_sha256':{p:sha(OUT/p) for p in files}}
    (OUT/'delivery-receipt.json').write_text(json.dumps(delivery,indent=2)+'\n')
    progress=BASE/'progress.json';d=json.loads(progress.read_text());phase=d['phases']['4']
    phase['status']='complete_patchy_site_coordinates_audited_opposite_atlas_failed_physiological_adapter_pending'
    for name in ['RESULTS.md','synapse-results.json','mapping-results.json','delivery-receipt.json']:
        p=str((OUT/name).relative_to(ROOT))
        if p not in phase['evidence']:phase['evidence'].append(p)
    d['next_action']='Resolve opposite-side glomerular registration and exact-root branch/physiology mappings before freezing a full-network graded adapter.'
    d['next']=d['next_action'];d['updated']=time.time();progress.write_text(json.dumps(d,indent=2)+'\n')
    ledger=ROOT/'reports/brain-integration/acceptance-ledger.json';d=json.loads(ledger.read_text())
    for step in d['requirements']:
        if step['step']==8:
            step['status']='complete_patchy_site_coordinates_partial_native_anatomy_full_adapter_pending'
            step['completion_proven']=False
            for name in ['RESULTS.md','delivery-receipt.json']:
                p=str((OUT/name).relative_to(ROOT/'reports/brain-integration'))
                if p not in step['evidence']:step['evidence'].append(p)
    d['updated']=time.time();ledger.write_text(json.dumps(d,indent=2)+'\n')
    print(json.dumps(delivery,indent=2))

if __name__=='__main__':main()
