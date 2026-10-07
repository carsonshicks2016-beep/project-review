"""Publish partial registration and successful exact-root geometry findings."""
from pathlib import Path
import json
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scripts.validate_local_glomerular_registration import ROOT,V1,V2,OUT,sha

BRANCH=ROOT/'reports/brain-integration/recovery/patchy-skeleton-site-audit-v1'

def main():
    primary=json.loads((OUT/'results.json').read_text());secondary=json.loads((OUT/'secondary-results.json').read_text())
    branch=json.loads((BRANCH/'results.json').read_text())
    assert primary['sources_reverified'] and secondary['sources_reverified'] and branch['sources_reverified']
    p=primary['evaluations'];s=secondary['evaluations']
    fig,axes=plt.subplots(1,3,figsize=(15,5),layout='constrained')
    for ax,values,cohort,title in [(axes[0],p,primary,'Fresh ORN → PN connections'),(axes[1],s,secondary,'Separate ORN → ORN source class')]:
        x=np.arange(2);w=.35
        ax.bar(x-w/2,[values[a]['unique_fraction']*100 for a in ['published_mirror','local_translation']],w,label='Unique enclosure')
        ax.bar(x+w/2,[values[a]['correct_unique_fraction']*100 for a in ['published_mirror','local_translation']],w,label='Correct label among enclosed')
        for center,metric in [(1-w/2,'unique_fraction'),(1+w/2,'correct_unique_fraction')]:
            value=values['local_translation'][metric]*100
            low,high=np.asarray(cohort['root_cluster_bootstrap_95_intervals'][metric])*100
            ax.errorbar(center,value,yerr=[[value-low],[high-value]],fmt='none',color='black',capsize=4)
        ax.axhline(70,color='gray',linestyle=':',alpha=.5);ax.axhline(80,color='gray',linestyle='--',alpha=.5)
        ax.set(xticks=x,xticklabels=['Original mirror','Local correction'],ylim=(0,105),title=title,ylabel='Percent')
    axes[0].legend(loc='lower left',fontsize=8)
    axes[1].text(.03,.04,'Whiskers: 95% neuron-cluster interval',transform=axes[1].transAxes,fontsize=8)
    a=pd.read_parquet(BRANCH/'skeleton-mapped-endpoints.parquet',columns=['distance_to_skeleton_nm'])
    distances=np.sort(a.distance_to_skeleton_nm.to_numpy()/1000)
    idx=np.linspace(0,len(distances)-1,2000).astype(int)
    axes[2].plot(distances[idx],(idx+1)/len(distances)*100)
    axes[2].axvline(1,color='gray',linestyle='--')
    axes[2].set(title='Exact neuron-branch alignment',xlabel='Distance to skeleton segment (µm)',ylabel='Cumulative endpoints (%)',ylim=(0,102))
    axes[2].text(.98,.12,f"{branch['within_1um_fraction']*100:.2f}% within 1 µm\n34 exact-root skeletons",transform=axes[2].transAxes,ha='right')
    fig.suptitle('Anatomical progress: improved registration, incomplete boundary validation')
    fig.savefig(OUT/'anatomical-followup.png',dpi=160);plt.close(fig)
    primary_glom=json.loads((OUT/'per-glomerulus-results.json').read_text())
    secondary_glom=json.loads((OUT/'secondary-per-glomerulus-results.json').read_text())
    pa=[r['glomerulus'] for r in primary_glom if r['candidate']=='local_translation' and r['point_gate_passed']]
    sa=[r['glomerulus'] for r in secondary_glom if r['candidate']=='local_translation' and r['point_gate_passed']]
    eligibility={'version':1,'primary_point_gates_passed':pa,'secondary_point_gates_passed':sa,
                 'both_source_class_point_gates_passed':sorted(set(pa)&set(sa)),
                 'DM1_primary':'No untouched root-pair probe available','DM1_secondary':'Coverage 68.99%, below the frozen 70% requirement',
                 'complete_bilateral_atlas_accepted':False,'full_network_adapter_ready':False,'electrical_compartments_created':0}
    (OUT/'eligibility.json').write_text(json.dumps(eligibility,indent=2)+'\n')
    report=f'''# Local registration improves labels; bilateral boundary validation remains incomplete

The frozen local correction improves opposite-side anatomical labeling, and an independent exact-root skeleton audit supports the source alignment. **This is not a validated bilateral glomerular atlas or an installed brain repair.** No controller, topology, connection count, neural dynamics, body behavior, recording or checkpoint was changed.

## Candidate and separation of fitting from testing

The candidate translates each published mirrored glomerular mesh to a neuron-balanced calibration center. Shapes, orientations, scales, names and thresholds are not fitted or retuned. Exact ORN root IDs determine a fixed 60% calibration / 40% probe split; all sites belonging to an ORN share its role. The fit uses the median of per-neuron coordinate medians rather than giving high-count connections extra votes. Native postsynaptic positions are in nanometers. Sparse calibration populations and shifts longer than the source mesh diagonal remain unavailable.

The source archive supplies **{primary['anchor_sites']:,}** cognate ORN→uniglomerular-PN sites; **{primary['calibration_sites']:,}** are in the fitting group. Parameters for **48** regions were frozen before probe outcomes were read. All **3,174** opposite-side root pairs inspected in the earlier atlas test were excluded from the fresh primary probe. No neuronal root is shared between fitting and that probe. The fit and its source hashes are retained.

The correction is an engineered registration grounded in labeled synapse coordinates. PN soma-side labels and cell-type annotations remain proxies for the intended anatomical population; the test is not independently traced boundary validation or a physiological measurement.

## Primary probe: pooled gate passes, individual coverage is limited

On **{primary['new_probe_sites']:,}** untouched ORN→PN sites from **{primary['new_probe_roots']}** ORNs:

| Same probe cohort | Unique enclosure | Correct labels among enclosed | Correct fraction of all sites |
| --- | ---: | ---: | ---: |
| Published global mirror | {p['published_mirror']['unique_fraction']*100:.2f}% | {p['published_mirror']['correct_unique_fraction']*100:.2f}% | {p['published_mirror']['correct_sample_fraction']*100:.2f}% |
| Frozen local correction | {p['local_translation']['unique_fraction']*100:.2f}% | {p['local_translation']['correct_unique_fraction']*100:.2f}% | {p['local_translation']['correct_sample_fraction']*100:.2f}% |

Neuron-cluster bootstrap 95% intervals are **81.74–85.99%** for unique enclosure and **96.46–98.28%** for label accuracy among enclosed sites. The lower bounds exceed the fixed 70% / 80% gates. The probe is a conservative, selected population of previously uninspected connection pairs, not a representative sample of all brain synapses.

Only nine regions have at least 50 fresh sites and five probe roots and pass their individual point gates: {', '.join(pa)}. Those individual checks are descriptive point gates, not per-region bootstrap-certified boundaries. DM1 has no untouched root-pair sample in this primary cohort, so this result does not validate DM1.

## Independent source-class check: coverage confidence fails

A second protocol tested the **same frozen candidate**, with no refitting, on **{secondary['probe_sites']:,}** cognate ORN→ORN sites in source neuropil AL_R, from **{secondary['probe_roots']}** presynaptic ORNs. Both endpoints' neurons belong to the original probe group. Autapses are excluded for this validation cohort only; the controller graph remains intact. No fitting ORN or prior source synapse ID is reused. The two probe cohorts can share held-out ORNs: independence here means source-class and synapse-record separation, not independent animals or entirely new neurons. Source AL_R is a coarse anatomical restriction, not a fitted mesh boundary.

| Same independent cohort | Unique enclosure | Correct labels among enclosed | Correct fraction of all sites |
| --- | ---: | ---: | ---: |
| Published global mirror | {s['published_mirror']['unique_fraction']*100:.2f}% | {s['published_mirror']['correct_unique_fraction']*100:.2f}% | {s['published_mirror']['correct_sample_fraction']*100:.2f}% |
| Frozen local correction | {s['local_translation']['unique_fraction']*100:.2f}% | {s['local_translation']['correct_unique_fraction']*100:.2f}% | {s['local_translation']['correct_sample_fraction']*100:.2f}% |

The unique-enclosure 95% interval is **69.57–76.51%**. Its lower bound is below 70%, so the secondary pooled gate **fails**, despite improved label accuracy (**91.08–95.07%** interval). No rounding, threshold relaxation or extra tuning is used to turn this into a pass.

DM1 has 158 sites from 27 presynaptic ORNs. Its corrected unique enclosure is **68.99%**, below the 70% requirement; correct labels among those enclosed reach **97.25%**. This indicates a remaining shape/coverage/overlap problem beyond a simple position error. Outside and multiply enclosed points stay unavailable. Four regions pass secondary individual point gates: {', '.join(sa)}. Only {', '.join(sorted(set(pa)&set(sa)))} passes individual point gates in both cohorts. The complete bilateral atlas is not accepted.

## Exact-root branch audit: passes geometric alignment

All **34** patchy neurons have cached full-resolution skeletons matching materialization 783, native nanometer coordinates and their recorded source SHA256. All are single connected trees with zero cycle rank. The audit projects all **{branch['endpoints_preserved']:,}** unchanged input/output endpoints onto their exact neuron skeleton segments, preserving source synapse IDs and directional coordinates.

The distance algorithm searches all potentially closer segments using conservative center/radius bounds. It is not a nearest-vertex approximation; tests include a long-segment case where nearest vertex and nearest center give the wrong answer, plus comparison against exhaustive projection.

- Median segment distance: **{branch['median_distance_nm']:.1f} nm**.
- 95th / 99th percentiles: **{branch['p95_distance_nm']:.1f} / {branch['p99_distance_nm']:.1f} nm**.
- Within 1 µm: **{branch['within_1um_fraction']*100:.2f}%**.
- Maximum: **{branch['max_distance_nm']/1000:.2f} µm**; zero endpoints beyond 10 µm.

These are geometric alignment measurements, not biological membrane-distance requirements. No distant site is dropped. Segment indices and positions are retained in the sibling `patchy-skeleton-site-audit-v1` package. Branch topology does not establish electrotonic isolation, axial resistance, membrane parameters, local GABA release, or peptide/receptor kinetics. Precomputed skeletons do not supply the required physiological fit.

## Verification, limitations and next gate

Six focused tests pass across the grouped registration and exact segment-search helpers. The secondary audit took **{secondary['wall_seconds']:.2f} s**, with peak RSS **{secondary['peak_rss_bytes']/1024**2:.1f} MiB**. The 34-root branch audit took **{branch['wall_seconds']:.2f} s**, with peak RSS **{branch['peak_rss_bytes']/1024**2:.1f} MiB**. These are analysis timings, not brain throughput. All current sources and frozen prior delivery receipts are reverified. No bulk download or recording/checkpoint deletion was needed.

A report-writing bug shadowed the archive identity with a loop label after the first fit and probe classification finished. The original worker and protocol are preserved. A separate finalizer completed saved statistics without repeating fitting/classification or changing parameters. `report-finalization-amendment.json` records source hashes and that limited repair. The corrected worker remains available for reproduction.

The next anatomical requirement is independently supported opposite-side boundary shape, particularly for DM1, with fresh prospective validation. Do not tune this candidate on these now-inspected probes and call the same data held out. A new candidate needs separate calibration and a genuinely fresh validation source or group. Keep the existing failed global mirror and this partially successful local candidate as distinct lineage records.

After boundary support is adequate, audit branch coupling and spike-to-local-activity/release/target units against the published physiological reference before freezing a separate full-network adapter. The published DC3/LN2P_c fit is still not a fit for the DM1/lLN2P_b pathway. Full-brain recovery/contrast, navigation and learning acceptance remain unpassed.

## Primary source links

- [Authors' version-783 synapse archive](https://doi.org/10.5281/zenodo.10676866), pinned and checksum-verified in the prior coordinate stage.
- [Official FlyWire mirroring documentation](https://fafbseg-py.readthedocs.io/en/latest/source/tutorials/flywire_mirror.html) and the pinned navis/flybrains sources in the previous package.
- [Authors' atlas package](https://github.com/natverse/hemibrainr), pinned in the first spatial audit. This local correction is our engineered method, not a claimed published right-side atlas.
- [Official version-783 skeleton retrieval documentation](https://fafbseg-py.readthedocs.io/en/latest/source/generated/fafbseg.flywire.get_skeletons.html); exact per-root URLs, hashes and local-only license notes are retained in the skeleton audit.
'''
    (OUT/'RESULTS.md').write_text(report)
    for package in [V1,V2]:
        receipt=json.loads((package/'delivery-receipt.json').read_text())
        for name,digest in receipt['artifacts_sha256'].items():
            if sha(package/name)!=digest:raise ValueError('Frozen prior delivery changed: '+name)
    progress=ROOT/'reports/brain-integration/recovery/progress.json';d=json.loads(progress.read_text());phase=d['phases']['4']
    phase['status']='local_registration_primary_pass_secondary_coverage_fail_exact_root_skeleton_alignment_audited'
    for f in [OUT/'RESULTS.md',OUT/'results.json',OUT/'secondary-results.json',OUT/'eligibility.json',BRANCH/'results.json']:
        name=str(f.relative_to(ROOT))
        if name not in phase['evidence']:phase['evidence'].append(name)
    d['next_action']='Resolve independently supported opposite-side glomerular boundary shape (DM1 coverage remains unpassed), then source branch coupling and physiological unit conversions before a full-network adapter.'
    d['next']=d['next_action'];d['updated']=time.time();progress.write_text(json.dumps(d,indent=2)+'\n')
    ledger=ROOT/'reports/brain-integration/acceptance-ledger.json';d=json.loads(ledger.read_text())
    for item in d['requirements']:
        if item['step']==8:
            item['status']=phase['status'];item['completion_proven']=False
            for f in [OUT/'RESULTS.md',BRANCH/'results.json']:
                name=str(f.relative_to(ROOT/'reports/brain-integration'))
                if name not in item['evidence']:item['evidence'].append(name)
    d['updated']=time.time();ledger.write_text(json.dumps(d,indent=2)+'\n')

if __name__=='__main__':main()
