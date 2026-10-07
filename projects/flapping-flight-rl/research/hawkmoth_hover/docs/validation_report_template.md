# Hawkmoth hover benchmark — validation report

**Status:** NOT_RUN  
**Run ID / immutable config hash:**  
**IBAMR version and commit:**  
**Date / host / compiler / MPI:**  
**Input, geometry, and executable hashes:**  

## Scope and fidelity boundary

State the paired fore/hind-wing simplification, source outline species, scaling, rigid-wing and fixed-body assumptions, flow model, and excluded effects.

## Input provenance and uncertainty

Link the parameter ledger and identify measured values, reported model values, raster-digitized values, modeling assumptions, and unresolved ambiguities.

## Numerical verification

| Check | Metric/norm | Result | Reference/expected trend | Gate |
|---|---|---:|---|---|
| Staggered solver verification | | | | |
| Incompressibility | | | | |
| Prescribed-boundary no-slip | | | | |
| Momentum balance | | | | |
| Zero-amplitude control | | | | |

## Periodicity and convergence

Include three consecutive cycle comparisons, the full 3x3 grid/time matrix, cycle-mean force/power changes for the two finest levels, phase-resolved differences, and any incomplete/stopped runs.

## Literature force comparison

State axes, sign convention, body/wing force inclusion, digitization method and uncertainty. Include overlay figures and discrepancies.

## Independent DPIV comparison

State specimen/data provenance, raw-file checksums, phase alignment, plane mapping, spatial/temporal resampling, comparison metrics and uncertainty. Explicitly retain the cross-study/non-synchronized limitation.

## Resources and outputs

| Run | Elapsed | CPU | MPI ranks | Peak RSS | Output size | Free disk before/after |
|---|---:|---:|---:|---:|---:|---:|
| | | | | | | |

List compact derived products and selected full-field snapshots. Attach exact command lines and logs.

## Gate decision

Set one status from the validation protocol. Explain every failed or unevaluated criterion. Do not infer controller performance from this CFD benchmark.
