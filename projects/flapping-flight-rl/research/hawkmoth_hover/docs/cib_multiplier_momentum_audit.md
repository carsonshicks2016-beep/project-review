# CIB multiplier and momentum-ledger source audit

**Date:** 2026-10-03  
**Status:** source-path, actual spread-integral, and analytic outer-boundary diagnostic checks completed; whole-domain CIB momentum discrepancy remains unresolved.  
**Gate:** Stage A remains **FAILED / UNVALIDATED**. This audit does not qualify the CIB open-sheet coupling or authorize crossflow, hover, PIV/literature comparison, or PPO.

## Question

Does the low-resolution one-step CIB matched-translation pilot's 33.1% normalized whole-domain momentum residual arise from a simple multiplier sign, time-level, spread scaling, omitted Lagrangian quadrature factor, or cell-average fluid-momentum quadrature?

## Source findings

### Multiplier time level

The CIB staggered operator evaluates the constraint system at `half_time = (current_time + new_time)/2`. It calls `setConstraintForce(L, half_time, -gamma)` before spreading the multiplier; `CIBMethod::setConstraintForce()` asserts that this data time equals its stored half-step time. `CIBMethod::postprocessIntegrateData()` reads the solved `lambda` vector after the fluid step and prints it while receiving `new_time`; the lambda dump's time header is therefore an output-step label, not proof that lambda is an endpoint-time force. The pilot's force-history row should be interpreted as the interval's midpoint CIB force paired with the old-to-new fluid momentum difference.

This makes a simple endpoint-versus-midpoint mismatch less likely as the explanation. It does not prove the discrete time-integrated equation closes, because the outer-boundary flux and stress quadrature still need verification against the solver's discretization.

### Sign and resultant meaning

The operator documents its momentum block as

```text
(C I + D L)u + Grad(P) - gamma S(lambda)
```

and the pilot uses `scale_spread_operator = 1`, `scale_interp_operator = 1`, and `normalize_spread_force = FALSE`. With `-lambda` spread on the left-hand side, the corresponding force on the fluid is `+S(lambda)` on the right-hand side. The pilot ledger's current positive-fluid-force convention (`dP/dt - lambda - outer_flux`) is consistent with that source convention.

`computeNetRigidGeneralizedForce()` sums the raw marker entries of lambda to form net translation force, and the CIB interface describes lambda as the constraint force. `LDataManager::spread()` passes those raw entries to the Eulerian spread kernel. The pinned 3-D six-point kernel forms tensor-product weights scaled by `1/(dx0 dx1 dx2)` and adds `weight * marker_force` to each Eulerian stencil value; this path does not apply a separate marker-area or marker-volume multiplier. Consequently, for full stencil support and a partition-of-unity kernel, the composite Eulerian volume integral is expected to equal the raw marker-force sum (subject to AMR composite weighting, support near boundaries, and the component-specific MAC quadrature). The source result narrowed the suspected quadrature error; the follow-up measurement below directly checks the actual side-centered field used by the saddle-point solve.

The `regulator` field is separate from force quadrature. Its default cell-volume value enters the mobility regularization term and should not be multiplied into the reported net force. The pilot's explicit settings leave `regularize_mob_factor = DELTA`; they do not change the raw `lambda` sign or spread scale.

### Actual spread capture result

The solver spreads lambda into a temporary side-centered operator output during each matrix application. The app's existing callback captures `IBHierarchyIntegrator::IB` body-force data, a distinct explicit-force channel; that callback cannot establish the CIB multiplier spread. IBAMR's optional `output_eul_lambda` is a cell-centered visualization path executed during postprocessing and uses the stored `X` data, so it is not the exact midpoint, side-centered force array applied inside `CIBStaggeredStokesOperator`. It must not be labeled as that array.

An environment-gated capture was added reversibly around `CIBMethod::spreadForce()` for one diagnostic run. It isolated the `A_U` side-centered increment, integrated it with IBAMR composite side weights, and compared it to `-d_scale_spread` times `computeNetRigidGeneralizedForce()` from the raw marker lambda. The solver equations were unchanged. Across 345 operator applications, the maximum absolute vector discrepancy was `1.9171e-9` and the maximum relative vector discrepancy was `2.2807e-8` (median relative discrepancy `4.3948e-12`; no rows exceeded `1e-6`). This verifies the spread scale, sign, and marker-resultant relation for the sampled one-level matched-flow case. It does not establish force accuracy or prove closure of the full fluid momentum balance.

A separate one-factor application diagnostic compared the existing cell-average momentum integral with a MAC side-centered integral using `HierarchyMathOps` composite side weights, which assign dual volumes and zero covered coarse sides. The case, physical inputs, solver, boundaries, lambda measurement, and boundary-flux code were unchanged. The normalized residual was `0.3309014942` with cell-average momentum and `0.3309014962` with MAC side weights; the dimensional residual vectors agree to the shown digits. Thus the cell-average momentum surrogate does not explain the 33.1% residual in this one-step result.

## Pilot result still unresolved

The retained one-step matched-translation result remains the only positive CIB control: new-time marker slip was `1.6043e-5 U_ref` RMS / `5.8989e-5 U_ref` maximum, pose error was roundoff, and the reported divergence was small. With the original boundary reconstruction, the whole-domain momentum residual was `(0.08404246, -0.00448268, 0.00138794)` code force units, norm `0.0841734`, or `0.3309015` of the summed dimensional term magnitudes. Replaying the same input, marker geometry, solver settings, and CIB executable with the corrected boundary reconstruction yielded `(0.08884705, -0.00457999, 0.00149812)`, normalized residual `0.3432944`. The small increase does not identify the cause; the result remains `DIAGNOSTIC_ONLY_NOT_PASSED`.

The custom outer-boundary terms have now been exercised in analytic no-structure flow, but their relation to the exact discrete IBAMR momentum equation remains unverified. `output_eul_lambda` and ordinary explicit-force channels cannot fill that gap.

## Analytic outer-boundary-ledger controls (2026-10-03)

A separate no-structure IBAMR application was built with the existing IBAMR 0.19.0 prefix to check the custom outer-face pressure, viscous, and advective quadrature independently of CIB. Uniform flow verifies cancellation of opposing advective fluxes. A parabolic channel profile verifies the expected pressure and wall-viscous resultants and was run at N=16, 32, and 64 spatial resolutions. The corrected face reconstruction uses a second-order one-sided pressure estimate at Neumann faces and a second-order one-sided normal derivative for tangential viscous traction at Dirichlet faces; the assumptions are explicit in `experiments/cib_open_disk/src/outer_momentum_flux.hpp`.

For the initialized analytic profile, the corrected pressure and viscous resultants equal their exact values (1 and -1 in the x direction) at all three resolutions. During the subsequent transient, the maximum residual normalized by the sum of dimensional momentum terms decreased from `1.9499%` at N=16 to `1.1265%` at N=32 and `0.5753%` at N=64. At N=32 over fixed duration 0.002, reducing dt from 0.001 to 0.0005 to 0.00025 changed this maximum from `1.1265%` to `1.0491%` to `1.0346%`; this indicates a spatially dominated residual approaching roughly a 1% floor for this short analytic case. These checks support the face-term implementation and show a refinement trend for this manufactured configuration; they do not prove a closed discrete momentum identity for the CIB solver case.

The corrected ledger was replayed on the unchanged one-step CIB matched-translation input. Its normalized residual changed from `0.3309015` to `0.3432944`, so correcting the simple boundary-face reconstruction did **not** resolve the CIB discrepancy. This is the decisive limitation of the analytic check: internally sensible outer-face values on a smooth no-structure flow do not yet account for the discrete CIB balance. Stage A remains **FAILED / UNVALIDATED**. The full study, including initial parser/diagnostic failures, remains preserved at `results/cib_boundary_ledger_analytic_local_20261003_v1/`; the pre-correction header is `outer_momentum_flux_pre_boundary_reconstruction.hpp`. The corrected matched-translation run record is in `matched_translation_boundary_v2/` and reports one MPI rank, 24.60 s elapsed, 0.707 GiB sampled peak RSS, and 55.279 GiB free disk afterward.

## Next diagnostic, before any crossflow

1. Preserve the completed actual-operator capture and same-input MAC-weight quadrature comparison under `results/cib_operator_spread_capture_local_20261003_v1/`; do not replace or rewrite the original pilot.
2. Use the retained no-structure uniform and Poiseuille checks as boundary-quadrature controls, then compare the corrected outer ledger against the actual discrete staggered momentum equation and face/ghost data on the same CIB step. The remaining question is not simply whether the analytic traction integrals are right; it is why those terms do not close with the captured CIB multiplier and measured fluid-momentum increment.
3. Do not run stationary-relative-flow CIB crossflow until that discrepancy is explained or bounded by a prospectively declared discrete error budget. Then, and only then, run the preregistered short discriminator with unchanged force, no-slip, divergence, and resource measurements.
4. Do not use the one-step null, operator-spread identity, or one momentum quadrature comparison as a Stage A pass. Keep the existing limits, thresholds, geometry, and scientific boundaries unchanged.

Preserve the original pilot unchanged. Do not tune thresholds, marker weights, geometry, or solver scales from the moth case. A resolved source identity alone is not a Stage A pass; the prescribed refinement controls and all existing gates remain required.

## Identities inspected

All IBAMR files below are from the pinned unpacked IBAMR 0.19.0 source tree at `/Users/REVIEW_USER/Applications/ibamr-iim-research/tmp/unpack/IBAMR-0.19.0` and were read without modification.

| Source | SHA-256 |
|---|---|
| `src/IB/CIBStaggeredStokesOperator.cpp` | `63164a2dc6f92bba853fa2ca215006fa22b04f3489da1ab47a16836e4561774b` |
| `src/IB/CIBMethod.cpp` | `3eb1df8799611c60ba1466688041a029e4dbc52d28d580c20635b6caf139c73c` |
| `src/IB/CIBSaddlePointSolver.cpp` | `081cea4d31b4910b266d27bbfa360839254516f9ebeeb6e29ff681b077c82c25` |
| `ibtk/src/lagrangian/LDataManager.cpp` | `5723d37e0a7218d440ff70cb97c8adeb4d368d1b407cd8d48082e7fef484bf11` |
| `ibtk/src/lagrangian/LEInteractor.cpp` | `a7b7aaef9a53bcb35f8939344e954fa4c54c41480ac4335d603282635f38353f` |
| `ibtk/src/lagrangian/fortran/lagrangian_interaction3d.f.m4` | `51f0deed9a9ab8db2703c70709001467bf8dd1ddba936501275f72c9506eece9` |

The inspected CIB pilot application SHA-256 is `afa4822e349b2e9341f7a5a1c178298da8b8df47adc5c5ea9024851585a941c1`; the one-step input SHA-256 is `1ac076bb2bfdf1f57b65c3dbc6fd8a7bc8c838abc594363ec561c215122e367b`. Machine-readable run evidence remains under `results/cib_method_pilot_local_20261003_v1/`.
