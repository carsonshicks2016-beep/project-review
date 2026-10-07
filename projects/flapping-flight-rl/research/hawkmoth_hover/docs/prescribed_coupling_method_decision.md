# Prescribed-wing coupling route decision

**Date:** 2026-10-03  
**Decision status:** CIB is the leading *pilot candidate*, not yet selected as the production method.  
**Project status:** Stage A remains **FAILED / UNVALIDATED**. No hawkmoth case, force comparison, PIV comparison, or PPO result is authorized by this decision.

## Decision question

What is the smallest credible route to impose a prescribed rigid wing velocity and obtain the corresponding fluid/wing constraint force, after the open-sheet DG-IIM callback was shown not to impose no-slip?

## Recommendation

Screen IBAMR 0.19.0's `CIBMethod` with `CIBStaggeredStokesSolver` on an isolated three-dimensional analytic control, then on the already-recorded open-disk marker geometry. The CIB solver is the best available native candidate found in the pinned build because its coupled saddle-point system includes an unknown Lagrange multiplier that enforces rigid-body constraints while solving for fluid velocity and pressure. The method supports prescribed rigid-body translational and angular velocities evaluated at current, midpoint, and new time levels. Rigid prescribed wings match that kinematic class.

This is a **go for a bounded method pilot only**. It is **not** a go for the hover benchmark, and it does not rehabilitate Stage A. The open-sheet use remains a research question: the pinned CIB examples demonstrate 2-D fully prescribed markers and 3-D rigid marker bodies, but none found demonstrates a fully prescribed 3-D open thin sheet with converged no-slip, force, and edge behavior.

## Pinned-source evidence

The audit used the unpacked IBAMR 0.19.0 source at `/Users/REVIEW_USER/Applications/ibamr-iim-research/tmp/unpack/IBAMR-0.19.0/` and the installed 3-D shared library at `/Users/REVIEW_USER/Applications/ibamr-iim-research/packages/IBAMR-0.19.0/lib/libIBAMR3d.dylib`.

1. `CIBStaggeredStokesSolver.h` states that it solves for the constraint Lagrange multiplier, fluid velocity and pressure, and free rigid-body velocities. In `CIBStaggeredStokesSolver.cpp::solveSystem()`, IBAMR obtains the constraint-force vector, nests it with the fluid and rigid-velocity unknowns, and invokes the saddle-point solve. This differs fundamentally from the current IIM callback, which prescribes a force used to construct jumps rather than solving an unknown traction to satisfy a velocity constraint.
2. `CIBMethod` is explicitly the rigid-body constraint implementation using standard IB markers. It exposes constrained COM translational and angular velocities. In `CIBMethod.cpp`, the callback is evaluated at the current, half-step, and new times and the prescribed components are installed in the rigid-body state.
3. The pinned `examples/CIB/ex0` README and example configure a fully prescribed body (all rigid DOF solve flags zero and a constrained-velocity callback), but that example is 2-D. `examples/CIB/ex1` and `ex4` provide 3-D rigid-marker cases; the 3-D examples do not, as checked here, supply a reference output for a fully prescribed open sheet. An explicit 3-D fully prescribed control must therefore be run before relying on that combination.
4. The installed `libIBAMR3d.dylib` exports `CIBMethod` and `CIBStaggeredStokesSolver` symbols (library SHA-256 `ad92cffa2c1450ee20eb392aac4874dfdb1fc6db2102f8f48f03dd2e19e40cc9`). The candidate is present in this local binary; this verifies availability, not correctness or runtime.
5. CIB's regularization field defaults to the Eulerian cell volume for each marker. If `weight_filenames` are supplied, source code sets the field to cell volume divided by the file's per-marker weight. That is part of the discrete marker coupling; it is not a measured wing thickness or anatomical volume. The resolution dependence and appropriate marker weights for the disk and wing must be recorded and tested rather than silently interpreted as geometry.

### Source identity hashes

These hashes pin the audited implementation files in the unpacked source tree:

| File | SHA-256 |
|---|---|
| `include/ibamr/CIBMethod.h` | `0abd213cfd55f1e248f9a037e7ea96a4dc3ca0830fb3cc6bb057f70914f3b1e7` |
| `src/IB/CIBMethod.cpp` | `3eb1df8799611c60ba1466688041a029e4dbc52d28d580c20635b6caf139c73c` |
| `include/ibamr/CIBStaggeredStokesSolver.h` | `08b45b37a7bcee889095767e1ca3b275b290619cf309dc3e2cf8de0d2dede7df` |
| `src/IB/CIBStaggeredStokesSolver.cpp` | `4bbe4be445bc8551419b72b66d846b00bf1b49cebb0fc2d7adc76fa280099a7b` |
| `examples/CIB/ex0/example.cpp` | `289fa2998a14738f3f461886a12d62755923ee80e99a4a9f066fbeb37fcbc413` |
| `examples/CIB/ex4/example.cpp` | `78ec3c8bb15c23e277bf49114e7f913a2fc8c06e28c291d8e7fac80fdf974d3c` |
| `src/IB/IBFEDirectForcingKinematics.cpp` | `6fc68d3d40b2dea7202c8c927ebb3e8e8ab4d7f9ff1644cca1554e972064d90f` |

The adjacent `ConstraintIBMethod` is a different implementation from `CIBMethod`. Its source derives a per-node volume by counting occupied Eulerian cells unless explicitly set. It is not the candidate selected here; do not conflate its volume heuristic with CIB's regularization-weight field.

## Alternatives screened

| Route | Finding | Disposition |
|---|---|---|
| Existing DG-IIM force callback plus coordinate hold | Source and runs show it prescribes force/jump data; `use_direct_forcing` freezes coordinates. The fixed open-disk test retained order-one surface velocity mismatch. `TAU_OUT` is one-sided on the open sheet. | Reject as no-slip/force route. Preserve it as a geometry and diagnostic prototype. |
| Custom prescribed-velocity/unknown-traction extension to IIM | Could retain a sharp FE interface, but must alter the coupled solver and IIM time integration/jump equations consistently; substantially more method code and sign verification are needed. | Keep as fallback if CIB's marker-sheet behavior/force is not adequate. |
| IBAMR `CIBMethod` + `CIBStaggeredStokesSolver` | Native coupled Lagrange-multiplier constraint, prescribed rigid-body velocity callbacks, 3-D implementation available. Open-sheet accuracy and weighting are unverified. | Screen first on analytic 3-D controls and current disk marker set. |
| `IBFEMethod` + `IBFEDirectForcingKinematics` | Prescribes rigid-body kinematics, but its pinned implementation computes the explicit correction `F_df = rho_s (U_b - U)/dt`. This is a useful comparison route, but is less attractive as the first choice than CIB's coupled saddle-point multiplier for a strict no-slip gate. | Do not implement unless the CIB pilot fails or comparison becomes necessary. |

## Smallest discriminating experiment

### Hypothesis

The CIB coupled constraint will reduce marker-interpolated surface-velocity error on the existing open-disk geometry to the preregistered Stage A scale while maintaining incompressibility and a force sign consistent with the independently integrated whole-domain momentum balance.

### Sequence and baseline

1. **3-D solver/control preflight:** build the new pilot in an isolated application target against the existing IBAMR 0.19.0 3-D library; run a short version-matched 3-D CIB reference/control. Then run a prescribed rigid-body null where the disk and initially uniform fluid translate together. The multiplier and flow perturbation should be zero to discretization/solver tolerance. This tests the 3-D prescribed-kinematics plumbing, but is not sufficient evidence of coupling strength.
2. **Primary discriminator:** use the existing disk's exact point coordinates and dimensions as one stationary rigid CIB marker body in the existing low-Re broadside crossflow setup. Baseline for physical/input values is the source-corrected fixed open-disk IIM `ETA_S=0` run in `results/open_disk_force_audit_local_20261003_v1/eta0_current_source_run/`; preserve rho, mu, background velocity, domain, initial state, disk point coordinates, physical boundaries, short observation horizon, and resource guards. The single changed factor is the coupling-method route (IIM callback/jumps to CIB rigid constraint and its native marker regularization). Since the interpolation/spreading operators are intrinsic to that route, report the method as one bundled algorithmic factor and do not claim to have isolated the kernel from the constraint formulation.
3. Begin with a one-step run, then a short 10-step extension only if finite and bounded. Do not lengthen the horizon, tune marker weights after seeing the crossflow outcome, or start refinement until the prospective weight policy, measurement definitions, and acceptance criteria are fixed.

### Measurements

- Prescribed rigid transform and per-marker target-position error, evaluated against the analytic disk pose; target velocity and actual interpolated fluid velocity at the same CIB marker locations/time level; RMS and max velocity error normalized by `U_ref`.
- Composite-AMR divergence volume-weighted RMS/max globally and in the same declared surface band used in Stage A.
- Constraint multiplier/resultant and torque with code/SI units correctly identified, plus solver residuals and prescribed/solved DOFs. Establish the sign by comparison to the independently coded whole-fluid-domain momentum ledger, not by trusting a variable name.
- Outer-boundary advection, pressure traction, and viscous traction; whole-domain fluid momentum change; signed dimensional residual and normalization. Preserve edge-local velocity/divergence and any force concentration metrics.
- Exact input, source/build/library hashes, run stdout/stderr, exit status, timestep history, wall time, sampled peak RSS, and free disk before/after.

### Prospective gates and stop rules

- Retain the existing no-slip thresholds: RMS ≤5% and max ≤20% of `U_ref`; target-position lag RMS ≤2% and max ≤10% of finest `dx`; normalized divergence RMS ≤`1e-4` and max ≤`1e-3`. Do not loosen them based on CIB results. A one-step matched-flow null may only be called a null-control pass; it cannot qualify nonzero-flow no-slip.
- Before the primary disk run, declare a nonzero-flow momentum-residual tolerance based on the matched-flow ledger, solver residuals, and discretization/time-step scale. No such nonzero-flow tolerance currently exists; until preregistered, force/ledger closure is diagnostic and cannot pass.
- Stop and preserve the run for nonfinite values, rising error/force without bound, timestep failure, conservative 7.5 GiB observed process-tree RSS stop (8 GiB hard ceiling), or free disk below 40 GiB. Use no more than 8 MPI ranks; start at one rank.
- If the primary case fails no-slip, divergence, edge regularity, or independently signed momentum closure, mark this CIB open-sheet route **FAILED / UNVALIDATED** and do not proceed to disk-drag, mesh/time convergence, moth, or PPO. Diagnose one factor at a time; return to custom IIM constraints or another explicit interface model only with a stated mathematical reason.
- If it passes this short screen, that only justifies a prospective two-resolution/two-time-step CIB disk study. It still does not pass all of Stage A or validate the hawkmoth physics.

## What this does and does not establish

- Established: source-supported 3-D CIB implementation and solver are present in the local IBAMR binary; the method solves a constraint multiplier with the fluid and exposes prescribed rigid-body velocity callbacks.
- Not established: a fully prescribed 3-D CIB reference run on this installation; open-sheet no-slip; surface force quadrature or convergence; disk edge behavior; momentum closure; wing-specific regularization policy; runtime or memory cost; or whether this method produces accurate hawkmoth forces/wake.
- No files in the current geometry, original prototype, or prior run records were overwritten. No CFD case, external service, or PPO training was launched during this source audit.


## Pilot execution update (2026-10-03)

The bounded one-step matched-translation CIB pilot has now been run. This supersedes the earlier “no CIB case was built or run” status above; it does **not** change Stage A's **FAILED / UNVALIDATED** state or authorize the primary disk crossflow experiment.

At the actual new-time Eulerian field, the sampled marker slip was `1.6043e-5 U_ref` RMS / `5.8989e-5 U_ref` maximum, target-position error was at roundoff, and the recorded composite-divergence metrics were within the one-step null scales. However, the whole-domain momentum residual norm was `0.0841734` code force units, equal to `0.330901` of the sum of the dimensional term magnitudes. This is not a residual pass. The coupled operator source supports interpreting `lambda` as force on the fluid with the sign currently used, but it does not yet establish matching time centering, the generalized resultant's quadrature, or equality to the actual discrete force spread.

The late-captured ordinary `IBHierarchyIntegrator` body-force field was zero. This field is not the CIB saddle-point multiplier or its spread. Two earlier reads failed because the temporary ordinary-force patch data was deallocated after the hierarchy advance; those failed attempts and the log-classified tighter-solver nonconvergence remain preserved. The successful callback probe is useful only as a data-flow distinction, not as a CIB force verification. See `results/cib_method_pilot_local_20261003_v1/study_summary.json` and per-run records.

### Updated decision

**Continue only with source-level and discrete-budget diagnosis. Do not yet run stationary crossflow.** Determine the exact time level and quadrature of CIB `lambda`, trace its operator-level spreading into the fluid equation, and reconcile the discrete fluid momentum change against the independently integrated pressure, viscous, and advective boundary terms. A new instrumentation attempt must capture the CIB quantity at a valid lifetime and must not relabel the ordinary explicit-force field as the multiplier. Define a prospective ledger tolerance from analytic/null and solver-error evidence before viewing a nonzero-flow result. Proceed to a short crossflow discriminator only after the force path and the 33.1% null-ledger discrepancy are resolved or bounded by a defensible discrete identity.

The one-step run took approximately 24.6 s and peaked near 0.70 GiB process-tree RSS on the low-resolution diagnostic configuration. These numbers are not scalable estimates for a refined disk or hawkmoth. Recheck free disk before every run; the latest retained observation was approximately 57.03 GiB, so the ≥40 GiB reserve remains binding. No geometry, CIB weights, thresholds, or original prototype were changed. No hawkmoth, PIV/literature comparison, PPO, commit, or PR was performed.

## Follow-up exact multiplier-path audit (2026-10-03)

The separate read-only audit in `cib_multiplier_momentum_audit.md` resolves source semantics that the pilot update above correctly marked unknown at the time. The operator calls `setConstraintForce(lambda, half_time, -gamma)` and spreads it into its side-centered momentum output; the operator equation therefore puts `+S(lambda)` on the fluid-equation right-hand side. The stored lambda is an interval-midpoint quantity even though postprocess output labels it with `new_time`. The generalized-force routine sums raw lambda entries; the selected IB_6 kernel scales its delta weights by Eulerian cell volume and does not apply a further marker measure. These facts support the current sign and intended resultant, but are not a numerical check of the actual AMR/MAC spread.

The `output_eul_lambda` visualization is cell-centered and runs after the solve; it does not expose the exact midpoint side field used inside each matrix application. The ordinary explicit IB body-force capture remains a separate channel. The 33.1% null-ledger residual is still unexplained, with actual operator-spread integration and independent boundary-ledger verification outstanding. The next action is a bounded, equation-preserving capture of the side-vector increment at the CIB spread call and a no-structure analytic check of the boundary quadrature. No crossflow, convergence, hover, or PPO is authorized; Stage A stays **FAILED / UNVALIDATED**.
