# IBAMR DG-IIM force and coupling audit

**Date:** 2026-10-03  
**Status:** source semantics partially resolved; open-sheet force and no-slip remain **UNVALIDATED**.

## Scope and source identity

This audit follows the open-disk DG-IIM app into the pinned IBAMR 0.19.0 source, rather than inferring force meaning from printed labels. It does not validate the model or pass Stage A.

- App: `experiments/open_disk_dg_iim/src/open_disk.cpp`, updated SHA-256 `0dedb88bcbff2f643fe79c35f9f2886736e427c63d22142639a4657b5d0fddf1`.
- IBAMR source used for inspection: `/Users/REVIEW_USER/Applications/ibamr-iim-research/tmp/unpack/IBAMR-0.19.0/`.
- Diagnostic executable: SHA-256 `22896404f8762e6ec8c4371d33625f012102a4148a616d48bad50e774569d3bb`.
- Fresh report-only reruns: `results/open_disk_force_audit_local_20261003_v1/eta0_current_source_run/` and `eta100_current_source_run/`.

## What the callback means in this build

The app registers `target_force_function` with `IIMethod::LagSurfaceForceFcnData` and declares `IIMethod::VELOCITY_SYSTEM_NAME` as its interpolated system. At the midpoint force evaluation, the callback receives FE surface velocity `U` and returns

\[
F_s = \kappa_s(X-x) - \eta_s U.
\]

Here `X` is the stored reference target, `x` the current FE surface coordinate, and `U` the IIM FE velocity evaluated at surface quadrature points. This is a user-specified Lagrangian surface-force density used to construct the IIM force and jump data. It is not an unknown Lagrange multiplier solved to make the velocity error zero.

For the disk input both pressure-jump and velocity-jump conditions are enabled. IBAMR 0.19.0 `IIMethod::computeLagrangianForce()` adds the callback to its surface force `F`, then constructs (with `n` the oriented current surface normal, `dA` reference area element, and `da` current area element)

\[
[p] = (F\cdot n)\,dA/da,
\qquad
[\nabla u]_{ij} = -(dA/da)\left(F_i-(F\cdot n)n_i\right)n_j.
\]

With both jump mechanisms active, the source removes the already represented residual force from the direct Lagrangian force vector. `IIMethod::spreadForce()` spreads that residual and `imposeJumpConditions()` inserts pressure and velocity-jump contributions into the Eulerian force data. Thus the callback can affect the flow through the IIM jump treatment. It still does not solve a kinematic constraint for prescribed surface velocity.

In `IBFESurfaceMethod::{forwardEulerStep,midpointStep,trapezoidalStep}`, `use_direct_forcing=TRUE` copies current Lagrangian coordinates to the new state rather than advancing them with interpolated FE velocity. This option holds coordinates; it does not impose no-slip on the Eulerian flow. The recorded zero coordinate lag alongside order-one velocity mismatch is consistent with that distinction.

## What `TAU_OUT` means

IBAMR names this FE system `Exterior traction system`. In `IIMethod::computeFluidTraction()`, its quadrature value is formed as

\[
\tau_{out} = WSS_{out} - P_{out} n,
\]

where the one-sided exterior pressure and wall-shear/viscous traction are reconstructed using the IIM side data, viscosity, and the oriented surface normal. `TAU_OUT` is therefore a one-sided exterior fluid-traction field, not the registered target-force callback and not the full-domain momentum residual. IBAMR's closed-body `examples/IIM/ex5/example.cpp::calculateFluidForceAndTorque()` integrates `TAU_OUT` and comments that resultant as the net force acting on the body.

That closed-body example does not settle the force interpretation for this **open sheet**. The disk has a free edge and no globally enclosed interior; `TAU_OUT` describes one oriented side. It is not automatically the total force on a thin wing exposed to flow on both sides. A two-sided sheet resultant requires a documented combination of interior and exterior tractions with the matching normals/sign convention, then an independent momentum-balance check. This open-disk study has not done that. So the software field's definition is now known, but the reported open-sheet vector is not promoted to a validated aerodynamic force.

## Units

The inputs are computational values (`rho=1`, `mu=100`, background velocity `U=1`, disk radius `R=0.5`) and declare no map to a specimen or SI scale. Consequently, the report now labels lengths, velocities, area, and force resultants as code units. No numerical `ETA_S` value is a measured or physically calibrated wing damping property. Under an eventual SI mapping, the callback must be dimensioned consistently as a surface force density: `kappa_s` would scale as force/area/length and `eta_s` as force/area/velocity; that dimensional observation does not calibrate either coefficient here.

## Diagnostic label repair and preservation check

The old output name “Integrated spring reaction” was false for nonzero `ETA_S`: the old postprocessor integrated the complete callback and negated it, including the velocity term. The old `[m/s]`, `[m^2]`, and `[m]` labels were also unsupported by the uncalibrated input.

The postprocessor now reports separately:

- the integrated spring term `∫ kappa_s (X-x) dA`;
- the integrated velocity-feedback term `∫ -eta_s U dA`;
- the integrated registered callback, explicitly as an IIM force-callback resultant;
- the integrated `TAU_OUT` exterior-side traction resultant with mesh-normal orientation stated;
- target lag, velocity mismatch, and area in code units.

The rebuilt app (one MPI rank) was run on the same 10-step, 0.05-code-time-unit inputs for `ETA_S=0` and `ETA_S=100`. The rebuilt-run slip values exactly match their prior source run to printed precision: respectively 1.01291/1.07434 and 0.911292/1.00120 RMS/max. At ETA=0, both callback terms and their sum are zero. At ETA=100, the spring term is zero and the velocity-feedback term equals the callback resultant `[-70.3859, 3.96448, 2.3884]` code-force units. This checks diagnostic decomposition and confirms the label-only change did not alter these short flow outputs; it is not independent force validation.

The rebuilt runs used approximately 1.46/1.47 GiB peak process-tree RSS, 17.4/16.4 s wall time, one rank, and retained at least 57.38 GiB free disk. The two initial relaunch attempts were rejected before solver startup because the guarded runner requires empty run directories; those preflight refusals are retained separately and are not CFD failures.

## Decision and next gate

The audit resolves source-level callback and `TAU_OUT` semantics, and repairs misleading output labels. It does **not** explain the remaining slip or validate a force on the open disk. Do not continue `ETA_S` tuning, infer drag from `TAU_OUT`, or proceed to the hawkmoth case. The next method task is to design/select a genuine prescribed-velocity constraint that solves for the needed surface traction, and separately implement an open-sheet two-sided force accounting path. Verify signs with analytic controls and close an independent domain momentum ledger before any force claim. Stage A remains **FAILED / UNVALIDATED**.

## Pinned IBAMR source locations inspected

All references are inside the IBAMR 0.19.0 source root listed above:

- `src/IB/IIMethod.cpp`: `computeLagrangianForce()` (callback assembly and pressure/velocity jump construction), `spreadForce()`, `imposeJumpConditions()`, and `computeFluidTraction()` (one-sided `TAU_OUT`).
- `src/IB/IBFESurfaceMethod.cpp`: the three coordinate-update methods and their `d_use_direct_forcing` branches.
- `examples/IIM/ex5/example.cpp`: closed-body `TAU_OUT` integration as net body force/torque.
