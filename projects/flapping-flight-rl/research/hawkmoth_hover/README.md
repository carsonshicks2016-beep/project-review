# Wake-resolved hawkmoth hover benchmark

This directory contains the research CFD benchmark and its verification record for the flapping-flight project. The planned biological case is a rigid, prescribed-kinematics, three-dimensional *Manduca sexta* hover simulation. It must establish a converged force and wake baseline before any physically meaningful reinforcement learning is attempted.

**Current status: Stage A is `FAILED / UNVALIDATED`.** The version-matched IBAMR 0.19.0 staggered-flow regression passes its recorded norms. In the one-step CIB matched-translation disk null, the captured actual spread integral matches the negative raw marker resultant within maximum relative error `2.281e-8` across 345 operator applications, and composite MAC dual-volume momentum weighting leaves the original normalized residual near `0.3309015`. Separate no-structure uniform/Poiseuille runs recover the initialized analytic outer-boundary resultants and show a short transient residual decreasing from `1.9499%` at N=16 to `0.5753%` at N=64; a fixed-resolution time-step study approaches a roughly 1% residual floor. Applying the corrected boundary reconstruction to the same CIB null changes its residual to `0.3432944`, so the ledger check has not resolved CIB momentum closure. These are diagnostic results, not a coupling pass. Earlier immersed-boundary implementations also have failed nonzero-relative-flow controls. No production hawkmoth hover, periodic response, literature-force comparison, independent DPIV comparison, or physically grounded PPO training has been completed. Preserve and interpret all runs through the dated report and result inventory; do not treat a matched-flow null or process exit as validation.

The project objective and staged go/no-go plan are in [`docs/project_goals.md`](docs/project_goals.md). The later PPO campaign has a **one-to-two-week target window and a hard maximum of 14 elapsed wall-clock days**, applied only after physics qualification and validation of the CFD-derived reduced-order model. There is no two-week deadline for prerequisite solver, CFD-baseline, or reduced-model work. The 14-day ceiling is a resource limit, not a forecast of achievable throughput or a guarantee that PPO will learn a successful hover policy. PPO training is intended to use the reduced model, with selected policies subsequently evaluated in full CFD; training directly in full 3-D CFD is not the planned route.

## Scientific question

Does a three-dimensional viscous Navier–Stokes calculation with an immersed moving wing pair reproduce the phase-resolved aerodynamic force pattern and wake topology of the selected published hawkmoth hover case, within defensible numerical and source uncertainty?

The model is a controlled reproduction of one published idealization, not a complete specimen reconstruction. The source authors traced an *Agrius convolvuli* wing/body outline, scaled wing chord to a reported *M. sexta* mean chord, and represented the fore- and hind-wings together as one synchronized pair. This benchmark inherits those boundaries. We will not add empirical LEV, wingtip-vortex, clap-and-fling, or lift multipliers.

## Contents

- `provenance.yaml`: parameter source, units, uncertainty, provenance, and unresolved interpretation.
- `case/`: solver input deck and case-specific configuration.
- `src/`: IBAMR application source and CMake definition.
- `tools/`: input preparation, post-processing, and acceptance-gate tools.
- `reference_data/ibamr_0.19.0_navier_stokes/`: upstream staggered-flow regression source, exact input, and expected output; build with `tools/build_solver_reference.sh` and run the prepared regression case with the guarded runner.
- `reference_data/`: cited paper figure used for explicitly identified digitization and the DPIV data README/provenance. The DPIV files are not included unless obtained from Dryad and recorded in the ledger.
- `docs/`: [project goals and roadmap](docs/project_goals.md), preregistered run protocol, Stage A failure report, feasibility log, [CIB multiplier/momentum source audit](docs/cib_multiplier_momentum_audit.md), output contract, and validation report template.
- `docs/agent_takeover.md`: continuation brief with research boundaries, current gates, evidence interpretation, exact reproduction commands, and next actions. `docs/results_inventory.md` lists every current result root and run/preparation record; refresh with `python3 tools/build_results_inventory.py`.
- `results/`: immutable short smoke/failure artifacts and `docs/feasibility_log.md`; these are diagnostic startup checks, not biological results.
- Generated meshes, raw fields, logs, and run products belong in ignored `output/` and `restart/` directories.

The owner-supplied `reference_data/user_supplied_dpiv/DPIV_DATA.mat` is retained with its SHA-256 and schema notes, but its source, units, phase mapping, and license are unverified. It is not yet treated as the official Dryad DPIV dataset or as validation ground truth; see its adjacent README and the reference-data ledger.

## Reproduction outline

1. Build pinned IBAMR and dependencies outside the repository with `tools/install_ibamr.sh`. Archive its logs and dependency versions.
2. Build this application with `IBAMR_PREFIX=/path/to/install tools/build_case.sh`.
3. Prepare a run, e.g. `python3 tools/prepare_run.py S0T0`. It creates an ignored run directory with the exact input deck, mesh files, hashes, and requested resource limits. The current manual outline trace is provisional and must be replaced/quantified before interpreting biology.
4. Run it with `python3 tools/run_case.py case/runs/S0T0 --executable build/hawkmoth_hover --ranks 1`. The wrapper refuses runs above eight MPI ranks, samples aggregate solver-process RSS every 0.2 seconds, and stops before free disk falls below 40 GB. The recorded RSS is a sampled peak, not a kernel-enforced memory ceiling.
5. Run Stage A before interpreting moth output. Build and run the bundled upstream 3-D manufactured-flow regression, compare its six error norms with `tools/verify_solver_reference.py`, and use `tools/prepare_verification.py` for immutable stationary, stationary-crossflow, and matched-translation controls. The app's optional diagnostics record realized timestep/CFL, target lag, IB-interpolated surface no-slip error, global and near-surface composite-grid divergence, physical-boundary margin, and a domain momentum ledger with separately integrated pressure, advection, and viscous terms. Prescribed target forces are evaluated at the explicit midpoint; nonzero target damping is made relative to prescribed surface velocity by a temporary, reversible force-reference shift. Stage A controls and full physics validation remain mandatory; a successful executable build or exact matched-flow null is not validation.
6. Evaluate force periodicity and convergence with `tools/force_gates.py` after the recorded CSV histories cover the declared full cycles. Do not change physics between levels apart from resolution and time-step controls.
7. Export phase-resolved force/moment/power and selected three-dimensional phase snapshots. A reproducible plane-extraction/export path for the 75%-span PIV comparison remains to be completed with the installed Silo/VisIt toolchain.
8. Compare force histories to the rigid-base-kinematics results in Nakata & Liu (2012), including the published cycle means (15.4 mN vertical and −4.4 mN horizontal for that paper's complete wing/body convention). Compare PIV only as a separate cross-study wake comparison.
9. Mark the physics benchmark validated only if every applicable gate in `docs/validation_protocol.md` passes and unexplained differences are resolved. PPO remains gated on the subsequent CFD-derived reduced-model validation and is not covered by a CFD physics pass alone.

## PPO compute plan (future stage)

The one-to-two-week PPO window, with a hard 14-day maximum, applies only to the eventual policy-training campaign, not to the time available to build and validate the physics model. It is not a claim that this Mac can execute a particular number of transitions or that a successful policy will result within that window. Before training, benchmark the final reduced environment's end-to-end transitions per second and plan the transition count, seeds, and evaluation reserve from that measured rate. For scale, 10 million transitions require about 16.5 transitions/s for one continuous week or 8.3/s for two; 100 million require about 165/s or 82.7/s. These are arithmetic rates, not measured throughput or a promise of PPO convergence. Count all seeds, rollouts, updates, restarts, and hyperparameter trials against the 14-day cap. Evaluate selected policies in full CFD afterward and report that cost separately. Details and claim boundaries are in `docs/project_goals.md`.

## Fidelity and sign conventions

- Flow: incompressible, transient, three-dimensional, viscous Navier–Stokes; staggered-grid IBAMR solver and adaptive Cartesian refinement.
- Domain: 4R across the two transverse axes and 16R along z, at 16 coarse cells per R. The longer wake-axis extent is a computational compromise; wake-boundary sensitivity remains a required measured gate.
- Boundary: thin, rigid immersed surface point cloud with prescribed periodic motion. The target-point penalty stiffness and Lagrangian spacing are numerical parameters that require sensitivity/convergence checks; they are not physical wing stiffness.
- `KINEMATICS_SCALE=0` holds the paired surfaces at their initial pose for the zero-motion control; `1` selects the declared hover harmonics. Verification histories separately record RMS/max target-position lag and IB-kernel-interpolated Eulerian no-slip velocity error; neither alone establishes physical validation.
- The timestep is additionally capped by a conservative moving-boundary displacement estimate, and the run history records realized `dt_s`. The run matrix uses 8192/16384/32768 nominal steps per period. Target stiffness and damping are numerical penalty parameters, not physical wing stiffness or material damping; they still require a converged sensitivity study. The rotating-target relative-damping implementation has not yet been exercised in a hover run.
- Body: fixed in the source model, but the current immersed mesh omits the body. The current force is wing-only, while the published table describes forces on wings and body; a direct force match is therefore not yet apples-to-apples. Add and document body geometry or a defensible force decomposition before calling the literature force gate satisfied.
- Geometry: paired-wing source-model simplification; no claim of a complete four-wing flexible moth.
- Power: report aerodynamic work delivered by the prescribed moving boundary to the fluid, with its sign convention and any inertial term stated. Do not call this muscle power or total metabolic power.
- Independent DPIV: another specimen/study. Align phase from wing position using documented landmarks and compare only spatial/temporal scales supported by the published vectors.

## Resource policy

Local solver runs are limited to 8 MPI ranks, an 8 GiB hard RSS ceiling (the runner triggers at 7.5 GiB to cover its 0.2 s sampling interval), and at least 40 GB free disk. Build/install artifacts live outside the tracked project. Save full fields only at selected phases; retain compact sampled planes and derived fields for routine comparisons. If the cap prevents convergence, preserve and report the feasibility limit rather than loosening it silently.

## References

- Nakata, T. & Liu, H. (2012), “Aerodynamic performance of a hovering hawkmoth with flexible wings: a computational approach,” *Proceedings of the Royal Society B*, 279:722–731. [Article](https://pmc.ncbi.nlm.nih.gov/articles/PMC3248719/), DOI: 10.1098/rspb.2011.1023.
- Mountcastle, A. M. & Daniel, T. L. (2010), “Aerodynamic and functional consequences of wing compliance,” with the associated hawkmoth DPIV data archived at [Dryad DOI 10.5061/dryad.tqjq2bw6d](https://datadryad.org/dataset/doi:10.5061/dryad.tqjq2bw6d). The dataset is an independent experiment and specimen.
- IBAMR 0.19.0: [build guide](https://ibamr.github.io/building), [solver FAQ](https://ibamr.github.io/faq), [examples](https://ibamr.github.io/examples).
