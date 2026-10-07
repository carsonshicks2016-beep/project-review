# Project goals and roadmap

## North-star objective

Build a reproducible, physically defensible simulation of *Manduca sexta* hawkmoth hovering flight, then use it to train a PPO controller that can stabilize hover and eventually perform controlled maneuvers. The flow model should resolve unsteady viscous aerodynamics and the shed wake, including leading-edge and wingtip vortices as consequences of the governing equations and geometry—not as fitted force bonuses.

The existing repository-root blade-element environment is a preserved exploratory prototype. Its named aerodynamic mechanisms are model assumptions, not resolved flow measurements. Prototype checkpoints, reward curves, or dashboard visualizations do not establish that a real or CFD-validated moth can hover.

## Research principles

1. **Physics before policy claims.** Do not train or report a physically meaningful flight policy until the moving-boundary method, force accounting, hover solution, numerical convergence, and reference comparisons pass their declared gates.
2. **Measure rather than tune toward a desired answer.** Freeze source geometry lineage, kinematics, solver settings, tolerances, and comparison procedures before interpreting production outcomes. Record any departure from the protocol.
3. **Keep resolved and reduced models distinct.** PPO will use a validated reduced-order model for tractable sampling. Reduced-model performance is not itself a CFD result; evaluate selected policies in the full CFD solver and report disagreement.
4. **Preserve negative results.** A failed coupling, convergence, or policy gate is a valid project result when inputs, hashes, logs, resource use, and interpretation are reproducible.
5. **Make scope explicit.** The first physical model is the chosen published rigid, paired-wing hover idealization. Do not call it a complete, flexible, four-wing specimen model.

## Roadmap and gates

### 1. Moving-boundary and force qualification — active, not passed

Qualify the solver and immersed interface on analytic controls before moth geometry. The CIB matched-flow null has very low sampled slip and divergence, but its whole-domain momentum residual remains unresolved: `0.3309015` under the original boundary reconstruction and `0.3432944` when the same input is replayed with the corrected boundary reconstruction. The actual side-centered spread increment agrees with the negative raw marker resultant within `2.281e-8` maximum relative error over 345 operator applications, while replacing the cell-average fluid momentum integral with composite MAC dual-volume weights leaves the old residual unchanged. Separate no-structure uniform and Poiseuille controls now exercise the outer-boundary terms; the initial analytic pressure and viscous resultants match their exact values, and the short transient residual decreases from `1.9499%` at N=16 to `0.5753%` at N=64. At N=32, reducing dt by four changes that residual from `1.1265%` to `1.0346%`, suggesting a roughly 1% short-case floor. This qualifies selected face terms and demonstrates an analytic refinement trend, but it does not reconcile the CIB momentum equation or validate its force solution. See `cib_multiplier_momentum_audit.md` and `results/cib_boundary_ledger_analytic_local_20261003_v1/`. Stage A remains **FAILED / UNVALIDATED**. Next derive the discrete staggered momentum balance for the same CIB step and identify the discrepancy before any stationary-relative-flow control. Do not advance to hawkmoth hover.

**Exit gate:** fixed stationary/moving controls meet the existing position, no-slip, divergence, solver-convergence, and independent momentum tolerances under the preregistered refinement and timestep checks. A matched-flow null alone is insufficient.

### 2. Converged prescribed-kinematics hover CFD

After Stage A passes, establish the prescribed rigid paired-wing hover solution with fully recorded geometry/kinematics provenance. Demonstrate periodic force and power histories and the protocol's spatial/time convergence. Compare force histories to the selected published case and wake structure to independent DPIV only within their declared source and measurement limits.

**Exit gate:** all applicable numerical and physical-comparison gates in `validation_protocol.md` pass, or the benchmark is reported `UNVALIDATED` with the unresolved differences stated.

### 3. CFD-derived reduced-order flight environment

Build a fast control environment from the qualified CFD database. Preserve unsteady effects needed for control, including phase and relevant wake/history state; do not reduce the system to instantaneous quasi-steady coefficients unless held-out CFD tests show that memory can be neglected. Couple aerodynamic loads to the declared rigid-body dynamics and expose bounded, documented actions/observations.

Test predictions on held-out CFD kinematics and trajectories, including stroke reversals and disturbances not used to fit the model. Predeclare force, moment, and state-trajectory error criteria before PPO training. If the ROM cannot reproduce those tests within the declared scope, revise or reject it before policy optimization.

**Exit gate:** the ROM passes its held-out comparison and conservation/physical-bound checks, and its exact data lineage and validity envelope are recorded.

### 4. PPO campaign — one-to-two-week target, 14-day hard maximum

Once the physics benchmark and reduced environment pass, run PPO in the reduced model. The planning target is **one to two weeks of training compute**, with an absolute cap of **14 elapsed wall-clock days for the full PPO campaign** on the available Mac. The clock begins when the first policy optimization/rollout run starts. Count rollouts, gradient updates, restarts, random seeds, and hyperparameter trials against the cap. Model implementation, CFD data generation, solver validation, and reduced-model qualification happen beforehand and are not hidden inside the training budget; these physics and model-building stages may take substantially longer, with no two-week deadline. Prioritize the most physically faithful model that passes its evidence gates; do not reduce physics fidelity just to fit an unmeasured transition target into this later PPO window.

Before the campaign, benchmark end-to-end environment throughput using the final observation/action path and vectorization. Choose the total transition and seed plan from measured throughput rather than assuming a training rate. For reference, 10 million transitions require about 16.5 transitions/s over seven continuous days or 8.3/s over fourteen; 100 million require about 165/s or 82.7/s. These are arithmetic planning rates, not claims about the Mac's achievable throughput. The 14-day limit is a resource budget, not a prediction that a particular transition count is feasible or a guarantee that PPO will learn a successful hover policy. Reserve time for checkpoints and held-out evaluation within the campaign.

Pre-register the hover-success definition, perturbation suite, evaluation seeds, observation/action timing, reward, PPO configuration, and promotion rules. Report learning curves and binary hover success separately from falls, divergence, timeouts, and numerical failures. Use multiple independent seeds where the measured budget permits; do not select a policy from reward alone. Preserve each candidate's immutable checkpoint and any observation-normalization state together with source, config, environment-data, and seed hashes.

**Claim boundary:** a policy that succeeds in the ROM is a reduced-model PPO result. It is not a physically validated hawkmoth controller until it survives the full-CFD evaluation below.

### 5. Full-CFD closed-loop policy evaluation

Replay selected policies against the high-fidelity solver under predeclared nominal and perturbed cases. Report the number and cost of episodes, force/state differences against the ROM, solver stability, and any failures. Treat this evaluation as a separately measured compute stage; the 14-day cap limits PPO training, not the physics validation or final CFD evaluation.

**Exit gate:** make a flight-control claim only for the tested policy, model scope, initial conditions, and perturbations that pass the declared full-CFD criteria. Any ROM-to-CFD failure remains a negative result, not something hidden by further ROM reward.

## Compute and reproducibility constraints

- Target local development and physics runs at the Apple Silicon Mac. For guarded local solver runs, use no more than 8 MPI ranks, stop before observed solver/process-tree RSS reaches 7.5 GiB (8 GiB hard ceiling), and preserve at least 40 GiB free disk.
- Do not estimate production runtime by scaling the current one-step, low-resolution CIB disk null. Its approximately 24.6-second wall time is startup-inclusive and is not representative of a hawkmoth cycle.
- Every result must include immutable inputs, source and executable/build identity, solver logs, resource records, and machine-readable interpretation. Distinguish process completion, linear/nonlinear solver convergence, numerical gate pass, and scientific validation status.
- External compute, purchased service, or cloud charges require a separate explicit decision. No resource is provisioned by this roadmap.

## Current stop/go decision

**Current state:** Stage A `FAILED / UNVALIDATED`; CIB matched-flow null and no-structure analytic-ledger controls are diagnostic-only; force/momentum closure unresolved. The actual side-centered multiplier spread has been measured and matches its raw marker resultant to numerical precision. The corrected outer-boundary reconstruction gives the expected initial pressure/viscous resultants and a decreasing short-case residual under spatial refinement, but the same CIB null's normalized residual changes slightly upward from `0.3309015` to `0.3432944`. Therefore the remaining task is to derive and compare the exact discrete staggered momentum balance on that CIB step, not to treat the smooth-flow boundary quadrature check as a solution. Do not run stationary-relative-flow CIB, PPO, full hover, literature/PIV force or wake comparisons, or full-CFD policy evaluation until the discrepancy is explained or prospectively bounded and the applicable gates pass.
