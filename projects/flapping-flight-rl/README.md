# Flapping Flight RL

An ongoing research project toward physically grounded hawkmoth flight control with reinforcement learning. The long-term objective is to model unsteady wing aerodynamics and their shed wake, then train a PPO controller to hover and maneuver. The project is **not yet a validated hawkmoth flight simulator**.

## Current status

- The original fast blade-element / Gymnasium environment remains as an exploratory prototype. Its LEV, clap-and-fling, rotational-circulation, and added-mass terms are reduced-order model assumptions; they do not resolve the three-dimensional flow or wingtip vortices. Its dashboard and checkpoints are prototype artifacts, not evidence of physical validation or a successful hawkmoth controller.
- The active physics work is under [`research/hawkmoth_hover/`](research/hawkmoth_hover/). The Stage A moving-boundary qualification remains **FAILED / UNVALIDATED**.
- A one-step IBAMR CIB matched-translation disk null has very low sampled slip and divergence, but its whole-domain momentum residual remains unresolved: 33.09015% with the original boundary reconstruction and 34.32944% after correcting the simple outer-face pressure/viscous reconstruction. The actual CIB spread integral matches the negative raw marker resultant within 2.281e-8 maximum relative error over 345 operator applications, and composite MAC momentum weighting leaves the original residual unchanged. Separate no-structure uniform/Poiseuille controls recover the initialized analytic boundary resultants and show the short transient residual decreasing from 1.9499% at N=16 to 0.5753% at N=64; at N=32, the time-step study approaches a roughly 1% floor. These tests partially qualify the outer-face ledger but do not reconcile the CIB discrete momentum balance. This is diagnostic evidence only: Stage A remains FAILED / UNVALIDATED, and no nonzero-relative-flow CIB control, production hover, literature-force comparison, independent PIV comparison, or physically grounded PPO training has been completed.
- No PPO training should be treated as a research result until the physics gates and reduced-model validation described in [`research/hawkmoth_hover/docs/project_goals.md`](research/hawkmoth_hover/docs/project_goals.md) are met.

## Research sequence

1. Qualify prescribed moving-boundary coupling and force/momentum accounting with analytic controls.
2. Establish a converged prescribed-kinematics hawkmoth hover CFD baseline and compare its force and wake to the declared references.
3. Build a reduced-order environment from validated CFD data and test it on held-out CFD trajectories.
4. Train PPO in that reduced model within a **one-to-two-week target window and a hard maximum of 14 elapsed wall-clock days** on the available Mac, then evaluate selected policies in full CFD.

The one-to-two-week window applies only to PPO policy optimization after the physics and reduced model are ready. It does not cap the time spent qualifying the solver, establishing the CFD baseline, or building and validating the reduced model; those stages may take substantially longer. The 14-day ceiling is a compute budget, not a forecast of achievable training throughput or a guarantee of a successful hover policy. PPO training in full three-dimensional CFD is not the planned route.

## Documentation

- [Project goals and roadmap](research/hawkmoth_hover/docs/project_goals.md)
- [Research benchmark README and reproduction notes](research/hawkmoth_hover/README.md)
- [Validation protocol and gates](research/hawkmoth_hover/docs/validation_protocol.md)
- [Stage A report](research/hawkmoth_hover/docs/stage_a_report.md)
- [CIB multiplier and momentum source audit](research/hawkmoth_hover/docs/cib_multiplier_momentum_audit.md)
- [Takeover guide and current next action](research/hawkmoth_hover/docs/agent_takeover.md)
- [Results inventory](research/hawkmoth_hover/docs/results_inventory.md)

## Legacy prototype

The repository-root Python environment and dashboard are retained to preserve the original prototype. They can support software experiments, but results from them must be labeled as prototype-model results and kept separate from the IBAMR research benchmark. See the detailed benchmark status and resource policy in the research README before running solver work.
