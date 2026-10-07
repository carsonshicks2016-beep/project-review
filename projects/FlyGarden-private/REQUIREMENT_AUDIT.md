# Original-plan completion audit

This audit supplements the delivery report. The project is usable, but the entire scientific plan is not complete. A failed gate remains failed; later gated experiments are not silently counted as delivered.

| Requirement | Authoritative evidence | Current disposition |
|---|---|---|
| Separate local project, loopback server, stopped-app launcher | launcher.py; cold-start.json; api-verification.json | Verified |
| Full available brain first; provenance and memory benchmark | provenance.json; brain-benchmark.json; combined-benchmark.json; application-memory.json | Verified full network; slower than real time |
| Persistent neural state, bounded decoded commands, shared elapsed simulation time | brain.py; engine.py; output-control.json; cold-start.json | Verified implemented sample-and-hold coupling |
| Actual walking body, stop/turn/reset, finite physics | body.py; body-acceptance.json; turn-direction.json | Verified finite runs; obstacle falls contradict robust walkability |
| Odor, annotated sensory and descending populations | neuron-mapping.json; brain.py | Engineered encodings implemented |
| Actual retinal and ascending contact/motion brain adapter | body.py; IMPLEMENTATION_REPORT.md | Incomplete; renderer inspection and supplied gait feedback do not prove neural integration |
| Plasticity, fixed non-learning weights, bounded signed weights | plasticity.py; brain.py; retention.json | Implemented experimental rule; biological fidelity unvalidated |
| Acquisition/reversal against frozen and shuffled controls | learning-evaluation.json; learning-evaluation-source.py | Completed neural screen failed; behavioral choice assay incomplete |
| Retention of learned parameters and complete-state continuation | retention.json; cold-start.json; arena-persistence.json | Verified modeled state; not behavioral retention |
| Generalization and threat association | learning-evaluation.json next_gate | Deferred after failed learning screen |
| Walls/blocks/shelter, LOS and bounded navigated predator | world.py; tests; predator-calibration.json | Implemented; obstacle walking unreliable |
| Shallow uneven terrain | body.py; delivery-summary.json | Editable 0.15 mm physical step implemented; two short traversal trials passed, broader terrain stability unvalidated |
| Sandbox, challenges, experiments, camera/senses, editor, progression | static/app.js; browser preview; api-verification.json | Implemented; gameplay completion is not a science badge |
| Immutable learned/scene saves, lineage, replay, interrupted jobs | storage.py; api-verification.json; interrupted-fixture.json; job-restart.json | Verified, including actual stopped-server restart |
| Tutorials, presets, dependencies, provenance, reports, native video | README.md; field guide in app; requirements.lock; reports | Delivered |

Next work: diagnose the failed neural readout and documented action pathway on independent diagnostic seeds; establish stable obstacle locomotion before extending terrain; add retinal/contact neural mappings only with traceable annotation evidence. Do not re-label a rule-based forager as the full-brain controller or change the failed evaluation sign to manufacture a pass.
