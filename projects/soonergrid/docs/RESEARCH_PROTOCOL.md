# SoonerGrid research protocol

Status: exploratory simulator; **not field validated**. The professional interface is not evidence of scientific validity. This document supersedes the earlier “research-grade” labels and headline savings in TODO.md and the archived dashboards.

## Evidence classes

| Artifact | Interpretation |
| --- | --- |
| `research_benchmark.json` | Fresh deterministic network experiments, input/source fingerprints, matched conditions, vehicle conservation ledger |
| `norman_network_processed.json` | OSM-derived geometry; road speeds, capacities and storage include assumed defaults |
| `norman_game_day_demand.json` | Assumed game-day demand; no observed count calibration supplied |
| Original baseline/autonomous JSON | Legacy engine output; lacks immutable source/run provenance and predates conservation repairs |
| Original resilience JSON | Prescribed curves and analytical sensitivity formulas, not twelve executed shock experiments |
| Original research dashboard | Synthetic playback and assumed ablation effects; not a trained-policy evaluation |

Do not pool these evidence classes or describe synthetic curves as empirical data. The historical `MARLSignalController` class name is retained for compatibility. Its implementation is a deterministic rule-based controller: there is no policy gradient, reward optimization, training dataset or learned checkpoint.

## Experiment design

Primary comparisons use the same processed network, demand, timestep, CAV fraction (0), and driver compliance (0.6). Control is either baseline fixed-time signals or the existing coordinated rules and incident response. The policy is a bundle of interventions: these runs do not isolate an individual component’s causal contribution.

The delivered eight-hour nominal/collision experiments use a 15-second step, selected to bound the cost of exploratory runs. This is **not a demonstrated converged timestep**. The default runner uses 5 seconds. Short links and whole-link transfer approximations require systematic convergence testing before performance interpretation. No random component is used in this runner; seed replication would repeat the same deterministic experiment, not produce valid confidence intervals. Demand multipliers are deterministic sensitivity cases, not sampling replicates.

Each artifact records creation time, complete configuration, input SHA-256 hashes, package-source digest, Python/NetworkX versions, elapsed wall time, summary and playback. A source change during execution invalidates publication. Only a complete experiment set is atomically published; interrupted work does not become a final artifact.

## Numerical contract

At every step:

`initial + generated = absorbed/exited + on-road occupancy + boundary queue`

Gateway demand exceeding available entry storage or capacity remains queued. Merge proposals are computed simultaneously and proportionally scaled to each target’s aggregate receiving/storage budget. Sink and dead-end absorption share one exit path; a vehicle cannot also move downstream in the same step. Closed receiving links admit no inter-link transfer.

The residual is computed before presentation rounding. Each run exposes generated, admitted, exited, remaining and external queued vehicles, plus the maximum absolute conservation residual. Ledger checks do not by themselves validate travel times, turn movements or route choice.

## Metrics and timing

- **Network TSTT:** left-endpoint integral of on-road occupancy, vehicle-hours. The modeled population uses fractional vehicle mass.
- **Boundary waiting:** integral of external queue occupancy after admission, vehicle-hours.
- **Total system time:** network TSTT plus boundary waiting. Report with unfinished demand; a finite-horizon total is not completed-trip travel time.
- **Exit count:** removal by the model’s sink/boundary rules. It does not demonstrate arrival at a traveler’s intended destination.
- **Speed-based delay:** legacy proxy relative to free-flow speed, not a trip-matched delay measurement.
- **Fuel:** legacy fixed multiplier of proxy delay, not an emissions model.
- **Playback:** stored timestep samples. Whole-run scorecards and ledgers stay fixed while the map and corridor chart scrub through the sampled trajectory. Last sample is before the horizon endpoint.
- **Recovery / resilience:** older heuristic fields remain in raw output but are not promoted as validated resilience scores in the new UI. A censored recovery must not be reported as an observed recovery time.

## Known structural limitations

1. Receiving-weighted flow splitting is not destination-specific route assignment; unrealistic cycles or return flows can occur.
2. Parking stall capacity, parked vehicles, and their subsequent release are not coupled to the engine.
3. Signal mapping uses street names and can associate many links with the first matching signal. It is not a validated intersection movement model.
4. Existing contraflow logic assumes lane configurations and spatial pairings. Physical clearance/occupancy and engineering approval are not established.
5. Incident effects, CAV headways, compliance and pedestrian factors are parameter assumptions; these are not microscopic driver trajectories.
6. Boundary admission uses nominal link capacity; dynamic incident effects govern subsequent flow. This limits boundary shock interpretation.
7. The existing whole-link scheme is not a verified implementation of Daganzo’s cell transmission model or a cumulative-count LTM; do not claim that equivalence.

## Validation needed for research claims

Collect independent approach counts, turning counts, signal plans, travel times, queue lengths, parking occupancy and incident timestamps. Define an explicit calibration objective, parameter ranges and acceptable errors before fitting. Reserve separate game days for validation. Replace aggregate routing and absorption assumptions, establish temporal/spatial discretization convergence, and cross-check a small network against a trusted reference implementation.

For stochastic experiments, use independent demand realizations with paired common random inputs across policies. Report paired effect estimates and uncertainty, and keep failed/censored runs visible. For intervention attribution, execute component ablations with all other settings fixed. No analytical curve should substitute for such a run.

This workflow follows the distinction between model checking, calibration and validation in [FHWA’s transportation analysis guidance](https://ops.fhwa.dot.gov/publications/fhwahop16072/mod4.htm). [FHWA’s Traffic Analysis Toolbox](https://ops.fhwa.dot.gov/trafficanalysistools/toolbox.htm) provides the broader modeling workflow. The [original network CTM paper](https://escholarship.org/content/qt9pz309w7/qt9pz309w7.pdf) is a theoretical reference, not certification of this implementation.
