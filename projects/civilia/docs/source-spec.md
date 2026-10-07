# Civilia Source Spec — Index

The canonical spec is [civilia-brainstorm.md](civilia-brainstorm.md) (~2,500 lines).
This index organizes it for stage planning. Mechanics coverage is tracked in
[coverage-ledger.md](coverage-ledger.md); the build order is
[full-staging-build-plan.md](full-staging-build-plan.md).

## Vision layer

| Section | What it pins down |
| --- | --- |
| North Star / Core Pillars | Emergent civilization sandbox; deterministic sim + LLM flavor + observer UI |
| MVP Shape / First Playable Scenario | Ashvale, ~10→32 agents, needs/jobs/rumors/chronicle |
| Signature Features | The Chronicle, Rumor River, Artifact Ledger, Culture Autopsy |

## Mechanics catalog (the "do not skip" backlog)

| Section | Ledger group |
| --- | --- |
| Civilization Simulator Systems | agents, social, culture, economy, politics, history |
| Additional Bells and Whistles | psychology, religion, law, language, fashion, generations, secret societies, architecture, propaganda, romance, disease, exploration, meta-tools |
| Realistic Mechanics Backlog | 16 mechanic families, ~160 named mechanics |

## Engineering layer

| Section | What it pins down |
| --- | --- |
| Technical Architecture Sketch | event bus, memory, beliefs, planner, LLM layer, experiment runner |
| Local LLM and Intelligence Stack | qwen3:4b / qwen3:8b / nomic-embed-text roles, invocation rules, routing table |
| Implementation Blueprint | runtime loop, determinism rules, data model, event ontology, action system, testing strategy |
| Resolved Start-Gate Decisions | stack, phases, cast size, persistence timing, LLM timing |

## UI direction

The original blueprint's "inspector-first" UI plan is superseded by the
**procedural living map** direction — see [living-map-design.md](living-map-design.md).
