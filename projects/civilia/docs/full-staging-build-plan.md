# Civilia Full-Staging Build Plan

## Summary

Build Civilia in context-sized stages, where each stage is small enough for one
focused implementation bout and ends with tests, a status report, and coverage
updates. The source spec is the Civilia brainstorm document (in this repo:
[civilia-brainstorm.md](civilia-brainstorm.md)).

Non-negotiables:

- Do not skip any saved mechanic; track every heading/mechanic in a coverage ledger.
- Deterministic sim first, LLM later.
- Event log is the spine.
- Procedural living map is the primary UI direction.
- Every stage must leave the project runnable and inspectable.

## Stage Protocol

Each stage must produce:

- runnable app or passing sim tests
- updated stage report
- updated coverage ledger mapping MD sections/mechanics to implemented status
- decision traces for any new agent behavior
- no hidden "we'll remember later" work

Create these planning artifacts in Stage 0:

- `docs/source-spec.md`: copied/organized Civilia spec
- `docs/coverage-ledger.md`: every MD heading and mechanic mapped to a stage
- `docs/stage-reports/`: one report per completed stage

Core interfaces to stabilize early:

- `World`, `Agent`, `Place`, `Building`, `Claim`, `Event`, `ActionDefinition`, `DecisionTrace`
- `Memory`, `Belief`, `Rumor`, `Artifact`, `CultureElement`, `ChronicleIssue`, `LLMJob`
- `SimEngine`, `ActionRegistry`, `EventBus`, `ReactionSystem`, `MapGenerator`, `OverlaySystem`, `LLMService`

## Phased Roadmap

| Stage | Goal | Complete When |
| --- | --- | --- |
| 0 | Spec ledger and project scaffold | React/Vite + Node + TS project exists; coverage ledger tracks every MD section |
| 1 | Core IDs, RNG, time | Stable IDs, seeded RNG, day phases, reproducible stepping |
| 2 | World/event kernel | World, Event, append-only event log, basic stepWorld |
| 3 | Ashvale seed | 10 agents, 5-6 places, initial resources, deterministic generation |
| 4 | Foundation UI | map shell, timeline, selected agent panel, step phase/day controls |
| 5 | Basic needs/actions | hunger, rest, money; work, buy food, eat, sleep, travel |
| 6 | Decision system | utility scoring, seeded randomness, full decision traces |
| 7 | Debug inspector | candidate scores, selected action, reasons, event causality visible |
| 8 | Persistence v0 | SQLite schema for worlds/events/snapshots; save/load basic runs |
| 9 | Procedural terrain v0 | seeded map with water, rivers, forests, hills, mountains, biomes, resources |
| 10 | Map zoom layers v0 | regional, settlement, human zoom states wired in UI |
| 11 | Buildings/occupancy | buildings, indoor/outdoor state, click building to see occupants/actions |
| 12 | Movement/path wear | agents move between places; repeated routes become paths/roads |
| 13 | Relationships | trust, affection, resentment, debt, kinship; talk/help/insult/apologize |
| 14 | Memory system | memories from events, importance, decay placeholder, agent memory panel |
| 15 | Belief system | facts vs rumors vs suspicions, confidence, source trust, belief inspector |
| 16 | Gossip/rumor v0 | gossip action creates structured rumor events and belief changes |
| 17 | Rumor River | variants, truth distance, believers/skeptics, spread graph, rumor geography |
| 18 | Economy pressure | goods, production, prices, shortages, wages, debt, favors, hoarding |
| 19 | Construction system | needs trigger building proposals; planned/in-progress/complete states |
| 20 | Materials/architecture | timber/clay/stone/reeds affect buildings, status, scarcity, repairs |
| 21 | Claims/borders | factions/cults/clubs/governments stake claims; disputed areas and incidents |
| 22 | Map overlays | people, rumors, economy, tension, factions, memory/history, culture, construction, environment, privacy |
| 23 | Speech/thought bubbles v0 | visible bubbles, color categories, preset/contextual phrase templates |
| 24 | Phrase inspection | click phrase to see surface meaning, motive, context, relationship effects |
| 25 | Artifact Ledger | items, provenance, ownership, symbolic/monetary value, item witnessing |
| 26 | Culture elements | rituals, taboos, slang, fashion, holidays, symbols, adoption tracking |
| 27 | Culture Autopsy | causal chains from events to customs/buildings/phrases/taboos |
| 28 | Chronicle v0 | event clustering, source-linked weekly issue, no LLM required yet |
| 29 | Ollama gateway | qwen3:4b, qwen3:8b, nomic-embed-text, queues, validation, cache |
| 30 | LLM-assisted text | rumor mutation, memory compression, dialogue flavor, Chronicle prose |
| 31 | Law/crime/justice | theft, accusation, complaints, trials, precedent, punishment, loopholes |
| 32 | Politics/governance | councils, legitimacy, coalitions, petitions, soft coups, crisis powers |
| 33 | Religion/myth | omens, relics, sects, shrines, saints, heresies, sacred/cursed geography |
| 34 | Families/demographics | households, marriage, inheritance, age cohorts, generational memory |
| 35 | Health/disease/care | sickness, medicine, quarantine, disability, grief, care burden |
| 36 | Education/language/fashion | schools, apprenticeship, language drift, slang, status clothing |
| 37 | Institutions | guilds, temples, schools, clubs, secret societies, mission drift |
| 38 | Environment/infrastructure | water politics, fire, sanitation, seasons, resource depletion, roads |
| 39 | Conflict/security | feuds, militia, banditry, guards, trauma, peace rituals |
| 40 | Migration/exploration | outsiders, refugees, diaspora, cultural contact, borderlands |
| 41 | Media/communication | books, newspapers, pamphlets, message decay, censorship, phrase provenance |
| 42 | Daily life texture | hobbies, leisure, domestic conflict, routines, food preference, mood spillover |
| 43 | Weird causal loops | bad winter, accidental prophet, institutional amnesia, cursed-by-consensus objects, archive lies |
| 44 | Meta-sim tools | timeline branching, seed comparison, counterfactuals, weirdness dial |
| 45 | Dev tools | scenario editor, memory inspector, belief map, artifact browser, law viewer, search tools |
| 46 | Visual polish pass | seasonal visuals, scars, ruins reuse, construction politics, public gathering heat |
| 47 | Scaling/performance | workers, LLM budget, batching, caching, larger populations, profiling |
| 48 | Full coverage audit | every MD heading/mechanic has status: implemented, tested, deferred-with-reason |

## Current Position

As of 2026-06-11 the project is at **checkpoint 12** (movement/path wear).

Honest status against earlier stages, from a code audit on the same date:

- Stages 0–7 are substantively done (scaffold, RNG/time, event kernel, Ashvale
  seed, UI, needs/actions, decision system, decision inspector). The Stage 0
  planning artifacts (`coverage-ledger.md`, `stage-reports/`) do not exist yet.
- Stage 8 (SQLite persistence) is **not implemented** — the world is in-memory only.
- Stage 9 (procedural terrain) is **not implemented** — the map is a fixed
  hand-placed layout in `src/ui/mapLayout.ts`.
- Stage 10 (zoom layers) is **not implemented**.
- Stage 11 is partially done: places have occupants and clickable place cards,
  but there is no building/indoor-outdoor model.
- Stage 12 is half done: agents travel between places, but repeated routes do
  not yet wear into paths/roads (current paths are static decoration).
- Some later stages already have early versions built out of order: 13
  (relationships + social actions), 16 (gossip/rumor v0), 18 (economy
  pressure, shallow), 28 (Chronicle v0, deterministic side complete but
  `pendingChronicle` requests are not yet consumed anywhere).
