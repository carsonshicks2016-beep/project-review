# Civilia Coverage Ledger

Maps every section/mechanic group of [civilia-brainstorm.md](civilia-brainstorm.md)
to its stage in [full-staging-build-plan.md](full-staging-build-plan.md) and its
implementation status. Update this file every stage.

Statuses: **done** (implemented + tested), **partial** (v0 exists, depth pending),
**planned** (not started), **deferred** (with reason).

Last updated: 2026-06-11 (stage 0–12 reinforcement pass).

## Vision & engineering foundations

| Spec section | Stage(s) | Status | Notes |
| --- | --- | --- | --- |
| North Star / Core Pillars / MVP Shape | 0–7 | done | Deterministic core loop live |
| Technical Architecture Sketch | 1–2, 29 | partial | Event log spine done; LLM/embeddings layer planned |
| Determinism and Randomness | 1 | done | Seeded order-independent streams (`rng.ts`), determinism tests |
| Canonical State vs Derived Interpretation | 2 | done | Rumors/speech never mutate canonical state |
| Core Data Model / Schema Sketches | 2 | partial | World/Agent/Place/Building/Event/Rumor/Memory/DecisionTrace live; Belief/Goal/Artifact/CultureElement/Law planned (15, 25, 26, 31) |
| Event Ontology | 2 | partial | 20+ typed events; lifecycle/politics/culture event families arrive with their stages |
| Action System / MVP Action Catalog | 5–6, 13 | partial | 14 actions incl. apologize + repay_favor; accuse/investigate/attend_meeting/write_note → stages 31, 41 |
| Agent Decision Formula | 6 | done | Utility + traits + relationships + seeded noise, squared-weight selection pool |
| Needs and Emotions | 5, 14 | partial | hunger/fatigue/belonging live; emotions model → stage 14 |
| Testing Strategy | all | partial | Determinism, action, chronicle, terrain, persistence, claims, speech suites |
| Persistence and Replay | 8 | partial | JSON snapshot save/load v0; SQLite deferred — no Node server yet, browser-only app (revisit at stage 29 when the Ollama gateway adds a server process) |
| Milestone Roadmap (old) | — | superseded | Replaced by the 48-stage plan |

## Local LLM stack

| Spec section | Stage(s) | Status | Notes |
| --- | --- | --- | --- |
| Model roles (qwen3:4b / 8b / nomic-embed-text) | 29–30 | planned | |
| Invocation rules / routing table | 29 | planned | Chronicle prompts already structured for handoff |
| Structured output policy / validation | 29 | planned | |
| Prompt context budgeting / performance strategy | 29, 47 | planned | |

## Signature features

| Feature | Stage(s) | Status | Notes |
| --- | --- | --- | --- |
| The Chronicle | 28 (det.), 30 (LLM) | partial | Event selection, clustering, faction prompt templates, reader UI, config done; articles unfulfilled until stage 29–30 (`world.pendingChronicle` awaits consumer) |
| Rumor River | 16 (v0), 17 | partial | Structured rumors, template mutation, graded conviction rosters, truth distance, **rumor lens overlay (live particle arcs)** done; persistent spread graph + geography → 17 |
| Artifact Ledger | 25 | planned | |
| Culture Autopsy | 27 | planned | |

## Living map (Carson's UI direction — see living-map-design.md)

| Requirement | Stage(s) | Status | Notes |
| --- | --- | --- | --- |
| Procedural terrain (ocean, lakes, rivers, forests, mountains, biomes, resources) | 9 | done | `src/sim/terrain.ts`, seeded value-noise fBm, landmark naming, determinism tests |
| Three zoom viewpoints (regional / settlement / human) | 10 | done | Scroll-zoom tiers in `WorldMap` |
| 16-bit pixel visual style | 46 (polish), v0 now | partial | Canvas tile renderer + pixel sprites v0 |
| Buildings with planned / in-progress / complete states | 11, 19 | partial | Building model, construction system v0 (need-triggered proposals, material gathering, build shifts) |
| Materials from terrain (timber/stone/clay/reeds) | 20 | partial | Gathering depletes terrain tiles; architecture/status effects of materials pending |
| Claims, borders, disputed zones | 21 | partial | Household claims v0, disputed-tile detection, border_dispute incidents; cults/governments/clubs claim when factions exist (37) |
| Indoor/outdoor occupancy, click building → who's inside doing what | 11 | done | `Agent.indoors`, BuildingCard panel |
| Movement + path wear → trails → roads | 12 | done | A* routing around water, wear accumulation + decay, road rendering |
| Speech/thought bubbles, color-coded | 23 | partial | v0: speech stored on events, thought bubbles derived from needs |
| Generated phrases from preset + contextual variables (rumors, prices, items) | 23 | partial | **Contextual dialogue engine: lines cite the strongest memory/belief about the listener** (speaking-time snapshots); relics/books/newspapers as sources → 25, 41 |
| Phrase inspection (motive: premeditation/kindness/deceit…) | 24 | partial | Full Reasoning Tree: motivation, historical context (cited memory → origin event day/phase), influencing knowledge (belief + confidence + heard-from), gossip reception factors |

## Mechanics backlog (16 families, summarized)

| Family | Stage(s) | Status | Notes |
| --- | --- | --- | --- |
| Individual Agents (traits, needs, secrets, goals, quirks) | 5–6, 14–15 | partial | Traits/needs done; **memories done** (formation, tone, decay, recall, decision bias, panel); beliefs → 15; goals, secrets, habits later |
| Social Simulation & Social Mechanics (favor economy, reputation lag, scapegoats, cliques…) | 13 | partial | trust/affection/resentment + talk/help/insult/**apologize** done with forgiveness thresholds; favor economy v0 (gift debts, repayment, debt pressure); grievance tracking; social graph UI. Reputation lag, scapegoats, cliques → later |
| Belief & Knowledge Mechanics (source trust, official vs street truth, conspiracies…) | 15 | partial | Graded beliefs done: source trust, disposition bias, inertia, witnessed-vs-hearsay asymmetry, corroboration, truth-vs-belief inspector. Conspiracies, official truth, credibility cascades → later |
| Culture Engine & Culture Mechanics (taboos, rituals, slang, symbols, fashion…) | 26–27, 36 | planned | |
| Economy & Economy Mechanics (production chains, credit, panic buying, hoarding…) | 18 | partial | Bread/grain/prices/shortages/theft v0; chains, debt, hoarding spiral planned |
| Politics & Law (precedent, loopholes, legitimacy, trials, selective enforcement…) | 31–32 | planned | |
| History System (eras, revision waves, archives as power) | 28, 41, 43 | planned | Event log foundation exists |
| Religion and Myth Engine | 33 | planned | |
| Language Drift | 36 | planned | Speech template engine is the seam |
| Fashion and Status Engine | 36 | planned | Status scalar exists |
| Generational Memory / Demographics / Romance & Family | 34 | partial | Households + cottage spin-off (household_formed) v0; marriage/inheritance/cohorts planned |
| Secret Societies / Institutions | 37 | planned | |
| Architecture Evolution / Environment & Infrastructure | 19–20, 38 | partial | Construction v0; walls/fire/sanitation/depletion planned |
| Disease / Health / Care | 35 | planned | |
| Conflict / Security | 39 | planned | border_dispute incidents are the seed |
| Migration / Exploration / Media / Daily Texture / Weird Loops / Meta-tools / Dev tools | 40–45 | planned | |

## Player roles

| Role | Stage(s) | Status |
| --- | --- | --- |
| Observer / Archivist | 4, 7, now | partial |
| God mode, Ruler, Scientist, Trickster, Journalist, Archaeologist | 44–45 | planned |
