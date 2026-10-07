# Civilia

An emergent AI civilization sandbox: a deterministic single-town social
simulator rendered as a living, procedurally generated pixel world. The
vision lives in [docs/civilia-brainstorm.md](docs/civilia-brainstorm.md); the
build order is the 48-stage plan in
[docs/full-staging-build-plan.md](docs/full-staging-build-plan.md) (currently
through stage 12, with v0s of 19/21/23/24 pulled forward for the living map —
see [docs/coverage-ledger.md](docs/coverage-ledger.md) for exact status).

The town of **Ashvale** sits on seeded terrain — ocean, lakes, rivers,
forests, mountains, marsh — with named landmarks that villagers mention in
conversation. Ten agents (so far) have needs, traits, relationships, jobs,
and money. Days advance in six phases; every agent scores its candidate
actions, adds seeded noise, and picks one. Every meaningful action emits a
structured event into an append-only log, every decision records a full
trace, and every social interaction produces an actual spoken line with a
motive analysis. Colonists propose buildings when needs demand them and
raise them from local timber, stone, clay, and reeds. Households claim
territory; disputed borders breed grudges and rumors.

No LLM is involved yet — by design. The deterministic core comes first;
the Ollama layer (qwen3:4b/8b + nomic-embed-text) arrives at stages 29–30.

## Run it

```bash
npm install
npm run dev        # living map UI (Vite)
npm test           # determinism + behavior tests (54)
npm run simulate   # headless 30-day run, prints a town report
npm run simulate -- mySeed 60   # custom seed and day count
```

## What to look at

- **The map** (default tab) — scroll to move between three viewpoints:
  - *Region*: biomes, named landmarks, and household claim territories —
    filled borders, with disputed ground stippled red.
  - *Settlement*: buildings appear — ghost outlines for planned, scaffolds
    with progress bars under construction, full sprites complete. Worn
    routes darken into trails, paths, and roads.
  - *Street*: villagers at house-relative scale. Speech bubbles are colored
    by category (chat, rumor, hostile, kind, status, alarm); thought bubbles
    surface pressing needs. **Click a bubble** to see why they said it —
    premeditation, genuine kindness, deceit, venting — with the numbers
    behind the judgment. **Click a building** to see everyone inside and
    outside and what each person is doing.
- **Stats bar**: bread stock/prices, coins in town, hunger, rumor count,
  hottest rumor, shortage/starvation alerts.
- **Town happenings ticker**: crimes witnessed, rumors mutating, border
  confrontations, constructions finished. Click to inspect the agent.
- **People tab**: the town's social web — household clusters, affection in
  green, grudges in red, open favor-debts in dashed gold. Hover an edge for
  both directions' numbers; click a villager to inspect them.
- **Agent drawer**: needs, traits, relationships, **what they remember**
  (first-person memories with emotional tone and fading strength), carried
  rumors, and the decision inspector (every candidate action, score, noise,
  reasons).
- **Log tab**: the full searchable event timeline.
- **💾 / 📂**: save and load the world (versioned JSON snapshot —
  mid-phase saves replay identically).
- **Seed box**: same seed ⇒ identical history, terrain and all.

## Architecture

```
src/sim/            deterministic simulation core (no UI dependencies)
  rng.ts              seeded, order-independent random streams
  types.ts            World / Agent / Place / Building / SimEvent / Rumor / traces
  terrain.ts          seeded biomes, rivers, materials, named landmarks, town siting
  movement.ts         A* routes; path wear: trail → path → road, nightly decay
  persistence.ts      versioned JSON snapshots (SQLite deferred to server stage)
  createWorld.ts      Ashvale: fixed cast, seed-derived traits, founding buildings
  stepWorld.ts        phase/day loop; dawn claims, end-of-day systems
  actions/            fourteen actions (work, food, social, repair, crime, travel, build)
  systems/            decisions, needs, events, memories, rumors, construction,
                      claims, occupancy, speech+motives, chronicle
src/ui/             React living-map inspector (reads the world, never mutates)
  components/WorldMap.tsx   canvas terrain + claims, SVG sprites, 3 zoom tiers
  components/PhraseCard.tsx motive inspector for any spoken line
  components/BuildingCard.tsx inside/outside occupants + construction state
scripts/            headless simulation runner
tests/              9 suites: determinism, actions, terrain, movement,
                    construction, claims, speech, persistence, chronicle
docs/               spec, build plan, coverage ledger, stage reports
```

Core rules the codebase enforces:

- Canonical truth is structured state; the event log is append-only.
- All randomness derives from labeled streams off the world seed — same
  seed, same history, even across save/load.
- Rumors and generated speech never modify canonical world state; speech is
  stored on its event at emit time with full motive provenance.
- Every agent decision — and every spoken line — is explainable from stored
  traces.

## Current cast

Four founding households — Calder, Bray, Fenn, Hale — plus whoever builds a
cottage and strikes out on their own. The Brays run the bakery and start
with a quiet grain surplus; one townsperson already suspects it. Two seeded
cross-household grudges give the gossip mill something to grind, and the
unemployed (Tessa, Ren, Ivy) have no income, which is where the trouble
usually starts.
