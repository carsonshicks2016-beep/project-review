# Civilia Living Map — Design Direction

Carson's direction (2026-06-11): the primary UI is a procedurally generated
living map, 16-bit pixel style, replacing the hand-placed town view.

## Core requirements (from Carson)

1. **Procedural terrain**: seeded map with oceans, lakes, rivers, trees,
   forests, mountains, hills, biomes, and landmarks.
2. **Colonists build their own buildings**, driven by need (crowding,
   shortage, status) and constrained by locally available materials
   (timber, stone, clay, reeds).
3. **Three zoom viewpoints**, transitioned by scrolling:
   - **Regional** (bird's eye): biomes and landmarks, with filled border
     areas showing the staked claims of local factions
     (households/cults/governments/clubs) and disputed border zones.
   - **Settlement**: buildings appear inside the territory, visibly
     **planned** (ghost outline), **in progress** (scaffold + progress), or
     **complete**.
   - **Human**: people rendered at height/size proportional to the houses.
     Speech bubbles and thought bubbles, color-coded by category (status
     update, chat, rumor, hostility, kindness, private thought).
4. **Generated speech**: every interaction (insult, compliment, gossip,
   help…) produces a phrase from a mix of preset templates and contextual
   variables — names, prices, rumors, relics, newspapers, books, any item
   in the game when applicable — that makes sense for the two people involved.
5. **Phrase inspection**: clicking a phrase opens a sub-panel explaining
   *why* they said it — premeditation, genuine kindness, deceit, venting,
   desperation — derived from both agents' traits, relationship, and the
   decision trace.
6. **Visible occupancy**: everyone doing things outside is visible; people
   indoors are shown via their building. Clicking a building lists everyone
   inside and what each is doing.

## Architecture rules for the map

- Terrain, claims, buildings, path wear, and utterances are **canonical sim
  state**, generated deterministically from the world seed. The renderer
  only reads.
- Generated speech is stored on the emitting event (`payload.speech`) with
  its motive analysis, so history is replayable and inspectable.
- Thought bubbles are presentation-layer: derived live from needs +
  latest decision trace, never stored as fake events.

## Extensions to build onto this (proposed)

Near-term (fits stages 9–24):

- **Landmark naming**: lakes, mountains, and forests get seeded procedural
  names at generation ("Cold Lake", "the Bray Pines") shown at regional
  zoom — and agents reference them in speech, so geography enters culture.
- **Path wear feedback loop**: worn routes are cheaper to walk, so trails
  self-reinforce into roads; roads that cross claims become border
  flashpoints; an old road outliving its destination becomes a cultural
  fossil.
- **Construction politics**: site selection inside a rival claim triggers
  disputes; material scarcity creates gathering trips into contested
  forest; a completed building shifts the claim map — architecture as
  politics.
- **Night map**: lit windows show who's home; movement at night is
  suspicious and generates better rumors.
- **Crowd heat**: gatherings (tavern evenings, festivals, mob scenes)
  visibly cluster at settlement zoom — the map's "weather of people".

Later (stages 25+):

- **Building interiors**: a fourth zoom tier — cutaway rooms, furniture
  from materials, agents at tables/beds; interior layout becomes status.
- **Phrase provenance**: utterances can cite a rumor, chronicle article,
  book, or relic as their source; clicking follows the chain (speech →
  rumor → originating event) — the Rumor River surfaced inside dialogue.
- **Terrain memory**: fires leave char tiles, abandoned buildings decay to
  ruins, ruins attract myths and squatters; the map itself accumulates
  history (Culture Autopsy gains physical anchors).
- **Seasonal palette + mechanics**: winter slows construction and gathering,
  freezes the marsh (new paths open), harvest colors autumn; "the One Bad
  Winter" becomes visible.
- **Migration at the edges**: newcomers arrive from the map border, camp
  outside the claims, and petition to settle — assimilation pressure and
  outsider scapegoating made spatial.
- **Border stones as artifacts**: claims are marked by physical stones with
  provenance; moving one is a crime, a rumor seed, and eventually a ritual.
- **Follow camera + cinematic mode**: pin an agent and the camera tracks
  their day; lets the observer "live with" one colonist.
