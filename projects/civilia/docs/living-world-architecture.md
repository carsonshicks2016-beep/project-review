# Civilia: The Living World Architecture (Stage 13+)

Carson's architectural vision (2026-06-11). This is the governing document
for the semantic zoom system, contextual dialogue, and supporting pillars.
Implementation status is tracked inline and in the
[coverage ledger](coverage-ledger.md).

## 1. The 3-Tier Semantic Zoom System

The camera doesn't just scale graphics; it changes information density and
the interaction layer — Google Earth into Street View.

### Tier 1: Macro View (The God Map)
- Procedural cartographic map: continents, oceans, mountains, biomes — **live (v0: biome canvas + named landmarks)**
- Political/cultural layer: filled border regions for claims of cults/governments/clubs — **live (household claims, strength-field borders; Voronoi-equivalent)**
- High-level god powers: migrations, weather, trade routes — *planned (stage 44 meta-tools)*
- **Fog of Subjectivity**: toggle to a faction's perspective; unexplored areas blank or "here be dragons" — *planned; needs per-household explored-tile tracking (awaiting visual direction)*

### Tier 2: Meso View (The Settlement)
- Buildings color-coded by state: Planned (blueprints), In-Progress (scaffolding), Completed — **live**
- Material-Based Architecture: appearance follows local biome/materials (adobe by desert lakes, logs and stone in the north) — *partially: recipes already adapt to local terrain; visual palette awaiting direction*
- Zoning, stockpiles, bottleneck observation — *planned (stage 18/19 deepening)*

### Tier 3: Micro View (The Ant Farm)
- Agents at building-proportional scale — **live**
- **Dollhouse Mode**: click/hover fades the roof for an X-ray cutaway of the interior — *planned (awaiting visual direction; occupant data already live via BuildingCard)*
- Speech and thought bubbles — **live**
- Click a bubble → Contextual Reasoning — **live**

## 2. The Dynamic Contextual Dialogue Engine

No generic barks. The line synthesizes from the exact simulation context:

1. **The Trigger**: Agent A decides to insult Agent B — **live**
2. **The Query**: A's memory graph is queried for the strongest negative
   edge to B — **live** (`peekMemories` by tone, falls back to damaging beliefs)
3. **The Generation**: context → templating engine (LLM pass at stage 30) — **live**
4. **The Output**: *"I saw you with my own eyes this very day, Mara. Once a
   thief, always a thief."* — **live, verified**

### The "Why?" Inspector (Contextual Reasoning UI) — **live**
- Motivation: deceit / premeditation / genuine kindness / venting / performative / desperate
- Historical context: the cited memory with the exact day it formed, days
  elapsed, dwell count, and the origin event (type + day + phase)
- Influencing knowledge: the cited belief with confidence at speaking time,
  source (witnessed/rumor/inference), and who told them — citations are
  **speaking-time snapshots**, so the receipt survives even after the
  memory itself fades
- For gossip: "How it landed" — the listener's belief-update factors
- Influencing items: *planned (needs Relics & Texts)*

## 3. Complementary Systems

### A. The "Relics & Texts" System — *planned (stage 25/41 pull-forward candidate)*
- Newspapers: Chronicle issues drop as physical items; readers update biases
  (needs the stage 29 Ollama gateway to fulfill articles first, or
  deterministic article stubs)
- Relics: unique items from historical events ("The Bloody Pitchfork of the
  Bread Riot"); holding grants status, makes enemies hostile, and feeds the
  dialogue engine's "influencing items"

### B. The Rumor River Overlay — **live (v0)**
- Lens toggle in the map HUD: recent tellings render as glowing arcs with
  traveling particles between teller and listener; faithful retellings
  purple, **mutations (deceit) pink**, particle size/speed scales with the
  rumor's emotional charge; terrain desaturates while the lens is on
- Full who-told-whom spread graph persistence → stage 17 (Rumor River proper)

### C. The Needs-Based Build AI — **live (v0)**
- Triggers: persistent bread shortages → granary; household crowding +
  pooled wealth → cottage
- Blueprint placed on buildable ground near the need; idle laborers and the
  unemployed pathfind to the site, gather local materials (recipes adapt to
  what the terrain offers), and raise it shift by shift
- Aggregate-need triggers (town hunger thresholds) and a mayor role →
  stages 18/32
