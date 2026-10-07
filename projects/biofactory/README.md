# BioFactory: Neural Ant Logistics Simulator

BioFactory is a 2D real-time biological logistics simulator. The current MVP focuses on one colony, 500 autonomous ant-like agents, multi-resource gathering, storage demand, resource/demand/traffic pheromones, and visible trail formation.

## Run

From `/Users/REVIEW_USER/Documents/New project 2`:

```bash
python3 -m biofactory.main
```

Direct script execution also works:

```bash
python3 biofactory/main.py
```

Run a headless smoke simulation:

```bash
python3 -m biofactory.main --headless --ticks 600
```

Run tests:

```bash
python3 -m unittest discover biofactory/tests
```

Run a render benchmark:

```bash
python3 -B -m biofactory.utils.render_benchmark --ants 500 --frames 60
python3 -B -m biofactory.utils.render_benchmark --ants 10000 --frames 30 --render-only
```

## MVP Controls

- `Space`: pause/play
- `.`: step one tick while paused
- `1`, `2`, `3`, `4`, `5`: simulation speed 1x, 2x, 5x, 20x, 100x
- `F`: toggle food pheromone overlay
- `I`: toggle water pheromone overlay
- `O`: toggle protein pheromone overlay
- `V`: toggle waste pheromone overlay
- `D`: toggle demand pheromone overlay
- `T`: toggle traffic pheromone overlay
- `B`: cycle visual mode: hybrid, circuit, natural
- `P`: cycle render quality: fast, balanced, quality
- `R`: toggle resource source glow
- `H`: toggle traffic heatmap tint
- `C`: center camera on colony
- `WASD` or arrow keys: pan
- Mouse wheel: zoom
- Left click: inspect nearest ant/cell

## Visual Modes

- `Hybrid`: default view; natural terrain with readable bio-circuit logistics.
- `Circuit`: dark factory-map view that emphasizes pheromone highways, demand, and traffic.
- `Natural`: quieter ecosystem view with reduced data glow and stronger terrain/resource texture.

## Current Scope

Implemented:

- Grid terrain with movement costs and obstacle cells.
- One colony with a queen/storage center and resource-specific demand pressure.
- Seeded typed storage chambers for leaves, food, water, protein, and waste.
- 500 continuous-position ants with lightweight neural brains plus local pheromone/resource/nest signals.
- Leaves, water, protein, and waste resource sources, pickup, carrying, and storage deposit.
- Food, water, protein, waste, demand, and traffic pheromone layers with diffusion and decay.
- Demand states with shortage urgency, critical boosts, saturation penalties, bottleneck alerts, and resource target priorities.
- Chamber-owned inventory with aggregate colony totals, chamber-local demand, typed cargo dropoff, overflow fallback, and chamber inspection.
- Resource-specific cargo colors, source glows, trail colors, colony storage totals, and cell inspection values.
- Congestion pressure from traffic pheromone.
- Pygame renderer with terrain, resources, glowing pheromones, ants, cargo, storage, stats, controls, and overlays.
- Procedural texture layer, ant LOD rendering, visual modes, cargo glints, and additive pheromone glow.
- Headless mode for long simulation runs.
- Unit tests for pheromone decay, demand calculation, inventory transfer, and resource pickup/drop mechanics.

Next major systems:

- Basic processing chains: fungus farm, nutrient processing, protein processing, composting.
- First production chambers that consume from typed storage and create processed outputs.

Deferred until after the logistics core stabilizes:

- Predators, rival colonies, weather, seasons, warfare, disease, replay, and complex production chains.
