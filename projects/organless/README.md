# Organless

Interactive computational tissue with a real learned allocation policy. Every surviving cell applies the same 8-input / 3-output softmax network (24 parameters). No cell IDs, coordinates, organ map, or target direction enters the network. A cell allocates a fraction of effort to sensing/feeding, transport, and movement; these are modeled mechanisms, not discovered biological processes.

## Run

`python3 -m http.server 4329 --bind 127.0.0.1 --directory dist`

Open http://127.0.0.1:4329. No application dependencies or build step.

- Cut with the pointer, or use **Remove a region**.
- **Move food** relocates the nutrient source.
- **Compare frozen roles** runs two identical snapshots from the latest injury for 400 steps, one adaptive and one with allocations fixed.
- Views expose dominant role, energy, local signal vectors, and changed roles.
- Keyboard on the canvas: Space pauses, Delete applies the preset injury, C/F select tools.

## Train and verify

`node scripts/train.mjs 70`

`node --test tests/sim.test.mjs`

Training uses deterministic cross-entropy evolutionary search: 18 candidates per generation, 5 elites, 70 generations, seeds 11 and 37, 340-step episodes. The objective is `2 * approach_distance + 30 * mean_energy - 15 * depleted_fraction`. Every training episode removes the same off-center region at step 160. Saved history is measured; the browser runs inference, not live optimization.

`dist/model.json` contains actual weights, seed, generation history, and evaluation. Six held-out seeds change initial energy and food direction; they are evaluated for 420 steps against equal allocation and a policy frozen at injury. This is not a held-out injury-shape benchmark. Arbitrary user cuts are experimental and need not improve with adaptation.

## Model boundaries

The body starts as 149 cells on a circular square lattice. Only immediate orthogonal neighbors exchange energy and chemical signal. An exposed edge samples a synthetic nutrient field and contributes directional sensing. Transport modulates conservative pairwise energy diffusion. A cell's motion contribution depends on its movement allocation, stored energy, and locally propagated signal. The body translates according to their aggregate thrust with velocity smoothing; this is an abstract motility model, not a soft-body mechanics solver. Disconnected pieces remain on the same rigid scaffold.

Feeding and metabolic costs are explicit. Allocation and field updates are synchronous. Energy is bounded to [0,1]; low energy reduces function, but cells do not spontaneously die. Cut cells remain absent: the experiment tests functional reassignment, not anatomical regrowth. Exposed boundaries can increase feeding after a cut; the frozen-role comparison holds the wound geometry fixed to isolate adaptation's contribution. The three functional mechanisms are designed; their spatial allocation is learned.

The preview starts after 140 simulated warm-up steps so the first view shows differentiated tissue. The step counter includes those steps. A source held at the same location is approached and eventually surrounded, so motion slows; move the food to keep exploring.

## Browser agent controls

Optional, feature-detected WebMCP: `read_tissue`, `damage_tissue`, `configure_tissue_view`. Tools share the UI's state and validation.
