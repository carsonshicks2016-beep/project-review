# CRUMPLE

A soft-body driving sandbox: outrun pursuit cars through a procedural city in a
car that genuinely deforms. No physics engine — the solver is written from
scratch, because the deformation model is the whole point.

## Run it

ES modules need a real origin, so `file://` won't work — serve the folder:

```sh
cd ~/physics-sandbox
python3 -m http.server 8123
```

Then open <http://localhost:8123>. Click to start (the click also unlocks audio).

## Controls

| Key | |
|---|---|
| `W` `A` `S` `D` / arrows | drive (`S` brakes, then reverses) |
| `Space` | handbrake — locks the rear, kicks the tail out |
| `C` | camera: chase → bumper → overhead |
| `B` | beam view — see the lattice, green intact → yellow bent → red broken |
| `T` | respawn: repairs the car and drops it on a random road |
| `R` | reset the whole run |

Survive as long as you can. A new pursuer joins every 24s up to seven. At 72%
chassis damage the engine dies and the run ends.

## How the deformation works

Each car is 20 chassis nodes + 4 wheel nodes wired by ~166 spring-damper
*beams*, solved at 1500 Hz. Past a strain threshold a beam **yields**: its rest
length migrates toward its current length, so the dent is permanent. Past a
higher threshold it snaps. The rendered body's vertices *are* the chassis nodes,
so the visible shape is the simulation state — there is no separate damage model
or swapped-in "damaged mesh".

Two things turned out to matter far more than expected, and both were found by
measuring rather than guessing:

- **Connectivity must be sparse.** The first version connected every node pair.
  That truss is effectively incompressible — an impact spreads strain over every
  beam at once, so none reaches yield and the car bounces off walls perfectly
  intact. Local beams now handle crumple; a short list of explicit long rails
  (`RAILS` in `vehicle.js`) carries global shape.
- **Plastic rate is what separates a crumple zone from a spring.** Stiffness sets
  the peak force; the *yield rate* decides whether that energy is absorbed or
  handed back. Too slow and the car stores the whole impact elastically and
  catapults itself backward.

Yield thresholds are deliberately set well above any driving load (~17 kN) and
well below crash loads. Set them too low and the car destroys itself under its
own braking — which it did, at first.

### Measured crash response

Head-on into a building, from rest afterward:

| Impact | Nose crush | Beams bent | Damage |
|---|---|---|---|
| 20 mph | 1 cm | 0 | 3% |
| 40 mph | 5 cm | 14 | 18% |
| 49 mph | 9 cm | 18 | 28% |
| 69 mph | 17 cm | 32 | 53% |
| 89 mph | 27 cm | 42 | 84% — totaled |

## Layout

| File | |
|---|---|
| `src/physics.js` | node/beam solver, plastic yield, collision, broadphase |
| `src/vehicle.js` | chassis lattice, body skin, tire & engine model |
| `src/city.js` | procedural road grid, buildings, barriers |
| `src/ai.js` | pursuit steering with whisker obstacle probes |
| `src/audio.js` | fully synthesized engine, impacts, tire squeal (no assets) |
| `src/main.js` | rendering, camera, HUD, game loop |

Cost is about 0.9 ms/frame of physics for six cars, and ~6.8k triangles — most
of the budget is free for more cars or a bigger city.

## Poking at it

`window.CRUMPLE` is exposed for stepping the sim without the render loop:

```js
CRUMPLE.simulate(3)   // advance 3 seconds of physics headlessly
CRUMPLE.stats()       // speed, damage, bent/broken beams, NaN check
```

Useful knobs: beam stiffness/yield in `vehicle.js` (`build()`), `SUBSTEP_HZ` in
`physics.js`, and `MAX_CHASERS` / `ESCALATE_EVERY` in `main.js`.
