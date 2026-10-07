# 3D Asset Credits

## Supra A80 hero car (`viewer3d/static/assets/supra_a80.glb`)

Based on **"Toyota Supra Mk4(A80)"** by **G1itchz**, licensed
**CC BY 4.0** (Creative Commons Attribution).

- Source: https://sketchfab.com/3d-models/toyota-supra-mk4a80-2f5a0eaf72e446a39b32604ffc833ad5
- License: https://creativecommons.org/licenses/by/4.0/
- Modifications (by this project): deleted the bundled ground/dummy planes,
  recolored the body paint to a metallic blue + gunmetal wheels, consolidated
  each wheel into a single `wheel_FL/FR/RL/RR` node with a hub-centered origin,
  flattened the Sketchfab transform wrapper, re-exported as glb.
  Processing script: `scratch/carmodel/process_sketchfab.py`.

The earlier hand-modeled voxel/parametric Supra (`supra_voxel.js`, and the
`scratch/carmodel/blockout.py` experiment) is original project work, no
attribution required.
