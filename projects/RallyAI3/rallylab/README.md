# Rally Research Lab

The local dashboard manages experiments and the Unity player. Start it from the project root with `./start-rally-lab.sh`, then open `http://127.0.0.1:8765`. The dashboard stays available while the browser is closed; training supervisors save their state under `.rally/`.

## First setup

1. Install Python dependencies with `.venv/bin/python -m pip install -r rallylab/requirements.txt` and dashboard dependencies with `npm --prefix dashboard install`.
2. Open the project in Unity 6000.4.11f1 once and wait for script compilation to finish. Unity must have an active editor license; batch mode without that license cannot build the player.
3. Start Rally Research Lab and select **Prepare player**. The Unity Editor bridge builds the player and writes a source/build and driving-contract manifest.
4. In Courses, generate the six-course starter library. Review its previews before using it for benchmark claims.
5. Create a specialist or generalist PPO experiment. Stop asks ML-Agents to finish and save; resume uses the same run directory and saved trainer state.
6. Prepare a frozen checkpoint before watching or evaluating it. Evaluation budgets default to 20 attempts for each selected course.

## Records and artifacts

Managed runs, manifests, model snapshots, and course bundles live in `.rally/`, which is machine-local and ignored by Git. Export experiment JSON or CSV from an experiment detail view. Existing `results/` entries are reported as legacy because they lack the managed course/build lineage.

Evaluation records use fixed simulation ticks, a directional start gate, ordered directional waypoint gates, and a finish gate. A run that leaves the legal road corridor is retained as an attempt but excluded from valid time records. Time-attack reward versions are separately labeled; they do not change the baseline PPO configuration.

## Limitations of this milestone

Generated road meshes and collision geometry are frozen in Unity asset bundles. The initial course families adjust the existing procedural generator and are not hand-authored rally stages. Specialist training, procedural generalist training, checkpoint packaging, and course-by-course evaluation are supported; a human driving benchmark and expert-authored tracks are not included.
