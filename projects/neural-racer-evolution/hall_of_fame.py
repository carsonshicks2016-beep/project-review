import json
import os
from brain import Genome

SAVE_DIR  = "hall_of_fame"
MAX_SAVED = 12


def save(genome, generation, fitness, tag=""):
    os.makedirs(SAVE_DIR, exist_ok=True)
    filename = os.path.join(
        SAVE_DIR,
        f"gen{generation:04d}_fit{int(max(fitness, 0)):08d}{tag}.json"
    )
    with open(filename, "w") as f:
        json.dump(
            {"generation": generation, "fitness": fitness, "genome": genome.to_dict()},
            f, indent=2
        )
    _prune()
    return filename


def load(filepath):
    with open(filepath) as f:
        d = json.load(f)
    return Genome.from_dict(d["genome"]), d["generation"], d["fitness"]


def list_all():
    if not os.path.exists(SAVE_DIR):
        return []
    files = sorted(
        [f for f in os.listdir(SAVE_DIR) if f.endswith(".json")],
        reverse=True,
    )
    return [os.path.join(SAVE_DIR, f) for f in files]


def load_best():
    saved = list_all()
    if not saved:
        return None, 0, 0.0
    return load(saved[0])


def _prune():
    for path in list_all()[MAX_SAVED:]:
        try:
            os.remove(path)
        except OSError:
            pass
