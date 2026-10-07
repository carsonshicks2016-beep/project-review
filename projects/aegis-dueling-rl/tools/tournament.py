"""
Aegis M5 — tournament + ELO between trained checkpoints.

The ELO + round-robin logic is SOLID and reusable. The match runner uses mlagents_envs to drive
a *built* (inference) Unity player and onnxruntime to pick actions per team — this part you must
point at your build and verify team/agent handling matches your scene (see NOTE in play_match).

Usage:
  python tools/tournament.py --env build/AegisDuel.app --checkpoints results/*/Duelist/*.onnx
Requires: mlagents-envs, onnxruntime, numpy.
"""
import argparse, glob, itertools, os
import numpy as np


# ----------------------------- ELO (solid) -----------------------------
def expected(ra, rb):
    return 1.0 / (1.0 + 10 ** ((rb - ra) / 400.0))


def update_elo(ra, rb, score_a, k=32):
    ea = expected(ra, rb)
    ra2 = ra + k * (score_a - ea)
    rb2 = rb + k * ((1 - score_a) - (1 - ea))
    return ra2, rb2


def run_ladder(names, results):
    """results: dict[(i,j)] -> (wins_i, wins_j, draws). Returns sorted (name, elo)."""
    elo = {n: 1200.0 for n in names}
    # replay games in random order a few times for stability
    games = []
    for (i, j), (wi, wj, d) in results.items():
        games += [(i, j, 1.0)] * wi + [(i, j, 0.0)] * wj + [(i, j, 0.5)] * d
    rng = np.random.default_rng(0)
    for _ in range(5):
        rng.shuffle(games)
        for i, j, s in games:
            elo[names[i]], elo[names[j]] = update_elo(elo[names[i]], elo[names[j]], s)
    return sorted(elo.items(), key=lambda kv: -kv[1])


# ----------------------- match runner (adapt to your build) -----------------------
def play_match(env_path, onnx_a, onnx_b, n_games=10, seed=0):
    """
    NOTE: returns (wins_a, wins_b, draws). Implement against YOUR build:
      - load env: UnityEnvironment(file_name=env_path, seed=seed, no_graphics=True)
      - both fighters share behavior 'Duelist'; distinguish the two agents by agent_id/team.
        (Easiest robust option: give the two fighters DIFFERENT behavior names in the duel scene,
         e.g. 'DuelistA'/'DuelistB', so each maps cleanly to one onnx policy.)
      - step loop: get_steps -> infer actions with onnxruntime -> set_actions -> env.step()
      - read terminal rewards to decide the winner of each game.
    The stepping API differs slightly by ml-agents version, so this is left as a clear stub
    rather than version-fragile code.
    """
    import onnxruntime as ort  # noqa: F401  (used in your implementation)
    raise NotImplementedError(
        "Wire play_match to your Unity duel build — see the docstring. ELO/bracket above is ready."
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", required=True, help="path to the built Unity duel player")
    ap.add_argument("--checkpoints", nargs="+", required=True, help="glob(s) of .onnx policies")
    ap.add_argument("--games", type=int, default=10)
    a = ap.parse_args()

    paths = []
    for c in a.checkpoints:
        paths += sorted(glob.glob(c))
    names = [os.path.basename(p) for p in paths]
    if len(paths) < 2:
        raise SystemExit("need >= 2 checkpoints")

    results = {}
    for i, j in itertools.combinations(range(len(paths)), 2):
        wi, wj, d = play_match(a.env, paths[i], paths[j], a.games)
        results[(i, j)] = (wi, wj, d)
        print(f"{names[i]} vs {names[j]}: {wi}-{wj}-{d}")

    print("\n=== ELO ladder ===")
    for rank, (name, elo) in enumerate(run_ladder(names, results), 1):
        print(f"{rank:2d}. {name:40s} {elo:7.1f}")


if __name__ == "__main__":
    main()
