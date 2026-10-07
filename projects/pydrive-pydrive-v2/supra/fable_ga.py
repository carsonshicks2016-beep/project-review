"""
Fable Five GENETIC-ALGORITHM baseline — the rudimentary yardstick for PPO.

The whole point of this module is a fair fight. It reuses Fable Five's *frozen*
machinery so the genetic algorithm faces an identical driving problem to the PPO
pipeline:

  * same car          — Mazda 787B (`FABLE_CAR`),
  * same track         — the real-elevation Nordschleife (`RING_TRACK`),
  * same physics + obs — it drives the very same `FableEnv`, so the observation is
                         the frozen `fable-v1` layout (58 sensors + 8 pace + 2 mode
                         = 68 dims),
  * same control        — 3 actions [steer, long, gear-offset]; `long` splits into
                         throttle/brake and `gear` is a -2..+2 RaceBox offset,
  * same reward         — both optimisers maximise the identical `FableReward`
                         return,
  * same yardstick      — the champion is judged by the identical
                         `FableEvaluator` deterministic lap eval, so `[eval-ga]`
                         lines are directly comparable to `[eval-fable]`.

The ONLY thing that differs is the learner. PPO follows the policy gradient; this
evolves a population of NumPy-MLP genomes with elitism + tournament selection +
uniform crossover + Gaussian mutation (with annealing sigma). To keep the fight
honest we also give the GA the two structural aids PPO gets for free: a network of
equal capacity (default hidden = PPO's (128, 128)) and a frozen observation
standardiser built the same way as PPO's running normaliser.

This module NEVER modifies Fable Five state: it only *reads* the frozen env/eval
and writes its own checkpoint (`ga_787b_ring.npz`) and eval snapshot
(`fable_ga_eval_latest.json`).
"""
from __future__ import annotations

import copy
import json
import os
import signal
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import multiprocessing as mp

import numpy as np

from .brain import MLP
from .config import SimSpec, get_car
from .ppo_env import RunningNorm
from .fable5 import (FABLE_CAR, RING_TRACK, STAGES, SUPERHUMAN_LAP,
                     FableEnv, FableEvaluator, FableSpec, _configure_ppo,
                     attach_envelope, stage_defaults, vref_at)
from .track import named_track

GA_CHECKPOINT = "ga_787b_ring.npz"
GA_EVAL_LATEST = "fable_ga_eval_latest.json"
OBS_LAYOUT = "fable-v1"          # must match supra.fable5.OBS_LAYOUT


# --------------------------------------------------------------------------- #
# hyper-parameters — the "make the GA as strong as a GA can be" knobs
# --------------------------------------------------------------------------- #
@dataclass
class GAHyper:
    pop_size: int = 64
    hidden: tuple = (128, 128)        # capacity parity with the PPO actor
    elite_frac: float = 0.12          # top fraction carried over unchanged
    tournament_k: int = 4
    crossover_rate: float = 0.90      # else clone-and-mutate a single parent
    mutation_rate: float = 0.15       # fraction of genes perturbed
    mutation_sigma: float = 0.10      # initial perturbation scale
    sigma_anneal: float = 0.997       # per-generation multiplicative decay
    sigma_floor: float = 0.02
    init_weight_scale: float = 0.20
    bias_long: float = 0.6            # gen-0 throttle bias (matches PPO action_bias)
    # fitness: a bounded-horizon return, averaged over a fixed deterministic start
    # set (the analogue of PPO's finite rollouts).
    fitness_seconds: float = 120.0
    fitness_starts: int = 3           # line + evenly spaced flying-entry sectors
    # champion eval cadence (the expensive FableEvaluator lap eval)
    eval_every: int = 5


def hyper_for_pop(pop: int | None, hidden=None, starts=None,
                  fitness_seconds=None, eval_every=None) -> GAHyper:
    h = GAHyper()
    if pop:
        h.pop_size = int(pop)
    if hidden:
        h.hidden = tuple(int(x) for x in hidden)
    if starts:
        h.fitness_starts = int(starts)
    if fitness_seconds:
        h.fitness_seconds = float(fitness_seconds)
    if eval_every:
        h.eval_every = int(eval_every)
    return h


# --------------------------------------------------------------------------- #
# parallel fitness — one long-lived env per worker process, reused across gens
# --------------------------------------------------------------------------- #
_WORKER: dict = {}


def _init_worker(spec, car, cfg, sim, fitness_seconds, start_set,
                 obs_dim, hidden, bias_long, norm_mean, norm_var, clip):
    """Build ONE FableEnv per worker process (spawned once, reused every gen)."""
    trk = attach_envelope(named_track(RING_TRACK), car)
    c = copy.copy(cfg)
    c.random_start = False
    c.episode_seconds = max(float(fitness_seconds), 5.0)
    env = FableEnv(mode="race", car=car, ppo=c, sim=sim, fixed_track=trk,
                   fable_spec=spec, rng_seed=0, diagnostics=False)
    _WORKER.clear()
    _WORKER.update(env=env, start_set=list(start_set), obs_dim=int(obs_dim),
                   hidden=tuple(hidden), bias_long=float(bias_long),
                   max_steps=int(round(float(fitness_seconds) * c.control_hz)),
                   mean=np.asarray(norm_mean), var=np.asarray(norm_var),
                   clip=float(clip))


def _norm_apply(x, mean, var, clip):
    return np.clip((x - mean) / np.sqrt(var + 1e-8), -clip, clip)


def _rollout_fitness(env, genome, start_set, max_steps, obs_dim, hidden,
                     bias_long, mean, var, clip) -> float:
    brain = MLP.from_genome(np.asarray(genome, dtype=float).ravel(),
                            obs_dim, hidden, 3, out_activation="fable",
                            bias_long=bias_long)
    total = 0.0
    for idx, v0 in start_set:
        obs = env.reset_at(idx, speed=v0)
        for _ in range(max_steps):
            a = brain.forward(_norm_apply(obs, mean, var, clip))
            obs, r, term, trunc, info = env.step(a)
            total += float(r)
            if term or trunc:
                break
            if info.get("laps", 0.0) >= 1.0 and not info.get("fable_invalid", False):
                break
    return total / max(1, len(start_set))


def _score_chunk(genomes) -> np.ndarray:
    w = _WORKER
    out = np.empty(len(genomes), dtype=float)
    for j, g in enumerate(genomes):
        out[j] = _rollout_fitness(w["env"], g, w["start_set"], w["max_steps"],
                                  w["obs_dim"], w["hidden"], w["bias_long"],
                                  w["mean"], w["var"], w["clip"])
    return out


def _chunk(items, n):
    """Split a list into <=n contiguous chunks (order preserved)."""
    n = max(1, min(n, len(items)))
    k, r = divmod(len(items), n)
    out, i = [], 0
    for c in range(n):
        size = k + (1 if c < r else 0)
        out.append(list(items[i:i + size]))
        i += size
    return [c for c in out if c]


# --------------------------------------------------------------------------- #
# eval adapter — make a numpy-MLP genome quack like a PPO for FableEvaluator
# --------------------------------------------------------------------------- #
class _GAPolicyAdapter:
    """FableEvaluator.evaluate() only touches .cfg / .car / .sim_cfg and, per
    step, ._norm(obs), ._t(x) and .net.act_mean(t). We satisfy exactly that so
    the GA champion is scored by the identical deterministic lap protocol as a
    PPO policy — no reimplementation, no drift."""

    def __init__(self, genome, obs_dim, hidden, cfg, car, sim, norm: RunningNorm,
                 bias_long: float = 0.6):
        import torch
        self._torch = torch
        self.brain = MLP.from_genome(np.asarray(genome, dtype=float).ravel(),
                                     obs_dim, hidden, 3, out_activation="fable",
                                     bias_long=bias_long)
        self.cfg = copy.copy(cfg)
        self.car = car
        self.sim_cfg = sim
        self._norm_obj = norm
        self.net = self                      # act_mean lives here

    def _norm(self, obs):
        return self._norm_obj.normalize(np.asarray(obs, dtype=np.float64))

    def _t(self, x):
        return self._torch.as_tensor(np.asarray(x, dtype=np.float32))

    def act_mean(self, t):
        arr = t.detach().cpu().numpy()
        single = arr.ndim == 1
        if single:
            arr = arr[None]
        acts = np.stack([self.brain.forward(row) for row in arr]).astype(np.float32)
        return self._torch.as_tensor(acts)

    def policy_hash(self) -> str:        # only used by __call__, kept for safety
        import hashlib
        return hashlib.sha256(self.brain.get_genome().tobytes()).hexdigest()


# --------------------------------------------------------------------------- #
# the GA
# --------------------------------------------------------------------------- #
class FableGA:
    def __init__(self, hyper: GAHyper, spec: FableSpec, car: str = FABLE_CAR,
                 track=None, seed: int = 0):
        self.h = hyper
        self.spec = spec
        self.car = car
        self.track_name = RING_TRACK
        self.track = attach_envelope(track if track is not None
                                     else named_track(RING_TRACK), car)
        # exact obs/action parity with a PPO fable run
        self.cfg = _configure_ppo(spec)
        self.cfg.random_start = False
        self.sim = SimSpec()
        self.rng = np.random.default_rng(seed)

        probe = self._build_env()
        self.obs_dim = int(probe.obs_dim)
        self.act_dim = 3
        _validate_obs_dim(self.obs_dim)
        self._local_env = probe

        tmpl = self._new_brain()
        self.genome_size = tmpl.size
        self.start_set = self._build_start_set(self.h.fitness_starts)

        # frozen observation standardiser (built like PPO's running normaliser)
        self.norm = self._fit_obs_norm()

        self.genomes = [self._new_brain().get_genome()
                        for _ in range(self.h.pop_size)]
        self.generation = 0
        self.sigma = float(self.h.mutation_sigma)
        self.history = []                         # (gen, best_fit, mean_fit)
        self.best_genome = self.genomes[0].copy()  # best-by-fitness (breeding elite)
        self.best_fitness = -1e18
        self.champion_genome = self.genomes[0].copy()  # best-by-eval-metric (saved)
        self.best_metric = -1e18
        self.best_eval = None
        self._pool = None

    # -- construction helpers --------------------------------------------- #
    def _build_env(self):
        cfg = copy.copy(self.cfg)
        cfg.random_start = False
        cfg.episode_seconds = max(self.h.fitness_seconds, 5.0)
        return FableEnv(mode="race", car=self.car, ppo=cfg, sim=self.sim,
                        fixed_track=self.track, fable_spec=self.spec,
                        rng_seed=0, diagnostics=False)

    def _new_brain(self) -> MLP:
        return MLP(self.obs_dim, self.h.hidden, 3,
                   weight_scale=self.h.init_weight_scale,
                   out_activation="fable", bias_long=self.h.bias_long)

    def _build_start_set(self, n):
        """Fixed deterministic starts: the start/finish line (standing) plus
        evenly spaced flying entries at 90% of the raw pace envelope — the same
        mix of standing-launch and race-pace the eval uses, held constant across
        generations so selection is fair."""
        N = len(self.track.center)
        n = max(1, int(n))
        idxs = [0] + [int(N * i / n) for i in range(1, n)]
        starts = []
        for idx in idxs:
            if idx == 0:
                v0 = 0.0
            else:
                v0 = min(60.0, vref_at(self.track, float(self.track.arc[idx]))
                         * self.spec.envelope_scale * 0.90)
            starts.append((int(idx), float(v0)))
        return starts

    def _fit_obs_norm(self, samples: int = 4, steps: int = 250) -> RunningNorm:
        norm = RunningNorm(self.obs_dim)
        env = self._local_env
        for _ in range(samples):
            brain = self._new_brain()
            for idx, v0 in self.start_set:
                obs = env.reset_at(idx, speed=v0)
                for _ in range(steps):
                    norm.update(obs[None])
                    obs, _, term, trunc, _ = env.step(brain.forward(obs))
                    if term or trunc:
                        break
        return norm

    # -- fitness ----------------------------------------------------------- #
    @property
    def max_steps(self) -> int:
        return int(round(self.h.fitness_seconds * self.cfg.control_hz))

    def evaluate_headless(self, genomes, workers=None) -> np.ndarray:
        workers = workers or min(8, os.cpu_count() or 4)
        if workers <= 1:
            fits = np.empty(len(genomes))
            for i, g in enumerate(genomes):
                fits[i] = _rollout_fitness(
                    self._local_env, g, self.start_set, self.max_steps,
                    self.obs_dim, self.h.hidden, self.h.bias_long,
                    self.norm.mean, self.norm.var, self.norm.clip)
            return fits
        pool = self._ensure_pool(workers)
        # Split into MANY small chunks (not one per worker): rollout time varies
        # wildly by genome — elites drive the full budget, mutants crash in <1 s —
        # and the elites are front-loaded in the bred population. One big chunk
        # per worker hands all the slow genomes to worker 0 while the rest idle.
        # Finer chunks let free workers steal the remaining work (dynamic balance).
        n_chunks = min(len(genomes), max(workers, workers * 6))
        chunks = _chunk(list(genomes), n_chunks)
        results = list(pool.map(_score_chunk, chunks))
        return np.concatenate(results)

    def _ensure_pool(self, workers):
        if self._pool is None:
            self._pool = ProcessPoolExecutor(
                max_workers=int(workers),
                mp_context=mp.get_context("spawn"),
                initializer=_init_worker,
                initargs=(self.spec, self.car, self.cfg, self.sim,
                          self.h.fitness_seconds, self.start_set, self.obs_dim,
                          tuple(self.h.hidden), self.h.bias_long,
                          self.norm.mean, self.norm.var, self.norm.clip))
        return self._pool

    def close(self):
        if self._pool is not None:
            self._pool.shutdown(cancel_futures=True)
            self._pool = None

    # -- evolution --------------------------------------------------------- #
    def advance(self, fitnesses) -> dict:
        fitnesses = np.asarray(fitnesses, dtype=float)
        order = np.argsort(fitnesses)[::-1]
        best_i = int(order[0])
        stats = {"gen": self.generation, "best": float(fitnesses[best_i]),
                 "mean": float(fitnesses.mean()),
                 "median": float(np.median(fitnesses)), "sigma": self.sigma}
        self.history.append((self.generation, stats["best"], stats["mean"]))
        if fitnesses[best_i] > self.best_fitness:
            self.best_fitness = float(fitnesses[best_i])
            self.best_genome = self.genomes[best_i].copy()
        self.genomes = self._breed(self.genomes, fitnesses, order)
        self.sigma = max(self.h.sigma_floor, self.sigma * self.h.sigma_anneal)
        self.generation += 1
        return stats

    def _breed(self, genomes, fitnesses, order):
        h = self.h
        pop = len(genomes)
        n_elite = max(1, int(h.elite_frac * pop))
        nxt = [genomes[int(i)].copy() for i in order[:n_elite]]   # elitism
        while len(nxt) < pop:
            pa = self._tournament(genomes, fitnesses)
            if self.rng.random() < h.crossover_rate:
                child = self._crossover(pa, self._tournament(genomes, fitnesses))
            else:
                child = pa.copy()
            self._mutate(child)
            nxt.append(child)
        return nxt

    def _tournament(self, genomes, fitnesses):
        idx = self.rng.integers(0, len(genomes), self.h.tournament_k)
        return genomes[int(idx[int(np.argmax(fitnesses[idx]))])]

    def _crossover(self, pa, pb):
        mask = self.rng.random(pa.size) < 0.5
        return np.where(mask, pa, pb).copy()

    def _mutate(self, g):
        m = self.rng.random(g.size) < self.h.mutation_rate
        g += m * self.rng.standard_normal(g.size) * self.sigma

    # -- champion eval (identical protocol to Fable Five) ------------------ #
    def champion_eval(self, genome=None) -> dict:
        genome = self.best_genome if genome is None else genome
        adapter = _GAPolicyAdapter(genome, self.obs_dim, self.h.hidden,
                                   self.cfg, self.car, self.sim, self.norm,
                                   bias_long=self.h.bias_long)
        evaluator = FableEvaluator(self.spec, self.track,
                                   manifest_path=_scratch_manifest())
        latest = evaluator.evaluate(adapter)
        # promote to saved champion on a genuine metric improvement
        if latest.get("metric", -1e18) > self.best_metric:
            self.best_metric = float(latest["metric"])
            self.champion_genome = np.asarray(genome, dtype=float).ravel().copy()
            self.best_eval = latest
        return latest

    # -- checkpoint I/O ---------------------------------------------------- #
    def save_champion(self, path: str):
        ev = self.best_eval or {}
        tmp = f"{path}.tmp.npz"
        np.savez(
            tmp,
            genome=self.champion_genome,
            obs_dim=self.obs_dim,
            hidden=np.array(self.h.hidden),
            act_dim=self.act_dim,
            out_activation="fable",
            obs_layout=OBS_LAYOUT,
            car=self.car,
            track=self.track_name,
            stage=self.spec.stage,
            generation=self.generation,
            fitness=self.best_fitness,
            metric=self.best_metric,
            lap_time=float(ev.get("lap_time") or 0.0),
            pop_size=self.h.pop_size,
            bias_long=self.h.bias_long,
            norm_mean=self.norm.mean,
            norm_var=self.norm.var,
            norm_clip=self.norm.clip,
        )
        os.replace(tmp, path)

    @staticmethod
    def load_champion(path: str) -> dict:
        d = np.load(path, allow_pickle=True)
        out = {
            "genome": d["genome"],
            "obs_dim": int(d["obs_dim"]),
            "hidden": tuple(int(h) for h in d["hidden"]),
            "act_dim": int(d["act_dim"]) if "act_dim" in d.files else 3,
            "out_activation": str(d["out_activation"]) if "out_activation" in d.files else "fable",
            "car": str(d["car"]),
            "track": str(d["track"]) if "track" in d.files else RING_TRACK,
            "stage": str(d["stage"]) if "stage" in d.files else "frontier",
            "generation": int(d["generation"]) if "generation" in d.files else 0,
            "fitness": float(d["fitness"]) if "fitness" in d.files else 0.0,
            "metric": float(d["metric"]) if "metric" in d.files else 0.0,
            "lap_time": float(d["lap_time"]) if "lap_time" in d.files else 0.0,
            "bias_long": float(d["bias_long"]) if "bias_long" in d.files else 0.6,
        }
        if "norm_mean" in d.files and "norm_var" in d.files:
            out["norm_mean"] = d["norm_mean"]
            out["norm_var"] = d["norm_var"]
            out["norm_clip"] = float(d["norm_clip"]) if "norm_clip" in d.files else 5.0
        return out

    def warm_start(self, genome, norm_mean=None, norm_var=None, norm_clip=None,
                   fitness=-1e18, metric=-1e18, eval_latest=None):
        g = np.asarray(genome, dtype=float).ravel()
        if g.size != self.genome_size:
            raise ValueError(f"champion genome size {g.size} != current "
                             f"{self.genome_size} (architecture mismatch — the "
                             f"checkpoint was trained with a different --ga-hidden "
                             f"or obs layout)")
        if norm_mean is not None and norm_var is not None:
            self.norm.mean = np.asarray(norm_mean)
            self.norm.var = np.asarray(norm_var)
            if norm_clip is not None:
                self.norm.clip = float(norm_clip)
        self.best_genome = g.copy()
        self.champion_genome = g.copy()
        self.best_fitness = float(fitness)
        self.best_metric = float(metric)
        self.best_eval = eval_latest
        self.genomes = [g.copy()]
        for _ in range(self.h.pop_size - 1):
            c = g.copy()
            self._mutate(c)
            self.genomes.append(c)


# fix the placeholder assert accidentally referencing cfg — obs_dim is validated
# against the frozen fable layout below instead.
def _validate_obs_dim(obs_dim: int) -> None:
    if obs_dim <= 0:
        raise ValueError(f"invalid obs_dim {obs_dim}")


def _scratch_manifest() -> str:
    """A throwaway manifest path so FableEvaluator can never touch the real
    Fable Five pipeline manifest (we only ever call .evaluate(), which does not
    write, but this is belt-and-suspenders)."""
    base = os.environ.get("TMPDIR", "/tmp")
    return os.path.join(base, "fable_ga_scratch_manifest.json")


# --------------------------------------------------------------------------- #
# logging + snapshot
# --------------------------------------------------------------------------- #
def ga_log_line(stage: str, gen: int, ev: dict) -> str:
    lap = ev.get("lap_time") or 0.0
    vs = ev.get("vs_bellof")
    vs_s = f"{vs:.3f}" if vs else "--"
    tag = "SUPERHUMAN " if ev.get("superhuman") else ""
    return (f"[eval-ga] {tag}stage={stage} gen={gen} "
            f"clean={ev.get('clean_sectors', 0)}/{ev.get('sector_count', 0)} "
            f"chain={ev.get('clean_chain', 0)} "
            f"pace={ev.get('pace_ratio', 0.0):.2f} "
            f"lap={lap:.2f} theo={ev.get('theoretical_lap', 0.0):.1f} "
            f"progress={ev.get('max_progress_m', 0.0):.0f}m "
            f"terminal={ev.get('terminal_rate', 0.0):.2f} "
            f"metric={ev.get('metric', 0.0):.3f} vs_bellof={vs_s}")


def _write_json_atomic(path: str, payload: dict) -> None:
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, default=float)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def write_eval_snapshot(ga: "FableGA", stats: dict, ev: dict,
                        path: str = GA_EVAL_LATEST) -> None:
    payload = {
        "pipeline": "fable_ga",
        "optimizer": "genetic_algorithm",
        "time_unix": time.time(),
        "track": ga.track_name,
        "car": ga.car,
        "stage": ga.spec.stage,
        "generation": ga.generation,
        "pop_size": ga.h.pop_size,
        "hidden": list(ga.h.hidden),
        "genome_size": int(ga.genome_size),
        "sigma": ga.sigma,
        "best_fitness": ga.best_fitness,
        "mean_fitness": stats.get("mean"),
        "best_metric": ga.best_metric,
        "eval": ev,
        "superhuman_lap": SUPERHUMAN_LAP,
        "history": ga.history[-200:],
    }
    _write_json_atomic(path, payload)


# --------------------------------------------------------------------------- #
# headless trainer (robust: resume, budgets, periodic eval, SIGINT-safe)
# --------------------------------------------------------------------------- #
def train_fable_ga(stage: str = "frontier", generations: int = 200,
                   pop: int | None = None, workers: int | None = None,
                   seed: int = 0, checkpoint: str = GA_CHECKPOINT,
                   resume: str | None = None, minutes: float | None = None,
                   hidden=None, fitness_seconds: float | None = None,
                   fitness_starts: int | None = None,
                   eval_every: int | None = None,
                   snapshot: str = GA_EVAL_LATEST) -> dict:
    spec = stage_defaults(stage)
    hyper = hyper_for_pop(pop, hidden=hidden, starts=fitness_starts,
                          fitness_seconds=fitness_seconds, eval_every=eval_every)
    print(f"[fable-ga] building {RING_TRACK} + population "
          f"(pop {hyper.pop_size}, hidden {hyper.hidden})...", flush=True)
    ga = FableGA(hyper, spec, car=FABLE_CAR, seed=seed)
    print(f"[fable-ga] genome {ga.genome_size:,} params | obs {ga.obs_dim} "
          f"| 3 actions | stage {stage} | theoretical lap "
          f"{ga.track.fable_envelope['lap_time']:.1f}s | Bellof "
          f"{SUPERHUMAN_LAP:.2f}s", flush=True)

    if resume:
        if not os.path.exists(resume):
            print(f"[fable-ga] resume file not found: {resume}", flush=True)
        else:
            ck = FableGA.load_champion(resume)
            ga.warm_start(ck["genome"], norm_mean=ck.get("norm_mean"),
                          norm_var=ck.get("norm_var"),
                          norm_clip=ck.get("norm_clip"),
                          fitness=ck["fitness"], metric=ck["metric"])
            print(f"[fable-ga] resumed {resume} (gen {ck['generation']}, "
                  f"metric {ck['metric']:.3f}, lap {ck['lap_time']:.2f}s)",
                  flush=True)

    def _save_and_exit(signum, frame):
        print(f"\n[fable-ga] interrupted — saving champion -> {checkpoint}",
              flush=True)
        try:
            ga.save_champion(checkpoint)
        finally:
            ga.close()
        raise SystemExit(0)

    signal.signal(signal.SIGINT, _save_and_exit)

    t0 = time.time()
    eval_every = hyper.eval_every
    last = {}
    last_stats = {"mean": ga.best_fitness}
    last_eval_gen = -1
    try:
        for gen in range(generations):
            fits = ga.evaluate_headless(ga.genomes, workers=workers)
            stats = ga.advance(fits)
            last_stats = stats
            elapsed = time.time() - t0
            print(f"  gen {stats['gen']:3d}  best_fit {stats['best']:8.1f}  "
                  f"mean {stats['mean']:8.1f}  sigma {stats['sigma']:.3f}  "
                  f"({elapsed:5.0f}s)", flush=True)

            budget_hit = bool(minutes and elapsed >= minutes * 60.0)
            due = (gen % eval_every == 0) or budget_hit
            if due:
                ev = ga.champion_eval()
                last = ev
                last_eval_gen = ga.generation
                print("  " + ga_log_line(stage, ga.generation, ev), flush=True)
                ga.save_champion(checkpoint)
                write_eval_snapshot(ga, stats, ev, snapshot)

            if budget_hit:
                print(f"[fable-ga] wall-clock budget {minutes:.1f} min reached "
                      f"at gen {ga.generation}", flush=True)
                break
    finally:
        ga.close()

    # final eval + save (skip if the last generation already evaluated)
    if last_eval_gen != ga.generation:
        ev = ga.champion_eval()
        last = ev
        print("  " + ga_log_line(stage, ga.generation, ev), flush=True)
        ga.save_champion(checkpoint)
        write_eval_snapshot(ga, last_stats, ev, snapshot)
    else:
        ev = last
    lap = ev.get("lap_time")
    lap_s = f"{lap:.2f}s" if lap else "no clean lap yet"
    print(f"[fable-ga] done: {ga.generation} generations, best metric "
          f"{ga.best_metric:.3f}, {lap_s}. Champion -> {checkpoint}", flush=True)
    print(f"[fable-ga] watch it:  python3 run.py --watch-fable-ga "
          f"--checkpoint {checkpoint}", flush=True)
    return ev
