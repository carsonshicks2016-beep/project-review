"""Held-out evaluation harness (North Star §10 / roadmap D1 + D4).

Given a fixed seed set, run deterministic episodes and report completion,
cleanliness, terminations, time vs theoretical minimum, and per-sector splits.

Determinism contract
--------------------
* Actions are **mean** actions — never sampled. Sampling inflates variance and
  makes runs incomparable.
* The observation normaliser is **frozen**. Updating it during eval makes
  results depend on eval order.

Actors before a trained policy exists
-------------------------------------
* ``ReferencePilotActor`` — drivability smoke path (NOT a baseline for claims).
* ``ConstantMeanActor`` — fixed mean action for ultra-cheap harness tests.
* ``ObsMeanActor`` — wraps anything with ``mean_action(obs)`` (C1's policy).

What C1 / C4 must provide
-------------------------
* **C1** (``rallyai.train.policy``): a policy exposing
  ``mean_action(obs: np.ndarray) -> np.ndarray`` of shape ``(4,)``. Sampling
  belongs in training rollouts only; eval never calls it.
* **C4** (``rallyai.train.checkpoint``): ``load_for_eval(path)`` returning
  ``(policy, normaliser, meta)`` where ``meta["has_normaliser"]`` is True and
  the normaliser is frozen (``normalize`` only).

Optimal times (D2)
------------------
If ``rallyai.stage.optimal`` exists it is used; otherwise callers may inject
``optimal_time_s`` / ``optimal_sector_times_s``. Missing optima become ``null``
in the report rather than inventing a number.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np

from rallyai.env import EnvConfig, RallyEnv
from rallyai.env.pilot import ReferencePilot
from rallyai.stage.generator import generate

# --------------------------------------------------------------------------- #
# Seed reservation — training and eval pools are disjoint by construction
# --------------------------------------------------------------------------- #

EVAL_SEED_MODULUS = 10
EVAL_SEED_RESIDUE = 7


def is_eval_seed(seed: int) -> bool:
    """Held-out seeds satisfy ``seed % 10 == 7``."""
    return int(seed) % EVAL_SEED_MODULUS == EVAL_SEED_RESIDUE


def is_train_seed(seed: int) -> bool:
    """Training may draw any seed that is not reserved for evaluation."""
    return not is_eval_seed(seed)


def held_out_seeds(n: int, *, start: int = EVAL_SEED_RESIDUE) -> list[int]:
    """First ``n`` reserved eval seeds: 7, 17, 27, … (or ``start`` if valid)."""
    if n < 0:
        raise ValueError("n must be non-negative")
    seed = int(start)
    if not is_eval_seed(seed):
        raise ValueError(
            f"start={start} is not an eval seed "
            f"(need seed % {EVAL_SEED_MODULUS} == {EVAL_SEED_RESIDUE})"
        )
    return [seed + EVAL_SEED_MODULUS * i for i in range(n)]


def assert_train_eval_disjoint(train_seeds: list[int], eval_seeds: list[int]) -> None:
    """Hard guard: any overlap means the harness is measuring memorisation."""
    overlap = sorted({int(s) for s in train_seeds} & {int(s) for s in eval_seeds})
    if overlap:
        raise AssertionError(f"train/eval seed pools overlap: {overlap}")
    bad_eval = [s for s in eval_seeds if not is_eval_seed(s)]
    if bad_eval:
        raise AssertionError(f"eval seeds outside reserved residue: {bad_eval}")
    bad_train = [s for s in train_seeds if not is_train_seed(s)]
    if bad_train:
        raise AssertionError(f"train seeds collide with reserved eval residue: {bad_train}")


# --------------------------------------------------------------------------- #
# Protocols — C1 / B3 / C4 fill these in; stubs cover the gap until then
# --------------------------------------------------------------------------- #

@runtime_checkable
class EvalPolicy(Protocol):
    """C1 must provide this. Eval calls ``mean_action`` only — never sample."""

    def mean_action(self, obs: np.ndarray) -> np.ndarray: ...


@runtime_checkable
class ObservationNormaliser(Protocol):
    """B3 / C4 normaliser. Must be frozen for the duration of an evaluation."""

    def normalize(self, obs: np.ndarray) -> np.ndarray: ...


class FrozenIdentityNormaliser:
    """Pass-through normaliser that refuses updates — the eval default.

    B3's real normaliser serialises ``(mean, var, count)`` beside the weights.
    Until that exists, identity + freeze is the honest smoke-test stand-in:
    it cannot silently drift mid-eval.
    """

    frozen: bool = True

    def normalize(self, obs: np.ndarray) -> np.ndarray:
        return np.asarray(obs, dtype=np.float32)

    def update(self, obs: np.ndarray) -> None:
        raise RuntimeError(
            "Observation normaliser is frozen during evaluation. "
            "Updating mid-eval makes results depend on episode order (roadmap B3/D1)."
        )


class ObsMeanActor:
    """Policy-mean actor: freeze normaliser, never sample."""

    def __init__(
        self,
        policy: EvalPolicy,
        normaliser: ObservationNormaliser | None = None,
        *,
        name: str = "policy_mean",
    ):
        self.policy = policy
        self.normaliser = normaliser or FrozenIdentityNormaliser()
        self.name = name

    def action(self, env: RallyEnv, obs: np.ndarray) -> np.ndarray:
        x = self.normaliser.normalize(obs)
        a = np.asarray(self.policy.mean_action(x), dtype=np.float32).reshape(-1)
        return a


class ReferencePilotActor:
    """Reference-pilot path for smoke tests. NOT a baseline for any claim."""

    name = "reference_pilot"

    def __init__(self) -> None:
        self._pilot: ReferencePilot | None = None
        self._track_id: int | None = None

    def action(self, env: RallyEnv, obs: np.ndarray) -> np.ndarray:
        del obs  # pilot reads the track/car, not the observation vector
        tid = id(env.track)
        if self._pilot is None or self._track_id != tid:
            self._pilot = ReferencePilot(env.track)
            self._track_id = tid
        return self._pilot.act(env.query, env.car)


class ConstantMeanActor:
    """Fixed mean action — harness wiring tests without a competent driver."""

    name = "constant_mean"

    def __init__(self, action: np.ndarray | None = None):
        if action is None:
            action = np.array([0.0, 0.35, 0.0, 0.0], dtype=np.float32)
        self._action = np.asarray(action, dtype=np.float32).reshape(4)

    def action(self, env: RallyEnv, obs: np.ndarray) -> np.ndarray:
        del env, obs
        return self._action.copy()


@runtime_checkable
class EpisodeActor(Protocol):
    name: str

    def action(self, env: RallyEnv, obs: np.ndarray) -> np.ndarray: ...


# --------------------------------------------------------------------------- #
# Optimal-time interface (D2) — imports stage/optimal when present
# --------------------------------------------------------------------------- #

OptimalTimeFn = Callable[[dict[str, Any]], float | None]
OptimalSectorTimesFn = Callable[[dict[str, Any], list[dict[str, Any]]], list[float | None]]


def _import_optimal_stage_time() -> OptimalTimeFn | None:
    try:
        from rallyai.stage import optimal as opt
    except ImportError:
        return None
    for name in (
        "theoretical_minimum_time",  # D2
        "optimal_time_s",
        "optimal_time",
        "stage_optimal_time",
    ):
        fn = getattr(opt, name, None)
        if callable(fn):
            return fn  # type: ignore[return-value]
    return None


def _sector_times_from_profile(
    stage: dict[str, Any],
    sectors: list[dict[str, Any]],
) -> list[float | None] | None:
    """Integrate D2's speed envelope between archetype boundaries."""
    try:
        from rallyai.stage.optimal import theoretical_minimum_profile
        from rallyai.track import Track
    except ImportError:
        return None
    profile = theoretical_minimum_profile(stage)
    track = Track(stage)
    s = np.asarray(track.s, dtype=np.float64)
    v = np.asarray(profile["v"], dtype=np.float64)
    if s.size < 2 or v.size != s.size:
        return None
    # Match D2's integrator: charge ds[i] to sample i's speed.
    ds = np.diff(s)
    dt = ds / np.maximum(v[:-1], 1e-6)
    cum = np.concatenate([[0.0], np.cumsum(dt)])

    def _t_at(s_query: float) -> float:
        return float(np.interp(s_query, s, cum, left=0.0, right=float(cum[-1])))

    return [_t_at(float(sec["s_end"])) - _t_at(float(sec["s_start"])) for sec in sectors]


def _import_optimal_sector_times() -> OptimalSectorTimesFn | None:
    try:
        from rallyai.stage import optimal as opt
    except ImportError:
        return None
    for name in ("optimal_sector_times_s", "optimal_sector_times", "sector_optimal_times"):
        fn = getattr(opt, name, None)
        if callable(fn):
            return fn  # type: ignore[return-value]

    # Fallback: derive sector splits from a whole-stage profile.
    if hasattr(opt, "theoretical_minimum_profile"):
        def _from_profile(
            stage: dict[str, Any], sectors: list[dict[str, Any]],
        ) -> list[float | None]:
            times = _sector_times_from_profile(stage, sectors)
            return times if times is not None else [None] * len(sectors)

        return _from_profile
    return None


# --------------------------------------------------------------------------- #
# Human baseline (D3) — offline path until F4
# --------------------------------------------------------------------------- #

_DEFAULT_HUMAN_BASELINE = (
    Path(__file__).resolve().parents[4] / "docs" / "baselines" / "human" / "times.json"
)


def load_human_baseline(
    path: str | Path | None = None,
) -> dict[tuple[int, int], dict[str, Any]]:
    """Load ``docs/baselines/human/times.json`` keyed by ``(seed, tier)``.

    Empty ``seeds`` (no drives yet) returns ``{}``. Not claimable until Carson
    fills finished keyboard times.
    """
    p = Path(path) if path is not None else _DEFAULT_HUMAN_BASELINE
    if not p.exists():
        return {}
    data = json.loads(p.read_text(encoding="utf-8"))
    out: dict[tuple[int, int], dict[str, Any]] = {}
    for row in data.get("seeds") or []:
        out[(int(row["seed"]), int(row["tier"]))] = dict(row)
    return out


def resolve_optimal_time(
    stage: dict[str, Any],
    *,
    optimal_time_s: float | None = None,
    optimal_fn: OptimalTimeFn | None = None,
) -> float | None:
    """Return a theoretical minimum, or None if neither injection nor D2 exist."""
    if optimal_time_s is not None:
        return float(optimal_time_s)
    fn = optimal_fn if optimal_fn is not None else _import_optimal_stage_time()
    if fn is None:
        return None
    value = fn(stage)
    return None if value is None else float(value)


def resolve_optimal_sector_times(
    stage: dict[str, Any],
    sectors: list[dict[str, Any]],
    *,
    optimal_sector_times_s: list[float] | None = None,
    optimal_fn: OptimalSectorTimesFn | None = None,
) -> list[float | None]:
    if optimal_sector_times_s is not None:
        if len(optimal_sector_times_s) != len(sectors):
            raise ValueError(
                f"optimal_sector_times_s length {len(optimal_sector_times_s)} "
                f"!= sectors {len(sectors)}"
            )
        return [float(t) for t in optimal_sector_times_s]
    fn = optimal_fn if optimal_fn is not None else _import_optimal_sector_times()
    if fn is None:
        # If only a whole-stage optimum exists, leave per-sector null.
        return [None] * len(sectors)
    values = list(fn(stage, sectors))
    if len(values) != len(sectors):
        raise ValueError("optimal sector-time function returned wrong length")
    return [None if v is None else float(v) for v in values]


# --------------------------------------------------------------------------- #
# Sectors (D4)
# --------------------------------------------------------------------------- #

def sectors_for_stage(stage: dict[str, Any]) -> list[dict[str, Any]]:
    """Archetype-boundary sectors from the generator, or one whole-stage sector."""
    meta = stage.get("meta") or {}
    raw = meta.get("sectors")
    if raw:
        out = []
        for sec in raw:
            out.append({
                "name": str(sec["name"]),
                "s_start": float(sec["s_start"]),
                "s_end": float(sec["s_end"]),
            })
        return out
    length = float(stage.get("length_m") or 0.0)
    return [{"name": "whole", "s_start": 0.0, "s_end": length}]


def _sector_times_from_crossings(
    sectors: list[dict[str, Any]],
    crossings: list[float | None],
    final_time_s: float,
    finished: bool,
) -> list[float | None]:
    """``crossings[i]`` is the clock when the car reached ``sectors[i].s_start``.

    Length is ``len(sectors) + 1``; the last entry is the clock at the final
    ``s_end`` (or None if the car never got there).
    """
    times: list[float | None] = []
    for i, _sec in enumerate(sectors):
        t0 = crossings[i]
        t1 = crossings[i + 1]
        if t0 is None:
            times.append(None)
            continue
        if t1 is not None:
            times.append(float(t1 - t0))
        elif finished and i == len(sectors) - 1:
            times.append(float(final_time_s - t0))
        else:
            # Partial sector — still report time spent inside it.
            times.append(float(final_time_s - t0))
    return times


# --------------------------------------------------------------------------- #
# Checkpoint loading hook (C4)
# --------------------------------------------------------------------------- #

def load_actor_from_checkpoint(checkpoint_path: str | Path) -> tuple[EpisodeActor, dict[str, Any]]:
    """Load a mean-action actor + frozen normaliser from a C4 checkpoint."""
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"checkpoint not found: {path}")
    try:
        from rallyai.train.checkpoint import load_for_eval
    except ImportError as exc:
        raise RuntimeError(
            "Checkpoint evaluation requires C4: "
            "rallyai.train.checkpoint.load_for_eval(path) -> "
            "(policy, normaliser, meta) with policy.mean_action and a frozen "
            "normaliser."
        ) from exc
    try:
        policy, normaliser, meta = load_for_eval(path)
    except Exception as exc:
        raise RuntimeError(f"failed to load checkpoint for eval: {exc}") from exc
    if not meta.get("has_normaliser", False):
        raise RuntimeError(
            "Checkpoint meta.has_normaliser is false — a policy restored without "
            "its observation normaliser is a different policy."
        )
    if hasattr(normaliser, "frozen"):
        normaliser.frozen = True
    elif hasattr(normaliser, "freeze"):
        normaliser.freeze()
    actor = ObsMeanActor(policy, normaliser, name=f"checkpoint:{path.name}")
    return actor, dict(meta)


# --------------------------------------------------------------------------- #
# Episode + aggregate evaluation
# --------------------------------------------------------------------------- #

TERMINATION_KEYS = ("finish", "crash", "off_course", "spun", "stuck", "timeout", "other")


@dataclass
class EpisodeResult:
    seed: int
    tier: int
    termination: str
    time_s: float
    finished: bool
    clean: bool
    off_course_s: float
    obstacle_contacts: int
    optimal_time_s: float | None
    time_vs_optimal: float | None
    human_best_s: float | None = None
    time_vs_human: float | None = None
    sectors: list[dict[str, Any]] = field(default_factory=list)
    deaths: list[dict[str, Any]] = field(default_factory=list)
    behavior: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "tier": self.tier,
            "termination": self.termination,
            "time_s": self.time_s,
            "finished": self.finished,
            "clean": self.clean,
            "off_course_s": self.off_course_s,
            "obstacle_contacts": self.obstacle_contacts,
            "optimal_time_s": self.optimal_time_s,
            "time_vs_optimal": self.time_vs_optimal,
            "human_best_s": self.human_best_s,
            "time_vs_human": self.time_vs_human,
            "sectors": self.sectors,
            "deaths": self.deaths,
            "behavior": self.behavior,
        }


def run_episode(
    stage: dict[str, Any],
    actor: EpisodeActor,
    *,
    seed: int,
    tier: int,
    env_config: EnvConfig | None = None,
    optimal_time_s: float | None = None,
    optimal_sector_times_s: list[float] | None = None,
    human_best_s: float | None = None,
) -> EpisodeResult:
    """Run one deterministic episode; sector clocks advance on arc-length crossings."""
    cfg = env_config or EnvConfig()
    cfg.record_frames = True
    env = RallyEnv(stage, cfg)
    obs, _ = env.reset(seed=seed)
    sectors = sectors_for_stage(stage)
    # crossings[i] = time at sectors[i].s_start; crossings[-1] = time at last s_end
    crossings: list[float | None] = [0.0] + [None] * len(sectors)
    next_boundary = 0  # index into sector s_end list (== crossings index - 0 for start done)

    terminated = truncated = False
    info: dict[str, Any] = {}
    
    steering_reversals = 0
    last_steer_sign = 0
    throttle_frames = 0
    brake_frames = 0
    total_frames = 0
    
    while not (terminated or truncated):
        action = actor.action(env, obs)
        
        steer = float(action[0])
        throttle = float(action[1])
        brake = float(action[2])
        
        steer_sign = 1 if steer > 0.05 else (-1 if steer < -0.05 else 0)
        if steer_sign != 0 and last_steer_sign != 0 and steer_sign != last_steer_sign:
            steering_reversals += 1
        if steer_sign != 0:
            last_steer_sign = steer_sign
            
        if throttle > 0.05:
            throttle_frames += 1
        if brake > 0.05:
            brake_frames += 1
        total_frames += 1

        obs, _, terminated, truncated, info = env.step(action)
        s = float(info["s"])
        t = float(info["time_s"])
        while next_boundary < len(sectors) and s >= float(sectors[next_boundary]["s_end"]):
            crossings[next_boundary + 1] = t
            next_boundary += 1

    termination = str(info.get("termination") or "other")
    finished = bool(info.get("finished")) or termination == "finish"
    time_s = float(info.get("time_s", env.time_s))
    if finished and crossings[-1] is None:
        crossings[-1] = time_s
        
    deaths = []
    if termination == "crash":
        crash_time = time_s
        trace = []
        if hasattr(env, "frames"):
            for frame in env.frames:
                if frame["t"] >= crash_time - 2.0:
                    trace.append(frame)
        
        crash_s = float(info.get("s", 0.0))
        crash_x = float(trace[-1]["x"]) if trace else 0.0
        crash_y = float(trace[-1]["y"]) if trace else 0.0
        
        deaths.append({
            "s": crash_s,
            "x": crash_x,
            "y": crash_y,
            "trace": trace
        })
        
    dist_km = float(info.get("s", 0.0)) / 1000.0
    behavior = {
        "steering_reversals_per_km": (steering_reversals / dist_km) if dist_km > 0.001 else 0.0,
        "throttle_ratio": (throttle_frames / total_frames) if total_frames > 0 else 0.0,
        "brake_ratio": (brake_frames / total_frames) if total_frames > 0 else 0.0,
    }

    # Prefer the env's per-stage bound when the caller did not inject one.
    opt = resolve_optimal_time(stage, optimal_time_s=optimal_time_s)
    if opt is None and getattr(env, "optimal_time_s", None) is not None:
        opt = float(env.optimal_time_s)
    ratio = (time_s / opt) if (opt is not None and opt > 0.0) else None
    human_ratio = (
        (time_s / human_best_s)
        if (human_best_s is not None and human_best_s > 0.0 and finished)
        else None
    )

    sector_opt = resolve_optimal_sector_times(
        stage, sectors, optimal_sector_times_s=optimal_sector_times_s,
    )
    sector_times = _sector_times_from_crossings(sectors, crossings, time_s, finished)
    sector_rows = []
    for sec, t_sec, t_opt in zip(sectors, sector_times, sector_opt, strict=True):
        row = {
            "name": sec["name"],
            "s_start": sec["s_start"],
            "s_end": sec["s_end"],
            "time_s": t_sec,
            "optimal_time_s": t_opt,
            "time_vs_optimal": (
                (t_sec / t_opt) if (t_sec is not None and t_opt is not None and t_opt > 0)
                else None
            ),
        }
        sector_rows.append(row)

    return EpisodeResult(
        seed=int(seed),
        tier=int(tier),
        termination=termination,
        time_s=time_s,
        finished=finished,
        clean=bool(info.get("clean", False)),
        off_course_s=float(info.get("off_course_s", 0.0)),
        obstacle_contacts=int(info.get("obstacle_contacts", 0)),
        optimal_time_s=opt,
        time_vs_optimal=ratio,
        human_best_s=human_best_s,
        time_vs_human=human_ratio,
        sectors=sector_rows,
        deaths=deaths,
        behavior=behavior,
    )


def _termination_breakdown(episodes: list[EpisodeResult]) -> dict[str, int]:
    counts = {k: 0 for k in TERMINATION_KEYS}
    for ep in episodes:
        key = ep.termination if ep.termination in counts else "other"
        counts[key] += 1
    return counts


def _eval_json_path(checkpoint: Path | None, out_dir: Path | None, tier: int) -> Path:
    if checkpoint is not None:
        return checkpoint.with_suffix(checkpoint.suffix + ".eval.json") \
            if checkpoint.suffix else Path(str(checkpoint) + ".eval.json")
    base = out_dir or Path(".")
    return base / f"eval_tier{tier}.json"


def evaluate(
    *,
    seeds: list[int] | None = None,
    tier: int = 0,
    checkpoint: str | Path | None = None,
    actor: EpisodeActor | None = None,
    stage_fn: Callable[[int, int], dict[str, Any]] | None = None,
    env_config: EnvConfig | None = None,
    out_path: str | Path | None = None,
    out_dir: str | Path | None = None,
    optimal_time_s: float | None = None,
    optimal_times_by_seed: dict[int, float] | None = None,
    optimal_sector_times_s: list[float] | None = None,
    human_baseline_path: str | Path | None = None,
    require_held_out: bool = True,
) -> dict[str, Any]:
    """Run held-out evaluation and write the full record beside the checkpoint.

    Returns the metrics ``eval`` block (including ``json_path``), plus richer
    fields (``terminations``, ``cleanliness``, ``episodes``) for callers.
    """
    seed_list = list(seeds) if seeds is not None else held_out_seeds(4)
    if require_held_out:
        bad = [s for s in seed_list if not is_eval_seed(s)]
        if bad:
            raise ValueError(
                f"seeds {bad} are not in the held-out reservation "
                f"(seed % {EVAL_SEED_MODULUS} == {EVAL_SEED_RESIDUE}). "
                f"Pass require_held_out=False only for deliberate non-claim runs."
            )

    ckpt_path = Path(checkpoint) if checkpoint is not None else None
    meta: dict[str, Any] = {}
    if actor is None:
        if ckpt_path is not None:
            actor, meta = load_actor_from_checkpoint(ckpt_path)
        else:
            actor = ReferencePilotActor()

    human = load_human_baseline(human_baseline_path)
    make_stage = stage_fn or (lambda seed, t: generate(seed, t))
    episodes: list[EpisodeResult] = []
    for seed in seed_list:
        stage = make_stage(int(seed), int(tier))
        opt = None
        if optimal_times_by_seed is not None and int(seed) in optimal_times_by_seed:
            opt = float(optimal_times_by_seed[int(seed)])
        elif optimal_time_s is not None:
            opt = float(optimal_time_s)
        human_row = human.get((int(seed), int(tier)))
        human_best = float(human_row["best_time_s"]) if human_row else None
        episodes.append(
            run_episode(
                stage,
                actor,
                seed=int(seed),
                tier=int(tier),
                env_config=env_config,
                optimal_time_s=opt,
                optimal_sector_times_s=optimal_sector_times_s,
                human_best_s=human_best,
            )
        )

    n = len(episodes) or 1
    finished = [ep for ep in episodes if ep.finished]
    clean = [ep for ep in episodes if ep.clean]
    completion_rate = len(finished) / n
    clean_rate = len(clean) / n
    mean_time = float(np.mean([ep.time_s for ep in finished])) if finished else float("nan")
    ratios = [ep.time_vs_optimal for ep in finished if ep.time_vs_optimal is not None]
    time_vs_optimal = float(np.mean(ratios)) if ratios else None
    human_ratios = [ep.time_vs_human for ep in finished if ep.time_vs_human is not None]
    time_vs_human = float(np.mean(human_ratios)) if human_ratios else None

    json_path = Path(out_path) if out_path is not None else _eval_json_path(
        ckpt_path, Path(out_dir) if out_dir is not None else None, tier,
    )
    json_path.parent.mkdir(parents=True, exist_ok=True)

    record = {
        "schema_version": 1,
        "kind": "eval",
        "checkpoint": str(ckpt_path) if ckpt_path is not None else None,
        "checkpoint_meta": meta,
        "actor": getattr(actor, "name", type(actor).__name__),
        "tier": int(tier),
        "seeds": [int(s) for s in seed_list],
        "seed_reservation": {
            "modulus": EVAL_SEED_MODULUS,
            "eval_residue": EVAL_SEED_RESIDUE,
            "rule": "train: seed % 10 != 7; eval: seed % 10 == 7",
        },
        "completion_rate": completion_rate,
        "clean_rate": clean_rate,
        "mean_time_s": None if np.isnan(mean_time) else mean_time,
        "time_vs_optimal": time_vs_optimal,
        "time_vs_human": time_vs_human,
        "human_baseline_seeds": len(human),
        "terminations": _termination_breakdown(episodes),
        "cleanliness": {
            "mean_off_course_s": float(np.mean([ep.off_course_s for ep in episodes])),
            "mean_obstacle_contacts": float(
                np.mean([ep.obstacle_contacts for ep in episodes])
            ),
            "total_off_course_s": float(sum(ep.off_course_s for ep in episodes)),
            "total_obstacle_contacts": int(sum(ep.obstacle_contacts for ep in episodes)),
        },
        "episodes": [ep.to_dict() for ep in episodes],
        "json_path": str(json_path),
    }
    json_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    # Metrics schema `eval` block + harness extras.
    return {
        "seeds": record["seeds"],
        "completion_rate": completion_rate,
        "mean_time_s": record["mean_time_s"],
        "time_vs_optimal": time_vs_optimal,
        "time_vs_human": time_vs_human,
        "clean_rate": clean_rate,
        "json_path": str(json_path),
        "terminations": record["terminations"],
        "cleanliness": record["cleanliness"],
        "episodes": record["episodes"],
        "actor": record["actor"],
    }
