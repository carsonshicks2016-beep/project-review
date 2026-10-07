"""Vectorised environments for rollout collection.

``SyncVecEnv`` is the reference implementation: in-process, bit-identical to a
bare ``RallyEnv`` at ``n=1``. ``AsyncVecEnv`` matches that step/reset contract
across worker processes, including ``info["final_observation"]`` on every
episode end, and revives dead or stalled workers instead of taking the run down.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from rallyai.env import EnvConfig, RallyEnv, RewardConfig
from rallyai.sense import SensorSpec
from rallyai.stage.generator import generate
from rallyai.track import Track

# Default stall budget for a single worker reply. Normal steps are sub-second;
# anything this long is a wedged worker, not a slow one.
DEFAULT_STALL_TIMEOUT_S = 30.0
MAX_RESPAWNS_PER_WORKER = 8


def _default_stage_fn(tier: int) -> Callable[[int], dict[str, Any]]:
    """Close over a mutable tier so ``set_tier`` reaches the generator."""

    state = {"tier": int(tier)}

    def stage_fn(seed: int) -> dict[str, Any]:
        return generate(int(seed), int(state["tier"]))

    stage_fn.state = state  # type: ignore[attr-defined]
    return stage_fn


class SyncVecEnv:
    """Batched in-process envs with auto-reset and truncation bootstrapping.

    When a sub-env finishes, the *new* observation is returned in the obs
    array and the ending observation is preserved in
    ``info["final_observation"]``. GAE needs that ending obs to bootstrap a
    truncated episode; losing it is the classic silent PPO bug.
    """

    def __init__(
        self,
        n: int,
        stage_fn: Callable[[int], dict[str, Any]] | None = None,
        *,
        env_config: EnvConfig | None = None,
        sensor_spec: SensorSpec | None = None,
        reward_config: RewardConfig | None = None,
        tier: int = 0,
    ) -> None:
        if n < 1:
            raise ValueError(f"n must be >= 1, got {n}")
        self.n = int(n)
        self.env_config = env_config or EnvConfig()
        self.sensor_spec = sensor_spec or SensorSpec()
        self.reward_config = reward_config or RewardConfig()
        self._tier = int(tier)
        if stage_fn is None:
            self._stage_fn = _default_stage_fn(self._tier)
            self._owns_stage_fn = True
        else:
            self._stage_fn = stage_fn
            self._owns_stage_fn = False

        # Build placeholder envs; ``reset`` loads the real stages.
        bootstrap = self._stage_fn(0)
        self.envs = [
            RallyEnv(
                bootstrap,
                config=self.env_config,
                sensor_spec=self.sensor_spec,
                reward_config=self.reward_config,
            )
            for _ in range(self.n)
        ]
        self.obs_dim = int(self.envs[0].observation_space.shape[0])
        self.act_dim = int(self.envs[0].action_space.shape[0])
        self._rngs: list[np.random.Generator] = [
            np.random.default_rng(i) for i in range(self.n)
        ]
        self._closed = False

    def set_tier(self, tier: int) -> None:
        self._tier = int(tier)
        if self._owns_stage_fn:
            self._stage_fn.state["tier"] = self._tier  # type: ignore[attr-defined]

    def _load_stage(self, index: int, stage_seed: int) -> None:
        stage = self._stage_fn(int(stage_seed))
        env = self.envs[index]
        env.stage = stage
        env.track = Track(stage)

    def reset(self, seed: int | None = None) -> np.ndarray:
        if self._closed:
            raise RuntimeError("SyncVecEnv is closed")
        if seed is None:
            seed = int(np.random.default_rng().integers(0, 2**31 - 1))
        seed = int(seed)
        # Independent streams per sub-env. Seeding them identically would give
        # n copies of one episode and a policy that overfits a single stage.
        ss = np.random.SeedSequence(seed)
        children = ss.spawn(self.n)
        obs = np.empty((self.n, self.obs_dim), dtype=np.float32)
        for i in range(self.n):
            self._rngs[i] = np.random.default_rng(children[i])
            # env 0 uses ``seed`` directly so n=1 matches a bare RallyEnv
            # constructed with ``stage_fn(seed)`` and ``reset(seed=seed)``.
            stage_seed = seed if i == 0 else seed + i
            self._load_stage(i, stage_seed)
            o, _ = self.envs[i].reset(seed=stage_seed)
            obs[i] = o
        return obs

    def step(
        self, actions: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
        if self._closed:
            raise RuntimeError("SyncVecEnv is closed")
        actions = np.asarray(actions, dtype=np.float32)
        if actions.shape != (self.n, self.act_dim):
            raise ValueError(
                f"actions shape {actions.shape} != ({self.n}, {self.act_dim})"
            )

        obs = np.empty((self.n, self.obs_dim), dtype=np.float32)
        rewards = np.empty(self.n, dtype=np.float32)
        terminated = np.zeros(self.n, dtype=bool)
        truncated = np.zeros(self.n, dtype=bool)
        infos: list[dict[str, Any]] = []

        for i in range(self.n):
            o, r, term, trunc, info = self.envs[i].step(actions[i])
            rewards[i] = r
            terminated[i] = term
            truncated[i] = trunc
            if term or trunc:
                # Ending obs for GAE truncation bootstrap; obs array gets the
                # post-reset observation so the next transition starts clean.
                info = dict(info)
                info["final_observation"] = np.asarray(o, dtype=np.float32).copy()
                stage_seed = int(self._rngs[i].integers(0, 2**31 - 1))
                self._load_stage(i, stage_seed)
                o, _ = self.envs[i].reset(seed=stage_seed)
            obs[i] = o
            infos.append(info)
        return obs, rewards, terminated, truncated, infos

    def close(self) -> None:
        self._closed = True
        self.envs.clear()


# --------------------------------------------------------------------------- #
# Async (subprocess) vectorised env
# --------------------------------------------------------------------------- #


def _rebuild_configs(
    env_config: dict[str, Any],
    sensor_spec: dict[str, Any],
    reward_config: dict[str, Any],
) -> tuple[EnvConfig, SensorSpec, RewardConfig]:
    return (
        EnvConfig(**env_config),
        SensorSpec(**sensor_spec),
        RewardConfig(**reward_config),
    )


def _load_env_stage(env: RallyEnv, seed: int, tier: int) -> None:
    """Workers generate from (seed, tier) — never ship the stage dict over the pipe."""
    stage = generate(int(seed), int(tier))
    env.stage = stage
    env.track = Track(stage)


def _limit_worker_threads() -> None:
    """Keep each worker single-threaded so N workers do not oversubscribe."""
    import os

    for key in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        os.environ[key] = "1"
    try:
        import torch

        torch.set_num_threads(1)
        if hasattr(torch, "set_num_interop_threads"):
            torch.set_num_interop_threads(1)
    except Exception:
        pass
    try:
        import threadpoolctl

        threadpoolctl.threadpool_limits(limits=1)
    except Exception:
        pass


def _async_worker(
    remote: Any,
    parent_remote: Any,
    n_envs: int,
    env_config: dict[str, Any],
    sensor_spec: dict[str, Any],
    reward_config: dict[str, Any],
    tier: int,
) -> None:
    parent_remote.close()
    _limit_worker_threads()

    cfg, sensors, reward = _rebuild_configs(env_config, sensor_spec, reward_config)
    bootstrap = generate(0, int(tier))
    envs = [
        RallyEnv(bootstrap, config=cfg, sensor_spec=sensors, reward_config=reward)
        for _ in range(n_envs)
    ]
    obs_dim = int(envs[0].observation_space.shape[0])
    act_dim = int(envs[0].action_space.shape[0])
    rngs = [np.random.default_rng(i) for i in range(n_envs)]
    current_tier = int(tier)

    try:
        while True:
            cmd, data = remote.recv()
            if cmd == "reset":
                seed = int(data)
                ss = np.random.SeedSequence(seed)
                children = ss.spawn(n_envs)
                out = []
                for i in range(n_envs):
                    rngs[i] = np.random.default_rng(children[i])
                    stage_seed = seed if i == 0 else seed + i
                    _load_env_stage(envs[i], stage_seed, current_tier)
                    o, _ = envs[i].reset(seed=stage_seed)
                    out.append(np.asarray(o, dtype=np.float32))
                remote.send(out)
            elif cmd == "step":
                actions = np.asarray(data, dtype=np.float32)
                out = []
                for i in range(n_envs):
                    o, r, term, trunc, info = envs[i].step(actions[i])
                    if term or trunc:
                        info = dict(info)
                        info["final_observation"] = np.asarray(o, dtype=np.float32).copy()
                        stage_seed = int(rngs[i].integers(0, 2**31 - 1))
                        _load_env_stage(envs[i], stage_seed, current_tier)
                        o, _ = envs[i].reset(seed=stage_seed)
                    out.append((
                        np.asarray(o, dtype=np.float32),
                        float(r),
                        bool(term),
                        bool(trunc),
                        info,
                    ))
                remote.send(out)
            elif cmd == "set_tier":
                current_tier = int(data)
                remote.send(True)
            elif cmd == "close":
                break
            elif cmd == "ping":
                remote.send((obs_dim, act_dim, n_envs))
            else:
                remote.send(("__error__", f"unknown command {cmd!r}"))
    except (KeyboardInterrupt, EOFError):
        pass
    except Exception:
        import traceback

        try:
            remote.send(("__error__", traceback.format_exc()))
        except Exception:
            pass
    finally:
        try:
            remote.close()
        except Exception:
            pass


class AsyncVecEnv:
    """Process-parallel vectorised env with worker revival.

    One ``spawn`` process per worker, each owning ``envs_per_worker`` RallyEnvs.
    Workers receive ``(seed, tier)`` and call ``generate`` themselves — the stage
    document never crosses the pipe. A dead or stalled worker is logged to the
    metrics JSONL, respawned, and the run continues with a short rollout for
    that shard rather than fabricating fake transitions beyond a clean boundary.
    """

    def __init__(
        self,
        n_workers: int,
        *,
        envs_per_worker: int = 1,
        env_config: EnvConfig | None = None,
        sensor_spec: SensorSpec | None = None,
        reward_config: RewardConfig | None = None,
        tier: int = 0,
        metrics_path: str | Path | None = None,
        run_id: str = "async",
        stall_timeout_s: float = DEFAULT_STALL_TIMEOUT_S,
    ) -> None:
        import multiprocessing as mp

        if n_workers < 1:
            raise ValueError(f"n_workers must be >= 1, got {n_workers}")
        if envs_per_worker < 1:
            raise ValueError(f"envs_per_worker must be >= 1, got {envs_per_worker}")

        # Inherit into spawn children before they re-import numpy/OpenBLAS.
        _limit_worker_threads()

        self.n_workers = int(n_workers)
        self.envs_per_worker = int(envs_per_worker)
        self.n = self.n_workers * self.envs_per_worker
        self.env_config = env_config or EnvConfig()
        self.sensor_spec = sensor_spec or SensorSpec()
        self.reward_config = reward_config or RewardConfig()
        self._tier = int(tier)
        self._metrics_path = Path(metrics_path) if metrics_path is not None else None
        self._run_id = str(run_id)
        self.stall_timeout_s = float(stall_timeout_s)
        self._ctx = mp.get_context("spawn")
        self._env_config_dict = asdict(self.env_config)
        self._sensor_spec_dict = asdict(self.sensor_spec)
        self._reward_config_dict = asdict(self.reward_config)
        self._respawns = [0] * self.n_workers
        self._short_rollout_workers: list[int] = []
        self.remotes: list[Any] = []
        self.procs: list[Any] = []
        self._closed = False

        for w in range(self.n_workers):
            remote, proc = self._spawn(w)
            self.remotes.append(remote)
            self.procs.append(proc)

        # Probe dims from worker 0 rather than building a local env (keeps parent light).
        self.remotes[0].send(("ping", None))
        if not self.remotes[0].poll(self.stall_timeout_s):
            raise TimeoutError("AsyncVecEnv worker 0 failed to answer ping")
        ping = self.remotes[0].recv()
        if isinstance(ping, tuple) and len(ping) == 2 and ping[0] == "__error__":
            raise RuntimeError(ping[1])
        self.obs_dim, self.act_dim, _ = ping

    # ------------------------------------------------------------------ #
    # lifecycle
    # ------------------------------------------------------------------ #

    def _spawn(self, worker_index: int) -> tuple[Any, Any]:
        remote, worker_remote = self._ctx.Pipe()
        proc = self._ctx.Process(
            target=_async_worker,
            args=(
                worker_remote,
                remote,
                self.envs_per_worker,
                self._env_config_dict,
                self._sensor_spec_dict,
                self._reward_config_dict,
                self._tier,
            ),
            daemon=True,
            name=f"rallyai-async-{worker_index}",
        )
        proc.start()
        worker_remote.close()
        return remote, proc

    def _log_worker(self, index: int, event: str, detail: str = "") -> None:
        if self._metrics_path is None:
            return
        line = {
            "schema_version": 1,
            "kind": "worker",
            "run_id": self._run_id,
            "wall_t": time.time(),
            "worker": {
                "index": int(index),
                "event": event,
                "detail": detail or "",
            },
        }
        self._metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with self._metrics_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(line, separators=(",", ":")) + "\n")

    def _kill_proc(self, worker_index: int) -> None:
        remote = self.remotes[worker_index]
        proc = self.procs[worker_index]
        try:
            remote.close()
        except Exception:
            pass
        try:
            if proc.is_alive():
                proc.kill()
            proc.join(timeout=2.0)
        except Exception:
            pass

    def _revive(self, worker_index: int, event: str, why: str) -> list[np.ndarray]:
        """Replace a dead/stalled worker; return fresh reset observations for its shard."""
        self._respawns[worker_index] += 1
        if self._respawns[worker_index] > MAX_RESPAWNS_PER_WORKER:
            raise RuntimeError(
                f"async worker {worker_index} died "
                f"{self._respawns[worker_index]} times — giving up. last: {why}"
            )
        self._log_worker(worker_index, event, why)
        self._kill_proc(worker_index)
        remote, proc = self._spawn(worker_index)
        self.remotes[worker_index] = remote
        self.procs[worker_index] = proc

        # Replay tier, then reset with a derived seed so the shard is live again.
        try:
            remote.send(("set_tier", self._tier))
            if not remote.poll(min(30.0, self.stall_timeout_s)):
                raise TimeoutError("revived worker set_tier timeout")
            ack = remote.recv()
            if isinstance(ack, tuple) and len(ack) == 2 and ack[0] == "__error__":
                raise RuntimeError(ack[1])
            bootstrap_seed = int(time.time_ns() % (2**31 - 1)) ^ (worker_index * 9973)
            remote.send(("reset", bootstrap_seed))
            if not remote.poll(min(60.0, max(self.stall_timeout_s, 5.0))):
                raise TimeoutError("revived worker reset timeout")
            reply = remote.recv()
            if isinstance(reply, tuple) and len(reply) == 2 and reply[0] == "__error__":
                raise RuntimeError(reply[1])
        except Exception as exc:
            return self._revive(worker_index, "died", f"bootstrap failed: {exc}")

        self._log_worker(worker_index, "revived", why)
        self._short_rollout_workers.append(worker_index)
        return list(reply)

    def _send(self, worker_index: int, msg: tuple[Any, Any]) -> None:
        try:
            self.remotes[worker_index].send(msg)
        except Exception:
            pass

    def _recv(self, worker_index: int) -> tuple[Any, str | None]:
        """Return ``(reply, None)`` or revive and return ``(reset_obs_list, reason)``."""
        remote = self.remotes[worker_index]
        try:
            if not remote.poll(self.stall_timeout_s):
                why = f"worker reply timeout after {self.stall_timeout_s:.1f}s"
                return self._revive(worker_index, "stalled", why), why
            reply = remote.recv()
            if isinstance(reply, tuple) and len(reply) == 2 and reply[0] == "__error__":
                why = reply[1] or "worker error"
                return self._revive(worker_index, "died", why), why
            return reply, None
        except (EOFError, ConnectionResetError, BrokenPipeError, OSError) as exc:
            why = repr(exc)
            return self._revive(worker_index, "died", why), why

    # ------------------------------------------------------------------ #
    # public API (mirrors SyncVecEnv)
    # ------------------------------------------------------------------ #

    def set_tier(self, tier: int) -> None:
        self._tier = int(tier)
        for w in range(self.n_workers):
            self._send(w, ("set_tier", self._tier))
        for w in range(self.n_workers):
            self._recv(w)

    def reset(self, seed: int | None = None) -> np.ndarray:
        if self._closed:
            raise RuntimeError("AsyncVecEnv is closed")
        if seed is None:
            seed = int(np.random.default_rng().integers(0, 2**31 - 1))
        seed = int(seed)
        self._short_rollout_workers.clear()
        for w in range(self.n_workers):
            # Each worker's shard starts at a distinct base seed.
            self._send(w, ("reset", seed + w * self.envs_per_worker))
        obs: list[np.ndarray] = []
        for w in range(self.n_workers):
            reply, _died = self._recv(w)
            obs.extend(reply)
        return np.stack(obs).astype(np.float32)

    def step(
        self, actions: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
        if self._closed:
            raise RuntimeError("AsyncVecEnv is closed")
        actions = np.asarray(actions, dtype=np.float32)
        if actions.shape != (self.n, self.act_dim):
            raise ValueError(
                f"actions shape {actions.shape} != ({self.n}, {self.act_dim})"
            )

        s = self.envs_per_worker
        for w in range(self.n_workers):
            self._send(w, ("step", actions[w * s:(w + 1) * s]))

        obs_list: list[np.ndarray] = []
        rewards: list[float] = []
        terminated: list[bool] = []
        truncated: list[bool] = []
        infos: list[dict[str, Any]] = []

        for w in range(self.n_workers):
            reply, died = self._recv(w)
            if died is not None:
                # Shard restarted mid-step: surface a clean episode boundary
                # (reward 0, truncated) so GAE sees a short rollout, not fake
                # dynamics. ``reply`` is the revived worker's reset obs list.
                for o in reply:
                    o_arr = np.asarray(o, dtype=np.float32)
                    obs_list.append(o_arr)
                    rewards.append(0.0)
                    terminated.append(False)
                    truncated.append(True)
                    infos.append({
                        "termination": "worker_died",
                        "final_observation": o_arr.copy(),
                        "worker_died": True,
                        "worker_index": w,
                    })
                continue
            for o, r, term, trunc, info in reply:
                obs_list.append(np.asarray(o, dtype=np.float32))
                rewards.append(float(r))
                terminated.append(bool(term))
                truncated.append(bool(trunc))
                infos.append(info)

        return (
            np.stack(obs_list).astype(np.float32),
            np.asarray(rewards, dtype=np.float32),
            np.asarray(terminated, dtype=bool),
            np.asarray(truncated, dtype=bool),
            infos,
        )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for w in range(self.n_workers):
            try:
                self.remotes[w].send(("close", None))
            except Exception:
                pass
            try:
                self.remotes[w].close()
            except Exception:
                pass
        for proc in self.procs:
            proc.join(timeout=1.0)
            if proc.is_alive():
                try:
                    proc.kill()
                    proc.join(timeout=1.0)
                except Exception:
                    pass
        self.remotes.clear()
        self.procs.clear()
