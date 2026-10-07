"""Disk-backed dashboard queries for the program hub.

All answers come from the filesystem (runs layout, HoF ``.pt`` files, eval JSON,
replay dirs). In-memory job handles are only used to refresh live run state —
never as the sole source of truth.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from rallyai.control.jobs import JobStore, _repo_root
from rallyai.train.checkpoint import HOF_SLOTS

CONTROL_VERSION = "0.1.0"

_SLOT_RE = re.compile(
    r"^(?P<run>.+)_(?P<slot>" + "|".join(HOF_SLOTS) + r")\.pt$"
)
_EVAL_NAME_RE = re.compile(r"\.eval\.json$", re.IGNORECASE)


def default_viewer_replays_dir() -> Path:
    return _repo_root() / "packages" / "viewer" / "public" / "replays"


def default_replay_dirs(runs_dir: Path) -> list[Path]:
    """Configured places the dashboard looks for replay JSON."""
    return [
        default_viewer_replays_dir(),
        Path(runs_dir) / "exports",
    ]


def _safe_stat(path: Path) -> tuple[float | None, int | None]:
    try:
        st = path.stat()
        return float(st.st_mtime), int(st.st_size)
    except OSError:
        return None, None


def _read_json_head(path: Path, max_bytes: int = 65_536) -> dict[str, Any] | None:
    try:
        raw = path.read_bytes()[:max_bytes]
        text = raw.decode("utf-8", errors="replace")
        # Truncated reads may break JSON; try full file if small enough.
        if len(raw) < path.stat().st_size and path.stat().st_size <= max_bytes * 4:
            text = path.read_text(encoding="utf-8")
        data = json.loads(text)
        return data if isinstance(data, dict) else None
    except (OSError, json.JSONDecodeError, UnicodeError):
        return None


def _looks_like_replay(path: Path) -> bool:
    if path.suffix.lower() != ".json":
        return False
    name = path.name.lower()
    if name in ("times.json",) or name.endswith(".schema.json"):
        return False
    if name.endswith(".eval.json"):
        return False
    data = _read_json_head(path)
    if data is None:
        return False
    if "frames" in data and isinstance(data.get("frames"), list):
        return True
    # Minimal / schema-valid replays always carry schema_version + source.
    return data.get("schema_version") == 1 and "source" in data


def _parse_hof_name(name: str) -> tuple[str, str] | None:
    m = _SLOT_RE.match(name)
    if not m:
        return None
    return m.group("run"), m.group("slot")


def _cheap_checkpoint_meta(path: Path) -> dict[str, Any]:
    """Pull scalar metadata without requiring a full training restore.

    Prefers a tiny torch load of the checkpoint dict (tests and real HoF files
    store ``timesteps`` / ``tier`` at the top level). Failures are soft — the
    file listing remains valid with mtime/size alone.
    """
    out: dict[str, Any] = {}
    try:
        import torch

        payload = torch.load(path, map_location="cpu", weights_only=False)
    except Exception:
        return out
    if not isinstance(payload, dict):
        return out
    for key in (
        "timesteps",
        "tier",
        "run_id",
        "policy_sha256",
        "obs_dim",
        "act_dim",
        "obs_layout_version",
        "stage",
    ):
        if key in payload:
            val = payload[key]
            if key in ("timesteps", "tier", "obs_dim", "act_dim", "obs_layout_version"):
                try:
                    out[key] = int(val)
                except (TypeError, ValueError):
                    out[key] = val
            else:
                out[key] = val
    return out


def _metrics_candidates(store: JobStore, run_id: str) -> list[Path]:
    paths: list[Path] = []
    nested = store.metrics_path(run_id)
    paths.append(nested)
    flat = store.runs_dir / f"{run_id}.jsonl"
    if flat not in paths:
        paths.append(flat)
    return paths


def _tail_metrics_lines(path: Path, max_lines: int = 64) -> list[dict[str, Any]]:
    if not path.exists() or not path.is_file():
        return []
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return []
    out: list[dict[str, Any]] = []
    for raw in text.splitlines()[-max_lines:]:
        raw = raw.strip()
        if not raw:
            continue
        try:
            row = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            out.append(row)
    return out


def _latest_update_snapshot(store: JobStore, run_id: str) -> dict[str, Any] | None:
    best: dict[str, Any] | None = None
    best_wall = float("-inf")
    for path in _metrics_candidates(store, run_id):
        for row in _tail_metrics_lines(path):
            if row.get("kind") not in ("update", "eval", "checkpoint", "run_end", "run_start"):
                continue
            wall = float(row.get("wall_t") or 0.0)
            if wall >= best_wall:
                best_wall = wall
                best = row
    return best


def _throughput_hint(snapshot: dict[str, Any] | None) -> dict[str, Any] | None:
    if not snapshot:
        return None
    thr = snapshot.get("throughput")
    if isinstance(thr, dict) and thr.get("steps_per_s") is not None:
        return {
            "steps_per_s": thr.get("steps_per_s"),
            "n_workers": thr.get("n_workers"),
            "rollout_s": thr.get("rollout_s"),
            "update_s": thr.get("update_s"),
            "source": "metrics",
            "run_id": snapshot.get("run_id"),
            "timesteps": snapshot.get("timesteps"),
        }
    return None


def discover_run_ids(store: JobStore) -> list[str]:
    """Union of control-room dirs (state.json) and flat CLI HoF / metrics stems."""
    found: set[str] = set()
    root = store.runs_dir
    if not root.exists():
        return []
    for child in root.iterdir():
        if child.is_dir() and (child / "state.json").exists():
            found.add(child.name)
        elif child.is_file():
            if child.name.endswith(".jsonl") and not child.name.endswith("_monitor.jsonl"):
                stem = child.name[: -len(".jsonl")]
                if stem:
                    found.add(stem)
            parsed = _parse_hof_name(child.name)
            if parsed:
                found.add(parsed[0])
    return sorted(found, reverse=True)


def run_exists(store: JobStore, run_id: str) -> bool:
    if (store.run_dir(run_id) / "state.json").exists():
        return True
    if (store.runs_dir / f"{run_id}.jsonl").exists():
        return True
    for path in store.runs_dir.glob(f"{run_id}_*.pt"):
        if path.is_file():
            return True
    nested = store.run_dir(run_id)
    if nested.is_dir():
        if any(nested.glob("*.pt")) or any(nested.glob("*.eval.json")):
            return True
    return False


def list_checkpoints(store: JobStore, run_id: str) -> list[dict[str, Any]]:
    """HoF / latest ``.pt`` files for a run, with mtime and cheap meta."""
    files: dict[str, Path] = {}

    def _consider(path: Path) -> None:
        if not path.is_file() or path.suffix != ".pt":
            return
        files[str(path.resolve())] = path

    nested = store.run_dir(run_id)
    if nested.is_dir():
        for path in nested.glob("*.pt"):
            _consider(path)

    for slot in HOF_SLOTS:
        _consider(store.runs_dir / f"{run_id}_{slot}.pt")
    for path in store.runs_dir.glob(f"{run_id}_*.pt"):
        _consider(path)

    rows: list[dict[str, Any]] = []
    for path in files.values():
        mtime, size = _safe_stat(path)
        parsed = _parse_hof_name(path.name)
        slot = parsed[1] if parsed else None
        row: dict[str, Any] = {
            "run_id": run_id,
            "name": path.name,
            "path": str(path),
            "slot": slot,
            "mtime": mtime,
            "size": size,
        }
        meta = _cheap_checkpoint_meta(path)
        for key, val in meta.items():
            if key == "run_id" and not val:
                continue
            row[key] = val
        row["run_id"] = run_id
        rows.append(row)

    rows.sort(key=lambda r: (-(r.get("mtime") or 0.0), r.get("slot") or "", r["name"]))
    return rows


def list_evals(store: JobStore, run_id: str) -> list[dict[str, Any]]:
    """Eval JSON records beside the run / its checkpoints."""
    files: dict[str, Path] = {}

    def _consider(path: Path) -> None:
        if not path.is_file() or not path.name.endswith(".json"):
            return
        name = path.name
        # Prefer *.eval.json; allow eval_tierN.json inside the run dir.
        if not (_EVAL_NAME_RE.search(name) or name.startswith("eval")):
            return
        files[str(path.resolve())] = path

    nested = store.run_dir(run_id)
    if nested.is_dir():
        for path in nested.rglob("*.json"):
            _consider(path)

    for path in store.runs_dir.glob(f"{run_id}*.eval.json"):
        _consider(path)
    for path in store.runs_dir.glob(f"{run_id}_*.pt.*.eval.json"):
        _consider(path)
    for path in store.runs_dir.glob(f"{run_id}_*.pt.eval.json"):
        _consider(path)

    rows: list[dict[str, Any]] = []
    for path in files.values():
        mtime, size = _safe_stat(path)
        row: dict[str, Any] = {
            "run_id": run_id,
            "name": path.name,
            "path": str(path),
            "mtime": mtime,
            "size": size,
        }
        data = _read_json_head(path)
        if data:
            for key in ("kind", "tier", "checkpoint", "completion_rate", "clean_rate"):
                if key in data:
                    row[key] = data[key]
            ev = data.get("eval") if isinstance(data.get("eval"), dict) else None
            if ev:
                for key in ("completion_rate", "clean_rate", "mean_time_s", "time_vs_optimal"):
                    if key in ev and key not in row:
                        row[key] = ev[key]
            if "tier" not in row and isinstance(data.get("config"), dict):
                tier = data["config"].get("tier")
                if tier is not None:
                    row["tier"] = tier
        rows.append(row)

    rows.sort(key=lambda r: (-(r.get("mtime") or 0.0), r["name"]))
    return rows


def list_replays(replay_dirs: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for root in replay_dirs:
        if not root.exists() or not root.is_dir():
            continue
        for path in sorted(root.rglob("*.json")):
            key = str(path.resolve())
            if key in seen:
                continue
            if not _looks_like_replay(path):
                continue
            seen.add(key)
            mtime, size = _safe_stat(path)
            data = _read_json_head(path) or {}
            meta = data.get("meta") if isinstance(data.get("meta"), dict) else {}
            frames = data.get("frames")
            n_frames = len(frames) if isinstance(frames, list) else None
            rows.append(
                {
                    "name": path.name,
                    "path": str(path),
                    "dir": str(root),
                    "mtime": mtime,
                    "size": size,
                    "source": data.get("source"),
                    "frames": n_frames,
                    "seed": meta.get("seed"),
                    "tier": meta.get("tier"),
                }
            )
    rows.sort(key=lambda r: (-(r.get("mtime") or 0.0), r["name"]))
    return rows


def count_checkpoints(store: JobStore) -> dict[str, Any]:
    by_slot: dict[str, int] = {s: 0 for s in HOF_SLOTS}
    by_run: dict[str, int] = {}
    total = 0
    root = store.runs_dir
    if not root.exists():
        return {"total": 0, "by_slot": by_slot, "by_run": by_run}

    seen: set[str] = set()
    candidates: list[Path] = []
    candidates.extend(root.glob("*.pt"))
    for child in root.iterdir():
        if child.is_dir():
            candidates.extend(child.glob("*.pt"))

    for path in candidates:
        key = str(path.resolve())
        if key in seen or not path.is_file():
            continue
        seen.add(key)
        total += 1
        parsed = _parse_hof_name(path.name)
        if parsed:
            run_id, slot = parsed
            by_slot[slot] = by_slot.get(slot, 0) + 1
            by_run[run_id] = by_run.get(run_id, 0) + 1
        else:
            # Nested non-HoF name: attribute to parent dir if it looks like a run.
            parent = path.parent.name
            if parent != root.name:
                by_run[parent] = by_run.get(parent, 0) + 1
            else:
                by_run["_other"] = by_run.get("_other", 0) + 1
    return {"total": total, "by_slot": by_slot, "by_run": by_run}


def dashboard_summary(store: JobStore) -> dict[str, Any]:
    listed = store.list_runs()
    active = [
        r
        for r in listed
        if r.get("state") in ("starting", "running", "stopping")
    ]

    # Prefer an active run's latest metrics; else most recently updated listed run;
    # else any discovered flat metrics stem.
    snapshot: dict[str, Any] | None = None
    candidates: list[str] = [str(r["run_id"]) for r in active]
    candidates.extend(str(r["run_id"]) for r in listed if str(r["run_id"]) not in candidates)
    for rid in discover_run_ids(store):
        if rid not in candidates:
            candidates.append(rid)

    best_wall = float("-inf")
    for rid in candidates:
        snap = _latest_update_snapshot(store, rid)
        if snap is None:
            continue
        wall = float(snap.get("wall_t") or 0.0)
        if wall >= best_wall:
            best_wall = wall
            snapshot = snap

    ckpt = count_checkpoints(store)
    return {
        "active_runs": active,
        "active_count": len(active),
        "runs_total": len(discover_run_ids(store)),
        "control_runs": len(listed),
        "latest_metrics": snapshot,
        "throughput": _throughput_hint(snapshot),
        "checkpoints": ckpt,
        "runs_dir": str(store.runs_dir),
    }
