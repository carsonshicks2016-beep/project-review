"""Work through a list of levels unattended: explore -> verify -> train -> evaluate.

    python -m smwrl.autopilot --levels YoshiIsland1,YoshiIsland3 --bar 0.6

Runs each level to a quality bar, then moves to the next. Everything is a
subprocess so one crashed trainer cannot take the whole night down, and every
decision is written to checkpoints/autopilot_<worker>.json so the morning
post-mortem does not depend on scrollback.

The judgement calls it makes, and why:

* **Quality bar is the unaided clear rate**, never the training log's `clear`
  column -- that one counts curriculum episodes which start next to the goal and
  clear trivially (YoshiIsland2 once read 25% while its true rate was 0%).

* **Patience.** If a level stops improving it re-explores to extend the archive
  and rebuilds the curriculum, because a stalled curriculum has been the cause
  every single time so far. If it stalls again it banks the best result and
  moves on rather than burning the night on one level.

* **It never blocks forever.** Every level has a step budget. Getting four
  levels to 60% is worth more than one level to 90%.
"""

from __future__ import annotations

import argparse
import fcntl
import gc
import hashlib
import json
import math
import os
import signal
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path
from contextlib import contextmanager

ROOT = Path(__file__).resolve().parent.parent
CKPT = ROOT / "checkpoints"
PY = str(ROOT / ".venv" / "bin" / "python")


def log(worker: str, msg: str) -> None:
    stamp = time.strftime("%H:%M:%S")
    print(f"[{stamp}] {worker}: {msg}", flush=True)


def run(cmd: list[str], timeout: float | None = None,
        output_limit: int = 2_000_000) -> tuple[int, str]:
    """Run a bounded subprocess and kill its complete worker process group.

    Training launches SubprocVecEnv children.  Killing only the direct Python
    process on timeout left emulator workers orphaned, so every job gets a new
    session and bounded TERM/KILL escalation.  Output is spooled to disk rather
    than retained unbounded in RAM for multi-hour jobs.
    """
    with tempfile.TemporaryFile() as output:
        process = subprocess.Popen(
            cmd, cwd=ROOT, stdout=output, stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        timed_out = False
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=10)
        except KeyboardInterrupt:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=10)
            raise
        output.flush()
        size = output.tell()
        output.seek(max(0, size - output_limit))
        text = output.read().decode("utf-8", errors="replace")
        if size > output_limit:
            text = "[earlier subprocess output truncated]\n" + text
        if timed_out:
            return 124, text + "\ntimed out; process group terminated"
        return int(process.returncode), text


def unaided_clear(level: str, episodes: int, checkpoint: Path,
                  seed: int = 0) -> dict:
    """Independent fresh-start evaluation result."""
    from smwrl.evaluate import evaluate

    r = evaluate(level, episodes=episodes, checkpoint=checkpoint, seed=seed)
    gc.collect()          # release the emulator slot for the next call
    if r.get("error"):
        return {"clear_rate": 0.0, "mean_progress": 0.0, "episodes": 0,
                "error": r["error"]}
    return r


def wilson_lower(rate: float, n: int, z: float = 1.96) -> float:
    """95% lower confidence bound for a Bernoulli clear rate."""
    if n <= 0:
        return 0.0
    denom = 1.0 + z * z / n
    centre = rate + z * z / (2 * n)
    margin = z * math.sqrt(rate * (1 - rate) / n + z * z / (4 * n * n))
    return max(0.0, (centre - margin) / denom)


def evaluation_score(evaluation: dict) -> tuple[float, float, float]:
    rate = float(evaluation["clear_rate"])
    n = int(evaluation["episodes"])
    return (wilson_lower(rate, n), rate, float(evaluation["mean_progress"]))


def snapshot_latest(level: str) -> Path | None:
    """Freeze the exact latest artifact that an evaluation will measure."""
    directory = CKPT / level
    latest = directory / "latest.zip"
    if not latest.exists():
        return None
    from smwrl.policy import copy_policy_metadata

    candidate = directory / "candidate_eval.zip"
    tmp = directory / "candidate_eval.tmp.zip"
    shutil.copyfile(latest, tmp)
    os.replace(tmp, candidate)
    copy_policy_metadata(latest, candidate)
    return candidate


def cleanup_candidate(candidate: Path | None) -> None:
    if candidate is None:
        return
    from smwrl.policy import metadata_path

    candidate.unlink(missing_ok=True)
    metadata_path(candidate).unlink(missing_ok=True)


def promote_best(level: str, source: Path, evaluation: dict,
                 incumbent: dict | None = None) -> bool:
    """Protect the exact independently evaluated candidate, never mutable latest."""
    if evaluation.get("error") or int(evaluation.get("episodes", 0)) <= 0:
        return False
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    if evaluation.get("model_sha256") != source_hash:
        raise ValueError("evaluation digest does not match the promotion candidate")
    score = evaluation_score(evaluation)
    if incumbent is not None and not incumbent.get("error"):
        if score <= evaluation_score(incumbent):
            return False

    from smwrl.policy import copy_policy_metadata

    directory = CKPT / level
    directory.mkdir(parents=True, exist_ok=True)
    best = directory / "best.zip"
    tmp_model = directory / "best.tmp.zip"
    shutil.copyfile(source, tmp_model)
    os.replace(tmp_model, best)
    copy_policy_metadata(source, best)
    payload = dict(evaluation)
    payload.update({"wilson_lower": score[0], "promoted": time.time(),
                    "source": source.name, "model_sha256": source_hash})
    metrics_path = directory / "best.json"
    tmp_json = directory / "best.tmp.json"
    tmp_json.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(tmp_json, metrics_path)
    return True


def evaluate_candidate(level: str, episodes: int, seed: int) -> tuple[dict, bool, dict | None]:
    """Evaluate a frozen latest and compare it to best on the same fixed suite."""
    candidate = snapshot_latest(level)
    if candidate is None:
        return ({"clear_rate": 0.0, "mean_progress": 0.0, "episodes": 0,
                 "error": "no latest policy"}, False, None)
    try:
        evaluation = unaided_clear(level, episodes, candidate, seed)
        best_path = CKPT / level / "best.zip"
        incumbent = (unaided_clear(level, episodes, best_path, seed)
                     if best_path.exists() else None)
        promoted = promote_best(level, candidate, evaluation, incumbent)
        return evaluation, promoted, incumbent
    finally:
        cleanup_candidate(candidate)


@contextmanager
def level_lock(level: str):
    """Prevent concurrent autopilots from overwriting one level's artifacts."""
    path = CKPT / level / ".autopilot.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+")
    try:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            yield False
        else:
            yield True
    finally:
        handle.close()


def coverage(level: str) -> tuple[float, int]:
    """How far back toward the level start the curriculum reaches, and how many
    cells know a route to the goal.

    A curriculum that only covers the last fifth of a level cannot teach the
    first four fifths, and training on it silently wastes hours -- overnight,
    every level except YoshiIsland2 had 13-24% coverage and every one of them
    finished at 0% unaided.
    """
    from smwrl.archive import Archive
    from smwrl.levels import LEVELS

    try:
        a = Archive.load(CKPT / level / "archive.pkl")
    except Exception:
        return 0.0, 0
    win = a.winning_cells
    # Only room 0 counts: a level always starts there, so cells in a later room
    # say nothing about whether the opening is covered. YoshiIsland3 had 31
    # winning cells, every one of them in room 1, and no route back to the start.
    xs = [c.x for c in win if c.room == 0]
    if not xs:
        return 0.0, len(win)
    frac = 1.0 - (min(xs) / max(1, LEVELS[level].length))
    return max(0.0, min(1.0, frac)), len(win)


def export_curriculum(level: str, worker: str) -> bool:
    """Rebuild curriculum.pkl from the latest periodically-saved archive.

    Exploration is allowed to time out, but it checkpoints archive.pkl while it
    runs and only exports curriculum.pkl on a clean exit.  Without this step the
    verifier trained against an older curriculum after every timed-out extend.
    """
    code, out = run([PY, "-m", "smwrl.explore", "--level", level,
                     "--export-only"], timeout=300)
    if code != 0:
        log(worker, f"  curriculum export failed (rc={code}): {out.strip()[-300:]}")
    return code == 0


def ensure_curriculum(level: str, worker: str, iters: int, extend: bool = False,
                      min_coverage: float = 0.5) -> bool:
    """Make sure a usable curriculum exists. Returns False if the level is hopeless."""
    apath = CKPT / level / "archive.pkl"
    if extend or not apath.exists():
        verb = "extending" if apath.exists() else "building"
        log(worker, f"{verb} frontier archive for {level} ({iters} excursions)")
        cmd = [PY, "-m", "smwrl.explore", "--level", level,
               "--iters", str(iters), "--rollout", "220"]
        if apath.exists():
            cmd.append("--resume")
        code, out = run(cmd, timeout=7200)
        tail = [ln for ln in out.splitlines() if ln.startswith(("cells:", "frontier:", "WARNING"))]
        log(worker, "  " + " | ".join(tail[-3:]) if tail else f"  explore rc={code}")
        if apath.exists():
            export_curriculum(level, worker)

    cov, nwin = coverage(level)
    log(worker, f"  curriculum reaches back {cov:.0%} of the level ({nwin} cells with a route)")
    if cov < min_coverage and not extend:
        log(worker, f"  below {min_coverage:.0%}; exploring more before training")
        run([PY, "-m", "smwrl.explore", "--level", level, "--iters", str(iters),
             "--rollout", "220", "--resume"], timeout=7200)
        if apath.exists():
            export_curriculum(level, worker)
        cov, nwin = coverage(level)
        log(worker, f"  now reaches back {cov:.0%} ({nwin} cells)")

    if cov < min_coverage:
        # verify() only checks the stages nearest the goal, so it happily passes
        # a curriculum covering the last 15% of a level. Training on that burned
        # 9M steps on YoshiIsland3 for 0%. Skip instead and come back to it.
        log(worker, f"  coverage still {cov:.0%} after exploring; skipping rather "
                    f"than training on a curriculum that cannot reach the start")
        return False

    code, out = run([PY, "-m", "smwrl.verify", "--level", level,
                     "--stages", "3", "--episodes", "8"], timeout=1800)
    ok = code == 0
    log(worker, f"  curriculum verify: {'PASS' if ok else 'FAIL'}")
    return ok


def train_chunk(level: str, steps: int, n_envs: int, bar: float) -> tuple[bool, str]:
    # Runs unattended, so cap it: a wedged trainer must not eat the whole night.
    # Sustained rate is ~600 steps/s; 150 is a generous floor before we call it.
    budget_s = max(3600, steps / 150)
    code, out = run([PY, "-m", "smwrl.train", "--level", level, "--resume",
                     "--steps", str(steps), "--n-envs", str(n_envs),
                     "--curriculum", "--curriculum-ratio", "0.4", "--ent-coef", "0.02",
                     "--save-every", "100000", "--stop-at-solo-clear", str(bar)],
                    timeout=budget_s)
    return code == 0, out


def do_level(level: str, args, worker: str, journal: list) -> dict:
    log(worker, f"===== {level} =====")
    entry = {"level": level, "started": time.time(), "chunks": []}

    if not ensure_curriculum(level, worker, args.explore_iters):
        if not ensure_curriculum(level, worker, args.explore_iters * 2, extend=True):
            log(worker, f"  {level}: no usable curriculum, skipping")
            entry["result"] = "no_curriculum"
            return entry

    best, best_prog, stale, done_steps = 0.0, 0.0, 0, 0
    budget = args.budget
    extensions = 0
    while done_steps < budget:
        trained, train_out = train_chunk(level, args.chunk, args.n_envs, args.bar)
        if not trained:
            tail = "\n".join(train_out.strip().splitlines()[-12:])
            log(worker, f"  trainer failed before the chunk completed:\n{tail}")
            entry["result"] = "training_error"
            entry["error"] = tail
            return entry
        done_steps += args.chunk
        evaluation = unaided_clear(level, args.eval_episodes)
        rate = float(evaluation["clear_rate"])
        prog = float(evaluation["mean_progress"])
        lower = wilson_lower(rate, int(evaluation["episodes"]))
        promoted = promote_best(level, evaluation) if not evaluation.get("error") else False
        entry["chunks"].append({"steps": done_steps, "clear": rate,
                                "wilson_lower": lower, "progress": prog,
                                "promoted_best": promoted})
        log(worker, f"  {done_steps/1e6:.1f}M steps -> unaided clear {rate:.0%}, "
                    f"95% lower {lower:.0%}, mean progress {prog:.0f}"
                    f"{' (promoted best)' if promoted else ''}")
        journal_write(worker, journal, entry)

        if lower >= args.bar:
            log(worker, f"  {level} PASSED: lower bound {lower:.0%} >= {args.bar:.0%}")
            entry["result"] = "passed"
            entry["clear_rate"] = rate
            return entry

        improved = rate > best + 0.02 or prog > best_prog * 1.15
        best, best_prog = max(best, rate), max(best_prog, prog)
        stale = 0 if improved else stale + 1

        # Don't abandon a level that is still getting better. YoshiIsland1 was
        # heading for its budget while unaided progress climbed 2321 -> 3400.
        if improved and done_steps + args.chunk >= budget and extensions < args.max_extensions:
            budget += args.budget // 2
            extensions += 1
            log(worker, f"  still improving at the budget; extending to "
                        f"{budget/1e6:.1f}M ({extensions}/{args.max_extensions})")
        if stale >= args.patience:
            # A stalled curriculum has been the cause every time so far.
            log(worker, f"  {level} stalled {stale} chunks; extending the archive")
            ensure_curriculum(level, worker, args.explore_iters, extend=True)
            stale = 0

    log(worker, f"  {level} hit its budget ({budget/1e6:.1f}M) at {best:.0%} unaided "
                f"(best progress {best_prog:.0f}); banking and moving on")
    entry["result"] = "budget_exhausted"
    entry["clear_rate"] = best
    return entry


def journal_write(worker: str, journal: list, current: dict | None = None) -> None:
    out = list(journal) + ([current] if current else [])
    p = CKPT / f"autopilot_{worker}.json"
    tmp = p.with_suffix(".tmp.json")
    tmp.write_text(json.dumps(out, indent=2))
    tmp.replace(p)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--levels", required=True)
    ap.add_argument("--worker", default="a")
    ap.add_argument("--bar", type=float, default=0.95,
                    help="95%% lower confidence bound required to pass")
    ap.add_argument("--chunk", type=int, default=1_500_000, help="steps between evaluations")
    ap.add_argument("--budget", type=int, default=9_000_000, help="max steps per level")
    ap.add_argument("--patience", type=int, default=2, help="stale chunks before re-exploring")
    ap.add_argument("--max-extensions", type=int, default=4,
                    help="times a still-improving level may exceed its budget")
    ap.add_argument("--n-envs", type=int, default=24)
    ap.add_argument("--explore-iters", type=int, default=8000)
    ap.add_argument("--eval-episodes", type=int, default=100)
    ap.add_argument("--min-free-gb", type=float, default=10.0)
    args = ap.parse_args()

    levels = [x.strip() for x in args.levels.split(",") if x.strip()]
    log(args.worker, f"autopilot starting on {len(levels)} level(s): {', '.join(levels)}")
    log(args.worker, f"bar={args.bar:.0%} chunk={args.chunk:,} budget={args.budget:,}/level")

    journal: list = []
    for lv in levels:
        free_gb = shutil.disk_usage(ROOT).free / 1e9
        if free_gb < args.min_free_gb:
            log(args.worker, f"only {free_gb:.1f} GB free; stopping before {lv}")
            journal.append({"level": lv, "result": "low_disk", "free_gb": free_gb})
            journal_write(args.worker, journal)
            break
        try:
            with level_lock(lv) as locked:
                if not locked:
                    log(args.worker, f"{lv} is already owned by another autopilot; skipping")
                    journal.append({"level": lv, "result": "locked"})
                else:
                    journal.append(do_level(lv, args, args.worker, journal))
        except KeyboardInterrupt:
            log(args.worker, "interrupted")
            break
        except Exception as e:      # never let one bad level end the night
            log(args.worker, f"  {lv} raised {type(e).__name__}: {e}")
            journal.append({"level": lv, "result": "error", "error": repr(e)})
        journal_write(args.worker, journal)

    log(args.worker, "=== summary ===")
    for e in journal:
        log(args.worker, f"  {e['level']:<20} {e.get('result','?'):<18} "
                         f"clear {e.get('clear_rate', 0):.0%}")
    journal_write(args.worker, journal)


if __name__ == "__main__":
    sys.exit(main())
