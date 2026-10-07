from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import time
import uuid
from contextlib import contextmanager

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("RALLY_LAB_DATA", ROOT / ".rally")).resolve()
UNITY = Path("/Applications/Unity/Hub/Editor/6000.4.11f1/Unity.app/Contents/MacOS/Unity")
PYTHON = ROOT / ".venv/bin/python"
ACTIVE = {"queued", "starting", "running", "stopping"}
RESUME_SAFE_LAB_RUNTIME_HASHES = {
    "b2082150e30791c20cfb1272693371315dcb0af596d4267faef9b2cc93cf2d6d",
    "d65c768e92644fa1b19593c4bef1e7f8cc2152d22ec2958681016dd295977e81",
}
_UNSET = object()


def read(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False))
    temporary.replace(path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def course_identity(manifest):
    source = manifest.get("source")
    definition = manifest.get("definition")
    if not source or not isinstance(definition, dict):
        return None
    payload = source + "\n" + json.dumps(definition, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def course_identity_valid(manifest):
    return bool(manifest.get("id")) and course_identity(manifest) == manifest.get("id")


def course_bundle_valid(manifest):
    bundle = Path(manifest.get("bundle", ""))
    return bundle.is_file() and bool(manifest.get("bundle_hash")) and sha(bundle) == manifest["bundle_hash"]


def course_asset_valid(manifest):
    return course_identity_valid(manifest) and course_bundle_valid(manifest)


def fingerprint():
    files = {}
    for directory in ("Assets", "Packages", "ProjectSettings"):
        for path in sorted((ROOT / directory).rglob("*")):
            if not path.is_file() or "RallyLabGenerated" in path.parts or path.name == "RallyLabGenerated.meta":
                continue
            if path.suffix in {".cs", ".shader", ".json", ".asset", ".unity", ".prefab", ".mat", ".meta"}:
                files[str(path.relative_to(ROOT))] = sha(path)
    return {"schema": 1, "hash": hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest(), "files": files}


def resume_build_compatibility(previous, current):
    if previous.get("contract") != current.get("contract"):
        return False, "Driving contract changed"
    old_source, new_source = previous.get("source", {}), current.get("source", {})
    if old_source.get("hash") == new_source.get("hash"):
        return True, []
    old_files, new_files = old_source.get("files", {}), new_source.get("files", {})
    changed = [path for path in sorted(set(old_files) | set(new_files)) if old_files.get(path) != new_files.get(path)]
    allowed = "Assets/Core/ML/LabRuntime.cs"
    if changed == [allowed] and {old_files.get(allowed), new_files.get(allowed)} == RESUME_SAFE_LAB_RUNTIME_HASHES:
        return True, changed
    return False, "Environment source changed: " + ", ".join(changed[:5])


def build_status():
    manifest = read(DATA / "build.json")
    if not manifest:
        return {"status": "legacy", "reason": "No managed build manifest"}
    current = fingerprint()
    executable = Path(manifest["executable"])
    good = executable.is_file() and sha(executable) == manifest["executable_hash"]
    return {**manifest, "status": "current" if current["hash"] == manifest["source"]["hash"] and good else "stale"}


@contextmanager
def db():
    DATA.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(DATA / "lab.sqlite", timeout=30)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("CREATE TABLE IF NOT EXISTS jobs (id TEXT PRIMARY KEY, kind TEXT, state TEXT, created REAL, updated REAL, spec TEXT, detail TEXT)")
    connection.execute("CREATE TABLE IF NOT EXISTS telemetry_events (job_id TEXT NOT NULL, seq INTEGER NOT NULL, kind TEXT NOT NULL, created REAL NOT NULL, payload TEXT NOT NULL, PRIMARY KEY(job_id, seq))")
    connection.execute("CREATE INDEX IF NOT EXISTS telemetry_kind_seq ON telemetry_events(job_id, kind, seq)")
    connection.execute("CREATE TABLE IF NOT EXISTS telemetry_offsets (job_id TEXT NOT NULL, path TEXT NOT NULL, offset INTEGER NOT NULL, PRIMARY KEY(job_id, path))")
    connection.execute("CREATE TABLE IF NOT EXISTS telemetry_episodes (job_id TEXT NOT NULL, seq INTEGER NOT NULL, source TEXT NOT NULL, source_offset INTEGER NOT NULL, outcome TEXT, valid INTEGER NOT NULL DEFAULT 0, seconds REAL, course_id TEXT, attempt INTEGER, payload TEXT NOT NULL, PRIMARY KEY(job_id, source, source_offset))")
    connection.execute("CREATE INDEX IF NOT EXISTS telemetry_episode_seq ON telemetry_episodes(job_id, seq)")
    connection.execute("CREATE TABLE IF NOT EXISTS telemetry_scalars (job_id TEXT NOT NULL, tag TEXT NOT NULL, step INTEGER NOT NULL, seq INTEGER NOT NULL, wall_time REAL, value REAL NOT NULL, PRIMARY KEY(job_id, tag, step))")
    connection.commit()
    try:
        yield connection
        connection.commit()
    finally:
        connection.close()


def jobs():
    with db() as conn:
        return [decode(row) for row in conn.execute("SELECT * FROM jobs ORDER BY created DESC")]


def decode(row):
    result = dict(row)
    for key in ("spec", "detail"):
        result[key] = json.loads(result[key])
    return result


def job(identifier):
    with db() as conn:
        row = conn.execute("SELECT * FROM jobs WHERE id=?", (identifier,)).fetchone()
    if row is None:
        raise ValueError("Unknown job")
    return decode(row)


def update(identifier, state=None, **details):
    with db() as conn:
        row = conn.execute("SELECT * FROM jobs WHERE id=?", (identifier,)).fetchone()
        previous = decode(row)
        now = time.time()
        next_state = state or previous["state"]
        if next_state != previous["state"]:
            _insert_event(conn, identifier, "lifecycle", {"from": previous["state"], "state": next_state}, now)
        conn.execute("UPDATE jobs SET state=?, updated=?, detail=? WHERE id=?",
                     (next_state, now, json.dumps({**previous["detail"], **details}), identifier))


def create(kind, spec):
    identifier = uuid.uuid4().hex[:16]
    path = DATA / "jobs" / identifier
    path.mkdir(parents=True)
    write(path / "spec.json", {"schema": 1, "kind": kind, **spec})
    with db() as conn:
        conn.execute("INSERT INTO jobs VALUES(?,?,?,?,?,?,?)", (identifier, kind, "queued", time.time(), time.time(), json.dumps(spec), "{}"))
        _insert_event(conn, identifier, "lifecycle", {"state": "queued", "created": True}, time.time())
    spawn(identifier)
    return job(identifier)


def spawn(identifier):
    path = DATA / "jobs" / identifier
    with (path / "supervisor.log").open("a") as log:
        process = subprocess.Popen([str(PYTHON), "-m", "rallylab.worker", identifier], cwd=ROOT,
                                   stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    import psutil
    update(identifier, supervisor_pid=process.pid, supervisor_created=psutil.Process(process.pid).create_time())


def reconcile():
    import psutil
    for item in jobs():
        if item["state"] not in ACTIVE:
            continue
        detail = item["detail"]
        try:
            process = psutil.Process(detail["supervisor_pid"])
            alive = abs(process.create_time() - detail["supervisor_created"]) < .01 and process.status() != psutil.STATUS_ZOMBIE
        except (KeyError, psutil.Error):
            alive = False
        if not alive and time.time() - item["updated"] > 5:
            update(item["id"], "interrupted", error="Supervisor exited; inspect logs and saved state before resuming")


def courses():
    return [read(p) for p in sorted((DATA / "courses").glob("*/manifest.json")) if read(p)]


def checkpoints():
    return [read(p) for p in sorted((DATA / "checkpoints").glob("*/manifest.json")) if read(p)]


def demonstrations():
    return [read(p) for p in sorted((DATA / "demonstrations").glob("*/manifest.json")) if read(p)]


def episodes(identifier):
    records = []
    for path in sorted((DATA / "jobs" / identifier).rglob("episodes-*.jsonl")):
        for line in path.read_text(errors="replace").splitlines():
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # A writer may still be completing the last line.
    return records


def _insert_event(conn, identifier, kind, payload, created=None):
    row = conn.execute("SELECT COALESCE(MAX(seq), 0) + 1 AS seq FROM telemetry_events WHERE job_id=?", (identifier,)).fetchone()
    seq = row["seq"]
    conn.execute("INSERT INTO telemetry_events VALUES(?,?,?,?,?)",
                 (identifier, seq, kind, created or time.time(), json.dumps(payload, allow_nan=False)))
    return seq


def append_telemetry(identifier, kind, payload, created=None):
    """Append one durable, monotonically sequenced event for a managed job."""
    with db() as conn:
        if kind == "checkpoint" and conn.execute(
                "SELECT 1 FROM telemetry_events WHERE job_id=? AND kind='checkpoint' AND json_extract(payload,'$.id')=? LIMIT 1",
                (identifier, payload.get("id"))).fetchone():
            return None
        if kind == "scalar":
            tag, step = str(payload["tag"]), int(payload["step"])
            if conn.execute("SELECT 1 FROM telemetry_scalars WHERE job_id=? AND tag=? AND step=?",
                            (identifier, tag, step)).fetchone():
                return None
            seq = _insert_event(conn, identifier, kind, payload, created)
            conn.execute("INSERT INTO telemetry_scalars VALUES(?,?,?,?,?,?)",
                         (identifier, tag, step, seq, payload.get("wall_time"), float(payload["value"])))
            return seq
        return _insert_event(conn, identifier, kind, payload, created)


def capture_job_samples(identifier, item=None, min_gap=25, cpu_percent=_UNSET):
    """Backfill sparse samples for supervisors started before telemetry was installed."""
    item = item or job(identifier)
    detail = item["detail"]
    now = time.time()
    sampled_cpu = detail.get("cpu_percent") if cpu_percent is _UNSET else cpu_percent
    with db() as conn:
        latest_resource = conn.execute(
            "SELECT created FROM telemetry_events WHERE job_id=? AND kind='resource' ORDER BY seq DESC LIMIT 1",
            (identifier,)).fetchone()
    sample_due = latest_resource is None or now - latest_resource["created"] >= min_gap
    if sample_due and any(value is not None for value in (sampled_cpu, detail.get("memory_bytes"), detail.get("children"))):
        heartbeat = detail.get("heartbeat")
        append_telemetry(identifier, "resource", {
            "source": "dashboard-sampler",
            "cpu_percent": sampled_cpu,
            "memory_bytes": detail.get("memory_bytes"),
            "children": detail.get("children"),
            "trainer_step": detail.get("trainer_step"),
            "steps_per_second": detail.get("steps_per_second"),
            "heartbeat": heartbeat,
            "stale": heartbeat is not None and now - heartbeat > 10,
        }, created=now)
    step, reward = detail.get("trainer_step"), detail.get("cumulative_reward")
    if step is not None and reward is not None:
        append_telemetry(identifier, "scalar", {
            "tag": "Environment/Cumulative Reward", "step": int(step),
            "value": float(reward), "wall_time": detail.get("last_summary_time") or now,
        }, created=detail.get("last_summary_time") or now)


def ingest_episode_events(identifier):
    """Index only newline-complete episode records; an in-progress final write waits."""
    folder = DATA / "jobs" / identifier
    if not folder.is_dir():
        return
    files = sorted(folder.rglob("episodes-*.jsonl"))
    with db() as conn:
        for path in files:
            source = str(path.resolve())
            offset_row = conn.execute("SELECT offset FROM telemetry_offsets WHERE job_id=? AND path=?",
                                      (identifier, source)).fetchone()
            offset = offset_row["offset"] if offset_row else 0
            try:
                size = path.stat().st_size
                if offset > size:
                    offset = 0
                with path.open("rb") as stream:
                    stream.seek(offset)
                    chunk = stream.read()
            except FileNotFoundError:
                continue
            last_newline = chunk.rfind(b"\n")
            if last_newline < 0:
                continue
            complete = chunk[:last_newline + 1]
            consumed = 0
            for line in complete.splitlines(keepends=True):
                line_offset = offset + consumed
                consumed += len(line)
                try:
                    record = json.loads(line)
                except (json.JSONDecodeError, UnicodeDecodeError):
                    continue
                if not isinstance(record, dict):
                    continue
                cursor = conn.execute("SELECT COALESCE(MAX(seq), 0) + 1 AS seq FROM telemetry_events WHERE job_id=?",
                                      (identifier,)).fetchone()["seq"]
                created = record.get("t")
                try:
                    created = float(created) if created is not None else time.time()
                except (TypeError, ValueError):
                    created = time.time()
                payload = json.dumps(record, allow_nan=False)
                inserted = conn.execute(
                    "INSERT OR IGNORE INTO telemetry_episodes(job_id,seq,source,source_offset,outcome,valid,seconds,course_id,attempt,payload) VALUES(?,?,?,?,?,?,?,?,?,?)",
                    (identifier, cursor, source, line_offset, record.get("outcome"),
                     int(bool(record.get("valid"))), record.get("seconds"),
                     record.get("courseId"), record.get("attempt"), payload))
                if inserted.rowcount:
                    conn.execute("INSERT INTO telemetry_events VALUES(?,?,?,?,?)",
                                 (identifier, cursor, "episode", created, payload))
            new_offset = offset + len(complete)
            conn.execute("INSERT INTO telemetry_offsets VALUES(?,?,?) ON CONFLICT(job_id,path) DO UPDATE SET offset=excluded.offset",
                         (identifier, source, new_offset))


def indexed_episodes(identifier, limit=None):
    with db() as conn:
        if limit is None:
            rows = conn.execute("SELECT payload FROM telemetry_episodes WHERE job_id=? ORDER BY seq", (identifier,)).fetchall()
        else:
            rows = conn.execute("SELECT payload FROM telemetry_episodes WHERE job_id=? ORDER BY seq DESC LIMIT ?",
                                (identifier, limit)).fetchall()[::-1]
    return [json.loads(row["payload"]) for row in rows]


def indexed_summary(identifier, expected=None):
    with db() as conn:
        total = conn.execute("SELECT COUNT(*) AS n FROM telemetry_episodes WHERE job_id=?", (identifier,)).fetchone()["n"]
        outcomes = {row["outcome"] or "Unknown": row["n"] for row in conn.execute(
            "SELECT outcome,COUNT(*) AS n FROM telemetry_episodes WHERE job_id=? GROUP BY outcome", (identifier,))}
        valid_rows = conn.execute(
            "SELECT seconds FROM telemetry_episodes WHERE job_id=? AND outcome='Finished' AND valid=1 AND seconds IS NOT NULL ORDER BY seconds",
            (identifier,)).fetchall()
        by_course = {row["course_id"]: row["n"] for row in conn.execute(
            "SELECT course_id,COUNT(*) AS n FROM telemetry_episodes WHERE job_id=? AND course_id IS NOT NULL GROUP BY course_id", (identifier,))}
    valid = [row["seconds"] for row in valid_rows]
    count = len(valid)
    interval = None
    if total:
        p, z = count / total, 1.96
        centre = (p + z*z/(2*total)) / (1 + z*z/total)
        margin = z * math.sqrt(p*(1-p)/total + z*z/(4*total*total)) / (1 + z*z/total)
        interval = [max(0, centre-margin), min(1, centre+margin)]
    median = None
    if count:
        middle = count // 2
        median = valid[middle] if count % 2 else (valid[middle-1] + valid[middle]) / 2
    return {"attempts": total, "episodes_completed": total, "requested": expected,
            "complete": expected is not None and total == expected,
            "finishes": count, "best": min(valid) if valid else None,
            "median": median, "finish_interval": interval, "outcomes": outcomes,
            "course_attempts": by_course}


def _sampled_events(conn, identifier, kind, limit):
    rows = conn.execute("SELECT seq,created,payload FROM telemetry_events WHERE job_id=? AND kind=? ORDER BY seq",
                        (identifier, kind)).fetchall()
    if len(rows) > limit:
        stride = (len(rows) - 1) / (limit - 1)
        rows = [rows[round(index * stride)] for index in range(limit)]
    return [{"seq": row["seq"], "created": row["created"], **json.loads(row["payload"])} for row in rows]


def telemetry_snapshot(identifier):
    with db() as conn:
        latest = conn.execute("SELECT COALESCE(MAX(seq),0) AS seq FROM telemetry_events WHERE job_id=?",
                              (identifier,)).fetchone()["seq"]
        history = {
            "episodes": _sampled_events(conn, identifier, "episode", 1600),
            "scalars": _sampled_events(conn, identifier, "scalar", 2400),
            "resources": _sampled_events(conn, identifier, "resource", 500),
            "checkpoints": _sampled_events(conn, identifier, "checkpoint", 250),
            "lifecycle": _sampled_events(conn, identifier, "lifecycle", 250),
        }
    return {"cursor": latest, "history": history}


def telemetry_events_after(identifier, cursor, limit=500):
    with db() as conn:
        rows = conn.execute("SELECT seq,kind,created,payload FROM telemetry_events WHERE job_id=? AND seq>? ORDER BY seq LIMIT ?",
                            (identifier, cursor, limit)).fetchall()
    events = [{"seq": row["seq"], "kind": row["kind"], "created": row["created"], **json.loads(row["payload"])}
              for row in rows]
    return events, (events[-1]["seq"] if events else cursor)


def summary(records, expected=None):
    import math
    import statistics
    valid = [r["seconds"] for r in records if r.get("outcome") == "Finished" and r.get("valid", False)]
    n, k = len(records), len(valid)
    interval = None
    if n:
        p, z = k / n, 1.96
        centre = (p + z*z/(2*n)) / (1 + z*z/n)
        margin = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / (1 + z*z/n)
        interval = [max(0, centre-margin), min(1, centre+margin)]
    outcomes = {}
    for row in records:
        key = row.get("outcome", "Unknown")
        outcomes[key] = outcomes.get(key, 0) + 1
    return {"attempts": n, "requested": expected, "complete": expected is not None and n == expected,
            "finishes": k, "best": min(valid) if valid else None,
            "median": statistics.median(valid) if valid else None,
            "finish_interval": interval, "outcomes": outcomes}
