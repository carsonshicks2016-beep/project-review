from __future__ import annotations

import asyncio
import csv
import io
import json
from pathlib import Path
import psutil
import threading
import time
from typing import Literal
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi import Query
from pydantic import BaseModel, Field

from . import core

_CPU_WATCH: dict[str, dict[int, psutil.Process]] = {}
_CPU_WATCH_LOCK = threading.Lock()


def sample_process_tree_cpu(item):
    """Sample a run's process tree using stable psutil objects for real deltas."""
    identifier = item["id"]
    pid = item.get("detail", {}).get("child_pid")
    if not isinstance(pid, int) or pid <= 0:
        with _CPU_WATCH_LOCK:
            _CPU_WATCH.pop(identifier, None)
        return None
    try:
        root = psutil.Process(pid)
        processes = [root, *root.children(recursive=True)]
    except psutil.Error:
        with _CPU_WATCH_LOCK:
            _CPU_WATCH.pop(identifier, None)
        return None

    with _CPU_WATCH_LOCK:
        watched = _CPU_WATCH.setdefault(identifier, {})
        current = {process.pid: process for process in processes}
        for stale_pid in watched.keys() - current.keys():
            watched.pop(stale_pid, None)
        total = 0.0
        measured = False
        for process_pid, process in current.items():
            previous = watched.get(process_pid)
            try:
                if previous is None:
                    process.cpu_percent(None)
                    watched[process_pid] = process
                else:
                    total += previous.cpu_percent(None)
                    measured = True
            except psutil.Error:
                watched.pop(process_pid, None)
        return total if measured else None


async def telemetry_sampler():
    while True:
        try:
            items = await asyncio.to_thread(core.jobs)
            for item in items:
                if item["state"] in core.ACTIVE:
                    cpu = await asyncio.to_thread(sample_process_tree_cpu, item)
                    await asyncio.to_thread(core.capture_job_samples, item["id"], item, min_gap=10, cpu_percent=cpu)
                else:
                    with _CPU_WATCH_LOCK:
                        _CPU_WATCH.pop(item["id"], None)
        except Exception:
            pass
        await asyncio.sleep(15)


@asynccontextmanager
async def lifespan(_app):
    sampler = asyncio.create_task(telemetry_sampler())
    try:
        yield
    finally:
        sampler.cancel()
        try:
            await sampler
        except asyncio.CancelledError:
            pass


app = FastAPI(title="Rally Research Lab", version="1.0", lifespan=lifespan)


@app.middleware("http")
async def local_only(request: Request, call_next):
    from urllib.parse import urlparse
    host = request.url.hostname
    origin = request.headers.get("origin")
    if host not in {"127.0.0.1", "localhost", "testserver"}:
        return JSONResponse({"detail": "Local access only"}, status_code=403)
    if request.method not in {"GET", "HEAD", "OPTIONS"} and origin:
        parsed = urlparse(origin)
        if parsed.hostname not in {"localhost", "127.0.0.1"} or parsed.port not in {request.url.port, 5173}:
            return JSONResponse({"detail": "Untrusted control origin"}, status_code=403)
    try:
        return await call_next(request)
    except ValueError as exc:
        return JSONResponse({"detail": str(exc)}, status_code=409)


class RunSpec(BaseModel):
    name: str = Field(min_length=1, max_length=80)
    mode: Literal["specialist", "generalist"] = "specialist"
    course: str = ""
    seed: int = Field(default=42, ge=0, le=2147483647)
    steps: int = Field(default=2000000, ge=1000, le=25000000)
    workers: int = Field(default=1, ge=1, le=4)
    reward: Literal["baseline-v1", "time-attack-v1", "time-attack-no-slip-v1"] = "baseline-v1"
    startingGear: Literal["neutral", "first"] = "neutral"
    parent: str = ""
    demonstrations: list[str] = Field(default_factory=list, max_length=30)
    imitationSteps: int = Field(default=10000, ge=1000, le=1000000)
    imitationMode: Literal["isolated", "joint"] = "isolated"


class CourseSpec(BaseModel):
    length: int = Field(default=1000, ge=1000, le=5000)
    name: str = Field(min_length=1, max_length=80)
    seed: int = Field(ge=0, le=2147483647)
    family: Literal["gentle", "technical", "crests"]
    rocks: float = Field(default=0, ge=0, le=2)
    scenery: bool = True
    suite: Literal["library", "development", "held-out"] = "library"


class CircuitSpec(BaseModel):
    circuit: Literal["nordschleife"] = "nordschleife"
    firstSector: int = Field(default=0, ge=0, le=15)
    sectorCount: int = Field(default=16, ge=1, le=16)


class CircuitPreviewSpec(BaseModel):
    camera: Literal["ThirdPerson", "Driver", "Hood", "Trackside", "Helicopter", "Cinematic"] = "ThirdPerson"


class DemonstrationSpec(BaseModel):
    course: str
    seed: int = Field(default=2026, ge=0, le=2147483647)
    timeScale: float = Field(default=10, ge=1, le=20)
    startingGear: Literal["neutral", "first"] = "first"


class EvaluationSpec(BaseModel):
    checkpoint: str
    courses: list[str] = Field(min_length=1, max_length=30)
    attempts: int = Field(default=20, ge=1, le=100)
    seed: int = Field(default=2026, ge=0, le=2147483647)
    deterministic: bool = False
    timeScale: float = Field(default=10, ge=1, le=20)
    startingGear: Literal["neutral", "first"] = "neutral"
    controlProbe: bool = False
    recordDemonstration: bool = False
    controlMode: Literal["constant-throttle", "waypoint-follow", "steering-step", "fixed-input"] = "constant-throttle"
    controlProbeSeconds: float = Field(default=0, ge=0, le=120)
    controlSteer: float = Field(default=0, ge=-0.25, le=0.25)
    controlDrive: float = Field(default=0.1, ge=-1, le=1)
    controlTargetSpeed: float = Field(default=8, ge=0, le=20)


def state():
    core.reconcile()
    items = core.jobs()
    for item in items:
        core.ingest_episode_events(item["id"])
        if item["kind"] == "evaluation":
            attempts = item["spec"].get("attempts", 0)
            expected_courses = item["spec"].get("courses", [])
            result = core.indexed_summary(item["id"], len(expected_courses) * attempts)
            result["course_attempts"] = {course: result["course_attempts"].get(course, 0) for course in expected_courses}
            result["complete"] = item["state"] == "completed" and all(count == attempts for count in result["course_attempts"].values())
            item["summary"] = result
        elif item["kind"] == "demonstration":
            result = core.indexed_summary(item["id"], 1)
            result["complete"] = item["state"] == "completed" and result["attempts"] == 1
            item["summary"] = result
        else:
            item["summary"] = core.indexed_summary(item["id"])
    return {"jobs": items, "courses": core.courses(), "checkpoints": core.checkpoints(),
            "demonstrations": core.demonstrations(), "time": time.time()}


@app.get("/api/state")
def get_state():
    return state()


@app.get("/api/build")
def get_build():
    return core.build_status()


def require_course(identifier: str):
    course = next((item for item in core.courses() if item["id"] == identifier), None)
    if not course or not core.course_asset_valid(course):
        raise ValueError(f"Course {identifier} is missing or failed its identity check")
    return course


def require_checkpoint(identifier: str, prepared: bool):
    checkpoint = next((item for item in core.checkpoints() if item["id"] == identifier), None)
    build = core.build_status()
    if not checkpoint or build["status"] != "current" or checkpoint.get("contract") != build.get("contract"):
        raise ValueError("Checkpoint or current compatible player is unavailable")
    policy = Path(checkpoint.get("policy", ""))
    if not policy.is_file() or core.sha(policy) != checkpoint["id"]:
        raise ValueError("Checkpoint policy is missing or modified")
    if prepared:
        bundle = Path(checkpoint.get("bundle", ""))
        if not checkpoint.get("prepared") or not bundle.is_file() or core.sha(bundle) != checkpoint.get("bundle_hash"):
            raise ValueError("Prepare an intact checkpoint bundle before viewing or evaluation")
    return checkpoint


@app.post("/api/build")
def build():
    return core.create("build", {})


@app.get("/api/events")
async def events(request: Request):
    async def stream():
        while not await request.is_disconnected():
            value = await asyncio.to_thread(state)
            yield f"data: {json.dumps(value)}\n\n"
            await asyncio.sleep(2)
    return StreamingResponse(stream(), media_type="text/event-stream")


@app.post("/api/runs")
def run(spec: RunSpec):
    if spec.mode == "specialist":
        course = require_course(spec.course)
        if course.get("legacyObstacleLayout"):
            raise ValueError("This course contains stale template road rocks. Choose one of the corrected course assets for new specialist training.")
        definition = course.get("definition", {})
        # Imported circuits count their barriers as obstacle objects; their eligibility is
        # the trainability gate on review, not the procedural zero-hazard audit.
        if course.get("scenery") is False and definition.get("topology") != "circuit":
            if course.get("rocks", 0) != 0 or definition.get("scenery") is not False or definition.get("obstacleObjects") != 0:
                raise ValueError("This no-hazards course is missing a verified zero-obstacle generation audit")
        if course["suite"] != "library" or course.get("review") != "reviewed":
            raise ValueError("Select a reviewed library course for specialist training")
    if core.build_status()["status"] != "current":
        raise ValueError("Build a current managed player first")
    if len(set(spec.demonstrations)) != len(spec.demonstrations):
        raise ValueError("Duplicate demonstrations")
    known = {d["id"]: d for d in core.demonstrations()}
    for identifier in spec.demonstrations:
        item = known.get(identifier)
        if not item or item.get("contract") != core.build_status()["contract"]:
            raise ValueError(f"Unknown or incompatible demonstration: {identifier}")
        if not Path(item["path"]).is_file() or core.sha(item["path"]) != item.get("sha256"):
            raise ValueError(f"Demonstration failed its integrity check: {identifier}")
    return core.create("training", {"schema": 1, **spec.model_dump()})


@app.post("/api/courses")
def course(spec: CourseSpec):
    return core.create("course", {"schema": 1, **spec.model_dump()})


@app.post("/api/circuits/import")
def import_circuit(spec: CircuitSpec):
    from .circuit import convert, variant
    data=convert()
    choice=variant(data,spec.firstSector,spec.sectorCount)
    return core.create("circuit", {"schema": 2, **spec.model_dump(), "revision": choice["revision"]})


@app.post("/api/circuits/{identifier}/preview")
def preview_circuit(identifier: str, spec: CircuitPreviewSpec):
    course=require_course(identifier)
    if course["definition"].get("topology")!="circuit":
        raise ValueError("Select an imported circuit")
    if core.build_status()["status"]!="current":
        raise ValueError("Prepare a current player before circuit review")
    return core.create("circuit-preview",{"course":identifier,**spec.model_dump()})


@app.post("/api/demonstrations/record")
def record_demonstration(spec: DemonstrationSpec):
    if core.build_status()["status"] != "current":
        raise ValueError("Build a current managed player first")
    course = require_course(spec.course)
    if course.get("legacyObstacleLayout"):
        raise ValueError("This course contains stale template road rocks. Record demonstrations on a corrected course asset instead.")
    if course["suite"] != "library" or course.get("review") != "reviewed":
        raise ValueError("Select a reviewed library course")
    return core.create("demonstration", {"schema": 1, **spec.model_dump()})


@app.post("/api/courses/{identifier}/review")
def review_course(identifier: str):
    manifest_path = next((p for p in (core.DATA / "courses").glob("*/manifest.json") if core.read(p, {}).get("id") == identifier), None)
    if manifest_path is None: raise ValueError("Unknown course")
    manifest = core.read(manifest_path)
    if not core.course_asset_valid(manifest): raise ValueError("Course asset failed its identity check")
    if manifest.get("definition", {}).get("topology") == "circuit" and manifest.get("trainability", {}).get("stage") != "validated":
        raise ValueError("Circuit visual inspection does not establish training readiness; runtime and bounded training validation remain pending")
    manifest["review"] = "reviewed"
    manifest["reviewed_at"] = time.time()
    core.write(manifest_path, manifest)
    return manifest


@app.patch("/api/courses/{identifier}/name")
async def rename_course(identifier: str, request: Request):
    payload = await request.json()
    name = payload.get("name")
    if not isinstance(name, str) or not name.strip() or len(name.strip()) > 80: raise ValueError("Name must contain 1 to 80 characters")
    manifest_path = next((p for p in (core.DATA / "courses").glob("*/manifest.json") if core.read(p, {}).get("id") == identifier), None)
    if manifest_path is None: raise ValueError("Unknown course")
    manifest = core.read(manifest_path)
    manifest["name"] = name.strip()
    core.write(manifest_path, manifest)
    return manifest


@app.post("/api/courses/seed-library")
def seed_library():
    existing = {(c["seed"], c["suite"]) for c in core.courses()}
    existing |= {(j["spec"].get("seed"), j["spec"].get("suite")) for j in core.jobs() if j["kind"] == "course" and j["state"] in core.ACTIVE}
    result = []
    for family_index, family in enumerate(("gentle", "technical", "crests")):
        for rocks in (0, 2):
            seed = 41000 + family_index * 100 + rocks
            if (seed, "library") not in existing:
                result.append(core.create("course", {"schema": 1, "name": f"{family.title()} / {'clear' if rocks == 0 else 'rocks'}", "seed": seed, "family": family, "rocks": rocks, "suite": "library"}))
    return result


@app.post("/api/suites/create")
def seed_suites():
    if any(j["kind"] == "suite" and j["state"] in core.ACTIVE for j in core.jobs()):
        raise ValueError("Benchmark suite generation is already queued")
    return core.create("suite", {"schema": 1, "name": "30-course development and 30-course held-out suites"})


@app.post("/api/jobs/{identifier}/stop")
def stop(identifier: str):
    item = core.job(identifier)
    if item["state"] not in core.ACTIVE: raise ValueError("Job is not active")
    (core.DATA / "jobs" / identifier / "stop").touch()
    core.update(identifier, "stopping")
    return core.job(identifier)


@app.post("/api/jobs/{identifier}/force")
def force(identifier: str):
    item = core.job(identifier)
    if item["state"] != "stopping": raise ValueError("Request a normal stop first")
    (core.DATA / "jobs" / identifier / "force").touch()
    return item


@app.post("/api/jobs/{identifier}/resume")
def resume(identifier: str):
    item = core.job(identifier)
    if item["kind"] != "training" or item["state"] not in {"stopped", "interrupted", "failed"}:
        raise ValueError("Only stopped training can resume")
    course_id = item["spec"].get("course")
    course = next((value for value in core.courses() if value["id"] == course_id), None)
    if course and course.get("legacyObstacleLayout"):
        raise ValueError("This run uses a course with stale template road rocks and cannot resume. Its saved results remain available; start a fresh run on a corrected course.")
    folder = core.DATA / "jobs" / identifier
    if not list((folder / "trainer").rglob("*.pt")): raise ValueError("No saved trainer state")
    for name in ("stop", "force"): (folder / name).unlink(missing_ok=True)
    core.update(identifier, "queued", error=None)
    core.spawn(identifier)
    return core.job(identifier)


@app.post("/api/checkpoints/{identifier}/prepare")
def prepare(identifier: str):
    require_checkpoint(identifier, prepared=False)
    return core.create("prepare", {"checkpoint": identifier})


@app.post("/api/evaluations")
def evaluate(spec: EvaluationSpec):
    if len(set(spec.courses)) != len(spec.courses): raise ValueError("Duplicate courses")
    if spec.recordDemonstration and not spec.controlProbe:
        raise ValueError("Demonstration recording requires an explicit control probe")
    require_checkpoint(spec.checkpoint, prepared=True)
    for identifier in spec.courses:
        require_course(identifier)
    return core.create("evaluation", spec.model_dump())


@app.post("/api/viewers")
def viewer(spec: EvaluationSpec):
    if len(spec.courses) != 1: raise ValueError("Select one course to watch")
    if spec.controlProbe: raise ValueError("Control probes run as bounded evaluations, not viewers")
    require_checkpoint(spec.checkpoint, prepared=True)
    require_course(spec.courses[0])
    spec = spec.model_copy(update={"attempts": 1})
    return core.create("viewer", spec.model_dump())


@app.get("/api/jobs/{identifier}")
def detail(identifier: str):
    item = core.job(identifier)
    folder = core.DATA / "jobs" / identifier
    log = ""
    for name in ("supervisor.log", "process.log", "unity.log"):
        path = folder / name
        if path.exists():
            with path.open("rb") as stream:
                stream.seek(max(0, path.stat().st_size - 12000))
                log += stream.read().decode(errors="replace")
    return {**item, "episodes": core.episodes(identifier), "log": log[-24000:]}


@app.get("/api/jobs/{identifier}/telemetry")
def telemetry(identifier: str, cursor: int | None = Query(default=None, ge=0), limit: int = Query(default=500, ge=1, le=1000)):
    item = core.job(identifier)
    core.ingest_episode_events(identifier)
    if item["state"] in core.ACTIVE:
        cpu = sample_process_tree_cpu(item)
        core.capture_job_samples(identifier, item, min_gap=10, cpu_percent=cpu)
    item = core.job(identifier)
    expected = None
    if item["kind"] == "evaluation":
        expected = len(item["spec"].get("courses", [])) * item["spec"].get("attempts", 0)
    elif item["kind"] == "demonstration":
        expected = 1
    result = core.indexed_summary(identifier, expected)
    if item["kind"] == "evaluation":
        result["complete"] = item["state"] == "completed" and all(
            result["course_attempts"].get(course, 0) == item["spec"].get("attempts", 0)
            for course in item["spec"].get("courses", []))
    elif item["kind"] == "demonstration":
        result["complete"] = item["state"] == "completed" and result["attempts"] == 1
    if cursor is None:
        stream = core.telemetry_snapshot(identifier)
        payload = {**stream, "job": item, "summary": result, "incremental": False}
        folder = core.DATA / "jobs" / identifier
        payload["manifest"] = core.read(folder / "run-manifest.json") or core.read(folder / "evaluation-manifest.json") or core.read(folder / "reference-manifest.json")
        payload["log"] = _job_log(folder)
        return payload
    events, next_cursor = core.telemetry_events_after(identifier, cursor, limit)
    return {"job": item, "summary": result, "events": events, "cursor": next_cursor,
            "incremental": True, "server_time": time.time(), "log": _job_log(core.DATA / "jobs" / identifier)}


def _job_log(folder: Path):
    log = ""
    for name in ("supervisor.log", "process.log", "unity.log"):
        path = folder / name
        if path.exists():
            with path.open("rb") as stream:
                stream.seek(max(0, path.stat().st_size - 12000))
                log += stream.read().decode(errors="replace")
    return log[-24000:]


@app.get("/api/jobs/{identifier}/export")
def export(identifier: str, format: str = "json"):
    item = core.job(identifier)
    records = core.episodes(identifier)
    if format == "csv":
        stream = io.StringIO()
        writer = csv.DictWriter(stream, fieldnames=sorted({k for row in records for k in row}) or ["outcome", "seconds"])
        writer.writeheader(); writer.writerows(records)
        return Response(stream.getvalue(), media_type="text/csv", headers={"Content-Disposition": f'attachment; filename="{identifier}.csv"'})
    return {"job": item, "episodes": records, "manifest": core.read(core.DATA / "jobs" / identifier / "run-manifest.json") or core.read(core.DATA / "jobs" / identifier / "evaluation-manifest.json")}


@app.get("/api/jobs/{identifier}/trajectory")
def trajectory(identifier: str, course: str, attempt: int):
    core.job(identifier)
    if course not in {c["id"] for c in core.courses()} or attempt < 1: raise ValueError("Unknown trajectory")
    path = core.DATA / "jobs" / identifier / "attempts" / course / f"trajectory-{attempt}.jsonl"
    if not path.exists(): raise HTTPException(404, "Trajectory unavailable")
    result = []
    for line in path.read_text().splitlines():
        try: result.append(json.loads(line))
        except json.JSONDecodeError: pass
    return result


@app.get("/api/legacy")
def legacy():
    return [{"name": path.name, "provenance": "unknown", "models": len(list(path.rglob("*.onnx")))}
            for path in (core.ROOT / "results").iterdir() if path.is_dir()]


inspection_dir = core.ROOT / "art-source/circuits/nordschleife/stage-c-review"
if inspection_dir.exists():
    app.mount("/circuit-review", StaticFiles(directory=inspection_dir, html=True), name="circuit-review")

dist = core.ROOT / "dashboard/dist"
if dist.exists():
    app.mount("/", StaticFiles(directory=dist, html=True), name="dashboard")
