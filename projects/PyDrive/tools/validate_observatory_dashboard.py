#!/usr/bin/env python3
"""Acceptance gates for the edition-safe Observatory Command Center surface.

This validator is intentionally read-only.  It imports the real Flask server,
uses its test client, submits only requests that must be rejected before a
session is created, and hashes the protected 2D viewer before and after all
checks.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import importlib.util
import json
import os
from pathlib import Path
import re
import sys
from typing import Any
from urllib.parse import urljoin, urlparse


ROOT = Path(__file__).resolve().parents[1]
SERVER_PATH = ROOT / "command-center" / "server.py"
VIEWER2_PATH = ROOT / "supra" / "viewer2.py"
VIEWER2_VALIDATOR_PATH = ROOT / "tools" / "validate_ring_2d_viewer.py"

# Recorded by the root implementation task before Observatory work began.
RECORDED_VIEWER2_SHA256 = (
    "d7bf749424bbba25d3a2cc0cb3890435b8b5987ca3692e47b23218dd253be0f6"
)
RECORDED_VIEWER2_VALIDATOR_SHA256 = (
    "c636027f5a4d0f912714c197e31cb615dd1b53c401e8d6cc69b20d7bfd6f2991"
)

EXPECTED_CARS = {"787b": "mazda787b", "919": "porsche_919evo"}
EXPECTED_LAYOUTS = {"787b": "fable-v1", "919": "fable-v2"}
FAILED: list[str] = []


def digest(path: Path) -> str:
    h = sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def gate(name: str, ok: bool, detail: str = "") -> bool:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}"
          + (f" — {detail}" if detail else ""))
    if not ok:
        FAILED.append(name)
    return ok


def cutover_gate(ok: bool, detail: str) -> bool:
    label = "PASS" if ok else "PENDING"
    print(f"  [{label}] legacy 3D cutover complete — {detail}")
    if not ok:
        FAILED.append("legacy 3D cutover complete")
    return ok


def load_server():
    # server.py imports its local observatory_api module by name.  Loading the
    # server through importlib does not add command-center/ to sys.path on its
    # own, so make the test-client bootstrap match a normal dashboard launch.
    command_center = str(SERVER_PATH.parent)
    if command_center not in sys.path:
        sys.path.insert(0, command_center)
    module_name = "_observatory_dashboard_acceptance_server"
    spec = importlib.util.spec_from_file_location(module_name, SERVER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot create import spec for {SERVER_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def response_json(response) -> Any:
    try:
        return response.get_json()
    except Exception:
        return None


def route_and_catalog_gates(server, client) -> dict[str, Any] | None:
    print("== authoritative Observatory catalogue ==")
    rules = list(server.app.url_map.iter_rules())
    catalog_rules = [rule for rule in rules
                     if rule.rule == "/api/observatory/catalog"]
    gate("exactly one Observatory catalog route", len(catalog_rules) == 1,
         f"count={len(catalog_rules)}")
    if catalog_rules:
        gate("catalog route is read-only",
             "GET" in catalog_rules[0].methods
             and "POST" not in catalog_rules[0].methods)

    response = client.get("/api/observatory/catalog")
    payload = response_json(response)
    if not gate("catalog responds with JSON", response.status_code == 200
                and isinstance(payload, dict), f"HTTP {response.status_code}"):
        return None
    gate("catalog schema is versioned",
         payload.get("schema") == "fable-observatory-catalog-v1")
    editions = payload.get("editions")
    if not gate("catalog has exactly the two registered editions",
                isinstance(editions, list)
                and [item.get("id") for item in editions] == ["787b", "919"]):
        return payload

    from supra.fable_editions import registry_payload

    registry = {item["id"]: item for item in registry_payload()}
    for item in editions:
        edition_id = item["id"]
        car_id = EXPECTED_CARS[edition_id]
        resolved = item.get("resolved_checkpoint")
        descriptor_fields = tuple(registry[edition_id])
        gate(f"{edition_id} descriptor comes from the server registry",
             all(item.get(key) == registry[edition_id].get(key)
                 for key in descriptor_fields))
        gate(f"{edition_id} descriptor identity",
             item.get("car_id") == car_id
             and item.get("observation_layout") == EXPECTED_LAYOUTS[edition_id])
        gate(f"{edition_id} resolved checkpoint exists",
             isinstance(resolved, dict) and resolved.get("playable") is True)
        if isinstance(resolved, dict):
            policy_hash = str(resolved.get("policy_sha256") or "")
            file_hash = str(resolved.get("file_sha256") or "")
            gate(f"{edition_id} resolved checkpoint is car-scoped",
                 resolved.get("edition") == edition_id
                 and resolved.get("car_id") == car_id
                 and resolved.get("observation_layout")
                 == EXPECTED_LAYOUTS[edition_id])
            gate(f"{edition_id} resolved checkpoint has exact hashes",
                 bool(re.fullmatch(r"[0-9a-f]{64}", policy_hash))
                 and bool(re.fullmatch(r"[0-9a-f]{64}", file_hash)))
            gate(f"{edition_id} compatibility class is authoritative",
                 resolved.get("compatibility_class")
                 == item.get("compatibility_class"))
        playable = [record for record in item.get("checkpoints", [])
                    if record.get("playable")]
        gate(f"{edition_id} playable catalogue never crosses cars",
             all(record.get("edition") == edition_id
                 and record.get("car_id") == car_id for record in playable),
             f"playable={len(playable)}")

    warning_919 = str(editions[1].get("authority_warning") or "").lower()
    gate("919 authority boundary is always present",
         "not faithful-v2" in warning_919 and "certification" in warning_919)
    return payload


def rejected_session_gates(client, catalog: dict[str, Any] | None) -> None:
    print("== fail-closed session creation ==")
    api = sys.modules.get("observatory_api")
    manager = getattr(api, "SESSION_MANAGER", None)
    if not gate("session manager is inspectable", manager is not None):
        return
    before = manager.status()
    before_ids = set(before.get("sessions", {}))

    editions = ((catalog or {}).get("editions") or [])
    by_id = {item.get("id"): item for item in editions}
    checkpoint_919 = (((by_id.get("919") or {}).get("resolved_checkpoint")
                       or {}).get("name") or "fable5_919_ring_best.pt")
    checkpoint_787 = (((by_id.get("787b") or {}).get("resolved_checkpoint")
                       or {}).get("name") or "fable5_ring_best.pt")
    cases = (
        ("wrong-car checkpoint", {
            "edition": "787b", "checkpoint": checkpoint_919,
            "mode": "replay", "seed": 7, "audio": False,
        }, {422}),
        ("unknown edition", {
            "edition": "unknown", "checkpoint": "active-best",
            "mode": "replay", "seed": 7, "audio": False,
        }, {400}),
        ("unknown checkpoint", {
            "edition": "787b", "checkpoint": "fable5_does_not_exist.pt",
            "mode": "replay", "seed": 7, "audio": False,
        }, {422}),
        ("checkpoint traversal", {
            "edition": "787b", "checkpoint": f"../{checkpoint_787}",
            "mode": "replay", "seed": 7, "audio": False,
        }, {422}),
        ("absolute checkpoint path", {
            "edition": "919", "checkpoint": str(ROOT / checkpoint_919),
            "mode": "replay", "seed": 7, "audio": False,
        }, {422}),
    )
    for label, body, expected_status in cases:
        response = client.post(
            "/api/observatory/sessions",
            json=body,
            headers={"X-Observatory-Browser": f"acceptance-{label.replace(' ', '-')}"},
        )
        error = response_json(response) or {}
        gate(f"{label} rejected", response.status_code in expected_status
             and response.status_code != 201,
             f"HTTP {response.status_code}: {error.get('error')}")
        now_ids = set(manager.status().get("sessions", {}))
        gate(f"{label} creates no session", now_ids == before_ids)

    after = manager.status()
    gate("all rejected requests leave session count unchanged",
         after.get("active") == before.get("active")
         and set(after.get("sessions", {})) == before_ids,
         f"before={before.get('active')} after={after.get('active')}")


def direct_launch_gates(client) -> None:
    """Prove a bare `/observatory/` client can create its documented default."""
    print("== direct Observatory launch default ==")
    response = client.post(
        "/api/observatory/sessions",
        json={"mode": "replay", "audio": False},
        headers={"X-Observatory-Browser": "dashboard-direct-launch"},
    )
    payload = response_json(response) or {}
    session_id = str(payload.get("session_id") or "")
    hello = payload.get("hello") or {}
    resolved = payload.get("resolved_checkpoint") or {}
    try:
        gate("bare Observatory launch resolves the documented 787B default",
             response.status_code == 201
             and hello.get("checkpoint", {}).get("car") == "mazda787b"
             and resolved.get("edition") == "787b",
             f"HTTP {response.status_code}: {resolved.get('name') or payload.get('error', '')}")
    finally:
        if session_id:
            cleanup = client.delete(f"/api/observatory/sessions/{session_id}")
            gate("direct-launch smoke session is released", cleanup.status_code == 200,
                 f"HTTP {cleanup.status_code}")


def fable_ga_state_gates(client) -> None:
    """The dashboard loads GA status with the rest of its initial data."""
    print("== Fable GA dashboard state ==")
    response = client.get("/api/fable-ga")
    payload = response_json(response) or {}
    champion = payload.get("champion")
    snapshot = payload.get("snapshot")
    gate("Fable GA state endpoint responds without a dashboard 404",
         response.status_code == 200 and isinstance(payload, dict),
         f"HTTP {response.status_code}")
    gate("Fable GA state has a stable empty-state schema",
         payload.get("checkpoint") == "ga_787b_ring.npz"
         and isinstance(payload.get("champion_exists"), bool)
         and isinstance(champion, dict)
         and isinstance(snapshot, dict)
         and payload.get("superhuman_lap") == 371.13)
    if payload.get("champion_exists"):
        gate("saved Fable GA champion reports its own identity",
             isinstance(champion.get("generation"), int)
             and champion.get("car") == "mazda787b"
             and champion.get("track") == "nordschleife")
    if snapshot:
        gate("Fable GA snapshot is recognisably a GA artifact",
             snapshot.get("pipeline") == "fable_ga"
             and snapshot.get("optimizer") == "genetic_algorithm")


def edition_state_gates(server, client) -> None:
    print("== edition-scoped state, diagnostics, and timeline ==")
    from supra.fable_editions import get_edition

    diagnostic_rows: dict[str, list[dict[str, Any]]] = {}
    timeline_rows: dict[str, list[dict[str, Any]]] = {}
    for edition_id, car_id in EXPECTED_CARS.items():
        state_response = client.get(f"/api/fable-five?edition={edition_id}")
        state = response_json(state_response) or {}
        resolved = state.get("resolved_checkpoint") or {}
        latest_diag = state.get("latest_diagnostic")
        gate(f"{edition_id} Fable state descriptor is scoped",
             state_response.status_code == 200
             and (state.get("edition") or {}).get("id") == edition_id
             and (state.get("edition") or {}).get("car_id") == car_id)
        gate(f"{edition_id} Fable state resolved checkpoint is scoped",
             not resolved or (resolved.get("edition") == edition_id
                              and resolved.get("car_id") == car_id))
        gate(f"{edition_id} active tasks are scoped",
             all(task.get("edition") == edition_id
                 for task in state.get("active_tasks", [])))
        gate(f"{edition_id} latest diagnostic is scoped",
             latest_diag is None
             or (latest_diag.get("edition") == edition_id
                 and latest_diag.get("car") == car_id))

        diag_response = client.get(f"/api/fable-diag?edition={edition_id}")
        diagnostics = response_json(diag_response)
        if not isinstance(diagnostics, list):
            diagnostics = []
        diagnostic_rows[edition_id] = diagnostics
        gate(f"{edition_id} diagnostics endpoint responds",
             diag_response.status_code == 200)
        gate(f"{edition_id} diagnostics contain no opposite-car rows",
             all(row.get("edition") == edition_id and row.get("car") == car_id
                 for row in diagnostics), f"rows={len(diagnostics)}")

        timeline_response = client.get(
            f"/api/fable-pit-log?edition={edition_id}&n=2000"
        )
        timeline = response_json(timeline_response) or {}
        rows = timeline.get("rows") if isinstance(timeline.get("rows"), list) else []
        timeline_rows[edition_id] = rows
        gate(f"{edition_id} timeline envelope is scoped",
             timeline_response.status_code == 200
             and timeline.get("edition") == edition_id
             and timeline.get("car") == car_id)
        gate(f"{edition_id} timeline contains no opposite-car rows",
             all(row.get("edition") == edition_id and row.get("car") == car_id
                 for row in rows), f"rows={len(rows)}")

    # A diagnostic visible to one edition must fail closed when requested from
    # the other edition, even when its exact public name is known.
    for source, destination in (("787b", "919"), ("919", "787b")):
        if not diagnostic_rows[source]:
            continue
        name = diagnostic_rows[source][0].get("name")
        response = client.get(
            f"/api/fable-diag/{name}?edition={destination}"
        )
        gate(f"{source} diagnostic is hidden from {destination}",
             response.status_code == 404, f"HTTP {response.status_code}")

    ids_787 = {(row.get("checkpoint"), row.get("policy_sha256"))
               for row in timeline_rows["787b"] if row.get("checkpoint")}
    ids_919 = {(row.get("checkpoint"), row.get("policy_sha256"))
               for row in timeline_rows["919"] if row.get("checkpoint")}
    gate("timeline checkpoint identities are disjoint", not (ids_787 & ids_919),
         f"787b={len(ids_787)} 919={len(ids_919)}")

    edition_787 = get_edition("787b")
    edition_919 = get_edition("919")
    synthetic_787 = {"car": edition_787.car_id,
                     "drivetrain_version": edition_787.drivetrain}
    synthetic_919 = {"car": edition_919.car_id,
                     "drivetrain_version": edition_919.drivetrain}
    gate("runtime event predicate rejects cross-edition stamps",
         server._event_matches_edition(synthetic_787, edition_787)
         and not server._event_matches_edition(synthetic_919, edition_787)
         and server._event_matches_edition(synthetic_919, edition_919)
         and not server._event_matches_edition(synthetic_787, edition_919))

    for path in ("/api/fable-five", "/api/fable-diag", "/api/fable-pit-log"):
        response = client.get(f"{path}?edition=unknown")
        gate(f"{path} rejects an unknown edition", response.status_code == 400,
             f"HTTP {response.status_code}")


def edition_command_gates(server, catalog: dict[str, Any] | None) -> None:
    print("== edition-safe Fable commands ==")
    editions = ((catalog or {}).get("editions") or [])
    by_id = {item.get("id"): item for item in editions}
    porsche = by_id.get("919") or {}
    mazda = by_id.get("787b") or {}
    resolved_919 = (porsche.get("resolved_checkpoint") or {}).get("name")
    resolved_787 = (mazda.get("resolved_checkpoint") or {}).get("name")
    for action in ("fable_watch", "fable_diagnose"):
        try:
            command = server.build_command(action, {"edition": "919"})
            checkpoint = command[command.index("--checkpoint") + 1]
            car = command[command.index("--car") + 1]
            valid = (
                car == EXPECTED_CARS["919"]
                and checkpoint == resolved_919
                and checkpoint != resolved_787
            )
        except Exception as exc:
            valid = False
            checkpoint = f"{type(exc).__name__}: {exc}"
        gate(f"919 {action} resolves its own active best", valid,
             f"checkpoint={checkpoint}")

    for label, params in (
        ("mismatched edition/car", {"edition": "919", "car": "mazda787b"}),
        ("cross-car explicit checkpoint", {
            "edition": "919", "checkpoint": resolved_787,
        }),
    ):
        try:
            server.build_command("fable_watch", params)
        except ValueError:
            rejected = True
        else:
            rejected = False
        gate(f"Fable watch rejects {label}", rejected)


def local_static_gates(client) -> None:
    print("== locally served Observatory build ==")
    response = client.get("/observatory/")
    html = response.get_data(as_text=True)
    if not gate("Observatory index is served after build",
                response.status_code == 200 and "text/html" in response.content_type,
                f"HTTP {response.status_code}"):
        return
    gate("index is the Observatory canvas, not an iframe shell",
         'id="stage"' in html and 'id="identity-checkpoint"' in html
         and "iframe" not in html.lower())
    gate("index declares no network dependency",
         not re.search(r"(?:src|href)=[\"'](?:https?:)?//", html, re.I))
    refs = re.findall(r"(?:src|href)=[\"']([^\"']+)[\"']", html, re.I)
    local_refs = []
    for ref in refs:
        parsed = urlparse(ref)
        if parsed.scheme or parsed.netloc or ref.startswith(("data:", "#")):
            continue
        local_refs.append(urljoin("/observatory/", ref))
    statuses = {ref: client.get(ref).status_code for ref in local_refs}
    gate("all built HTML modules and styles are locally served",
         bool(statuses) and all(status == 200 for status in statuses.values()),
         ", ".join(f"{ref}={status}" for ref, status in statuses.items()))

    vehicle_manifests = {}
    for car in ("mazda787b", "porsche_919evo"):
        manifest_path = f"/observatory/assets/observatory/vehicles/{car}/asset_manifest.json"
        vehicle_manifests[car] = response_json(client.get(manifest_path)) or {}
    required = (
        "/observatory/audio/pcm-jitter-processor.js",
        "/observatory/assets/observatory/world/world.manifest.json",
        "/observatory/assets/observatory/world/world_truth.glb",
        "/observatory/assets/observatory/world/world_visual.glb",
        "/observatory/assets/observatory/vehicles/mazda787b/asset_manifest.json",
        "/observatory/assets/observatory/vehicles/mazda787b/"
        + str(vehicle_manifests["mazda787b"].get("entry_glb", "missing.glb")),
        "/observatory/assets/observatory/vehicles/porsche_919evo/asset_manifest.json",
        "/observatory/assets/observatory/vehicles/porsche_919evo/"
        + str(vehicle_manifests["porsche_919evo"].get("entry_glb", "missing.glb")),
    )
    required_status = {path: client.get(path).status_code for path in required}
    gate("world, both cars, and AudioWorklet are local",
         all(status == 200 for status in required_status.values()),
         ", ".join(f"{Path(path).name}={status}"
                   for path, status in required_status.items()))

    world_response = client.get(
        "/observatory/assets/observatory/world/world.manifest.json"
    )
    world = response_json(world_response) or {}
    gate("world manifest declares deterministic offline generation",
         (world.get("generator") or {}).get("deterministic") is True
         and (world.get("generator") or {}).get("network_required") is False
         and (world.get("track") or {}).get("sample_count") == 6944)


def supported_launch_surface_gates(client) -> None:
    """Keep dashboard affordances aligned with routes that actually exist.

    The older `/3d/` viewer remains a separate, supported surface.  Observatory
    acceptance must not turn that unrelated migration into a permanent failure.
    It does, however, fail if the dashboard advertises the never-implemented
    `/observatory-v2/` route.
    """
    print("== supported Observatory launch surface ==")
    dashboard_html = (ROOT / "command-center" / "static" / "index.html").read_text(
        encoding="utf-8"
    )
    dashboard_js = (ROOT / "command-center" / "static" / "app.js").read_text(
        encoding="utf-8"
    )
    ghost = client.get("/observatory-v2/")
    gate("no unimplemented Observatory V2 route is advertised",
         "/observatory-v2/" not in dashboard_html
         and "/observatory-v2/" not in dashboard_js
         and ghost.status_code == 404,
         f"route HTTP {ghost.status_code}")
    gate("dashboard exposes only the supported 3D launch controls",
         all(control in dashboard_html for control in (
             'id="fb-watch-best-3d"', 'id="fb-watch-selected-3d"',
             'id="obs-open-replay"', 'id="obs-open-follow"',
         ))
         and "-v2" not in dashboard_html,
         "four current Observatory controls, no V2 aliases")


def two_d_hash_gates(expected_viewer: str,
                     expected_validator: str,
                     before: dict[str, str]) -> None:
    print("== protected 2D alternative ==")
    gate("viewer2 matches the recorded pre-work hash",
         before["viewer2"] == expected_viewer,
         f"actual={before['viewer2']} expected={expected_viewer}")
    gate("2D smoke validator matches the recorded pre-work hash",
         before["validator"] == expected_validator,
         f"actual={before['validator']} expected={expected_validator}")
    after = {
        "viewer2": digest(VIEWER2_PATH),
        "validator": digest(VIEWER2_VALIDATOR_PATH),
    }
    gate("dashboard acceptance checks did not edit either 2D file",
         after == before,
         f"viewer2={after['viewer2']} validator={after['validator']}")
    print("  hash record:")
    print(json.dumps({
        str(VIEWER2_PATH.relative_to(ROOT)): before["viewer2"],
        str(VIEWER2_VALIDATOR_PATH.relative_to(ROOT)): before["validator"],
    }, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--expect-viewer2-sha256",
        default=os.environ.get(
            "OBSERVATORY_VIEWER2_SHA256", RECORDED_VIEWER2_SHA256
        ),
        help="pre-work supra/viewer2.py SHA-256",
    )
    parser.add_argument(
        "--expect-2d-validator-sha256",
        default=os.environ.get(
            "OBSERVATORY_2D_VALIDATOR_SHA256",
            RECORDED_VIEWER2_VALIDATOR_SHA256,
        ),
        help="pre-work tools/validate_ring_2d_viewer.py SHA-256",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    os.chdir(ROOT)
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    before = {
        "viewer2": digest(VIEWER2_PATH),
        "validator": digest(VIEWER2_VALIDATOR_PATH),
    }
    try:
        server = load_server()
    except Exception as exc:
        gate("Command Center imports", False,
             f"{type(exc).__name__}: {exc}")
        two_d_hash_gates(
            args.expect_viewer2_sha256,
            args.expect_2d_validator_sha256,
            before,
        )
        return 1

    client = server.app.test_client()
    catalog = route_and_catalog_gates(server, client)
    rejected_session_gates(client, catalog)
    direct_launch_gates(client)
    fable_ga_state_gates(client)
    edition_state_gates(server, client)
    edition_command_gates(server, catalog)
    local_static_gates(client)
    supported_launch_surface_gates(client)
    two_d_hash_gates(
        args.expect_viewer2_sha256,
        args.expect_2d_validator_sha256,
        before,
    )

    if FAILED:
        print("\nObservatory dashboard acceptance FAILED:")
        for name in FAILED:
            print(f"  - {name}")
        return 1
    print("\nAll Observatory dashboard acceptance gates passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
