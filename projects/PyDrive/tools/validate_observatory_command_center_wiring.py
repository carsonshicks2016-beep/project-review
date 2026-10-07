#!/usr/bin/env python3
"""Bounded HTTP/DOM/source smoke for Command Center Observatory wiring.

This deliberately does not launch or control a browser.  It verifies the live
HTTP surface, required controls, exact frontend URL construction, and the
registry-derived allowlist used by the general checkpoint garage.  The separate
Playwright acceptance owns real popup/click behavior.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
from typing import Any
from urllib.parse import urlencode, urljoin
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORT_ROOT = ROOT / "runtime" / "observatory-qa"


class Audit:
    def __init__(self, base_url: str) -> None:
        self.base_url = base_url.rstrip("/")
        self.checks: list[dict[str, Any]] = []

    def gate(self, name: str, condition: bool, detail: str = "") -> None:
        passed = bool(condition)
        self.checks.append({"name": name, "pass": passed, "detail": detail})
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}"
              + (f" — {detail}" if detail else ""))

    def fetch(self, path: str) -> tuple[int, str, bytes]:
        request = Request(urljoin(self.base_url + "/", path.lstrip("/")),
                          headers={"Accept": "*/*"})
        with urlopen(request, timeout=15) as response:
            return response.status, response.headers.get_content_type(), response.read()

    @property
    def passed(self) -> bool:
        return all(row["pass"] for row in self.checks)


def compact_javascript(source: str) -> str:
    return re.sub(r"\s+", "", source)


def expected_url(edition: str, checkpoint: str, mode: str) -> str:
    # URLSearchParams preserves insertion order in openObservatory().
    return "/observatory/?" + urlencode({
        "edition": edition,
        "checkpoint": checkpoint,
        "mode": mode,
    })


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8877")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    audit = Audit(args.base_url)
    print("== live Command Center HTTP and DOM ==")
    try:
        root_status, root_type, root_body = audit.fetch("/")
        js_status, js_type, js_body = audit.fetch("/static/app.js")
        cat_status, cat_type, cat_body = audit.fetch("/api/observatory/catalog")
        ck_status, ck_type, ck_body = audit.fetch("/api/checkpoints")
    except Exception as exc:
        audit.gate("live Command Center is reachable", False,
                   f"{type(exc).__name__}: {exc}")
        return 1

    html = root_body.decode("utf-8", errors="replace")
    javascript = js_body.decode("utf-8", errors="replace")
    compact = compact_javascript(javascript)
    audit.gate("GET / loads the Command Center", root_status == 200
               and root_type == "text/html" and len(root_body) > 1000,
               f"HTTP {root_status}, {len(root_body)} bytes")
    audit.gate("Command Center app.js is live", js_status == 200
               and "javascript" in js_type and len(js_body) > 1000,
               f"HTTP {js_status}, {len(js_body)} bytes")

    controls = {
        "fb-watch-best-3d": "Watch Best 3D",
        "fb-watch-selected-3d": "Watch Selected 3D",
        "obs-open-replay": "Open Best Replay",
        "obs-open-follow": "Follow Active",
    }
    for control_id, label in controls.items():
        pattern = (rf'<button[^>]*\bid=["\']{re.escape(control_id)}["\'][^>]*>'
                   rf'\s*{re.escape(label)}\s*</button>')
        audit.gate(f"DOM exposes {label}", bool(re.search(pattern, html, re.I | re.S)),
                   f"#{control_id}")
    audit.gate("DOM exposes Fable, Observatory, and Checkpoints panels",
               all(f'id="tab-{name}"' in html
                   for name in ("fable", "observatory", "checkpoints")))

    print("== exact Observatory launch mapping ==")
    audit.gate("openObservatory targets only /observatory/ with ordered query",
               "constparams=newURLSearchParams({edition:ed,checkpoint,mode});"
               "window.open(`/observatory/?${params}`,\"_blank\",\"noopener\");"
               in compact)
    expected_handlers = {
        "Fable Watch Best 3D uses selected edition replay":
            '$("#fb-watch-best-3d").onclick=()=>openObservatory({edition:fableEdition().id,checkpoint:"active-best",mode:"replay"});',
        "Fable Watch Selected 3D uses selected edition and checkpoint":
            '$("#fb-watch-selected-3d").onclick=()=>openObservatory({edition:fableEdition().id,checkpoint:fableCheckpoint(),mode:"replay"});',
        "Observatory hub Replay uses hub edition":
            '$("#obs-open-replay").onclick=()=>openObservatory({edition:observatoryHubEdition()?.id,checkpoint:"active-best",mode:"replay",});',
        "Observatory hub Follow uses follow-active-best mode":
            '$("#obs-open-follow").onclick=()=>openObservatory({edition:observatoryHubEdition()?.id,checkpoint:"active-best",mode:"follow-active-best",});',
    }
    for name, fragment in expected_handlers.items():
        audit.gate(name, fragment in compact)
    audit.gate("selected checkpoint list is edition-car scoped",
               "constexpectedCar=fableEdition()?.car;" in compact
               and ")&&c.car===expectedCar);" in compact)

    print("== authoritative catalogue and checkpoint allowlist ==")
    catalog = json.loads(cat_body)
    checkpoints = json.loads(ck_body)
    editions = catalog.get("editions") or []
    audit.gate("catalog and checkpoint endpoints are JSON",
               cat_status == 200 and cat_type == "application/json"
               and ck_status == 200 and ck_type == "application/json")
    audit.gate("catalog exposes exactly 787b and 919",
               [item.get("id") for item in editions] == ["787b", "919"])

    playable: dict[str, tuple[str, dict[str, Any]]] = {}
    rejected: list[dict[str, Any]] = []
    for edition in editions:
        for record in edition.get("checkpoints") or []:
            if record.get("playable") is True:
                playable[record.get("name")] = (edition.get("id"), record)
            else:
                rejected.append(record)
    checkpoint_by_name = {row.get("name"): row for row in checkpoints}
    playable_metadata_ok = all(
        record.get("track") == "nordschleife"
        and record.get("track_profile") == "nordschleife-full-20.832km"
        and record.get("mode") == "race"
        and name.endswith(".pt")
        and record.get("car_id") == checkpoint_by_name.get(name, {}).get("car")
        and checkpoint_by_name.get(name, {}).get("fable_pipeline") is True
        and checkpoint_by_name.get(name, {}).get("track") == "nordschleife"
        for name, (_, record) in playable.items()
    )
    audit.gate("every View-in-3D-eligible record is Nordschleife Fable PPO",
               bool(playable) and playable_metadata_ok,
               f"eligible={len(playable)}")
    audit.gate("all rejected registry records are classified incompatible",
               bool(rejected)
               and all(row.get("classification") == "incompatible"
                       for row in rejected),
               f"rejected={len(rejected)}")
    audit.gate("general checkpoint button resolves through registry catalogue",
               "constobservable=observatoryCheckpoint(c.name);" in compact
               and 'if(observable&&observable.classification!=="incompatible"){' in compact
               and 'mk("Viewin3D",()=>openObservatory({' in compact
               and "edition:observable.edition,checkpoint:c.name,mode:\"replay\"," in compact)
    audit.gate("registry lookup itself rejects non-playable records",
               'if(record.playable!==false&&record.classification!=="incompatible"){' in compact)

    url_matrix: list[dict[str, str]] = []
    for edition in editions:
        edition_id = str(edition.get("id"))
        resolved = edition.get("resolved_checkpoint") or {}
        selected = str(resolved.get("name") or "")
        url_matrix.extend([
            {"surface": "fable-best-3d", "edition": edition_id,
             "url": expected_url(edition_id, "active-best", "replay")},
            {"surface": "fable-selected-3d", "edition": edition_id,
             "url": expected_url(edition_id, selected, "replay")},
            {"surface": "hub-replay", "edition": edition_id,
             "url": expected_url(edition_id, "active-best", "replay")},
            {"surface": "hub-follow", "edition": edition_id,
             "url": expected_url(edition_id, "active-best", "follow-active-best")},
        ])
    audit.gate("every edition resolves an exact representative URL matrix",
               len(url_matrix) == 8
               and all(row["url"].startswith(
                   f"/observatory/?edition={row['edition']}&checkpoint=")
                       for row in url_matrix))

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report_path = (args.report or
                   DEFAULT_REPORT_ROOT / f"dashboard-wiring-http-source-{stamp}.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "schema": "observatory-command-center-http-source-smoke-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "base_url": audit.base_url,
        "pass": audit.passed,
        "checks": audit.checks,
        "url_matrix": url_matrix,
        "eligible_checkpoint_count": len(playable),
        "eligible_checkpoints": [
            {"name": name, "edition": edition_id,
             "classification": record.get("classification")}
            for name, (edition_id, record) in sorted(playable.items())
        ],
        "browser_click_gate": "owned by tools/validate_observatory_playwright.mjs",
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("\n== expected browser URL matrix ==")
    for row in url_matrix:
        print(f"  {row['surface']:18s} {row['edition']:4s} {row['url']}")
    print(f"\nreport: {report_path}")
    if audit.passed:
        print("Command Center Observatory HTTP/DOM/source smoke passed.")
        return 0
    print("Command Center Observatory HTTP/DOM/source smoke FAILED.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
