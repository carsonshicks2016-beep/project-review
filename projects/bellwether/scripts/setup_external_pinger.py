"""Create or update the cron-job.org trigger for the cloud audit workflow.

Secrets are read from hidden prompts (or environment variables for automation)
and sent only to cron-job.org's API. They are never written to disk.
"""

from __future__ import annotations

import argparse
import getpass
import json
import os
import sys
from typing import Any

import requests

API_ROOT = "https://api.cron-job.org"
JOB_TITLE = "Bellwether watcher audit"
DISPATCH_URL = (
    "https://api.github.com/repos/carsonshicks2016-beep/bellwether/"
    "actions/workflows/watcher-audit.yml/dispatches"
)


def job_config(github_token: str) -> dict[str, Any]:
    """Return the cron-job.org DetailedJob payload."""
    return {
        "title": JOB_TITLE,
        "url": DISPATCH_URL,
        "enabled": True,
        "saveResponses": False,
        "requestMethod": 1,  # POST
        "requestTimeout": 30,
        "redirectSuccess": False,
        "schedule": {
            "timezone": "UTC",
            "expiresAt": 0,
            "hours": [-1],
            "mdays": [-1],
            "minutes": [7, 22, 37, 52],
            "months": [-1],
            "wdays": [-1],
        },
        "notification": {
            "onFailure": True,
            "onFailureCount": 2,
            "onSuccess": True,
            "onDisable": True,
        },
        "extendedData": {
            "headers": {
                "Accept": "application/vnd.github+json",
                "Authorization": f"Bearer {github_token}",
                "Content-Type": "application/json",
                "X-GitHub-Api-Version": "2022-11-28",
            },
            "body": json.dumps(
                {"ref": "main", "inputs": {"trigger_source": "cron-job.org"}}
            ),
        },
    }


def api_request(
    method: str,
    path: str,
    api_key: str,
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    try:
        response = requests.request(
            method,
            f"{API_ROOT}{path}",
            json=payload,
            timeout=30,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
        )
        response.raise_for_status()
    except requests.HTTPError as exc:
        detail = exc.response.text.strip() if exc.response is not None else ""
        status = exc.response.status_code if exc.response is not None else "unknown"
        raise RuntimeError(
            f"cron-job.org returned HTTP {status}: {detail or exc}"
        ) from exc
    except requests.RequestException as exc:
        raise RuntimeError(f"could not reach cron-job.org: {exc}") from exc
    return response.json() if response.content else {}


def upsert_job(api_key: str, github_token: str) -> tuple[str, int]:
    jobs = api_request("GET", "/jobs", api_key).get("jobs", [])
    existing = next((job for job in jobs if job.get("title") == JOB_TITLE), None)
    payload = {"job": job_config(github_token)}
    if existing:
        job_id = int(existing["jobId"])
        api_request("PATCH", f"/jobs/{job_id}", api_key, payload)
        return "updated", job_id
    result = api_request("PUT", "/jobs", api_key, payload)
    return "created", int(result["jobId"])


def secret(env_name: str, prompt: str) -> str:
    value = os.environ.get(env_name) or getpass.getpass(prompt)
    if not value.strip():
        raise RuntimeError(f"{env_name} cannot be empty")
    return value.strip()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print a redacted job payload without calling cron-job.org",
    )
    args = parser.parse_args()

    if args.dry_run:
        print(json.dumps({"job": job_config("<redacted>")}, indent=2))
        return 0

    try:
        api_key = secret("CRON_JOB_ORG_API_KEY", "cron-job.org API key: ")
        github_token = secret("GITHUB_DISPATCH_TOKEN", "GitHub fine-grained token: ")
        action, job_id = upsert_job(api_key, github_token)
        details = api_request("GET", f"/jobs/{job_id}", api_key).get("jobDetails", {})
    except (RuntimeError, ValueError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"{action} cron-job.org job {job_id}: {details.get('title', JOB_TITLE)}")
    print(f"enabled: {details.get('enabled')}; next execution: {details.get('nextExecution')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
