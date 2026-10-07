"""HTTP client for the WHOOP v2 developer API.

Handles the three things that bite on long backfills: the 100 req/min ceiling,
429s with Retry-After, and access tokens expiring mid-pagination.
"""

from __future__ import annotations

import time
from collections import deque
from collections.abc import Iterator
from datetime import datetime, timezone
from typing import Any

import httpx

from . import auth
from .config import API_BASE, PAGE_LIMIT, RATE_LIMIT_PER_MIN


class WhoopAPIError(RuntimeError):
    pass


def iso(dt: datetime) -> str:
    """WHOOP wants RFC 3339 with an explicit UTC zone."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


class WhoopClient:
    def __init__(self, timeout: float = 30.0):
        self._http = httpx.Client(timeout=timeout, base_url=API_BASE)
        self._calls: deque[float] = deque()

    def close(self) -> None:
        self._http.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def _throttle(self) -> None:
        """Client-side spacing so we stay under 100 requests per rolling minute."""
        now = time.monotonic()
        while self._calls and now - self._calls[0] > 60:
            self._calls.popleft()
        if len(self._calls) >= RATE_LIMIT_PER_MIN - 5:  # small safety margin
            sleep_for = 60 - (now - self._calls[0]) + 0.1
            if sleep_for > 0:
                time.sleep(sleep_for)
        self._calls.append(time.monotonic())

    def get(self, path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        params = {k: v for k, v in (params or {}).items() if v is not None}

        for attempt in range(6):
            self._throttle()
            token = auth.access_token()
            r = self._http.get(
                path,
                params=params,
                headers={"Authorization": f"Bearer {token}"},
            )

            if r.status_code == 200:
                return r.json()

            if r.status_code == 401 and attempt == 0:
                # Token died mid-flight; force a rotation and retry once.
                auth.refresh(force=True)
                continue

            if r.status_code == 429:
                wait = float(r.headers.get("Retry-After", 2 ** attempt))
                time.sleep(min(wait, 60))
                continue

            if r.status_code >= 500:
                time.sleep(2 ** attempt)
                continue

            raise WhoopAPIError(f"GET {path} -> {r.status_code}: {r.text[:300]}")

        raise WhoopAPIError(f"GET {path} failed after retries")

    def paginate(
        self,
        path: str,
        start: datetime | None = None,
        end: datetime | None = None,
        max_records: int | None = None,
    ) -> Iterator[dict[str, Any]]:
        """Yield every record across pages, oldest page first."""
        token: str | None = None
        seen = 0
        while True:
            page = self.get(
                path,
                {
                    "limit": PAGE_LIMIT,
                    "start": iso(start) if start else None,
                    "end": iso(end) if end else None,
                    "nextToken": token,
                },
            )
            for rec in page.get("records", []):
                yield rec
                seen += 1
                if max_records and seen >= max_records:
                    return
            token = page.get("next_token")
            if not token:
                return

    # ------------------------------------------------------------ endpoints

    def cycles(self, start=None, end=None, max_records=None):
        return self.paginate("/v2/cycle", start, end, max_records)

    def recoveries(self, start=None, end=None, max_records=None):
        return self.paginate("/v2/recovery", start, end, max_records)

    def sleeps(self, start=None, end=None, max_records=None):
        return self.paginate("/v2/activity/sleep", start, end, max_records)

    def workouts(self, start=None, end=None, max_records=None):
        return self.paginate("/v2/activity/workout", start, end, max_records)

    def profile(self) -> dict[str, Any]:
        return self.get("/v2/user/profile/basic")

    def body(self) -> dict[str, Any]:
        return self.get("/v2/user/measurement/body")
