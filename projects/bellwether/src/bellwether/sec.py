"""SEC EDGAR helpers: ticker -> CIK resolution (Phase 1) and the filing
submissions feed (Phase 2 watcher).

EDGAR requires a real User-Agent ("Name email") or it IP-bans you, and asks for
<= 10 req/s. We hard-sleep 0.15s between requests everywhere (see edgar_get).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import requests

from . import config

TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
ARCHIVES = "https://www.sec.gov/Archives/edgar/data/{cik_int}/{acc_nodash}/{doc}"
CACHE = config.ROOT / "data" / "company_tickers.json"
_RATE_SLEEP = 0.15


def _headers() -> dict:
    return {"User-Agent": config.require("SEC_USER_AGENT")}


def edgar_get(url: str, retries: int = 3) -> requests.Response:
    """GET an EDGAR URL with the required UA, a hard rate-limit sleep, and
    exponential backoff on 429/503."""
    for attempt in range(retries + 1):
        time.sleep(_RATE_SLEEP)
        r = requests.get(url, headers=_headers(), timeout=30)
        if r.status_code in (429, 503) and attempt < retries:
            time.sleep(2 ** attempt)  # 1s, 2s, 4s
            continue
        r.raise_for_status()
        return r
    r.raise_for_status()
    return r


def fetch_submissions(cik: str) -> dict:
    """Fetch a company's submissions JSON (CIK is 10-digit, zero-padded)."""
    return edgar_get(SUBMISSIONS_URL.format(cik=cik)).json()


def parse_recent(submissions: dict, cik: str) -> list[dict]:
    """Flatten EDGAR's parallel `filings.recent` arrays into a list of dicts.

    Each dict: cik, form_type, accession_no, filed_date, acceptance_datetime,
    url (primary document). acceptance_datetime is the sacred point-in-time
    anchor (e.g. '2024-01-05T16:30:21.000Z').
    """
    recent = submissions.get("filings", {}).get("recent", {})
    cik_int = str(int(cik))  # un-pad for the Archives path
    out = []
    accs = recent.get("accessionNumber", [])
    for i in range(len(accs)):
        acc = accs[i]
        doc = recent.get("primaryDocument", [""] * len(accs))[i]
        out.append({
            "cik": cik,
            "form_type": recent.get("form", [""] * len(accs))[i],
            "accession_no": acc,
            "filed_date": recent.get("filingDate", [""] * len(accs))[i],
            "acceptance_datetime": recent.get("acceptanceDateTime", [""] * len(accs))[i],
            "url": ARCHIVES.format(
                cik_int=cik_int, acc_nodash=acc.replace("-", ""), doc=doc
            ) if doc else "",
        })
    return out


def _load_ticker_map() -> dict[str, str]:
    """Return {TICKER: 10-digit CIK}, cached to data/company_tickers.json."""
    if not CACHE.exists():
        data = edgar_get(TICKERS_URL).json()
        CACHE.write_text(json.dumps(data))
    else:
        data = json.loads(CACHE.read_text())
    out = {}
    for row in data.values():
        out[row["ticker"].upper()] = str(row["cik_str"]).zfill(10)
    return out


def resolve_ciks(tickers: list[str]) -> dict[str, str | None]:
    """Map each ticker to its CIK (or None if EDGAR doesn't list it)."""
    m = _load_ticker_map()
    return {t: m.get(t.upper()) for t in tickers}
