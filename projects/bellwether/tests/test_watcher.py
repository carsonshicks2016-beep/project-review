"""Phase 2 tests: EDGAR parse (offline) + dedupe."""

from bellwether import db, sec, watcher

SAMPLE = {
    "filings": {
        "recent": {
            "accessionNumber": ["0001234567-24-000001", "0001234567-24-000002"],
            "form": ["8-K", "4"],
            "filingDate": ["2024-03-01", "2024-03-02"],
            "acceptanceDateTime": [
                "2024-03-01T16:30:21.000Z",
                "2024-03-02T09:15:00.000Z",
            ],
            "primaryDocument": ["d12345.htm", "form4.xml"],
        }
    }
}


def test_parse_recent_zips_arrays_and_builds_url():
    rows = sec.parse_recent(SAMPLE, "0000012345")
    assert len(rows) == 2
    r0 = rows[0]
    assert r0["form_type"] == "8-K"
    assert r0["accession_no"] == "0001234567-24-000001"
    assert r0["acceptance_datetime"] == "2024-03-01T16:30:21.000Z"
    # Archives URL: un-padded CIK, de-dashed accession, primary doc
    assert r0["url"].endswith("/12345/000123456724000001/d12345.htm")
    assert rows[1]["url"].endswith("form4.xml")  # Form 4 is XML


def test_store_filing_dedupes(tmp_path):
    conn = db.init_db(tmp_path / "t.db")
    # filings.cik is a FK -> companies; seed the parent (as real backfill does).
    conn.execute("INSERT INTO companies (cik, ticker) VALUES (?,?)",
                 ("0000012345", "TEST"))
    rows = sec.parse_recent(SAMPLE, "0000012345")
    first = sum(watcher.store_filing(conn, f) for f in rows)
    second = sum(watcher.store_filing(conn, f) for f in rows)
    conn.close()
    assert first == 2   # both new
    assert second == 0  # both already seen
