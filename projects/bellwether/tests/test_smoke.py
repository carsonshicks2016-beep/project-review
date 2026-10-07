"""Phase 0 smoke tests: package imports and the DB schema bootstraps."""

import bellwether
from bellwether import db


def test_package_imports():
    assert bellwether.__version__


def test_init_db_creates_all_tables(tmp_path):
    conn = db.init_db(tmp_path / "test.db")
    tables = set(db.table_names(conn))
    conn.close()
    assert {"companies", "filings", "signals", "prices", "outcomes"} <= tables
