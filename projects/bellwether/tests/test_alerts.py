import sqlite3
import pytest
from unittest.mock import patch
from bellwether import alerts


def _make_conn():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute("""
        CREATE TABLE companies (
            cik TEXT PRIMARY KEY, ticker TEXT, name TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE filings (
            id INTEGER PRIMARY KEY, cik TEXT, form_type TEXT,
            accession_no TEXT, acceptance_datetime TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE signals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filing_id INTEGER, catalyst_type TEXT, direction TEXT,
            materiality INTEGER, confidence REAL, summary TEXT,
            evidence_quote TEXT, model TEXT, created_at TEXT
        )
    """)
    conn.execute("INSERT INTO companies VALUES ('0001', 'TST', 'Test Corp')")
    conn.execute("INSERT INTO filings VALUES (1, '0001', '8-K', 'ACC-001', '2026-06-01T10:00:00Z')")
    conn.execute(
        "INSERT INTO signals VALUES (1, 1, 'guidance_change', 'bullish', 8, 0.9, "
        "'Raised guidance.', 'We are raising guidance.', 'gemini', '2026-06-01T10:01:00Z')"
    )
    conn.commit()
    return conn


def test_dedup_prevents_second_alert():
    conn = _make_conn()

    with patch("bellwether.alerts._send_email") as mock_send:
        with patch("bellwether.config.ALERT_MATERIALITY_MIN", 7):
            with patch("bellwether.config.ALERT_CONFIDENCE_MIN", 0.7):
                with patch("bellwether.config.ALERT_FROM_EMAIL", "user@example.com"):
                    with patch("bellwether.config.ALERT_TO_EMAIL", "user@example.com"):
                        with patch("bellwether.config.ALERT_GMAIL_APP_PASSWORD", "secret"):
                            # First call — should send
                            result1 = alerts.maybe_send_alert(conn, 1)
                            assert result1 is True
                            assert mock_send.call_count == 1

                            # Second call for the same signal — should NOT send
                            result2 = alerts.maybe_send_alert(conn, 1)
                            assert result2 is False
                            assert mock_send.call_count == 1  # still only 1


def test_low_materiality_skips_alert():
    conn = _make_conn()
    # Override the signal to have materiality 3
    conn.execute("UPDATE signals SET materiality = 3 WHERE id = 1")
    conn.commit()

    with patch("bellwether.alerts._send_email") as mock_send:
        result = alerts.maybe_send_alert(conn, 1)
        assert result is False
        mock_send.assert_not_called()
