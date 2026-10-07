"""Phase 5 — Alert delivery via Gmail SMTP.

Sends a formatted email whenever a high-conviction signal is extracted,
with built-in deduplication so the same signal never fires twice.
"""

from __future__ import annotations

import smtplib
import sqlite3
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

from . import config
from .logging_setup import get_logger

log = get_logger("bellwether.alerts")

_SEC_FILING_URL = "https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={cik}&type={form_type}&dateb=&owner=include&count=10"
_SEC_DIRECT_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{accession_clean}/{accession}.txt"

DIRECTION_EMOJI = {"bullish": "🟢", "bearish": "🔴", "neutral": "⚪"}


def _ensure_alerts_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS alerts_sent (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            signal_id  INTEGER NOT NULL UNIQUE,
            sent_at    TEXT NOT NULL
        )
        """
    )
    conn.commit()


def _already_sent(conn: sqlite3.Connection, signal_id: int) -> bool:
    row = conn.execute(
        "SELECT 1 FROM alerts_sent WHERE signal_id = ?", (signal_id,)
    ).fetchone()
    return row is not None


def _mark_sent(conn: sqlite3.Connection, signal_id: int) -> None:
    import datetime
    conn.execute(
        "INSERT INTO alerts_sent (signal_id, sent_at) VALUES (?, ?)",
        (signal_id, datetime.datetime.now(datetime.timezone.utc).isoformat()),
    )
    conn.commit()


def _send_email(subject: str, html_body: str, text_body: str) -> None:
    """Send via Gmail SMTP using an App Password. Supports multiple recipients."""
    from_email = config.require("ALERT_FROM_EMAIL")
    to_emails = [e.strip() for e in config.require("ALERT_TO_EMAIL").split(",") if e.strip()]
    app_password = config.require("ALERT_GMAIL_APP_PASSWORD")

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = f"Bellwether 🔔 <{from_email}>"
    msg["To"] = ", ".join(to_emails)

    msg.attach(MIMEText(text_body, "plain"))
    msg.attach(MIMEText(html_body, "html"))

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(from_email, app_password)
        server.sendmail(from_email, to_emails, msg.as_string())


def _build_message(signal: dict, filing: dict, company: dict) -> tuple[str, str, str]:
    """Returns (subject, html_body, text_body)."""
    ticker = company["ticker"]
    name = company["name"]
    catalyst = signal["catalyst_type"].replace("_", " ").title()
    direction = signal["direction"]
    materiality = signal["materiality"]
    confidence = int(signal["confidence"] * 100)
    summary = signal["summary"]
    quote = signal["evidence_quote"]
    form_type = filing["form_type"]
    accession = filing["accession_no"]
    cik = filing["cik"]

    direction_emoji = DIRECTION_EMOJI.get(direction, "⚪")
    filing_url = f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={cik}&type={form_type}&dateb=&owner=include&count=10"

    subject = f"[Bellwether] {direction_emoji} {ticker} — {catalyst} · Materiality {materiality}/10"

    html_body = f"""
<html><body style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; max-width: 600px; margin: 0 auto; padding: 20px; color: #1a1a1a;">

  <div style="border-left: 4px solid {'#22c55e' if direction == 'bullish' else '#ef4444' if direction == 'bearish' else '#6b7280'}; padding-left: 16px; margin-bottom: 24px;">
    <h2 style="margin: 0 0 4px 0; font-size: 22px;">{direction_emoji} {ticker} <span style="color: #6b7280; font-weight: 400;">({name})</span></h2>
    <p style="margin: 0; color: #6b7280; font-size: 14px;">{form_type} · {catalyst} · {direction.title()}</p>
  </div>

  <table style="width: 100%; border-collapse: collapse; margin-bottom: 20px;">
    <tr>
      <td style="padding: 8px 12px; background: #f9fafb; border-radius: 6px; font-size: 13px; color: #6b7280; width: 40%;">Materiality</td>
      <td style="padding: 8px 12px; font-weight: 700; font-size: 18px;">{materiality} / 10</td>
    </tr>
    <tr>
      <td style="padding: 8px 12px; font-size: 13px; color: #6b7280;">Confidence</td>
      <td style="padding: 8px 12px; font-weight: 600;">{confidence}%</td>
    </tr>
    <tr>
      <td style="padding: 8px 12px; background: #f9fafb; border-radius: 6px; font-size: 13px; color: #6b7280;">Direction</td>
      <td style="padding: 8px 12px;">{direction_emoji} {direction.title()}</td>
    </tr>
  </table>

  <h3 style="font-size: 14px; text-transform: uppercase; letter-spacing: 0.05em; color: #6b7280; margin-bottom: 8px;">Summary</h3>
  <p style="margin: 0 0 20px 0; line-height: 1.6;">{summary}</p>

  <h3 style="font-size: 14px; text-transform: uppercase; letter-spacing: 0.05em; color: #6b7280; margin-bottom: 8px;">Evidence Quote</h3>
  <blockquote style="margin: 0 0 24px 0; padding: 12px 16px; background: #f9fafb; border-left: 3px solid #d1d5db; border-radius: 4px; font-style: italic; line-height: 1.6; color: #374151;">
    "{quote}"
  </blockquote>

  <a href="{filing_url}" style="display: inline-block; background: #1a1a1a; color: white; text-decoration: none; padding: 10px 20px; border-radius: 6px; font-size: 14px; font-weight: 500;">
    View {form_type} Filing →
  </a>

  <p style="margin-top: 32px; font-size: 11px; color: #9ca3af;">
    Bellwether · Personal research tool · Not investment advice<br>
    Accession: {accession}
  </p>

</body></html>
"""

    text_body = f"""BELLWETHER SIGNAL ALERT
=======================
{direction_emoji} {ticker} ({name}) — {form_type}
Catalyst: {catalyst}
Direction: {direction.title()}
Materiality: {materiality}/10
Confidence: {confidence}%

SUMMARY
{summary}

EVIDENCE QUOTE
"{quote}"

FILING LINK
{filing_url}

---
Accession: {accession}
Bellwether · Personal research tool · Not investment advice
"""

    return subject, html_body, text_body


def maybe_send_alert(conn: sqlite3.Connection, signal_id: int) -> bool:
    """Send an alert for the given signal_id if it passes thresholds and hasn't been sent.

    Returns True if an alert was sent, False otherwise.
    """
    _ensure_alerts_table(conn)

    if _already_sent(conn, signal_id):
        log.debug("Signal %d already alerted, skipping.", signal_id)
        return False

    # Load signal + filing + company data
    row = conn.execute(
        """
        SELECT s.id as signal_id, s.catalyst_type, s.direction, s.materiality,
               s.confidence, s.summary, s.evidence_quote,
               f.form_type, f.accession_no, f.cik,
               c.ticker, c.name
        FROM signals s
        JOIN filings f ON s.filing_id = f.id
        JOIN companies c ON f.cik = c.cik
        WHERE s.id = ?
        """,
        (signal_id,),
    ).fetchone()

    if not row:
        log.warning("Signal %d not found, cannot alert.", signal_id)
        return False

    row = dict(row)

    # Threshold check
    if row["materiality"] < config.ALERT_MATERIALITY_MIN:
        log.info(
            "Signal %d materiality %d < threshold %d, skipping alert.",
            signal_id, row["materiality"], config.ALERT_MATERIALITY_MIN,
        )
        return False

    if row["confidence"] < config.ALERT_CONFIDENCE_MIN:
        log.info(
            "Signal %d confidence %.2f < threshold %.2f, skipping alert.",
            signal_id, row["confidence"], config.ALERT_CONFIDENCE_MIN,
        )
        return False

    # Check creds are configured
    if not config.ALERT_FROM_EMAIL or not config.ALERT_TO_EMAIL or not config.ALERT_GMAIL_APP_PASSWORD:
        log.warning(
            "Alert credentials not configured (ALERT_FROM_EMAIL / ALERT_TO_EMAIL / "
            "ALERT_GMAIL_APP_PASSWORD). Skipping email delivery."
        )
        return False

    signal = {k: row[k] for k in ("catalyst_type", "direction", "materiality", "confidence", "summary", "evidence_quote")}
    filing = {k: row[k] for k in ("form_type", "accession_no", "cik")}
    company = {k: row[k] for k in ("ticker", "name")}

    subject, html_body, text_body = _build_message(signal, filing, company)

    try:
        _send_email(subject, html_body, text_body)
        _mark_sent(conn, signal_id)
        log.info(
            "Alert sent → %s for signal %d (%s %s mat=%d)",
            config.ALERT_TO_EMAIL, signal_id, row["ticker"],
            row["catalyst_type"], row["materiality"],
        )
        return True
    except Exception as e:
        log.error("Failed to send alert for signal %d: %s", signal_id, e)
        return False
