"""Configuration loaded from environment / .env file.

Secrets live in .env (gitignored). Phase 0 runs fully offline — the API keys
are only needed once Phase 1/3 actually call those services.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# Project root = two levels up from this file (src/bellwether/config.py).
ROOT = Path(__file__).resolve().parents[2]

load_dotenv(ROOT / ".env")

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
TRIAGE_PROVIDER = os.getenv("TRIAGE_PROVIDER", "anthropic")
TRIAGE_MODEL = os.getenv("TRIAGE_MODEL", "claude-3-5-haiku-20241022")
DEEP_READ_PROVIDER = os.getenv("DEEP_READ_PROVIDER", "anthropic")
DEEP_READ_MODEL = os.getenv("DEEP_READ_MODEL", "claude-3-5-sonnet-20241022")
FMP_API_KEY = os.getenv("FMP_API_KEY", "")
TIINGO_API_KEY = os.getenv("TIINGO_API_KEY", "")
SEC_USER_AGENT = os.getenv("SEC_USER_AGENT", "")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
USE_MOCK_LLM = os.getenv("USE_MOCK_LLM", "false").lower() in ("true", "1", "yes")

# Phase 5 — Alert config
ALERT_MATERIALITY_MIN = int(os.getenv("ALERT_MATERIALITY_MIN", "7"))
ALERT_CONFIDENCE_MIN = float(os.getenv("ALERT_CONFIDENCE_MIN", "0.7"))
ALERT_FROM_EMAIL = os.getenv("ALERT_FROM_EMAIL", "")
ALERT_TO_EMAIL = os.getenv("ALERT_TO_EMAIL", "")
ALERT_GMAIL_APP_PASSWORD = os.getenv("ALERT_GMAIL_APP_PASSWORD", "")

# Database path, resolved relative to the project root unless absolute.
_db = os.getenv("BELLWETHER_DB", "bellwether.db")
DB_PATH = Path(_db) if os.path.isabs(_db) else ROOT / _db


def require(name: str) -> str:
    """Return a required config value, or raise a clear error if it's unset.

    Use this at the call site that actually needs the key (Phase 1/3), so
    Phase 0 never trips on missing secrets.
    """
    value = globals().get(name, "")
    if not value:
        raise RuntimeError(
            f"{name} is not set. Copy .env.example to .env and fill it in."
        )
    return value
