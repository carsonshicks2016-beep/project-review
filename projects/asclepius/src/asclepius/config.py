"""Paths, constants, and app-credential loading."""

from __future__ import annotations

import json
import os
from pathlib import Path

API_BASE = "https://api.prod.whoop.com/developer"
AUTH_URL = "https://api.prod.whoop.com/oauth/oauth2/auth"
TOKEN_URL = "https://api.prod.whoop.com/oauth/oauth2/token"

# The six member-data scopes, plus `offline` which is what makes WHOOP issue a
# refresh token at all. Without `offline` you re-authorize by hand every hour.
SCOPES = [
    "read:recovery",
    "read:cycles",
    "read:workout",
    "read:sleep",
    "read:profile",
    "read:body_measurement",
    "offline",
]

# Hard ceiling from the OpenAPI spec; the API rejects anything larger.
PAGE_LIMIT = 25

# Documented limits are 100 req/min and 10k req/day per client.
RATE_LIMIT_PER_MIN = 100

HOME = Path(os.environ.get("ASCLEPIUS_HOME", Path.home() / ".asclepius"))
APP_FILE = HOME / "app.json"
CREDS_FILE = HOME / "credentials.json"
DB_FILE = HOME / "whoop.db"
# Holds the `state` of an authorization that has been started but not yet
# completed, so the browser trip and the paste-back need not happen inside one
# terminal session. See `auth.finish`.
PENDING_FILE = HOME / "pending_auth.json"

# WHOOP's dashboard rejects http:// redirects, so a plain localhost callback
# cannot be registered. https://localhost still keeps the authorization code on
# this machine — the browser fails to connect, but the code lands in the address
# bar, which is all the manual flow needs. No third party ever receives it.
DEFAULT_REDIRECT_URI = "https://localhost:8177/callback"


def ensure_home() -> Path:
    HOME.mkdir(mode=0o700, parents=True, exist_ok=True)
    return HOME


class ConfigError(RuntimeError):
    pass


def load_app_credentials() -> tuple[str, str, str]:
    """Return (client_id, client_secret, redirect_uri).

    Environment wins over the on-disk file so the MCP server can be configured
    entirely through its launcher config if you prefer.
    """
    client_id = os.environ.get("WHOOP_CLIENT_ID")
    client_secret = os.environ.get("WHOOP_CLIENT_SECRET")
    redirect_uri = os.environ.get("WHOOP_REDIRECT_URI")

    if not (client_id and client_secret) and APP_FILE.exists():
        data = json.loads(APP_FILE.read_text())
        client_id = client_id or data.get("client_id")
        client_secret = client_secret or data.get("client_secret")
        redirect_uri = redirect_uri or data.get("redirect_uri")

    if not client_id or not client_secret:
        raise ConfigError(
            "No WHOOP app credentials found. Create an app at "
            "https://developer-dashboard.whoop.com/ and either set "
            "WHOOP_CLIENT_ID / WHOOP_CLIENT_SECRET, or write them to "
            f"{APP_FILE}."
        )

    return client_id, client_secret, redirect_uri or DEFAULT_REDIRECT_URI
