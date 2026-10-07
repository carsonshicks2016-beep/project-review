"""OAuth 2.0 authorization-code flow and token lifecycle.

Two things about WHOOP's OAuth are easy to get wrong and are handled here:

1. Refresh tokens rotate. The moment you exchange one, it is dead. If the new
   token is lost before it hits disk, the only recovery is a full re-authorize.
   Every write therefore goes through an atomic replace, and refreshes are
   serialized under a file lock so two processes cannot burn the same token.
2. The refresh request must carry `scope=offline` or WHOOP returns the new
   access token without a new refresh token, silently ending the chain.
"""

from __future__ import annotations

import fcntl
import json
import os
import secrets
import sys
import threading
import time
import urllib.parse
import webbrowser
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any

import httpx

from .config import (
    AUTH_URL,
    CREDS_FILE,
    PENDING_FILE,
    SCOPES,
    TOKEN_URL,
    ConfigError,
    ensure_home,
    load_app_credentials,
)

# Refresh this many seconds before the token actually expires.
EXPIRY_MARGIN = 120


class AuthError(RuntimeError):
    pass


def _write_atomic(path: Path, payload: dict[str, Any]) -> None:
    ensure_home()
    tmp = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
    tmp.write_text(json.dumps(payload, indent=2))
    os.chmod(tmp, 0o600)
    os.replace(tmp, path)


def load_tokens() -> dict[str, Any] | None:
    if not CREDS_FILE.exists():
        return None
    return json.loads(CREDS_FILE.read_text())


def save_tokens(tok: dict[str, Any]) -> dict[str, Any]:
    record = {
        "access_token": tok["access_token"],
        "refresh_token": tok.get("refresh_token"),
        "expires_at": time.time() + float(tok.get("expires_in", 3600)),
        "scope": tok.get("scope", ""),
        "token_type": tok.get("token_type", "bearer"),
    }
    _write_atomic(CREDS_FILE, record)
    return record


@contextmanager
def _refresh_lock():
    """Serialize refreshes across processes so a rotating token is never raced."""
    ensure_home()
    lock_path = CREDS_FILE.with_suffix(".lock")
    fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def _post_token(data: dict[str, str]) -> dict[str, Any]:
    with httpx.Client(timeout=30) as c:
        r = c.post(
            TOKEN_URL,
            data=data,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
    if r.status_code != 200:
        raise AuthError(f"Token endpoint returned {r.status_code}: {r.text[:500]}")
    return r.json()


def refresh(force: bool = False) -> dict[str, Any]:
    """Return valid tokens, refreshing if they are expired or nearly so."""
    client_id, client_secret, _ = load_app_credentials()

    with _refresh_lock():
        # Re-read inside the lock: another process may have just refreshed.
        tokens = load_tokens()
        if tokens is None:
            raise AuthError("Not authorized yet. Run `uv run whoop-auth` first.")

        if not force and time.time() < tokens["expires_at"] - EXPIRY_MARGIN:
            return tokens

        if not tokens.get("refresh_token"):
            raise AuthError(
                "No refresh token on file. Re-run `uv run whoop-auth` and make "
                "sure the `offline` scope is granted."
            )

        payload = _post_token(
            {
                "grant_type": "refresh_token",
                "refresh_token": tokens["refresh_token"],
                "client_id": client_id,
                "client_secret": client_secret,
                # Required, or WHOOP drops the refresh token from the response.
                "scope": "offline",
            }
        )
        # Carry the old refresh token forward only if WHOOP omitted a new one.
        payload.setdefault("refresh_token", tokens["refresh_token"])
        return save_tokens(payload)


def access_token() -> str:
    return refresh()["access_token"]


# ---------------------------------------------------------------- authorize


class _CallbackHandler(BaseHTTPRequestHandler):
    result: dict[str, str] = {}

    def do_GET(self):  # noqa: N802
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path.rstrip("/") not in ("/callback", ""):
            self.send_response(404)
            self.end_headers()
            return
        q = urllib.parse.parse_qs(parsed.query)
        _CallbackHandler.result = {k: v[0] for k, v in q.items()}
        ok = "code" in _CallbackHandler.result
        body = (
            b"<h2>WHOOP authorized.</h2><p>You can close this tab.</p>"
            if ok
            else b"<h2>Authorization failed.</h2><p>Check the terminal.</p>"
        )
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):  # silence the default stderr logging
        pass


def _exchange(code: str, client_id: str, client_secret: str, redirect_uri: str):
    return save_tokens(
        _post_token(
            {
                "grant_type": "authorization_code",
                "code": code,
                "client_id": client_id,
                "client_secret": client_secret,
                "redirect_uri": redirect_uri,
            }
        )
    )


def _params_from(landed: str) -> dict[str, str]:
    q = urllib.parse.parse_qs(urllib.parse.urlparse(landed.strip()).query)
    got = {k: v[0] for k, v in q.items()}
    if not got:
        raise AuthError(
            "No query parameters found in that URL. Paste the entire address "
            "including everything after the '?'."
        )
    return got


def _warn_if_no_offline(rec: dict[str, Any]) -> None:
    if "offline" not in set(rec.get("scope", "").split()):
        print(
            "\nWarning: `offline` was not granted, so there is no refresh token. "
            "You will have to re-authorize every hour.",
            file=sys.stderr,
        )


def finish(landed: str) -> dict[str, Any]:
    """Complete an authorization that an earlier run started.

    The browser trip and the paste-back do not have to happen inside one
    terminal session: the pending `state` is on disk, so closing the prompt —
    or pasting the URL somewhere else by mistake — does not cost the round trip.
    """
    client_id, client_secret, redirect_uri = load_app_credentials()

    if not PENDING_FILE.exists():
        raise AuthError(
            "No authorization is in progress. Run `asclepius-auth` to get a "
            "fresh URL, authorize in the browser, then pass the redirected "
            "address back with --finish."
        )
    pending = json.loads(PENDING_FILE.read_text())

    got = _params_from(landed)
    if "error" in got:
        raise AuthError(f"{got['error']}: {got.get('error_description', '')}")
    if got.get("state") != pending.get("state"):
        raise AuthError(
            "State mismatch — that URL is from a different authorization run. "
            "Start over with `asclepius-auth`."
        )
    if "code" not in got:
        raise AuthError(f"No authorization code in the response: {got}")

    rec = _exchange(
        got["code"], client_id, client_secret,
        pending.get("redirect_uri", redirect_uri),
    )
    PENDING_FILE.unlink(missing_ok=True)
    _warn_if_no_offline(rec)
    return rec


def authorize(manual: bool = False) -> dict[str, Any]:
    client_id, client_secret, redirect_uri = load_app_credentials()
    # WHOOP requires a state parameter of at least 8 characters.
    state = secrets.token_urlsafe(16)

    url = AUTH_URL + "?" + urllib.parse.urlencode(
        {
            "response_type": "code",
            "client_id": client_id,
            "redirect_uri": redirect_uri,
            "scope": " ".join(SCOPES),
            "state": state,
        }
    )

    # Record the state before sending anyone to the browser, so `--finish` can
    # complete this round trip even if the prompt below never gets the paste.
    _write_atomic(PENDING_FILE, {"state": state, "redirect_uri": redirect_uri})

    # A local listener can only catch a plain-http redirect. WHOOP does not
    # allow registering one, so anything else has to go through the paste flow —
    # detected here rather than left to hang on a callback that never arrives.
    scheme = urllib.parse.urlparse(redirect_uri).scheme
    if not manual and scheme != "http":
        print(
            f"\nRedirect URI is '{scheme}://', which a local listener cannot "
            "receive. Switching to manual mode."
        )
        manual = True

    if manual:
        print("\nOpen this URL and authorize:\n")
        print(url + "\n")
        print(
            "Your browser will then fail to load the redirect — that is expected\n"
            "and means the authorization code stayed on your machine. Copy the\n"
            "full URL out of the address bar and paste it below.\n"
        )
        try:
            webbrowser.open(url)
        except Exception:  # noqa: BLE001
            pass
        print(
            "If this prompt goes away before you paste, nothing is lost — run\n"
            "`asclepius-auth --finish '<that URL>'` instead.\n"
        )
        got = _params_from(input("Redirected URL: "))
    else:
        parsed = urllib.parse.urlparse(redirect_uri)
        port = parsed.port or 80
        server = HTTPServer(("127.0.0.1", port), _CallbackHandler)
        thread = threading.Thread(target=server.handle_request, daemon=True)
        thread.start()

        print(f"\nListening on {redirect_uri}\nOpening browser...\n")
        print(f"If it does not open, visit:\n{url}\n")
        webbrowser.open(url)

        thread.join(timeout=300)
        server.server_close()
        got = _CallbackHandler.result
        if not got:
            raise AuthError("Timed out waiting for the callback.")

    if "error" in got:
        raise AuthError(f"{got['error']}: {got.get('error_description', '')}")
    if got.get("state") != state:
        raise AuthError("State mismatch — aborting, the response may be forged.")
    if "code" not in got:
        raise AuthError(f"No authorization code in the response: {got}")

    rec = _exchange(got["code"], client_id, client_secret, redirect_uri)
    PENDING_FILE.unlink(missing_ok=True)
    _warn_if_no_offline(rec)
    return rec


def main() -> int:
    argv = sys.argv[1:]
    try:
        if "--finish" in argv:
            rest = [a for a in argv[argv.index("--finish") + 1:] if a.strip()]
            if not rest:
                print(
                    "Usage: asclepius-auth --finish '<redirected URL>'",
                    file=sys.stderr,
                )
                return 2
            rec = finish(rest[0])
        else:
            rec = authorize(manual="--manual" in argv)
    except (AuthError, ConfigError) as e:
        print(f"\nAuthorization failed: {e}", file=sys.stderr)
        return 1
    print(f"\nAuthorized. Tokens written to {CREDS_FILE} (mode 0600).")
    print(f"Scopes granted: {rec.get('scope', '(none reported)')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
