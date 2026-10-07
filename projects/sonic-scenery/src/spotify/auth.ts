/**
 * Spotify Authorization Code + PKCE flow for a desktop (Electron) app.
 *
 * PKCE needs only the client id and a registered loopback redirect URI; no
 * client secret is stored. We spin up a one-shot localhost HTTP listener to
 * receive the `?code=` redirect, exchange it for tokens, persist them, and
 * transparently refresh the access token before it expires.
 *
 * Docs: https://developer.spotify.com/documentation/web-api/tutorials/code-pkce-flow
 */
import { createHash, randomBytes } from "node:crypto";
import { createServer, type IncomingMessage, type ServerResponse } from "node:http";
import { mkdir, readFile, writeFile } from "node:fs/promises";
import { homedir } from "node:os";
import { dirname, join } from "node:path";

const TOKEN_URL = "https://accounts.spotify.com/api/token";
const AUTHORIZE_URL = "https://accounts.spotify.com/authorize";

/** Scope needed to read the user's current playback. */
const SCOPES = ["user-read-currently-playing", "user-read-playback-state"];

/** Refresh this many ms before the token actually expires, to avoid races. */
const REFRESH_SKEW_MS = 60_000;

interface StoredTokens {
  accessToken: string;
  /** May be absent if Spotify chose not to rotate it; we keep the old one. */
  refreshToken: string;
  /** Epoch ms when the access token expires. */
  expiresAt: number;
}

interface TokenResponse {
  access_token: string;
  token_type: string;
  expires_in: number;
  refresh_token?: string;
  scope?: string;
}

export interface AuthConfig {
  clientId: string;
  /** e.g. http://127.0.0.1:8888/callback */
  redirectUri: string;
  /** Where to persist the token cache. Defaults to ~/.sonic-scenery/spotify-token.json */
  tokenStorePath?: string;
}

function defaultTokenStorePath(): string {
  return join(homedir(), ".sonic-scenery", "spotify-token.json");
}

function base64Url(buf: Buffer): string {
  return buf.toString("base64").replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, "");
}

function generateCodeVerifier(): string {
  // 64 random bytes → ~86 char verifier (Spotify allows 43–128 chars).
  return base64Url(randomBytes(64));
}

function codeChallenge(verifier: string): string {
  return base64Url(createHash("sha256").update(verifier).digest());
}

/**
 * Manages Spotify OAuth tokens: loads a cached token, runs the interactive
 * PKCE flow when needed, and refreshes the access token on demand.
 */
export class SpotifyAuth {
  private readonly clientId: string;
  private readonly redirectUri: string;
  private readonly tokenStorePath: string;

  private tokens: StoredTokens | null = null;
  /** De-dupes concurrent refreshes into a single in-flight request. */
  private refreshing: Promise<string> | null = null;

  constructor(config: AuthConfig) {
    this.clientId = config.clientId;
    this.redirectUri = config.redirectUri;
    this.tokenStorePath = config.tokenStorePath ?? defaultTokenStorePath();
  }

  /**
   * Returns a valid access token, performing the interactive login the first
   * time and refreshing transparently thereafter.
   */
  async getAccessToken(): Promise<string> {
    if (!this.tokens) {
      this.tokens = await this.loadTokens();
    }
    if (!this.tokens) {
      this.tokens = await this.authorizeInteractive();
      await this.saveTokens();
    }
    if (Date.now() >= this.tokens.expiresAt - REFRESH_SKEW_MS) {
      return this.refreshAccessToken();
    }
    return this.tokens.accessToken;
  }

  /** Force a token refresh (e.g. after a 401). Coalesces concurrent calls. */
  async refreshAccessToken(): Promise<string> {
    if (this.refreshing) return this.refreshing;
    this.refreshing = this.doRefresh().finally(() => {
      this.refreshing = null;
    });
    return this.refreshing;
  }

  private async doRefresh(): Promise<string> {
    if (!this.tokens?.refreshToken) {
      // No refresh token available — fall back to a fresh interactive login.
      this.tokens = await this.authorizeInteractive();
      await this.saveTokens();
      return this.tokens.accessToken;
    }
    const body = new URLSearchParams({
      grant_type: "refresh_token",
      refresh_token: this.tokens.refreshToken,
      client_id: this.clientId,
    });
    const res = await fetch(TOKEN_URL, {
      method: "POST",
      headers: { "Content-Type": "application/x-www-form-urlencoded" },
      body,
    });
    if (!res.ok) {
      const text = await res.text().catch(() => "");
      // Refresh token revoked/expired → re-run interactive login.
      if (res.status === 400 || res.status === 401) {
        this.tokens = await this.authorizeInteractive();
        await this.saveTokens();
        return this.tokens.accessToken;
      }
      throw new Error(`Token refresh failed (${res.status}): ${text}`);
    }
    const data = (await res.json()) as TokenResponse;
    this.tokens = {
      accessToken: data.access_token,
      // Spotify may or may not rotate the refresh token; keep the old one if not.
      refreshToken: data.refresh_token ?? this.tokens.refreshToken,
      expiresAt: Date.now() + data.expires_in * 1000,
    };
    await this.saveTokens();
    return this.tokens.accessToken;
  }

  /**
   * Runs the interactive PKCE flow: opens the system browser to the consent
   * page and waits for the loopback redirect carrying the authorization code.
   */
  private async authorizeInteractive(): Promise<StoredTokens> {
    const verifier = generateCodeVerifier();
    const challenge = codeChallenge(verifier);
    const state = base64Url(randomBytes(16));

    const redirect = new URL(this.redirectUri);
    const authUrl = new URL(AUTHORIZE_URL);
    authUrl.searchParams.set("client_id", this.clientId);
    authUrl.searchParams.set("response_type", "code");
    authUrl.searchParams.set("redirect_uri", this.redirectUri);
    authUrl.searchParams.set("code_challenge_method", "S256");
    authUrl.searchParams.set("code_challenge", challenge);
    authUrl.searchParams.set("state", state);
    authUrl.searchParams.set("scope", SCOPES.join(" "));

    const code = await this.waitForRedirectCode(redirect, state, authUrl.toString());
    return this.exchangeCodeForTokens(code, verifier);
  }

  /**
   * Starts a localhost HTTP server on the redirect URI's host/port, prints the
   * authorize URL (and tries to open it), and resolves with the `code` once the
   * browser hits the redirect path. The server is always torn down afterward.
   */
  private waitForRedirectCode(redirect: URL, expectedState: string, authUrl: string): Promise<string> {
    const host = redirect.hostname || "127.0.0.1";
    const port = Number(redirect.port || "80");
    const callbackPath = redirect.pathname || "/callback";

    return new Promise<string>((resolve, reject) => {
      const server = createServer((req: IncomingMessage, res: ServerResponse) => {
        if (!req.url) {
          res.writeHead(400).end("Bad request");
          return;
        }
        const reqUrl = new URL(req.url, `http://${host}:${port}`);
        if (reqUrl.pathname !== callbackPath) {
          res.writeHead(404).end("Not found");
          return;
        }
        const error = reqUrl.searchParams.get("error");
        const code = reqUrl.searchParams.get("code");
        const returnedState = reqUrl.searchParams.get("state");

        const finish = (status: number, message: string) => {
          res.writeHead(status, { "Content-Type": "text/html; charset=utf-8" });
          res.end(
            `<!doctype html><meta charset="utf-8"><title>Sonic Scenery</title>` +
              `<body style="font-family:system-ui;background:#111;color:#eee;display:flex;` +
              `align-items:center;justify-content:center;height:100vh;margin:0">` +
              `<div style="text-align:center"><h2>${message}</h2>` +
              `<p>You can close this tab and return to Sonic Scenery.</p></div></body>`,
          );
          server.close();
        };

        if (error) {
          finish(400, "Authorization failed");
          reject(new Error(`Spotify authorization error: ${error}`));
          return;
        }
        if (returnedState !== expectedState) {
          finish(400, "State mismatch");
          reject(new Error("OAuth state mismatch — possible CSRF, aborting."));
          return;
        }
        if (!code) {
          finish(400, "Missing authorization code");
          reject(new Error("Redirect did not include an authorization code."));
          return;
        }
        finish(200, "Connected to Spotify");
        resolve(code);
      });

      server.on("error", reject);
      server.listen(port, host, () => {
        // eslint-disable-next-line no-console
        console.log(
          `[spotify] Authorize Sonic Scenery by opening this URL in your browser:\n${authUrl}`,
        );
        void this.tryOpenBrowser(authUrl);
      });
    });
  }

  /** Best-effort: open the consent page in the user's default browser. */
  private async tryOpenBrowser(url: string): Promise<void> {
    try {
      const { spawn } = await import("node:child_process");
      const platform = process.platform;
      const cmd = platform === "darwin" ? "open" : platform === "win32" ? "cmd" : "xdg-open";
      const args = platform === "win32" ? ["/c", "start", "", url] : [url];
      const child = spawn(cmd, args, { stdio: "ignore", detached: true });
      child.on("error", () => {
        /* user can open the URL manually; logged above */
      });
      child.unref();
    } catch {
      /* non-fatal: the URL is already logged for manual opening */
    }
  }

  private async exchangeCodeForTokens(code: string, verifier: string): Promise<StoredTokens> {
    const body = new URLSearchParams({
      grant_type: "authorization_code",
      code,
      redirect_uri: this.redirectUri,
      client_id: this.clientId,
      code_verifier: verifier,
    });
    const res = await fetch(TOKEN_URL, {
      method: "POST",
      headers: { "Content-Type": "application/x-www-form-urlencoded" },
      body,
    });
    if (!res.ok) {
      const text = await res.text().catch(() => "");
      throw new Error(`Token exchange failed (${res.status}): ${text}`);
    }
    const data = (await res.json()) as TokenResponse;
    return {
      accessToken: data.access_token,
      refreshToken: data.refresh_token ?? "",
      expiresAt: Date.now() + data.expires_in * 1000,
    };
  }

  private async loadTokens(): Promise<StoredTokens | null> {
    try {
      const raw = await readFile(this.tokenStorePath, "utf8");
      const parsed = JSON.parse(raw) as Partial<StoredTokens>;
      if (typeof parsed.accessToken === "string" && typeof parsed.expiresAt === "number") {
        return {
          accessToken: parsed.accessToken,
          refreshToken: typeof parsed.refreshToken === "string" ? parsed.refreshToken : "",
          expiresAt: parsed.expiresAt,
        };
      }
      return null;
    } catch {
      // Missing or corrupt cache → treat as "not logged in".
      return null;
    }
  }

  private async saveTokens(): Promise<void> {
    if (!this.tokens) return;
    try {
      await mkdir(dirname(this.tokenStorePath), { recursive: true });
      await writeFile(this.tokenStorePath, JSON.stringify(this.tokens, null, 2), {
        mode: 0o600,
      });
    } catch (err) {
      // eslint-disable-next-line no-console
      console.warn(`[spotify] Could not persist token cache: ${(err as Error).message}`);
    }
  }
}
