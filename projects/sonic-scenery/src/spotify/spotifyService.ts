/**
 * Agent A — Spotify metadata service.  Brief: docs/tasks/agent-A-spotify.md
 *
 *  - PKCE OAuth against the Spotify Web API.
 *  - Poll /me/player/currently-playing; detect trackId changes.
 *  - Fetch artist genres (/artists/{id}); extract album-art palette (node-vibrant).
 *  - Emit a TrackContext on every track change.
 *
 * NOTE: Audio Features / Analysis endpoints are deprecated (Nov 2024) and MUST
 * NOT be relied on. Musical character beyond metadata comes from Agent B.
 *
 * Implementation is split across:
 *   - auth.ts     PKCE OAuth + token persistence/refresh
 *   - poller.ts   currently-playing loop + track-change detection
 *   - palette.ts  album-art → Palette (node-vibrant)
 */
import type { TrackContext } from "../contracts";
import { SpotifyAuth } from "./auth";
import { CurrentlyPlayingPoller } from "./poller";

export type TrackChangeHandler = (ctx: TrackContext) => void;

export interface SpotifyService {
  start(): Promise<void>;
  stop(): void;
}

export interface SpotifyServiceOptions {
  /** Override env-derived client id (mainly for tests). */
  clientId?: string;
  /** Override env-derived redirect URI (mainly for tests). */
  redirectUri?: string;
  /** Poll cadence in ms (default 2000). */
  pollIntervalMs?: number;
  /** Where to cache OAuth tokens; defaults to ~/.sonic-scenery/spotify-token.json */
  tokenStorePath?: string;
}

function readConfig(options: SpotifyServiceOptions): { clientId: string; redirectUri: string } {
  const clientId = options.clientId ?? process.env["SPOTIFY_CLIENT_ID"];
  const redirectUri = options.redirectUri ?? process.env["SPOTIFY_REDIRECT_URI"];
  if (!clientId) {
    throw new Error(
      "Missing SPOTIFY_CLIENT_ID. Set it in your environment (see .env.example).",
    );
  }
  if (!redirectUri) {
    throw new Error(
      "Missing SPOTIFY_REDIRECT_URI. Set it in your environment (see .env.example).",
    );
  }
  return { clientId, redirectUri };
}

export function createSpotifyService(
  onTrackChange: TrackChangeHandler,
  options: SpotifyServiceOptions = {},
): SpotifyService {
  let auth: SpotifyAuth | null = null;
  let poller: CurrentlyPlayingPoller | null = null;

  return {
    async start(): Promise<void> {
      if (poller) return; // already started
      const { clientId, redirectUri } = readConfig(options);
      auth = new SpotifyAuth({
        clientId,
        redirectUri,
        tokenStorePath: options.tokenStorePath,
      });
      // Ensure we're authenticated (runs interactive login on first use)
      // before we begin polling, so the first tick has a valid token.
      await auth.getAccessToken();
      poller = new CurrentlyPlayingPoller(auth, onTrackChange, {
        pollIntervalMs: options.pollIntervalMs,
      });
      poller.start();
    },

    stop(): void {
      poller?.stop();
      poller = null;
      auth = null;
    },
  };
}
