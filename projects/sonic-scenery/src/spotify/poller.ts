/**
 * Currently-playing poller.
 *
 * Polls GET /me/player/currently-playing on an interval, detects when the
 * playing trackId changes, and (on a change) assembles a full TrackContext —
 * fetching artist genres and an album-art palette — before invoking the
 * onTrackChange handler exactly once per new track.
 *
 * Handles gracefully, without spamming the handler:
 *   - 204 No Content / `is_playing: false` / null item → nothing playing
 *   - pause + resume of the same track → no re-emit
 *   - seeking within a track → no re-emit (only progressMs would change)
 *   - 401 → refresh the token once and retry
 *   - 429 → honor Retry-After
 *   - transient network/5xx errors → log and keep polling
 *
 * Endpoints used are strictly the allowed ones:
 *   GET /me/player/currently-playing, GET /artists/{id}.
 * NEVER audio-features / audio-analysis / recommendations / related-artists.
 */
import type { SpotifyAuth } from "./auth";
import { extractPalette } from "./palette";
import type { Palette, TrackContext } from "../contracts";

const API_BASE = "https://api.spotify.com/v1";
const DEFAULT_POLL_INTERVAL_MS = 2000;

export type TrackChangeHandler = (ctx: TrackContext) => void;

// --- Minimal typings for the subset of Spotify responses we read. ---

interface SpotifyImage {
  url: string;
  width: number | null;
  height: number | null;
}

interface SpotifyArtistRef {
  id: string;
  name: string;
}

interface SpotifyAlbum {
  images: SpotifyImage[];
}

interface SpotifyTrack {
  id: string | null;
  name: string;
  duration_ms: number;
  artists: SpotifyArtistRef[];
  album: SpotifyAlbum;
}

interface CurrentlyPlayingResponse {
  is_playing: boolean;
  progress_ms: number | null;
  /** `track` | `episode` | null. We only build context for tracks. */
  currently_playing_type?: string;
  item: SpotifyTrack | null;
}

interface SpotifyArtist {
  id: string;
  name: string;
  genres: string[];
}

export interface PollerOptions {
  pollIntervalMs?: number;
}

export class CurrentlyPlayingPoller {
  private readonly auth: SpotifyAuth;
  private readonly onTrackChange: TrackChangeHandler;
  private readonly pollIntervalMs: number;

  private timer: ReturnType<typeof setTimeout> | null = null;
  private running = false;
  /** trackId of the most recently emitted context; guards against re-emits. */
  private lastTrackId: string | null = null;
  /** Whether we've already logged the "nothing playing" state (avoid spam). */
  private loggedIdle = false;

  constructor(auth: SpotifyAuth, onTrackChange: TrackChangeHandler, options: PollerOptions = {}) {
    this.auth = auth;
    this.onTrackChange = onTrackChange;
    this.pollIntervalMs = options.pollIntervalMs ?? DEFAULT_POLL_INTERVAL_MS;
  }

  start(): void {
    if (this.running) return;
    this.running = true;
    void this.tick();
  }

  stop(): void {
    this.running = false;
    if (this.timer) {
      clearTimeout(this.timer);
      this.timer = null;
    }
  }

  private scheduleNext(delayMs: number = this.pollIntervalMs): void {
    if (!this.running) return;
    this.timer = setTimeout(() => void this.tick(), delayMs);
  }

  private async tick(): Promise<void> {
    if (!this.running) return;
    try {
      const playing = await this.fetchCurrentlyPlaying();
      await this.handlePlaying(playing);
      this.scheduleNext();
    } catch (err) {
      if (err instanceof RateLimitError) {
        // Honor Retry-After; don't count this as a failure.
        this.scheduleNext(err.retryAfterMs);
        return;
      }
      // eslint-disable-next-line no-console
      console.warn(`[spotify] Poll error (will retry): ${(err as Error).message}`);
      this.scheduleNext();
    }
  }

  private async handlePlaying(playing: CurrentlyPlayingResponse | null): Promise<void> {
    const item = playing?.item ?? null;
    const isTrack =
      playing != null &&
      item != null &&
      typeof item.id === "string" &&
      (playing.currently_playing_type === undefined ||
        playing.currently_playing_type === "track");

    // Nothing useful playing (204, paused with no item, ad, or podcast episode).
    if (!isTrack || !item || item.id == null) {
      if (!this.loggedIdle) {
        // eslint-disable-next-line no-console
        console.log("[spotify] Nothing playing (idle).");
        this.loggedIdle = true;
      }
      // Reset so that resuming the same track after a long idle re-emits.
      this.lastTrackId = null;
      return;
    }
    this.loggedIdle = false;

    // Same track still playing (paused/seek/normal progress) → no re-emit.
    if (item.id === this.lastTrackId) return;

    // New track → build and emit context exactly once.
    const ctx = await this.buildTrackContext(item, playing.progress_ms ?? 0);
    this.lastTrackId = item.id;
    this.onTrackChange(ctx);
  }

  private async buildTrackContext(item: SpotifyTrack, progressMs: number): Promise<TrackContext> {
    const artUrl = this.pickLargestImage(item.album.images);
    const primaryArtist = item.artists[0];

    // Fetch genres + palette in parallel; both degrade gracefully.
    const [genres, palette] = await Promise.all([
      primaryArtist ? this.fetchArtistGenres(primaryArtist.id) : Promise.resolve<string[]>([]),
      this.safePalette(artUrl),
    ]);

    return {
      trackId: item.id as string,
      title: item.name,
      artist: item.artists.map((a) => a.name).join(", "),
      genres,
      durationMs: item.duration_ms,
      progressMs,
      palette,
      artUrl,
    };
  }

  private async safePalette(artUrl: string): Promise<Palette> {
    return extractPalette(artUrl);
  }

  /** Largest album image by area; first entry is already largest per Spotify. */
  private pickLargestImage(images: SpotifyImage[]): string {
    if (images.length === 0) return "";
    let best = images[0]!;
    let bestArea = (best.width ?? 0) * (best.height ?? 0);
    for (const img of images) {
      const area = (img.width ?? 0) * (img.height ?? 0);
      if (area > bestArea) {
        best = img;
        bestArea = area;
      }
    }
    return best.url;
  }

  /**
   * Fetches artist genres. Genres are artist-level and frequently empty —
   * we pass them through verbatim and NEVER fabricate. Any failure yields [].
   */
  private async fetchArtistGenres(artistId: string): Promise<string[]> {
    try {
      const artist = await this.apiGet<SpotifyArtist>(`/artists/${artistId}`);
      return Array.isArray(artist?.genres) ? artist.genres : [];
    } catch (err) {
      // eslint-disable-next-line no-console
      console.warn(`[spotify] Could not fetch genres for ${artistId}: ${(err as Error).message}`);
      return [];
    }
  }

  /**
   * GET /me/player/currently-playing. Returns null for 204 (no active session).
   */
  private async fetchCurrentlyPlaying(): Promise<CurrentlyPlayingResponse | null> {
    const res = await this.authedFetch(`${API_BASE}/me/player/currently-playing`);
    if (res.status === 204) return null;
    if (!res.ok) {
      throw await this.toError(res);
    }
    // Some clients return an empty body even with 200; guard against it.
    const text = await res.text();
    if (!text) return null;
    return JSON.parse(text) as CurrentlyPlayingResponse;
  }

  private async apiGet<T>(path: string): Promise<T> {
    const res = await this.authedFetch(`${API_BASE}${path}`);
    if (!res.ok) {
      throw await this.toError(res);
    }
    return (await res.json()) as T;
  }

  /**
   * Performs an authorized GET, refreshing the token once on a 401 and retrying.
   */
  private async authedFetch(url: string): Promise<Response> {
    const token = await this.auth.getAccessToken();
    let res = await fetch(url, { headers: { Authorization: `Bearer ${token}` } });
    if (res.status === 401) {
      const fresh = await this.auth.refreshAccessToken();
      res = await fetch(url, { headers: { Authorization: `Bearer ${fresh}` } });
    }
    return res;
  }

  private async toError(res: Response): Promise<Error> {
    if (res.status === 429) {
      const retryAfter = Number(res.headers.get("retry-after") ?? "1");
      return new RateLimitError((Number.isFinite(retryAfter) ? retryAfter : 1) * 1000);
    }
    const text = await res.text().catch(() => "");
    return new Error(`Spotify API ${res.status}: ${text}`);
  }
}

/** Thrown on HTTP 429 so the poller can honor Retry-After. */
class RateLimitError extends Error {
  readonly retryAfterMs: number;
  constructor(retryAfterMs: number) {
    super(`Rate limited; retry after ${retryAfterMs}ms`);
    this.name = "RateLimitError";
    this.retryAfterMs = retryAfterMs;
  }
}
