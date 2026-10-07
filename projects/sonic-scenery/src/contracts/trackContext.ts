/**
 * TrackContext — emitted by the Spotify service on every track change.
 * Themes and *seeds* the world (deterministic per trackId).
 *
 * Owner: Agent A (src/spotify/). Consumer: Agent C (generation).
 */
export interface Palette {
  /** Hex colors (e.g. "#1a2b3c"). */
  primary: string;
  secondary: string;
  accent: string;
  bg: string;
}

export interface TrackContext {
  trackId: string;
  title: string;
  artist: string;
  /** Artist genres from Spotify. Often sparse/empty — gen has a mood fallback. */
  genres: string[];
  durationMs: number;
  progressMs: number;
  /** Palette extracted from album art (node-vibrant). */
  palette: Palette;
  artUrl: string;
}
