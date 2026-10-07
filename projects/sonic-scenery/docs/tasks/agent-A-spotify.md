# Agent A — Spotify metadata service

**Files:** `src/spotify/`  ·  **Contract out:** `TrackContext`

## Goal
Authenticate to Spotify and emit a `TrackContext` every time the playing track
changes.

## Scope
- PKCE OAuth (Authorization Code + PKCE). Store/refresh tokens locally.
- Poll `GET /me/player/currently-playing` (~2s); detect `trackId` changes;
  handle pause/seek/no-active-session gracefully.
- Fetch artist genres via `GET /artists/{id}`.
- Download album art; extract a `Palette` with `node-vibrant`.
- Call the `onTrackChange(ctx)` handler with a populated `TrackContext`.

## Hard constraints
- **Do NOT use** Audio Features / Audio Analysis / Recommendations endpoints —
  deprecated Nov 2024, return 403 for new apps.
- Genres are artist-level and frequently empty; pass through as-is (gen handles
  the fallback). Never fabricate genres.

## Acceptance
- Playing a track logs a valid `TrackContext` (real palette from album art).
- Switching tracks emits exactly one new `TrackContext`.
- Survives token expiry (auto-refresh) and "nothing playing".
