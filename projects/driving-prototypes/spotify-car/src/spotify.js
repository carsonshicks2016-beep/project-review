/* ============================================================
   Spotify Web API — PKCE Auth + Currently Playing
   ============================================================ */

const CLIENT_ID = 'db41ef355ddf47fbad291efcee80fc68';
const REDIRECT_URI = `${window.location.origin}/callback`;
const SCOPES = 'user-read-currently-playing user-read-playback-state';
const AUTH_URL = 'https://accounts.spotify.com/authorize';
const TOKEN_URL = 'https://accounts.spotify.com/api/token';

let accessToken = null;
let refreshToken = null;
let tokenExpiry = 0;

let currentTrack = null;
let pollInterval = null;

// ——— PKCE helpers ———

function generateRandomString(length) {
  const chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-._~';
  const array = new Uint8Array(length);
  crypto.getRandomValues(array);
  return Array.from(array, b => chars[b % chars.length]).join('');
}

async function sha256(plain) {
  const encoder = new TextEncoder();
  const data = encoder.encode(plain);
  return crypto.subtle.digest('SHA-256', data);
}

function base64urlEncode(buffer) {
  const bytes = new Uint8Array(buffer);
  let str = '';
  bytes.forEach(b => str += String.fromCharCode(b));
  return btoa(str).replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/, '');
}

// ——— Auth flow ———

export async function startLogin() {
  const verifier = generateRandomString(128);
  const challenge = base64urlEncode(await sha256(verifier));

  sessionStorage.setItem('spotify_verifier', verifier);

  const params = new URLSearchParams({
    client_id: CLIENT_ID,
    response_type: 'code',
    redirect_uri: REDIRECT_URI,
    scope: SCOPES,
    code_challenge_method: 'S256',
    code_challenge: challenge,
  });

  window.location.href = `${AUTH_URL}?${params.toString()}`;
}

export async function handleCallback() {
  const params = new URLSearchParams(window.location.search);
  const code = params.get('code');
  if (!code) return false;

  const verifier = sessionStorage.getItem('spotify_verifier');
  if (!verifier) return false;

  try {
    const response = await fetch(TOKEN_URL, {
      method: 'POST',
      headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
      body: new URLSearchParams({
        client_id: CLIENT_ID,
        grant_type: 'authorization_code',
        code,
        redirect_uri: REDIRECT_URI,
        code_verifier: verifier,
      }),
    });

    const data = await response.json();
    if (data.access_token) {
      accessToken = data.access_token;
      refreshToken = data.refresh_token;
      tokenExpiry = Date.now() + data.expires_in * 1000;
      sessionStorage.removeItem('spotify_verifier');
      // Clean URL
      window.history.replaceState({}, document.title, '/');
      return true;
    }
  } catch (err) {
    console.error('Token exchange failed:', err);
  }
  return false;
}

async function refreshAccessToken() {
  if (!refreshToken) return false;
  try {
    const response = await fetch(TOKEN_URL, {
      method: 'POST',
      headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
      body: new URLSearchParams({
        client_id: CLIENT_ID,
        grant_type: 'refresh_token',
        refresh_token: refreshToken,
      }),
    });

    const data = await response.json();
    if (data.access_token) {
      accessToken = data.access_token;
      if (data.refresh_token) refreshToken = data.refresh_token;
      tokenExpiry = Date.now() + data.expires_in * 1000;
      return true;
    }
  } catch (err) {
    console.error('Token refresh failed:', err);
  }
  return false;
}

async function ensureToken() {
  if (Date.now() >= tokenExpiry - 60000) {
    return refreshAccessToken();
  }
  return true;
}

// ——— Track polling ———

async function fetchCurrentTrack() {
  if (!accessToken) return null;
  await ensureToken();

  try {
    const res = await fetch('https://api.spotify.com/v1/me/player/currently-playing', {
      headers: { Authorization: `Bearer ${accessToken}` }
    });

    if (res.status === 204 || res.status === 202) {
      return null; // nothing playing
    }

    const data = await res.json();
    if (data && data.item) {
      return {
        name: data.item.name,
        artist: data.item.artists.map(a => a.name).join(', '),
        album: data.item.album.name,
        artUrl: data.item.album.images[0]?.url || '',
        isPlaying: data.is_playing,
        progressMs: data.progress_ms,
        durationMs: data.item.duration_ms,
      };
    }
  } catch (err) {
    console.error('Failed to fetch track:', err);
  }
  return null;
}

/**
 * Start polling for current track. Calls onChange(track) when track changes.
 */
export function startPolling(onChange) {
  const poll = async () => {
    const track = await fetchCurrentTrack();
    if (track && (!currentTrack || currentTrack.name !== track.name || currentTrack.artist !== track.artist)) {
      currentTrack = track;
      onChange(track);
    } else if (!track && currentTrack) {
      currentTrack = null;
      onChange(null);
    }
  };

  poll(); // initial fetch
  pollInterval = setInterval(poll, 4000);
}

export function stopPolling() {
  if (pollInterval) clearInterval(pollInterval);
}

export function isLoggedIn() {
  return accessToken !== null;
}

export function getCurrentTrack() {
  return currentTrack;
}
