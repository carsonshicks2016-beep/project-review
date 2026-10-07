import { getTokens, saveTokens, upsertRecord, setSetting, type TokenSet } from './db.js';
import fs from 'node:fs';

const API = 'https://api.prod.whoop.com/developer/v2';
const TOKEN = 'https://api.prod.whoop.com/oauth/oauth2/token';
const collections = [
  { category: 'cycles', path: '/cycle', scope: 'read:cycles' },
  { category: 'recovery', path: '/recovery', scope: 'read:recovery' },
  { category: 'sleep', path: '/activity/sleep', scope: 'read:sleep' },
  { category: 'workouts', path: '/activity/workout', scope: 'read:workout' },
] as const;

type OAuthTokenResponse = { access_token: string; refresh_token?: string; expires_in: number };
type Page = { records?: Record<string, unknown>[]; next_token?: string };

function credentials() {
  const clientId = process.env.WHOOP_CLIENT_ID;
  const clientSecret = process.env.WHOOP_CLIENT_SECRET;
  if (!clientId || !clientSecret) throw new Error('WHOOP OAuth credentials are not configured in .env.');
  return { clientId, clientSecret };
}

export async function exchangeCode(code: string): Promise<void> {
  const { clientId, clientSecret } = credentials();
  const response = await fetch(TOKEN, {
    method: 'POST', headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
    body: new URLSearchParams({ grant_type: 'authorization_code', code, client_id: clientId, client_secret: clientSecret, redirect_uri: redirectUri() }),
  });
  if (!response.ok) throw new Error(`WHOOP token exchange failed (${response.status}).`);
  saveTokenResponse(await response.json() as OAuthTokenResponse);
}

export function redirectUri() { return process.env.WHOOP_REDIRECT_URI || 'http://localhost:8787/auth/callback'; }

export function authorizationUrl(state: string) {
  const { clientId } = credentials();
  const url = new URL('https://api.prod.whoop.com/oauth/oauth2/auth');
  url.search = new URLSearchParams({
    response_type: 'code', client_id: clientId, redirect_uri: redirectUri(),
    scope: 'offline read:cycles read:recovery read:sleep read:workout read:body_measurement', state,
  }).toString();
  return url.toString();
}

function saveTokenResponse(value: OAuthTokenResponse, previous?: TokenSet) {
  if (!value.access_token) throw new Error('WHOOP did not return an access token.');
  saveTokens({
    accessToken: value.access_token,
    refreshToken: value.refresh_token || previous?.refreshToken || '',
    expiresAt: Date.now() + Number(value.expires_in || 3600) * 1000,
  });
}

export function importBootstrapToken(): boolean {
  const file = process.env.WHOOP_TOKEN_BOOTSTRAP_FILE;
  if (!file || getTokens()) return false;
  try {
    const value = JSON.parse(fs.readFileSync(file, 'utf8')) as { access_token?: string; refresh_token?: string; expiry?: string; expires_in?: number };
    if (!value.access_token || !value.refresh_token) return false;
    const expiresAt = value.expiry ? Date.parse(value.expiry) : Date.now() + Number(value.expires_in || 0) * 1000;
    saveTokens({ accessToken: value.access_token, refreshToken: value.refresh_token, expiresAt: Number.isFinite(expiresAt) ? expiresAt : 0 });
    setSetting('token_source', 'imported');
    return true;
  } catch {
    return false;
  }
}

async function refreshTokens(current: TokenSet): Promise<TokenSet> {
  if (!current.refreshToken) throw new Error('WHOOP authorization expired. Reconnect WHOOP to continue.');
  const { clientId, clientSecret } = credentials();
  const response = await fetch(TOKEN, {
    method: 'POST', headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
    body: new URLSearchParams({ grant_type: 'refresh_token', refresh_token: current.refreshToken, client_id: clientId, client_secret: clientSecret }),
  });
  if (!response.ok) throw new Error(`WHOOP token refresh failed (${response.status}); reconnect WHOOP if needed.`);
  const json = await response.json() as OAuthTokenResponse;
  if (!json.refresh_token) json.refresh_token = current.refreshToken;
  saveTokenResponse(json, current);
  return getTokens()!;
}

async function getAccessToken(force = false): Promise<string> {
  let current = getTokens();
  if (!current) throw new Error('Connect your WHOOP account to sync data.');
  if (force || current.expiresAt < Date.now() + 60_000) current = await refreshTokens(current);
  return current.accessToken;
}

async function apiGet(path: string, query?: URLSearchParams): Promise<unknown> {
  let accessToken = await getAccessToken();
  const call = () => fetch(`${API}${path}${query?.size ? `?${query}` : ''}`, { headers: { Authorization: `Bearer ${accessToken}`, Accept: 'application/json' } });
  let response = await call();
  if (response.status === 401) {
    accessToken = await getAccessToken(true);
    response = await call();
  }
  if (response.status === 404) return null;
  if (!response.ok) {
    if (response.status === 429) throw new Error('WHOOP rate limit reached. Wait a little and sync again.');
    throw new Error(`WHOOP data request failed (${response.status}) for ${path}.`);
  }
  return response.json();
}

function recordId(category: string, value: Record<string, unknown>): string {
  const id = value.id ?? value.sleep_id ?? value.cycle_id ?? value.start;
  if (id === undefined || id === null) throw new Error(`WHOOP returned a ${category} record without an identifier.`);
  return String(id);
}

function storeRecord(category: string, value: Record<string, unknown>) {
  const id = recordId(category, value);
  const now = new Date().toISOString();
  upsertRecord({
    category, record_id: id,
    start_at: typeof value.start === 'string' ? value.start : (typeof value.created_at === 'string' ? value.created_at : null),
    end_at: typeof value.end === 'string' ? value.end : null,
    updated_at: typeof value.updated_at === 'string' ? value.updated_at : null,
    timezone_offset: typeof value.timezone_offset === 'string' ? value.timezone_offset : null,
    raw_json: JSON.stringify(value), synced_at: now,
  });
}

async function syncCollection(category: string, path: string): Promise<number> {
  let nextToken: string | undefined;
  let count = 0;
  let pages = 0;
  do {
    const query = new URLSearchParams({ limit: '25' });
    if (nextToken) query.set('nextToken', nextToken);
    const result = await apiGet(path, query) as Page | null;
    const records = result?.records ?? [];
    for (const record of records) { storeRecord(category, record); count += 1; }
    nextToken = result?.next_token;
    pages += 1;
    if (pages > 1000) throw new Error(`Stopped ${category} sync after an unexpected number of pages.`);
  } while (nextToken);
  return count;
}

export async function syncWhoop() {
  const synced: Record<string, number> = {};
  const startedAt = new Date().toISOString();
  setSetting('last_sync_started', startedAt);
  for (const collection of collections) synced[collection.category] = await syncCollection(collection.category, collection.path);
  const body = await apiGet('/user/measurement/body') as Record<string, unknown> | null;
  if (body) {
    const now = new Date().toISOString();
    upsertRecord({ category: 'body_measurement', record_id: 'current', start_at: null, end_at: null, updated_at: null, timezone_offset: null, raw_json: JSON.stringify(body), synced_at: now });
    synced.body_measurement = 1;
  }
  else synced.body_measurement = 0;
  const finishedAt = new Date().toISOString();
  setSetting('last_sync', finishedAt);
  setSetting('last_sync_started', startedAt);
  return { synced, finishedAt };
}

export async function validateConnection() {
  try {
    await apiGet('/cycle', new URLSearchParams({ limit: '1' }));
    return { connected: true, message: null as string | null };
  } catch (error) {
    return { connected: false, message: error instanceof Error ? error.message : 'WHOOP connection could not be verified.' };
  }
}

export function authConfigured() {
  return Boolean(process.env.WHOOP_CLIENT_ID && process.env.WHOOP_CLIENT_SECRET);
}

export { collections };
