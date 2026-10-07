import Database from 'better-sqlite3';
import crypto from 'node:crypto';
import fs from 'node:fs';
import path from 'node:path';
import os from 'node:os';

const dataDir = path.resolve('data');
fs.mkdirSync(dataDir, { recursive: true, mode: 0o700 });
const db = new Database(path.join(dataDir, 'whoop.sqlite'));
try { fs.chmodSync(path.join(dataDir, 'whoop.sqlite'), 0o600); } catch { /* new db is created below */ }
db.pragma('journal_mode = WAL');
db.pragma('foreign_keys = ON');

db.exec(`
  CREATE TABLE IF NOT EXISTS records (
    category TEXT NOT NULL,
    record_id TEXT NOT NULL,
    start_at TEXT,
    end_at TEXT,
    updated_at TEXT,
    timezone_offset TEXT,
    raw_json TEXT NOT NULL,
    synced_at TEXT NOT NULL,
    PRIMARY KEY(category, record_id)
  );
  CREATE INDEX IF NOT EXISTS records_start ON records(start_at);
  CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT NOT NULL);
  CREATE TABLE IF NOT EXISTS events (
    id TEXT PRIMARY KEY, label TEXT NOT NULL, category TEXT NOT NULL,
    start_at TEXT NOT NULL, end_at TEXT, notes TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL
  );
  CREATE TABLE IF NOT EXISTS experiments (
    id TEXT PRIMARY KEY, name TEXT NOT NULL, hypothesis TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL
  );
  CREATE TABLE IF NOT EXISTS experiment_entries (
    id TEXT PRIMARY KEY, experiment_id TEXT NOT NULL REFERENCES experiments(id) ON DELETE CASCADE,
    date TEXT NOT NULL, condition TEXT NOT NULL, notes TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL
  );
`);

function encryptionKey(): Buffer {
  const value = process.env.TOKEN_ENCRYPTION_KEY ?? '';
  if (!/^[a-f\d]{64}$/i.test(value)) {
    throw new Error('TOKEN_ENCRYPTION_KEY must be a 64-character hex value. See .env.example.');
  }
  return Buffer.from(value, 'hex');
}

export function encryptSecret(value: string): string {
  const iv = crypto.randomBytes(12);
  const cipher = crypto.createCipheriv('aes-256-gcm', encryptionKey(), iv);
  const encrypted = Buffer.concat([cipher.update(value, 'utf8'), cipher.final()]);
  return [iv.toString('hex'), cipher.getAuthTag().toString('hex'), encrypted.toString('hex')].join(':');
}

export function decryptSecret(value: string): string {
  const [iv, tag, encrypted] = value.split(':');
  const decipher = crypto.createDecipheriv('aes-256-gcm', encryptionKey(), Buffer.from(iv, 'hex'));
  decipher.setAuthTag(Buffer.from(tag, 'hex'));
  return Buffer.concat([decipher.update(Buffer.from(encrypted, 'hex')), decipher.final()]).toString('utf8');
}

export type TokenSet = { accessToken: string; refreshToken: string; expiresAt: number };

export function saveTokens(tokens: TokenSet) {
  const insert = db.prepare('INSERT INTO settings(key,value) VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value');
  const tx = db.transaction(() => {
    insert.run('tokens', encryptSecret(JSON.stringify(tokens)));
    insert.run('connection', 'connected');
  });
  tx();
  try { fs.chmodSync(path.join(dataDir, 'whoop.sqlite'), 0o600); } catch { /* permission adjustment is best effort */ }
}

export function getTokens(): TokenSet | null {
  const row = db.prepare("SELECT value FROM settings WHERE key='tokens'").get() as { value: string } | undefined;
  if (!row) return null;
  try { return JSON.parse(decryptSecret(row.value)) as TokenSet; } catch { return null; }
}

export function clearTokens() {
  db.prepare("DELETE FROM settings WHERE key IN ('tokens', 'connection')").run();
}

export function getSetting(key: string): string | null {
  return (db.prepare('SELECT value FROM settings WHERE key=?').get(key) as { value: string } | undefined)?.value ?? null;
}

export function setSetting(key: string, value: string) {
  db.prepare('INSERT INTO settings(key,value) VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value').run(key, value);
}

export type WhoopRecord = {
  category: string; record_id: string; start_at: string | null; end_at: string | null;
  updated_at: string | null; timezone_offset: string | null; raw_json: string; synced_at: string;
};

export function upsertRecord(record: WhoopRecord) {
  db.prepare(`INSERT INTO records(category,record_id,start_at,end_at,updated_at,timezone_offset,raw_json,synced_at)
    VALUES(@category,@record_id,@start_at,@end_at,@updated_at,@timezone_offset,@raw_json,@synced_at)
    ON CONFLICT(category,record_id) DO UPDATE SET start_at=excluded.start_at,end_at=excluded.end_at,
      updated_at=excluded.updated_at,timezone_offset=excluded.timezone_offset,raw_json=excluded.raw_json,synced_at=excluded.synced_at`).run(record);
}

export function queryRecords(filters: { category?: string; start?: string; end?: string; search?: string; limit?: number; offset?: number } = {}) {
  const where: string[] = [];
  const args: unknown[] = [];
  if (filters.category) { where.push('category=?'); args.push(filters.category); }
  if (filters.start) { where.push("COALESCE(start_at,updated_at,'')>=?"); args.push(filters.start); }
  if (filters.end) { where.push("COALESCE(start_at,updated_at,'')<=?"); args.push(filters.end); }
  if (filters.search) { where.push('(raw_json LIKE ? OR category LIKE ?)'); args.push(`%${filters.search}%`, `%${filters.search}%`); }
  const clause = where.length ? `WHERE ${where.join(' AND ')}` : '';
  const limit = Math.min(Math.max(filters.limit ?? 100, 1), 500);
  const offset = Math.max(filters.offset ?? 0, 0);
  const rows = db.prepare(`SELECT * FROM records ${clause} ORDER BY COALESCE(start_at,updated_at) DESC LIMIT ? OFFSET ?`).all(...args, limit, offset) as WhoopRecord[];
  const total = (db.prepare(`SELECT COUNT(*) AS n FROM records ${clause}`).get(...args) as { n: number }).n;
  return { records: rows.map(row => ({ ...row, raw: JSON.parse(row.raw_json) })), total, limit, offset };
}

export function allRecords(filters: { category?: string; start?: string; end?: string } = {}) {
  const where: string[] = [];
  const args: unknown[] = [];
  if (filters.category) { where.push('category=?'); args.push(filters.category); }
  if (filters.start) { where.push("COALESCE(start_at,updated_at,'')>=?"); args.push(filters.start); }
  if (filters.end) { where.push("COALESCE(start_at,updated_at,'')<=?"); args.push(filters.end); }
  const rows = db.prepare(`SELECT * FROM records ${where.length ? `WHERE ${where.join(' AND ')}` : ''} ORDER BY COALESCE(start_at,updated_at) DESC`).all(...args) as WhoopRecord[];
  return rows.map(row => ({ ...row, raw: JSON.parse(row.raw_json) })) as Array<WhoopRecord & { raw: Record<string, unknown> }>;
}

export function allEventAndExperimentData() {
  return { events: listEvents(), experiments: listExperiments() };
}

export function recordCounts() {
  return db.prepare('SELECT category, COUNT(*) AS count, MAX(synced_at) AS last_synced FROM records GROUP BY category ORDER BY category').all();
}

export function summaryRows() {
  return db.prepare('SELECT category,record_id,start_at,end_at,timezone_offset,raw_json FROM records ORDER BY COALESCE(start_at,updated_at) DESC').all() as Array<{ category: string; record_id: string; start_at: string | null; end_at: string | null; timezone_offset: string | null; raw_json: string }>;
}

export function addEvent(input: { label: string; category: string; startAt: string; endAt?: string; notes?: string }) {
  const id = crypto.randomUUID(); const createdAt = new Date().toISOString();
  db.prepare('INSERT INTO events(id,label,category,start_at,end_at,notes,created_at) VALUES(?,?,?,?,?,?,?)').run(id,input.label,input.category,input.startAt,input.endAt ?? null,input.notes ?? '',createdAt);
  return getEvent(id);
}

export function listEvents() { return db.prepare('SELECT * FROM events ORDER BY start_at DESC').all(); }
function getEvent(id: string) { return db.prepare('SELECT * FROM events WHERE id=?').get(id); }
export function deleteEvent(id: string) { return db.prepare('DELETE FROM events WHERE id=?').run(id).changes > 0; }

export function createExperiment(input: { name: string; hypothesis?: string }) {
  const id = crypto.randomUUID(); const createdAt = new Date().toISOString();
  db.prepare('INSERT INTO experiments(id,name,hypothesis,created_at) VALUES(?,?,?,?)').run(id,input.name,input.hypothesis ?? '',createdAt);
  return db.prepare('SELECT * FROM experiments WHERE id=?').get(id);
}
export function listExperiments() {
  const experiments = db.prepare('SELECT * FROM experiments ORDER BY created_at DESC').all() as Array<{ id: string; name: string; hypothesis: string; created_at: string }>;
  const entries = db.prepare('SELECT * FROM experiment_entries ORDER BY date DESC').all() as Array<{ id: string; experiment_id: string; date: string; condition: string; notes: string; created_at: string }>;
  return experiments.map(experiment => ({ ...experiment, entries: entries.filter(entry => entry.experiment_id === experiment.id) }));
}
export function addExperimentEntry(input: { experimentId: string; date: string; condition: string; notes?: string }) {
  const id = crypto.randomUUID(); const createdAt = new Date().toISOString();
  db.prepare('INSERT INTO experiment_entries(id,experiment_id,date,condition,notes,created_at) VALUES(?,?,?,?,?,?)').run(id,input.experimentId,input.date,input.condition,input.notes ?? '',createdAt);
  return db.prepare('SELECT * FROM experiment_entries WHERE id=?').get(id);
}
export function deleteExperiment(id: string) { return db.prepare('DELETE FROM experiments WHERE id=?').run(id).changes > 0; }

export const storageInfo = { directory: dataDir, platform: os.platform() };
