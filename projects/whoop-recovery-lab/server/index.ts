import 'dotenv/config';
import express from 'express';
import crypto from 'node:crypto';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { createServer as createViteServer } from 'vite';
import {
  addEvent, addExperimentEntry, allRecords, clearTokens, createExperiment, deleteEvent,
  deleteExperiment, getTokens, getSetting, listEvents, listExperiments, queryRecords,
  recordCounts, summaryRows,
} from './db.js';
import { buildDailySeries, csvCell, eventRecoveryResponse, fieldUnit, flattenValues, recoveryInsights } from './analytics.js';
import { authConfigured, authorizationUrl, exchangeCode, importBootstrapToken, redirectUri, syncWhoop } from './whoop.js';

const app = express();
const port = Number(process.env.PORT || 8787);
const pendingStates = new Map<string, number>();
const thisDir = path.dirname(fileURLToPath(import.meta.url));

importBootstrapToken();

app.disable('x-powered-by');
app.set('trust proxy', false);
app.use((_request, response, next) => {
  response.setHeader('X-Content-Type-Options', 'nosniff');
  response.setHeader('Referrer-Policy', 'no-referrer');
  response.setHeader('X-Frame-Options', 'DENY');
  next();
});
app.use(express.json({ limit: '1mb' }));
app.use('/api', (request, response, next) => {
  const origin = request.get('origin');
  const localOrigins = [`http://localhost:${port}`, `http://127.0.0.1:${port}`];
  if (origin && !localOrigins.includes(origin)) return response.status(403).json({ error: 'This local app accepts requests only from its own page.' });
  next();
});

const asText = (value: unknown, limit = 250) => typeof value === 'string' ? value.trim().slice(0, limit) : '';
const parseDate = (value: unknown) => {
  const date = asText(value, 50);
  return date && Number.isFinite(Date.parse(date)) ? date : null;
};

app.get('/api/health', (_request, response) => response.json({ ok: true, localOnly: true }));

app.get('/api/connection', (_request, response) => {
  const tokens = getTokens();
  response.json({
    connected: Boolean(tokens), configured: authConfigured(),
    tokenExpired: Boolean(tokens && tokens.expiresAt < Date.now()),
    lastSync: getSetting('last_sync'), lastSyncStarted: getSetting('last_sync_started'),
    counts: recordCounts(), redirectUri: redirectUri(),
    tokenSource: getSetting('token_source') ?? 'oauth',
    setupHint: !authConfigured() ? 'Add WHOOP_CLIENT_ID and WHOOP_CLIENT_SECRET to .env.' : null,
  });
});

app.get('/api/auth/url', (_request, response) => {
  try {
    const state = crypto.randomBytes(18).toString('hex');
    pendingStates.set(state, Date.now() + 10 * 60 * 1000);
    response.json({ url: authorizationUrl(state) });
  } catch (error) { response.status(400).json({ error: error instanceof Error ? error.message : 'WHOOP OAuth is not configured.' }); }
});

app.get('/auth/callback', async (request, response) => {
  const code = asText(request.query.code, 4096);
  const state = asText(request.query.state, 256);
  const expires = pendingStates.get(state);
  pendingStates.delete(state);
  for (const [key, expiry] of pendingStates) if (expiry < Date.now()) pendingStates.delete(key);
  if (!code || !expires || expires < Date.now()) return response.status(400).send('WHOOP authorization could not be confirmed. Return to Recovery Lab and try again.');
  try {
    await exchangeCode(code);
    response.redirect('/?connected=1');
  } catch (error) {
    const message = error instanceof Error ? error.message : 'WHOOP authorization failed.';
    response.status(400).send(`${message} You can close this page and return to Recovery Lab.`);
  }
});

app.post('/api/disconnect', (_request, response) => {
  clearTokens();
  response.json({ connected: false, message: 'WHOOP disconnected. Your locally saved data is retained.' });
});

app.post('/api/sync', async (_request, response) => {
  try { response.json(await syncWhoop()); }
  catch (error) { response.status(502).json({ error: error instanceof Error ? error.message : 'Sync failed.' }); }
});

app.get('/api/summary', (_request, response) => {
  const rows = summaryRows().map(row => ({ ...row, raw: JSON.parse(row.raw_json) as Record<string, any> }));
  const daily = buildDailySeries(rows).slice(-180);
  const insights = recoveryInsights(daily);
  const events = listEvents() as Array<{ id: string; label: string; category: string; start_at: string; end_at: string | null; notes: string }>;
  const eventLinks = events.map(event => {
    const eventDay = event.start_at.slice(0, 10);
    const nearest = daily.filter(point => Math.abs(Date.parse(`${point.date}T12:00:00Z`) - Date.parse(event.start_at)) <= 3 * 86_400_000);
    return { ...event, nearestDays: nearest, date: eventDay, recoveryResponse: eventRecoveryResponse(daily, eventDay, event.end_at?.slice(0, 10) ?? eventDay) };
  });
  const coverage = Object.fromEntries(recordCounts().map((row: any) => [row.category, row.count]));
  response.json({ daily, insights, events: eventLinks, experiments: listExperiments(), coverage, totalRecords: rows.length });
});

app.get('/api/events', (_request, response) => response.json(listEvents()));
app.post('/api/events', (request, response) => {
  const label = asText(request.body?.label, 160);
  const category = asText(request.body?.category, 80) || 'stress';
  const startAt = parseDate(request.body?.startAt);
  const endAt = request.body?.endAt ? parseDate(request.body.endAt) : null;
  if (!label || !startAt || (request.body?.endAt && !endAt) || (endAt && Date.parse(endAt) < Date.parse(startAt))) return response.status(400).json({ error: 'Add a label and valid start time; end time must be after the start.' });
  response.status(201).json(addEvent({ label, category, startAt, endAt: endAt ?? undefined, notes: asText(request.body?.notes, 1000) }));
});
app.delete('/api/events/:id', (request, response) => response.json({ deleted: deleteEvent(asText(request.params.id, 80)) }));

app.get('/api/experiments', (_request, response) => response.json(listExperiments()));
app.post('/api/experiments', (request, response) => {
  const name = asText(request.body?.name, 160);
  if (!name) return response.status(400).json({ error: 'An experiment name is required.' });
  response.status(201).json(createExperiment({ name, hypothesis: asText(request.body?.hypothesis, 600) }));
});
app.post('/api/experiments/:id/entries', (request, response) => {
  const experimentId = asText(request.params.id, 80);
  const date = asText(request.body?.date, 20);
  const condition = asText(request.body?.condition, 80);
  if (!/^\d{4}-\d{2}-\d{2}$/.test(date) || !Number.isFinite(Date.parse(`${date}T00:00:00Z`)) || !['Yes','No','Partial'].includes(condition)) return response.status(400).json({ error: 'Choose a valid date and condition.' });
  try { response.status(201).json(addExperimentEntry({ experimentId, date, condition, notes: asText(request.body?.notes, 1000) })); }
  catch { response.status(404).json({ error: 'Experiment not found.' }); }
});
app.delete('/api/experiments/:id', (request, response) => response.json({ deleted: deleteExperiment(asText(request.params.id, 80)) }));

app.get('/api/records', (request, response) => {
  const type = asText(request.query.type, 60);
  const search = asText(request.query.search, 160);
  const start = parseDate(request.query.start);
  const end = parseDate(request.query.end);
  const limit = Math.min(Math.max(Number(request.query.limit) || 100, 1), 500);
  const offset = Math.max(Number(request.query.offset) || 0, 0);
  response.json(queryRecords({ category: type || undefined, search: search || undefined, start: start || undefined, end: end || undefined, limit, offset }));
});

app.get('/api/export', (request, response) => {
  const category = asText(request.query.type, 60);
  const start = parseDate(request.query.start);
  const end = parseDate(request.query.end);
  const records = allRecords({ category: category || undefined, start: start || undefined, end: end || undefined });
  const format = request.query.format === 'csv' ? 'csv' : 'json';
  const stamp = new Date().toISOString().slice(0, 10);
  if (format === 'json') {
    response.setHeader('Content-Disposition', `attachment; filename="whoop-export-${stamp}.json"`);
    return response.json(records.map(record => ({ category: record.category, id: record.record_id, start: record.start_at, end: record.end_at, updated: record.updated_at, timezoneOffset: record.timezone_offset, syncedAt: record.synced_at, raw: record.raw })));
  }
  const lines = [['category','record_id','start_at','end_at','timezone_offset','field','value','unit'].map(csvCell).join(',')];
  for (const record of records) for (const field of flattenValues(record.raw)) lines.push([record.category,record.record_id,record.start_at,record.end_at,record.timezone_offset,field.path,field.value,fieldUnit(field.path)].map(csvCell).join(','));
  response.setHeader('Content-Type', 'text/csv; charset=utf-8');
  response.setHeader('Content-Disposition', `attachment; filename="whoop-export-${stamp}.csv"`);
  response.send(lines.join('\r\n'));
});

app.use('/api', (_request, response) => response.status(404).json({ error: 'API route not found.' }));

if (process.env.NODE_ENV === 'production') {
  app.use(express.static(path.resolve(thisDir, '../dist')));
  app.get('*', (_request, response) => response.sendFile(path.resolve(thisDir, '../dist/index.html')));
} else {
  const vite = await createViteServer({ server: { middlewareMode: true }, appType: 'spa' });
  app.use(vite.middlewares);
}

app.listen(port, '127.0.0.1', () => {
  console.log(`Recovery Lab is running locally at http://localhost:${port}`);
  if (process.env.WHOOP_REDIRECT_URI !== `http://localhost:${port}/auth/callback`) console.log(`Register this WHOOP redirect URI if OAuth asks for it: http://localhost:${port}/auth/callback`);
});
