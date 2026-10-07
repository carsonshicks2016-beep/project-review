import express from 'express';
import { spawn } from 'node:child_process';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const PROJECTS_FILE = path.join(ROOT, 'config', 'projects.json');
const PORT = Number(process.env.PORT || 9000);
const app = express();
app.disable('x-powered-by');
app.use(express.json({ limit: '16kb' }));

const processes = new Map();
const archivedLogs = new Map();
const LIMIT = 400;

function catalog() {
  const parsed = JSON.parse(fs.readFileSync(PROJECTS_FILE, 'utf8'));
  if (!Array.isArray(parsed)) throw new Error('Project catalog must be an array');
  const ids = new Set();
  for (const p of parsed) {
    if (!/^[a-z0-9][a-z0-9-]*$/.test(p.id) || ids.has(p.id)) throw new Error(`Invalid or duplicate project id: ${p.id}`);
    ids.add(p.id);
    if (typeof p.name !== 'string' || typeof p.path !== 'string' || !path.isAbsolute(p.path)) throw new Error(`Invalid project entry: ${p.id}`);
    if (p.command && (!Array.isArray(p.command) || p.command.some((part) => typeof part !== 'string'))) throw new Error(`Commands must be argument arrays: ${p.id}`);
    if (p.persistentService) {
      let healthUrl;
      try { healthUrl = new URL(p.healthUrl); } catch { throw new Error(`Invalid service health URL: ${p.id}`); }
      if (healthUrl.protocol !== 'http:' || !['127.0.0.1', 'localhost', '[::1]'].includes(healthUrl.hostname)) {
        throw new Error(`Service health checks must target loopback: ${p.id}`);
      }
    }
  }
  return parsed;
}

function findProject(id) { return catalog().find((p) => p.id === id); }
function exists(p) { try { return fs.statSync(p.path).isDirectory(); } catch { return false; } }
async function serviceHealthy(p) {
  if (!p.persistentService || !p.healthUrl) return false;
  try {
    const response = await fetch(p.healthUrl, { signal: AbortSignal.timeout(900) });
    return response.ok;
  } catch { return false; }
}
function log(id, text) {
  const entry = processes.get(id);
  const logs = entry?.logs ?? archivedLogs.get(id) ?? [];
  logs.push(text);
  while (logs.length > LIMIT) logs.shift();
  archivedLogs.set(id, logs);
  if (entry) for (const client of entry.clients) client.write(`data: ${JSON.stringify({ text })}\n\n`);
}

async function projectView(p) {
  const live = processes.get(p.id);
  const present = exists(p);
  const alive = live && live.child.exitCode === null;
  const healthy = present && await serviceHealthy(p);
  const state = !present ? 'missing' : healthy ? 'running' : alive ? (p.persistentService ? 'starting' : 'running') : 'stopped';
  return { ...p, exists: present, state, pid: alive ? live.child.pid : null,
    uptime: alive ? Math.floor((Date.now() - live.startedAt) / 1000) : null };
}

app.get('/api/projects', async (_req, res) => {
  try { res.json({ projects: await Promise.all(catalog().map(projectView)) }); }
  catch (e) { res.status(500).json({ error: e.message }); }
});

app.get('/api/system', (_req, res) => {
  const mem = os.totalmem();
  res.json({ hostname: os.hostname(), platform: os.platform(), cpus: os.cpus().length,
    cpuModel: os.cpus()[0]?.model ?? 'Unknown', loadAvg: os.loadavg()[0].toFixed(2),
    ramUsagePercent: Math.round((1 - os.freemem() / mem) * 100), totalMemGb: (mem / 2 ** 30).toFixed(1),
    activeServicesCount: processes.size, totalProjectsCount: catalog().length });
});

app.post('/api/projects/:id/start', async (req, res) => {
  const project = findProject(req.params.id);
  if (!project) return res.status(404).json({ error: 'Project not found' });
  if (!exists(project)) return res.status(409).json({ error: `Folder is missing: ${project.path}` });
  if (!project.command) return res.status(400).json({ error: 'This project has no configured launch command.' });
  if (await serviceHealthy(project)) return res.json({ state: 'running', pid: null, url: project.url ?? null, alreadyAvailable: true });
  const existing = processes.get(project.id);
  if (existing && existing.child.exitCode === null) return res.json({ state: project.persistentService ? 'starting' : 'running', pid: existing.child.pid });

  const [executable, ...args] = project.command;
  const persistent = Boolean(project.persistentService);
  const child = spawn(executable, args, { cwd: project.path, detached: process.platform !== 'win32',
    shell: false, stdio: persistent ? 'ignore' : ['ignore', 'pipe', 'pipe'], env: { ...process.env, FORCE_COLOR: '1', PYTHONUNBUFFERED: '1' } });
  const entry = { child, startedAt: Date.now(), logs: [], clients: new Set() };
  processes.set(project.id, entry);
  log(project.id, `[Mission Control] Started ${project.name} (PID ${child.pid})\n`);
  if (!persistent) {
    child.stdout.on('data', (chunk) => log(project.id, chunk.toString()));
    child.stderr.on('data', (chunk) => log(project.id, chunk.toString()));
  } else {
    child.unref();
  }
  child.on('error', (error) => log(project.id, `[Launch error] ${error.message}\n`));
  child.on('close', (code, signal) => {
    log(project.id, `\n[Mission Control] Exited (code ${code ?? 'none'}, signal ${signal ?? 'none'}).\n`);
    if (processes.get(project.id) === entry) processes.delete(project.id);
  });
  return res.status(202).json({ state: 'starting', pid: child.pid, url: project.url ?? null });
});

app.post('/api/projects/:id/stop', (req, res) => {
  const project = findProject(req.params.id);
  if (!project) return res.status(404).json({ error: 'Project not found' });
  if (project.persistentService) return res.status(409).json({ error: 'This service is managed from its own dashboard and is kept running independently of Mission Control.' });
  const entry = processes.get(project.id);
  if (!entry) return res.json({ state: 'stopped' });
  try { process.kill(process.platform === 'win32' ? entry.child.pid : -entry.child.pid, 'SIGTERM'); }
  catch { try { entry.child.kill('SIGTERM'); } catch {} }
  const child = entry.child;
  const timer = setTimeout(() => {
    if (processes.get(project.id) === entry && child.exitCode === null) {
      try { process.kill(process.platform === 'win32' ? child.pid : -child.pid, 'SIGKILL'); } catch {}
    }
  }, 2500);
  timer.unref();
  res.json({ state: 'stopping', pid: child.pid });
});

app.get('/api/projects/:id/logs', (req, res) => {
  if (!findProject(req.params.id)) return res.status(404).json({ error: 'Project not found' });
  const live = processes.get(req.params.id);
  res.json({ logs: live?.logs ?? archivedLogs.get(req.params.id) ?? [], running: Boolean(live), pid: live?.child.pid ?? null });
});

app.get('/api/projects/:id/logs/stream', (req, res) => {
  if (!findProject(req.params.id)) return res.status(404).end();
  res.set({ 'Content-Type': 'text/event-stream', 'Cache-Control': 'no-cache', Connection: 'keep-alive' });
  res.flushHeaders();
  const live = processes.get(req.params.id);
  res.write(`data: ${JSON.stringify({ text: (live?.logs ?? archivedLogs.get(req.params.id) ?? []).join('') || '[No output yet]\n' })}\n\n`);
  if (live) {
    live.clients.add(res);
    req.on('close', () => live.clients.delete(res));
  }
});

for (const action of ['reveal', 'open-terminal', 'open-editor']) {
  app.post(`/api/projects/:id/${action}`, (req, res) => {
    const p = findProject(req.params.id);
    if (!p) return res.status(404).json({ error: 'Project not found' });
    if (!exists(p)) return res.status(409).json({ error: 'Project folder is missing.' });
    const command = action === 'open-terminal' ? ['open', '-a', 'Terminal', p.path]
      : action === 'open-editor' ? ['open', '-a', 'Visual Studio Code', p.path] : ['open', p.path];
    const child = spawn(command[0], command.slice(1), { stdio: 'ignore', shell: false });
    child.on('error', (e) => console.warn(`Could not run ${action}: ${e.message}`));
    res.json({ ok: true });
  });
}

app.use('/embedded/:id', (req, res, next) => {
  let project;
  try { project = findProject(req.params.id); } catch { return res.status(500).end(); }
  if (!project || project.type !== 'embed' || !exists(project)) return res.status(404).end();
  const root = path.resolve(project.path);
  const relative = decodeURIComponent(req.path === '/' ? (project.staticIndex || 'index.html') : req.path.slice(1));
  const file = path.resolve(root, relative);
  if (file !== root && !file.startsWith(root + path.sep)) return res.status(403).end();
  if (req.path === '/' || req.path === '/index.html' || req.path === `/${project.staticIndex || 'index.html'}`) {
    return res.sendFile(project.staticIndex || 'index.html', { root });
  }
  express.static(root, { fallthrough: false })(req, res, next);
});

app.use(express.static(path.join(ROOT, 'public'), { extensions: ['html'] }));
app.use((err, _req, res, _next) => {
  console.error(err);
  if (!res.headersSent) res.status(err.status || 500).json({ error: 'Request failed.' });
});

const server = app.listen(PORT, '127.0.0.1', () => {
  console.log(`Mission Control ready at http://127.0.0.1:${PORT} (${catalog().length} projects)`);
});

function shutdown() {
  for (const [id, entry] of processes) {
    if (findProject(id)?.persistentService) continue;
    try { process.kill(process.platform === 'win32' ? entry.child.pid : -entry.child.pid, 'SIGTERM'); } catch {}
  }
  server.close(() => process.exit(0));
  setTimeout(() => process.exit(0), 3000).unref();
}
process.on('SIGINT', shutdown);
process.on('SIGTERM', shutdown);
