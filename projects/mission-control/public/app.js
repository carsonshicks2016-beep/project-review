const $ = (s) => document.querySelector(s);
const grid = $('#projects');
let projects = [];
let filter = 'all';
let query = '';
let eventSource;

const esc = (value = '') => String(value).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[c]);
const kind = (p) => !p.exists ? 'missing' : p.state === 'running' ? 'running' : p.state === 'starting' ? 'starting' : p.persistentService ? 'offline' : !p.command ? 'nocommand' : 'ready';

async function request(url, options) {
  const response = await fetch(url, options);
  const data = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(data.error || `Request failed (${response.status})`);
  return data;
}

async function refresh() {
  try {
    const [projectData, system] = await Promise.all([request('/api/projects'), request('/api/system')]);
    projects = projectData.projects;
    if ($('#notice').textContent.startsWith('Mission Control could not load')) $('#notice').hidden = true;
    $('#host').textContent = `${system.hostname} · ${system.ramUsagePercent}% memory used`;
    $('#project-count').textContent = projects.length;
    $('#running-count').textContent = projects.filter((p) => p.state === 'running').length;
    draw();
  } catch (error) {
    $('#notice').hidden = false;
    $('#notice').textContent = `Mission Control could not load its project list: ${error.message}`;
  }
}

function draw() {
  const visible = projects.filter((p) => {
    const state = kind(p);
    if (filter !== 'all' && state !== filter && !(filter === 'missing' && state === 'offline')) return false;
    const text = `${p.name} ${p.subtitle || ''} ${p.description || ''} ${p.path}`.toLowerCase();
    return text.includes(query.toLowerCase().trim());
  });
  if (!visible.length) {
    grid.innerHTML = '<div class="empty">No projects match. Try another search or filter.</div>';
    return;
  }
  grid.innerHTML = visible.map((p) => {
    const state = kind(p);
    const label = { missing: 'Folder missing', offline: 'Offline', starting: 'Starting', nocommand: 'Browse only', running: 'Running', ready: 'Ready' }[state];
    const statusClass = state === 'offline' || state === 'starting' ? 'nocommand' : state;
    let actions = `<button class="action" data-action="reveal" data-id="${esc(p.id)}">Show folder</button>`;
    if (p.type === 'embed' && p.exists) actions += `<a class="action primary" href="/embedded/${encodeURIComponent(p.id)}/" target="_blank" rel="noreferrer">Open project ↗</a>`;
    else if (state === 'running') actions += `<a class="action primary" href="${esc(p.url || '#')}" target="_blank" rel="noreferrer">Open app ↗</a>${p.persistentService ? '' : `<button class="action stop" data-action="stop" data-id="${esc(p.id)}">Stop</button>`}`;
    else if (state === 'starting') actions += '<span class="action" aria-live="polite">Starting service…</span>';
    else if (p.command && p.exists) actions += `<button class="action primary" data-action="start" data-id="${esc(p.id)}">Launch</button>`;
    if (p.command && p.exists && !p.persistentService) actions += `<button class="action spacer" data-action="logs" data-id="${esc(p.id)}">Output</button>`;
    return `<article class="card"><div class="card-top"><div class="emoji">${esc(p.icon || '◈')}</div><div class="title"><h2>${esc(p.name)}</h2><div class="subtitle">${esc(p.subtitle || (p.category === 'python-project' ? 'Python workspace' : p.category || 'Project'))}</div></div><span class="status ${statusClass}">${label}</span></div><p class="description">${esc(p.description || 'Project folder in your workspace.')}</p><div class="path" title="${esc(p.path)}">${esc(p.path)}</div><div class="card-foot">${actions}</div></article>`;
  }).join('');
}

grid.addEventListener('click', async (event) => {
  const button = event.target.closest('[data-action]');
  if (!button) return;
  const { action, id } = button.dataset;
  if (action === 'logs') return showLogs(id);
  button.disabled = true;
  const previous = button.textContent;
  button.textContent = action === 'start' ? 'Launching…' : action === 'stop' ? 'Stopping…' : 'Opening…';
  try {
    const data = await request(`/api/projects/${encodeURIComponent(id)}/${action}`, { method: 'POST' });
    if (action === 'start' && data.state === 'starting') showNotice('RallyAI3 is starting. Its app link will appear as soon as the local server is ready.');
    if (action === 'start' && data.alreadyAvailable) showNotice('RallyAI3 is already running. Open app is ready.');
    if (action === 'reveal' && data.ok) return;
    await refresh();
  } catch (error) {
    showNotice(error.message);
    button.disabled = false;
    button.textContent = previous;
  }
});

function showNotice(message) {
  const notice = $('#notice');
  notice.hidden = false;
  notice.textContent = message;
  clearTimeout(showNotice.timer);
  showNotice.timer = setTimeout(() => { notice.hidden = true; }, 6500);
}

async function showLogs(id) {
  const project = projects.find((p) => p.id === id);
  $('#logs-title').textContent = `${project?.name || id} · output`;
  $('#logs').textContent = 'Connecting…';
  $('#logs-dialog').showModal();
  if (eventSource) eventSource.close();
  eventSource = new EventSource(`/api/projects/${encodeURIComponent(id)}/logs/stream`);
  eventSource.onmessage = (event) => {
    try { $('#logs').textContent = JSON.parse(event.data).text; $('#logs').scrollTop = $('#logs').scrollHeight; } catch {}
  };
  eventSource.onerror = () => {};
}

$('#close-logs').addEventListener('click', () => { $('#logs-dialog').close(); eventSource?.close(); });
$('#logs-dialog').addEventListener('close', () => eventSource?.close());
$('#search').addEventListener('input', (event) => { query = event.target.value; draw(); });
$('#filters').addEventListener('click', (event) => {
  const button = event.target.closest('[data-filter]');
  if (!button) return;
  filter = button.dataset.filter;
  document.querySelectorAll('.filter').forEach((item) => item.classList.toggle('active', item === button));
  draw();
});
$('#refresh').addEventListener('click', refresh);
window.addEventListener('keydown', (event) => { if (event.key === '/' && document.activeElement.tagName !== 'INPUT') { event.preventDefault(); $('#search').focus(); } });
refresh();
setInterval(refresh, 5000);
