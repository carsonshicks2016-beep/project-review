'use strict';
// The promotion gauntlet replays frozen candidates against one fixed panel. It never
// trains and never promotes: the verdict below is evidence for a decision you make.
(function () {
  const modes = () => [$('gauntlet-assisted').checked && 'assisted', $('gauntlet-raw').checked && 'raw'].filter(Boolean);
  const chosen = () => Array.from($('gauntlet-candidates').selectedOptions, o => o.value);
  let comparison = null, comparisonId = null, inFlight = false, pace = null;

  // Hundreds of games is a multi-day commitment, so quote it from this project's own
  // finished evaluations rather than leaving the caller to find out.
  function duration(games) {
    if (!pace?.seconds_per_game) return '';
    const hours = games * pace.seconds_per_game / 3600;
    const figure = hours < 48 ? `${hours.toFixed(1)} hours` : `${(hours / 24).toFixed(1)} days`;
    return ` Roughly ${figure} of emulator time, at the ${Math.round(pace.seconds_per_game)} s median game of your ${fmt(pace.trials)} finished evaluations.`;
  }

  function controllerPolicies() {
    const all = (state?.checkpoints || []).filter(c => c.config?.action_set === 'controller' && !c.path.includes('opponent-policy'));
    const first = all.find(c => chosen().includes(c.path));
    const character = first?.config?.character;
    return character ? all.filter(c => c.config.character === character) : all;
  }

  function plan() {
    const n = chosen().length, m = modes().length, roster = $('gauntlet-roster').checked;
    const episodes = Math.max(1, Number($('gauntlet-episodes').value) || 0);
    if (n < 2 || n > 4 || !m) return null;
    // Mirrors the server's job plan: every candidate against each CPU condition, plus
    // every ordered pair of candidates and each candidate against itself.
    const jobs = m * (n * (2 + (roster ? 5 : 0)) + n * n);
    return { jobs, games: jobs * episodes, episodes, short: episodes < 40 };
  }

  function renderPlan() {
    const select = $('gauntlet-candidates'), keep = new Set(chosen());
    const entries = controllerPolicies();
    const signature = JSON.stringify(entries.map(c => [c.path, c.name, c.steps]));
    if (select.dataset.signature !== signature) {
      select.replaceChildren(...entries.map(c => new Option(`${c.name} · ${fmt(c.steps)} steps`, c.path, false, keep.has(c.path))));
      select.dataset.signature = signature;
    }
    const p = plan();
    const busy = !state?.availability?.can_start;
    $('gauntlet-launch').disabled = !p || busy || inFlight;
    $('gauntlet-plan').innerHTML = !p
      ? 'Select two to four full-controller policies of the same character, incumbent first, and at least one set of controller rules.'
      : `${fmt(p.jobs)} jobs · ${fmt(p.games)} games, every candidate in both ports and against itself.`
        + esc(duration(p.games)) + ' '
        + (p.short ? '<strong>Below 40 games per condition the gate reports insufficient evidence by design.</strong> ' : '')
        + 'The suite owns the emulator until it finishes, and resumes from completed jobs if it is stopped.';
  }

  function label(snapshots, index) {
    const s = snapshots[index];
    if (!s) return `candidate ${index}`;
    return `${fmt(s.steps)} steps${index === 0 ? ' · incumbent' : ''}`;
  }

  function describe(row, snapshots) {
    if (row.kind === 'cpu') return [`CPU ${row.level} ${esc(row.character)}`, label(snapshots, row.p1)];
    if (row.p1 === row.p2) return ['Self mirror', label(snapshots, row.p1)];
    return [`${esc(label(snapshots, row.p1))} <small>port 1</small> vs ${esc(label(snapshots, row.p2))} <small>port 2</small>`, 'head to head'];
  }

  function renderResults() {
    const card = $('gauntlet-results');
    if (!comparison) { card.hidden = true; return; }
    card.hidden = false;
    const { snapshots, rows, gate, status } = comparison;
    const done = rows.length, total = comparison.jobs;
    $('gauntlet-progress-line').textContent =
      `${esc(status?.phase || 'waiting')} · ${fmt(done)} of ${fmt(total)} jobs · ${fmt(comparison.episodes)} games per condition · seed ${esc(comparison.seed)}`;
    $('gauntlet-body').innerHTML = rows.map(row => {
      const [condition, who] = describe(row, snapshots);
      const ci = row.interval ? `${percent(row.interval[0])}–${percent(row.interval[1])}` : '—';
      return `<tr><td class="name">${condition}<small>${esc(who)}</small></td><td>${esc(row.mode)}</td>`
        + `<td><div class="rate"><strong>${percent(row.games ? row.wins / row.games : null)}</strong>`
        + `<small class="interval">95% CI ${ci}</small><small>${fmt(row.wins)} wins / ${fmt(row.games)} games</small></div></td>`
        + `<td>${row.override_rate == null ? '—' : percent(row.override_rate)}</td>`
        + `<td>${row.valid ? 'counted' : '<strong>rejected</strong>'}</td></tr>`;
    }).join('') || '<tr><td colspan="5" class="muted">No completed jobs yet.</td></tr>';
    $('gauntlet-verdict').innerHTML = !gate ? '' : gate.candidates.map(c => {
      const name = `${label(snapshots, c.candidate)} · ${snapshots[c.candidate]?.source || 'unknown source'}`;
      return `<div class="constraint"><strong>${esc(name)}: ${c.eligible ? 'clears the promotion gate' : 'not promoted'}</strong>`
        + (c.reasons.length ? `<ul>${c.reasons.map(r => `<li>${esc(r)}</li>`).join('')}</ul>` : '')
        + '</div>';
    }).join('') + (gate ? `<p class="footnote">Intervals are Wilson intervals adjusted across ${fmt(gate.comparisons)} planned comparisons, requiring at least ${fmt(gate.minimum_games)} games per condition, a head-to-head lower bound above 50%, balanced self mirrors, and no CPU regression beyond ${percent(gate.cpu_noninferiority_margin)}. Promotion stays a manual decision.</p>` : '');
  }

  async function poll() {
    renderPlan();
    const run = (state?.runs || []).find(r => r.mode === 'compare');
    if (!run) { comparison = null; comparisonId = null; renderResults(); return; }
    if (run.id !== comparisonId) { comparison = null; comparisonId = run.id; }
    try { comparison = await api(`/api/compare/${encodeURIComponent(run.id)}`); }
    catch (_) { comparison = null; }
    renderResults();
  }

  $('gauntlet-launch').addEventListener('click', () => act(async () => {
    inFlight = true; renderPlan();
    try {
      const body = { candidates: chosen(), episodes: Number($('gauntlet-episodes').value),
                     roster: $('gauntlet-roster').checked, modes: modes() };
      const { id } = await api('/api/compare', body);
      comparisonId = id; comparison = null;
      message('Comparison started. Candidates were copied; the originals are untouched.');
    } finally { inFlight = false; }
  }));
  for (const id of ['gauntlet-candidates', 'gauntlet-episodes', 'gauntlet-roster', 'gauntlet-assisted', 'gauntlet-raw'])
    $(id).addEventListener('change', renderPlan);
  api('/api/compare').then(measured => { pace = measured; renderPlan(); }).catch(() => {});
  setInterval(poll, 3000);
  poll();
})();
