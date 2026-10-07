/* Universal Leaderboard & Fleet Benchmark Controller (Tab 08) */

(() => {
  let rankingsData = [];
  let activeFilter = 'all';
  let pollTimer = null;
  let isScanning = false;

  // DOM Elements
  const scanBtn = document.getElementById('rank-scan-btn');
  const scopeSelect = document.getElementById('rank-scope-select');
  const searchInput = document.getElementById('rank-search');
  const filterPills = document.querySelectorAll('#rankings-panel .diag-filter-pill');
  const progressWrap = document.getElementById('rank-progress-wrap');
  const progressBar = document.getElementById('rank-progress-bar');
  const progressMsg = document.getElementById('rank-progress-msg');
  const progressPct = document.getElementById('rank-progress-pct');
  const progressDetail = document.getElementById('rank-progress-detail');
  const podiumSection = document.getElementById('rank-podium-section');
  const podiumCards = document.getElementById('rank-podium-cards');
  const tableBody = document.getElementById('rank-table-body');
  const totalCountBadge = document.getElementById('rank-total-count');
  const cachedTimeTag = document.getElementById('rank-cached-time');

  async function init() {
    setupEventListeners();
    await fetchCurrentRankings();
  }

  function setupEventListeners() {
    if (scanBtn) {
      scanBtn.addEventListener('click', triggerScan);
    }

    if (searchInput) {
      searchInput.addEventListener('input', () => renderTable());
    }

    filterPills.forEach(pill => {
      pill.addEventListener('click', () => {
        filterPills.forEach(p => p.classList.remove('active'));
        pill.classList.add('active');
        activeFilter = pill.dataset.filter || 'all';
        renderTable();
      });
    });

    if (tableBody) {
      tableBody.addEventListener('click', handleTableActions);
    }

    if (podiumCards) {
      podiumCards.addEventListener('click', handleTableActions);
    }
  }

  // Fetch currently cached or live rankings
  async function fetchCurrentRankings() {
    try {
      const resp = await fetch('/api/diagnostics/rankings');
      if (!resp.ok) return;
      const data = await resp.json();

      if (data.is_scanning) {
        startPollingProgress();
      } else if (data.rankings && data.rankings.length > 0) {
        rankingsData = data.rankings;
        updateCachedTimestamp(data.completed_at);
        renderPodium();
        renderTable();
        updateCategoryCounts();
      }
    } catch (err) {
      console.warn('Could not fetch fleet rankings:', err);
    }
  }

  function updateCachedTimestamp(ts) {
    if (!cachedTimeTag) return;
    if (ts) {
      const d = new Date(ts * 1000);
      cachedTimeTag.textContent = `LAST SCANNED: ${d.toLocaleDateString()} ${d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}`;
      cachedTimeTag.className = 'tag';
    } else {
      cachedTimeTag.textContent = 'READY TO SCAN';
      cachedTimeTag.className = 'tag';
    }
  }

  // Trigger Fleet Scan
  async function triggerScan() {
    if (isScanning) return;
    const scope = scopeSelect ? scopeSelect.value : 'distinct';

    if (scanBtn) {
      scanBtn.disabled = true;
      scanBtn.textContent = '⚡ Scanning Fleet…';
    }
    if (progressWrap) progressWrap.hidden = false;
    if (progressBar) progressBar.style.width = '2%';
    if (progressMsg) progressMsg.textContent = 'Starting fleet benchmark…';
    if (progressPct) progressPct.textContent = '0%';
    if (progressDetail) progressDetail.textContent = `Targeting scope: ${scope === 'distinct' ? 'Distinct Policies (~86 files)' : 'Exhaustive History (All files)'}`;

    try {
      const resp = await fetch(`/api/diagnostics/rankings/scan?scope=${encodeURIComponent(scope)}`, {
        method: 'POST'
      });
      if (!resp.ok) {
        const err = await resp.json();
        throw new Error(err.detail || 'Failed to start scan');
      }
      startPollingProgress();
    } catch (err) {
      alert(`Scan failed: ${err.message}`);
      if (scanBtn) {
        scanBtn.disabled = false;
        scanBtn.textContent = '⚡ Scan & Rank All Files';
      }
      if (progressWrap) progressWrap.hidden = true;
    }
  }

  function startPollingProgress() {
    isScanning = true;
    if (scanBtn) {
      scanBtn.disabled = true;
      scanBtn.textContent = '⚡ Scanning Fleet…';
    }
    if (progressWrap) progressWrap.hidden = false;

    if (pollTimer) clearInterval(pollTimer);
    pollTimer = setInterval(async () => {
      try {
        const resp = await fetch('/api/diagnostics/rankings/status');
        if (!resp.ok) return;
        const status = await resp.json();

        if (status.is_scanning) {
          const total = status.total || 1;
          const current = status.current || 0;
          const pct = Math.min(100, Math.round((current / total) * 100));

          if (progressBar) progressBar.style.width = `${Math.max(4, pct)}%`;
          if (progressPct) progressPct.textContent = `${pct}%`;
          if (progressMsg) progressMsg.textContent = `Evaluating artifact ${current} of ${total}…`;
          if (progressDetail) progressDetail.textContent = status.current_file ? status.current_file.split('/').slice(-2).join('/') : '';
        } else {
          // Finished!
          clearInterval(pollTimer);
          pollTimer = null;
          isScanning = false;

          if (scanBtn) {
            scanBtn.disabled = false;
            scanBtn.textContent = '⚡ Re-scan Fleet 🔄';
          }
          if (progressWrap) progressWrap.hidden = true;

          if (status.rankings) {
            rankingsData = status.rankings;
            updateCachedTimestamp(status.completed_at || (Date.now() / 1000));
            renderPodium();
            renderTable();
            updateCategoryCounts();

            if (typeof window.message === 'function') {
              window.message(`Fleet benchmark complete! Ranked ${rankingsData.length} artifacts.`);
            }
          }
        }
      } catch (e) {
        console.warn('Error polling status:', e);
      }
    }, 600);
  }

  function updateCategoryCounts() {
    const allCount = rankingsData.length;
    const ckptCount = rankingsData.filter(r => r.file_type === 'checkpoint').length;
    const dsCount = rankingsData.filter(r => r.file_type === 'dataset').length;
    const repCount = rankingsData.filter(r => r.file_type === 'replay').length;
    const gciCount = rankingsData.filter(r => r.file_type === 'gamecube_save').length;

    const elAll = document.getElementById('rank-count-all');
    const elCkpt = document.getElementById('rank-count-ckpt');
    const elDs = document.getElementById('rank-count-ds');
    const elRep = document.getElementById('rank-count-rep');
    const elGci = document.getElementById('rank-count-gci');

    if (elAll) elAll.textContent = allCount;
    if (elCkpt) elCkpt.textContent = ckptCount;
    if (elDs) elDs.textContent = dsCount;
    if (elRep) elRep.textContent = repCount;
    if (elGci) elGci.textContent = gciCount;

    if (totalCountBadge) totalCountBadge.textContent = `${allCount} ARTIFACTS RANKED`;
  }

  // Render Top 3 Champions Podium
  function renderPodium() {
    if (!podiumSection || !podiumCards) return;
    if (rankingsData.length === 0) {
      podiumSection.hidden = true;
      return;
    }

    const topThree = rankingsData.slice(0, 3);
    podiumSection.hidden = false;
    podiumCards.innerHTML = '';

    const tiers = [
      { rank: 1, medal: '🥇', title: 'Grand Champion', cls: 'podium-gold' },
      { rank: 2, medal: '🥈', title: 'Runner-Up Challenger', cls: 'podium-silver' },
      { rank: 3, medal: '🥉', title: 'Bronze Contender', cls: 'podium-bronze' }
    ];

    topThree.forEach((item, idx) => {
      const tier = tiers[idx] || tiers[2];
      const score = item.composite_score || 0;
      const card = document.createElement('div');
      card.className = `card podium-card ${tier.cls}`;

      const radar = item.radar || {};
      const char = item.character || 'SYSTEM';
      const steps = item.steps ? `${Math.round(item.steps / 1000)}k steps` : (item.file_type === 'gamecube_save' ? '100% Save' : (item.apm ? `${item.apm} APM` : 'Dataset'));

      card.innerHTML = `
        <div class="cardtop">
          <div style="display:flex;align-items:center;gap:6px;">
            <span class="podium-medal">${tier.medal}</span>
            <div><span class="eyebrow">${tier.title.toUpperCase()}</span><h3 style="margin:0;font-size:13px;">${escapeHtml(item.name || item.target)}</h3></div>
          </div>
          <span class="diag-rank-badge ${getRankClass(item.rank)}">${item.rank}</span>
        </div>
        <div class="podium-body">
          <div class="podium-score-row">
            <div class="podium-score-circle">
              <strong>${score.toFixed(1)}</strong>
              <small>INDEX</small>
            </div>
            <div class="podium-meta">
              <div><b>Character:</b> ${escapeHtml(char)}</div>
              <div><b>Scope:</b> ${steps}</div>
              <div><b>Health:</b> <span style="color: ${item.health_grade === 'EXCELLENT' ? 'var(--mint)' : 'var(--orange)'}">${item.health_grade || 'NORMAL'}</span></div>
            </div>
          </div>
          <div class="podium-radar-mini">
            <div class="radar-mini-bar" title="Threat Reaction: ${Math.round(radar.reaction || 0)}%"><label>REACT</label><div class="mini-track"><i style="width:${Math.round(radar.reaction || 0)}%;"></i></div></div>
            <div class="radar-mini-bar" title="Recovery: ${Math.round(radar.recovery || 0)}%"><label>DRIFT</label><div class="mini-track"><i style="width:${Math.round(radar.recovery || 0)}%;"></i></div></div>
            <div class="radar-mini-bar" title="Confirms: ${Math.round(radar.confirms || 0)}%"><label>CONF</label><div class="mini-track"><i style="width:${Math.round(radar.confirms || 0)}%;"></i></div></div>
          </div>
          <div class="podium-actions">
            <button data-action="inspect" data-path="${escapeHtml(item.path)}" class="primary" type="button" style="margin:0;padding:6px 10px;font-size:10px;">Inspect 🔬</button>
            ${item.file_type === 'checkpoint' ? `<button data-action="arm" data-path="${escapeHtml(item.path)}" class="subtle" type="button" style="padding:6px 10px;font-size:10px;">Arm ↗</button>` : ''}
          </div>
        </div>
      `;
      podiumCards.appendChild(card);
    });
  }

  // Render Full Standings Table
  function renderTable() {
    if (!tableBody) return;

    const query = searchInput ? searchInput.value.toLowerCase().trim() : '';

    const filtered = rankingsData.filter(item => {
      if (activeFilter !== 'all' && item.file_type !== activeFilter) return false;
      if (query) {
        const text = `${item.name} ${item.target} ${item.path} ${item.character} ${item.rank}`.toLowerCase();
        if (!text.includes(query)) return false;
      }
      return true;
    });

    if (filtered.length === 0) {
      tableBody.innerHTML = `<tr><td colspan="8" class="muted" style="text-align: center; padding: 30px;">No artifacts match the current filter.</td></tr>`;
      return;
    }

    tableBody.innerHTML = filtered.map(item => {
      const pos = item.rank_position || 1;
      const medal = pos === 1 ? '🥇' : pos === 2 ? '🥈' : pos === 3 ? '🥉' : `#${pos}`;
      const score = (item.composite_score || 0).toFixed(1);
      const scoreColor = item.composite_score >= 75 ? 'var(--mint)' : item.composite_score >= 45 ? 'var(--orange)' : 'var(--red)';
      const radar = item.radar || {};
      const char = item.character || 'SYSTEM';

      let scopeDesc = '—';
      if (item.steps) scopeDesc = `${(item.steps / 1000).toFixed(0)}k decisions`;
      else if (item.file_size_bytes) scopeDesc = `${Math.round(item.file_size_bytes / 1024)} KB`;
      else if (item.samples) scopeDesc = `${Math.round(item.samples / 1000)}k frames`;
      else if (item.apm) scopeDesc = `${item.apm} APM`;

      const typeBadge = item.file_type === 'checkpoint' ? 'CHECKPOINT' : item.file_type === 'gamecube_save' ? 'GCI SAVE' : item.file_type === 'replay' ? 'REPLAY' : 'DATASET';

      return `
        <tr>
          <td style="font-weight: 700; font-size: 13px;">${medal}</td>
          <td>
            <strong>${escapeHtml(item.name || item.target)}</strong>
            <small style="font-family: var(--mono); color: var(--muted);">${escapeHtml(item.path)}</small>
          </td>
          <td><span class="tag" style="font-size: 8px; padding: 2px 6px;">${typeBadge}</span></td>
          <td><span class="diag-rank-badge ${getRankClass(item.rank)}" style="font-size: 9px; padding: 3px 6px;">${escapeHtml(item.rank)}</span></td>
          <td style="font-family: var(--mono); font-size: 10px;">${scopeDesc}</td>
          <td>
            <div class="table-radar-meter" title="React: ${Math.round(radar.reaction || 0)}% · Drift: ${Math.round(radar.recovery || 0)}% · Conf: ${Math.round(radar.confirms || 0)}% · Disc: ${Math.round(radar.discipline || 0)}% · Rob: ${Math.round(radar.robustness || 0)}%">
              <span class="meter-bar" style="width: ${Math.round(radar.reaction || 0)}%; background: #7df5da;" title="React"></span>
              <span class="meter-bar" style="width: ${Math.round(radar.recovery || 0)}%; background: #b49aff;" title="Drift"></span>
              <span class="meter-bar" style="width: ${Math.round(radar.confirms || 0)}%; background: #ff9fb4;" title="Conf"></span>
              <span class="meter-bar" style="width: ${Math.round(radar.discipline || 0)}%; background: #ffd27d;" title="Disc"></span>
            </div>
          </td>
          <td style="text-align: right; font-family: var(--mono); font-weight: 700; font-size: 14px; color: ${scoreColor};">
            ${score}
          </td>
          <td style="text-align: right;">
            <div style="display: flex; gap: 6px; justify-content: flex-end;">
              <button data-action="inspect" data-path="${escapeHtml(item.path)}" type="button" class="subtle" style="padding: 4px 8px; font-size: 10px;" title="View Full Stress Dossier in Diagnostic Lab">Inspect 🔬</button>
              ${item.file_type === 'checkpoint' ? `
                <button data-action="arm" data-path="${escapeHtml(item.path)}" data-char="${escapeHtml(char)}" type="button" class="subtle" style="padding: 4px 8px; font-size: 10px;" title="Set as Starting Policy in Mission Control">Arm ↗</button>
                <button data-action="play" data-path="${escapeHtml(item.path)}" type="button" class="subtle" style="padding: 4px 8px; font-size: 10px;" title="Challenge in Arena">Fight ⚔️</button>
              ` : ''}
            </div>
          </td>
        </tr>
      `;
    }).join('');
  }

  function handleTableActions(e) {
    const btn = e.target.closest('[data-action]');
    if (!btn) return;
    const action = btn.dataset.action;
    const path = btn.dataset.path;
    if (!path) return;

    if (action === 'inspect') {
      if (typeof window.showTab === 'function') {
        window.showTab('diagnostic');
        if (typeof window.selectDiagnosticTarget === 'function') {
          window.selectDiagnosticTarget(path);
        }
      }
    } else if (action === 'arm') {
      const ckptSel = document.getElementById('checkpoint');
      if (ckptSel) {
        ckptSel.value = path;
        delete ckptSel.dataset.userExplicitEmpty;
      }
      const charSel = document.getElementById('agent-character');
      if (charSel && btn.dataset.char && btn.dataset.char !== 'UNKNOWN') {
        charSel.value = btn.dataset.char;
        charSel.dataset.userChanged = 'true';
      }
      if (typeof window.updateSelectors === 'function') window.updateSelectors();
      if (typeof window.renderControls === 'function') window.renderControls();
      if (typeof window.message === 'function') {
        window.message(`Armed policy “${path.split('/').pop()}” in Mission Control.`);
      }
      const launchForm = document.getElementById('launch-form');
      if (launchForm) launchForm.scrollIntoView({ behavior: 'smooth', block: 'center' });
    } else if (action === 'play') {
      const chalSel = document.getElementById('challenge-policy');
      if (chalSel) chalSel.value = path;
      if (typeof window.showTab === 'function') window.showTab('arena');
      if (typeof window.renderMission === 'function') window.renderMission();
    }
  }

  function getRankClass(rank) {
    if (!rank) return 'rank-c';
    if (rank.includes('S-TIER')) return 'rank-s';
    if (rank.includes('A-TIER')) return 'rank-a';
    if (rank.includes('B-TIER')) return 'rank-b';
    if (rank.includes('C-TIER')) return 'rank-c';
    return 'rank-f';
  }

  function escapeHtml(str) {
    if (!str) return '';
    return String(str)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;');
  }

  // Hook for app.js tab switching
  window.rankingsTabActivated = () => {
    fetchCurrentRankings();
  };

  // Bootstrap
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
