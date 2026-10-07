/* Extreme Diagnostic & Stress-Testing Lab Controller */

(() => {
  let diagnosticTargets = [];
  let currentStressLevel = 'extreme';
  let activeTarget = null;
  let isExecuting = false;

  // DOM Elements
  const dropzone = document.getElementById('diag-dropzone');
  const fileInput = document.getElementById('diag-file-input');
  const targetSelect = document.getElementById('diag-target-select');
  const runBtn = document.getElementById('run-diagnostic-btn');
  const gauntletCard = document.getElementById('gauntlet-progress');
  const resultsContainer = document.getElementById('diag-results');
  const stressBtns = document.querySelectorAll('.diag-stress-btn');
  const filterPills = document.querySelectorAll('.diag-filter-pill');

  // Initialize
  async function init() {
    setupEventListeners();
    await loadTargets();
  }

  function setupEventListeners() {
    // Dropzone Events
    if (dropzone && fileInput) {
      dropzone.addEventListener('click', () => fileInput.click());
      dropzone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropzone.classList.add('drag-over');
      });
      dropzone.addEventListener('dragleave', () => dropzone.classList.remove('drag-over'));
      dropzone.addEventListener('drop', handleFileDrop);
      fileInput.addEventListener('change', handleFileSelect);
    }

    // Stress Intensity Toggle
    stressBtns.forEach(btn => {
      btn.addEventListener('click', () => {
        stressBtns.forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        currentStressLevel = btn.dataset.level || 'extreme';
      });
    });

    // Filter Pills
    filterPills.forEach(pill => {
      pill.addEventListener('click', () => {
        filterPills.forEach(p => p.classList.remove('active'));
        pill.classList.add('active');
        renderTargetOptions(pill.dataset.filter || 'all');
      });
    });

    // Target Selection Change
    if (targetSelect) {
      targetSelect.addEventListener('change', () => {
        activeTarget = targetSelect.value;
        updateDropzoneLabel(activeTarget);
      });
    }

    // Run Button
    if (runBtn) {
      runBtn.addEventListener('click', executeGauntlet);
    }
  }

  // Load project targets from server
  async function loadTargets() {
    try {
      const resp = await fetch('/api/diagnostics/targets');
      if (!resp.ok) return;
      const data = await resp.json();
      diagnosticTargets = data.targets || [];
      renderTargetOptions('all');
    } catch (e) {
      console.warn('Could not fetch diagnostic targets:', e);
    }
  }

  function renderTargetOptions(filter) {
    if (!targetSelect) return;
    targetSelect.innerHTML = '';

    const filtered = diagnosticTargets.filter(t => {
      if (filter === 'checkpoint') return t.type === 'checkpoint';
      if (filter === 'dataset') return t.type === 'dataset';
      if (filter === 'replay') return t.type === 'replay';
      return true;
    });

    if (filtered.length === 0) {
      const opt = document.createElement('option');
      opt.value = '';
      opt.textContent = 'No matching project targets found';
      targetSelect.appendChild(opt);
      activeTarget = null;
      return;
    }

    // Group options
    const groups = {
      'Policy Checkpoints & Champions': filtered.filter(t => t.type === 'checkpoint'),
      'GameCube Memory Cards (.gci)': filtered.filter(t => t.type === 'gamecube_save'),
      'Demonstration Datasets (.npz)': filtered.filter(t => t.type === 'dataset'),
      'Recent Match Replays (.slp)': filtered.filter(t => t.type === 'replay')
    };

    for (const [grpLabel, items] of Object.entries(groups)) {
      if (items.length === 0) continue;
      const optGroup = document.createElement('optgroup');
      optGroup.label = grpLabel;
      items.forEach(item => {
        const opt = document.createElement('option');
        opt.value = item.path;
        let desc = item.name;
        if (item.character && item.character !== 'UNKNOWN') desc += ` (${item.character})`;
        if (item.steps) desc += ` · ${Math.round(item.steps / 1000)}k steps`;
        opt.textContent = desc;
        optGroup.appendChild(opt);
      });
      targetSelect.appendChild(optGroup);
    }

    if (targetSelect.options.length > 0) {
      targetSelect.selectedIndex = 0;
      activeTarget = targetSelect.value;
      updateDropzoneLabel(activeTarget);
    }
  }

  function updateDropzoneLabel(targetPath) {
    const copyTitle = document.getElementById('diag-dropzone-title');
    const copySub = document.getElementById('diag-dropzone-sub');
    if (!copyTitle || !copySub) return;

    if (!targetPath) {
      copyTitle.textContent = 'Drop any model, replay, or dataset here';
      copySub.textContent = 'Supports .zip policy checkpoints, .slp replays, and .npz tournament datasets';
      return;
    }

    const t = diagnosticTargets.find(x => x.path === targetPath);
    if (t) {
      copyTitle.textContent = `Target Armed: ${t.name}`;
      copySub.textContent = `Ready for ${currentStressLevel.toUpperCase()} stress gauntlet · Click to replace or drag new file`;
    } else {
      copyTitle.textContent = `File Loaded: ${targetPath.split('/').pop()}`;
      copySub.textContent = `Ready for ${currentStressLevel.toUpperCase()} stress gauntlet · Click to replace`;
    }
  }

  // Drag and Drop File Handlers
  async function handleFileDrop(e) {
    e.preventDefault();
    dropzone.classList.remove('drag-over');
    if (e.dataTransfer && e.dataTransfer.files.length > 0) {
      await uploadAndArmFile(e.dataTransfer.files[0]);
    }
  }

  async function handleFileSelect(e) {
    if (e.target.files && e.target.files.length > 0) {
      await uploadAndArmFile(e.target.files[0]);
    }
  }

  async function uploadAndArmFile(file) {
    const name = file.name;
    const ext = name.split('.').pop().toLowerCase();
    if (!['zip', 'slp', 'npz', 'gci', 'raw'].includes(ext)) {
      alert(`Unsupported file format '.${ext}'. Please drop a .zip, .gci, .slp, or .npz file.`);
      return;
    }

    updateDropzoneLabel(null);
    const copyTitle = document.getElementById('diag-dropzone-title');
    const copySub = document.getElementById('diag-dropzone-sub');
    if (copyTitle) copyTitle.textContent = `Uploading ${name}...`;
    if (copySub) copySub.textContent = `${(file.size / (1024 * 1024)).toFixed(2)} MB · Streaming to diagnostic chamber`;

    try {
      const resp = await fetch(`/api/diagnostics/upload?filename=${encodeURIComponent(name)}`, {
        method: 'POST',
        body: file
      });
      if (!resp.ok) {
        const err = await resp.json();
        throw new Error(err.detail || 'Upload failed');
      }
      const data = await resp.json();
      activeTarget = data.path;

      // Add to targets if not present
      if (!diagnosticTargets.some(t => t.path === data.path)) {
        diagnosticTargets.unshift({
          path: data.path,
          name: `[Uploaded] ${data.filename}`,
          type: ext === 'zip' ? 'checkpoint' : (ext === 'gci' || ext === 'raw') ? 'gamecube_save' : ext === 'slp' ? 'replay' : 'dataset',
          size_bytes: data.size_bytes
        });
        renderTargetOptions('all');
        if (targetSelect) targetSelect.value = data.path;
      }
      updateDropzoneLabel(activeTarget);

      // Automatically execute gauntlet on upload
      await executeGauntlet();
    } catch (err) {
      alert(`Failed to upload diagnostic target: ${err.message}`);
      updateDropzoneLabel(activeTarget);
    }
  }

  // Execute Gauntlet
  async function executeGauntlet() {
    if (!activeTarget) {
      alert('Please select or upload a target file first.');
      return;
    }
    if (isExecuting) return;

    isExecuting = true;
    if (runBtn) {
      runBtn.disabled = true;
      runBtn.textContent = 'Executing Gauntlet…';
    }
    if (resultsContainer) resultsContainer.hidden = true;
    if (gauntletCard) gauntletCard.hidden = false;

    // Animate gauntlet stages
    const stepEls = document.querySelectorAll('.gauntlet-step');
    let currentStep = 0;
    const progressTimer = setInterval(() => {
      if (currentStep < stepEls.length) {
        stepEls.forEach((el, idx) => {
          el.classList.toggle('active', idx === currentStep);
          if (idx < currentStep) el.classList.add('done');
        });
        currentStep++;
      }
    }, 280);

    try {
      const resp = await fetch('/api/diagnostics/probe', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          target: activeTarget,
          stress_level: currentStressLevel
        })
      });

      clearInterval(progressTimer);
      stepEls.forEach(el => {
        el.classList.remove('active');
        el.classList.add('done');
      });

      if (!resp.ok) {
        const errData = await resp.json();
        throw new Error(errData.detail || 'Diagnostic execution failed');
      }

      const dossier = await resp.json();
      setTimeout(() => {
        if (gauntletCard) gauntletCard.hidden = true;
        renderDossier(dossier);
      }, 400);

    } catch (err) {
      clearInterval(progressTimer);
      if (gauntletCard) gauntletCard.hidden = true;
      alert(`Gauntlet Failed: ${err.message}`);
    } finally {
      isExecuting = false;
      if (runBtn) {
        runBtn.disabled = false;
        runBtn.textContent = 'Execute Diagnostic Gauntlet ↗';
      }
    }
  }

  // Render Full Diagnostic Dossier
  function renderDossier(data) {
    if (!resultsContainer) return;
    resultsContainer.hidden = false;

    // 1. Overall Score & Rank
    const scoreVal = document.getElementById('diag-score-value');
    const scoreRank = document.getElementById('diag-rank-badge');
    const healthBadge = document.getElementById('diag-health-badge');
    const targetName = document.getElementById('diag-target-name');

    if (scoreVal) scoreVal.textContent = data.composite_score.toFixed(1);
    if (targetName) targetName.textContent = data.target;

    if (scoreRank) {
      scoreRank.textContent = data.rank;
      scoreRank.className = 'diag-rank-badge ' + getRankClass(data.rank);
    }

    if (healthBadge) {
      const grade = data.health_grade || 'NORMAL';
      healthBadge.textContent = `HEALTH: ${grade}`;
      healthBadge.className = 'diag-health-badge ' + (
        grade === 'EXCELLENT' ? 'health-excellent' :
        grade === 'WARNING' ? 'health-warning' : 'health-critical'
      );
    }

    // Target Specs
    const specChar = document.getElementById('diag-spec-char');
    const specType = document.getElementById('diag-spec-type');
    const specSize = document.getElementById('diag-spec-size');
    if (specChar) specChar.textContent = data.character || data.players?.P1 || 'SYSTEM';
    if (specType) specType.textContent = (data.file_type || 'checkpoint').toUpperCase();
    if (specSize) {
      if (data.total_parameters) specSize.textContent = `${(data.total_parameters / 1000).toFixed(0)}k Params`;
      else if (data.duration_seconds) specSize.textContent = `${data.duration_seconds}s Replay`;
      else if (data.samples) specSize.textContent = `${(data.samples / 1000).toFixed(0)}k Frames`;
      else if (data.file_size_bytes) specSize.textContent = `${(data.file_size_bytes / 1024).toFixed(0)} KB Save`;
    }

    // Action button to import into Policy Vault / Dolphin Memory Card
    const diagImportBtn = document.getElementById('diag-import-btn');
    if (diagImportBtn) {
      const isSaveOrModel = data.file_type === 'checkpoint' || data.file_type === 'gamecube_save' || (activeTarget && (activeTarget.endsWith('.zip') || activeTarget.endsWith('.gci') || activeTarget.endsWith('.raw')));
      if (isSaveOrModel) {
        diagImportBtn.hidden = false;
        diagImportBtn.textContent = (data.file_type === 'gamecube_save' || activeTarget?.endsWith('.gci') || activeTarget?.endsWith('.raw'))
          ? '📥 Install to Dolphin Card A'
          : '📥 Import to Policy Vault';
        diagImportBtn.onclick = () => {
          if (typeof window.openImportDialog === 'function') {
            window.openImportDialog(activeTarget, {
              name: data.target || data.name,
              character: data.character
            });
          }
        };
      } else {
        diagImportBtn.hidden = true;
      }
    }

    // 2. Render 5-Axis Combat Radar
    drawRadarChart(data.radar || {
      reaction: 75, recovery: 75, confirms: 75, discipline: 75, robustness: 75
    });

    // 3. Render Quantitative Stress Dimensions
    renderStressDimensions(data);

    // 4. Render Deep-Dive Section
    renderDeepDive(data);

    // 5. Render Expert Recommendations
    renderRecommendations(data.recommendations || []);

    // Scroll smoothly into results
    resultsContainer.scrollIntoView({ behavior: 'smooth', block: 'start' });
  }

  function getRankClass(rank) {
    if (rank.includes('S-TIER')) return 'rank-s';
    if (rank.includes('A-TIER')) return 'rank-a';
    if (rank.includes('B-TIER')) return 'rank-b';
    if (rank.includes('C-TIER')) return 'rank-c';
    return 'rank-f';
  }

  // Draw 5-Axis Canvas Radar
  function drawRadarChart(radar) {
    const canvas = document.getElementById('diag-radar-canvas');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const w = canvas.width;
    const h = canvas.height;
    const cx = w / 2;
    const cy = h / 2;
    const r = Math.min(cx, cy) - 65;

    ctx.clearRect(0, 0, w, h);

    const axes = [
      { key: 'awareness', label: 'State Awareness' },
      { key: 'reaction', label: 'Threat Reaction' },
      { key: 'recovery', label: 'Blastzone Drift' },
      { key: 'confirms', label: 'Lethal Confirms' },
      { key: 'discipline', label: 'Action Discipline' },
      { key: 'robustness', label: 'Noise Robustness' }
    ];

    const angleStep = (Math.PI * 2) / axes.length;

    // Draw concentric polygon grid
    const rings = [0.2, 0.4, 0.6, 0.8, 1.0];
    rings.forEach(lvl => {
      ctx.beginPath();
      for (let i = 0; i < axes.length; i++) {
        const angle = i * angleStep - Math.PI / 2;
        const x = cx + Math.cos(angle) * (r * lvl);
        const y = cy + Math.sin(angle) * (r * lvl);
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.closePath();
      ctx.strokeStyle = lvl === 1.0 ? '#434b6e' : '#23293f';
      ctx.lineWidth = 1;
      ctx.stroke();
    });

    // Draw radial axes & labels
    axes.forEach((axis, i) => {
      const angle = i * angleStep - Math.PI / 2;
      const x = cx + Math.cos(angle) * r;
      const y = cy + Math.sin(angle) * r;

      ctx.beginPath();
      ctx.moveTo(cx, cy);
      ctx.lineTo(x, y);
      ctx.strokeStyle = '#2d3452';
      ctx.lineWidth = 1;
      ctx.stroke();

      // Label text with directional alignment
      const cosA = Math.cos(angle);
      const sinA = Math.sin(angle);
      const lx = cx + cosA * (r + 14);
      const ly = cy + sinA * (r + 14);
      ctx.fillStyle = '#9aa5c4';
      ctx.font = '10px ui-monospace, Menlo, monospace';
      ctx.textAlign = cosA < -0.2 ? 'right' : cosA > 0.2 ? 'left' : 'center';
      ctx.textBaseline = sinA < -0.2 ? 'bottom' : sinA > 0.2 ? 'top' : 'middle';
      ctx.fillText(axis.label, lx, ly);
    });

    // Draw data polygon
    ctx.beginPath();
    axes.forEach((axis, i) => {
      const val = Math.max(0, Math.min(100, radar[axis.key] ?? 50)) / 100;
      const angle = i * angleStep - Math.PI / 2;
      const x = cx + Math.cos(angle) * (r * val);
      const y = cy + Math.sin(angle) * (r * val);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.closePath();

    // Fill polygon
    const grad = ctx.createRadialGradient(cx, cy, 10, cx, cy, r);
    grad.addColorStop(0, 'rgba(125, 245, 218, 0.45)');
    grad.addColorStop(1, 'rgba(180, 154, 255, 0.25)');
    ctx.fillStyle = grad;
    ctx.fill();

    ctx.strokeStyle = '#7df5da';
    ctx.lineWidth = 2.5;
    ctx.stroke();

    // Draw point nodes
    axes.forEach((axis, i) => {
      const val = Math.max(0, Math.min(100, radar[axis.key] ?? 50)) / 100;
      const angle = i * angleStep - Math.PI / 2;
      const x = cx + Math.cos(angle) * (r * val);
      const y = cy + Math.sin(angle) * (r * val);

      ctx.beginPath();
      ctx.arc(x, y, 4, 0, Math.PI * 2);
      ctx.fillStyle = '#7df5da';
      ctx.fill();
      ctx.strokeStyle = '#101625';
      ctx.lineWidth = 2;
      ctx.stroke();
    });
  }

  // Render Quantitative Stress Dimensions
  function renderStressDimensions(data) {
    const container = document.getElementById('diag-dimensions-container');
    if (!container) return;
    container.innerHTML = '';

    const dims = [
      {
        name: 'Threat Reaction Latency',
        val: data.stress_tests?.threat_reaction_score ?? data.radar?.reaction ?? 0,
        detail: 'Immediate point-blank rushdown & shield / dash-away responses under hostile attack frames.'
      },
      {
        name: 'Blastzone Recovery Drift',
        val: data.stress_tests?.deep_recovery_score ?? data.radar?.recovery ?? 0,
        detail: 'Ledge drift accuracy and jump conservation when dropped at deep offstage coordinates (-140, -75).'
      },
      {
        name: 'Lethal Confirm Sensitivity',
        val: data.stress_tests?.lethal_confirm_score ?? data.radar?.confirms ?? 0,
        detail: data.character === 'JIGGLYPUFF'
          ? `Rest execution calibration on popups & tech-chases. Neutral whiff risk: ${data.stress_tests?.neutral_rest_whiff_risk_pct ?? 0}%.`
          : 'High-damage lethal confirm triggers (shine / smashes / up-air) in advantage.'
      },
      {
        name: 'Action Entropy & Discipline',
        val: data.stress_tests?.discipline_score ?? data.radar?.discipline ?? 0,
        detail: `Decision variety vs degenerate button looping. Entropy: ${data.stress_tests?.action_entropy_bits ?? 2.5} bits.`
      },
      {
        name: 'Perturbation Robustness',
        val: data.stress_tests?.noise_robustness_pct ?? data.radar?.robustness ?? 0,
        detail: `Action invariance under Gaussian observation noise (σ=${currentStressLevel === 'extreme' ? '0.08' : '0.04'}).`
      }
    ];

    dims.forEach(dim => {
      const card = document.createElement('div');
      card.className = 'diag-dim-card';
      const colorClass = dim.val >= 75 ? '' : dim.val >= 50 ? 'amber' : 'rose';

      card.innerHTML = `
        <div class="diag-dim-head">
          <h4>${dim.name}</h4>
          <span class="diag-dim-val">${dim.val.toFixed(1)}%</span>
        </div>
        <div class="diag-dim-track">
          <div class="diag-dim-fill ${colorClass}" style="width: ${dim.val}%"></div>
        </div>
        <div class="diag-dim-detail">${dim.detail}</div>
      `;
      container.appendChild(card);
    });
  }

  // Render Deep Dive (Checkpoint Layers, Replay Stats, Dataset Stats)
  function renderDeepDive(data) {
    const container = document.getElementById('diag-deepdive-container');
    if (!container) return;
    container.innerHTML = '';

    if (data.file_type === 'checkpoint') {
      // 1. Neural Weights & Parameters Card
      const weightCard = document.createElement('div');
      weightCard.className = 'card';
      weightCard.innerHTML = `
        <div class="cardtop">
          <h2>Neural Parameter Health</h2>
          <span>${data.architecture?.toUpperCase() || 'FEEDFORWARD'}</span>
        </div>
        <div style="padding: 16px;">
          <div style="display: flex; gap: 20px; margin-bottom: 16px; font-family: var(--mono); font-size: 11px;">
            <div><small style="color: var(--muted); display: block;">TOTAL PARAMETERS</small><strong>${(data.total_parameters || 0).toLocaleString()}</strong></div>
            <div><small style="color: var(--muted); display: block;">DEAD NEURONS (0.0)</small><strong style="color: ${data.dead_neuron_pct > 5 ? 'var(--orange)' : 'var(--mint)'}">${data.dead_neuron_pct || 0}%</strong></div>
            <div><small style="color: var(--muted); display: block;">NAN / INF WEIGHTS</small><strong style="color: ${data.nan_weights > 0 ? 'var(--red)' : 'var(--mint)'}">${data.nan_weights || 0}</strong></div>
          </div>
          <div class="tablewrap">
            <table class="layer-table">
              <thead>
                <tr><th>Layer Name</th><th>Shape</th><th>L2 Norm</th><th>Mean ± Std</th><th>Zeros</th></tr>
              </thead>
              <tbody>
                ${(data.layer_stats || []).map(l => `
                  <tr>
                    <td>${l.name}</td>
                    <td>[${l.shape.join(', ')}]</td>
                    <td>${l.norm.toFixed(2)}</td>
                    <td>${l.mean.toFixed(3)} ± ${l.std.toFixed(3)}</td>
                    <td>${l.zeros}</td>
                  </tr>
                `).join('')}
              </tbody>
            </table>
          </div>
        </div>
      `;
      container.appendChild(weightCard);

      // 2. Behavioral Stress Parameters Card
      const stressCard = document.createElement('div');
      stressCard.className = 'card';
      stressCard.innerHTML = `
        <div class="cardtop">
          <h2>Stress Gauntlet Profile</h2>
          <span>${currentStressLevel.toUpperCase()} INTENSITY</span>
        </div>
        <div style="padding: 18px; font-family: var(--mono); font-size: 11px; line-height: 2;">
          <div><b>Threat Scenarios:</b> 64 Simulated Frame Trajectories</div>
          <div><b>Blastzone Depths:</b> 4 Extreme Edge Vectors (-110, 120, -140, 150)</div>
          <div><b>Rest / Confirm Protocol:</b> Active (Ground + Air Popups)</div>
          <div><b>Entropy Baseline:</b> 2.50 bits (Shannon Information)</div>
          <div><b>Noise Injection:</b> Gaussian σ=0.08 on 1836-dim tensor</div>
          <div><b>Recurrent State:</b> ${data.stress_tests?.lstm_retention_score ? `${data.stress_tests.lstm_retention_score}% Retention` : 'N/A (Feedforward)'}</div>
        </div>
      `;
      container.appendChild(stressCard);
    } else if (data.file_type === 'replay') {
      const replayCard = document.createElement('div');
      replayCard.className = 'card';
      replayCard.innerHTML = `
        <div class="cardtop">
          <h2>Match Replay Breakdown</h2>
          <span>${data.duration_seconds} SECONDS</span>
        </div>
        <div style="padding: 18px; font-family: var(--mono); font-size: 11px; line-height: 1.9;">
          <div><b>Matchup:</b> ${data.players?.P1} vs ${data.players?.P2}</div>
          <div><b>Total Frames:</b> ${data.total_frames?.toLocaleString()} frames</div>
          <div><b>Actions Per Minute (APM):</b> <strong style="color: var(--mint);">${data.apm}</strong></div>
          <div class="divider" style="margin: 12px 0;"></div>
          <div><b>Rest Executions:</b> ${data.rest_telemetry?.total_rests || 0} (${data.rest_telemetry?.hits || 0} hits, ${data.rest_telemetry?.whiffs || 0} whiffs)</div>
          <div><b>Rest Hit Rate:</b> <strong style="color: var(--mint);">${data.rest_telemetry?.hit_rate_pct || 0}%</strong></div>
          <div><b>Confirms Breakdown:</b> Up-Throw: ${data.rest_telemetry?.confirms?.up_throw || 0} · Crouch-Cancel: ${data.rest_telemetry?.confirms?.crouch_cancel || 0} · Tech-Chase: ${data.rest_telemetry?.confirms?.tech_chase || 0}</div>
        </div>
      `;
      container.appendChild(replayCard);
    } else if (data.file_type === 'dataset') {
      const dsCard = document.createElement('div');
      dsCard.className = 'card';
      dsCard.innerHTML = `
        <div class="cardtop">
          <h2>Demonstration Dataset Profile</h2>
          <span>${data.file_size_mb} MB ARCHIVE</span>
        </div>
        <div style="padding: 18px; font-family: var(--mono); font-size: 11px; line-height: 1.9;">
          <div><b>Total Frames:</b> ${data.samples?.toLocaleString()} decision frames</div>
          <div><b>Pro Games Ingested:</b> ${data.games_covered} tournament games</div>
          <div><b>Mean Return:</b> ${data.returns?.mean} (std: ${data.returns?.std})</div>
          <div class="divider" style="margin: 12px 0;"></div>
          <div><b>Stick Neutral:</b> ${data.action_distribution?.stick_neutral_ratio}%</div>
          <div><b>Button A (Attack):</b> ${data.action_distribution?.btn_a_press_pct}% · <b>Button B (Special):</b> ${data.action_distribution?.btn_b_press_pct}%</div>
          <div><b>Button Y (Jump):</b> ${data.action_distribution?.btn_y_jump_pct}% · <b>Shield Trigger:</b> ${data.action_distribution?.trigger_shield_pct}%</div>
        </div>
      `;
      container.appendChild(dsCard);
    } else if (data.file_type === 'gamecube_save') {
      const gcCard = document.createElement('div');
      gcCard.className = 'card';
      gcCard.innerHTML = `
        <div class="cardtop">
          <h2>GameCube Memory Card Save</h2>
          <span>DOLPHIN CARD A</span>
        </div>
        <div style="padding: 18px; font-family: var(--mono); font-size: 11px; line-height: 1.9;">
          <div><b>Game Title:</b> Super Smash Bros. Melee (${data.game_code || 'GALE01'})</div>
          <div><b>Card File Size:</b> ${(data.file_size_bytes || 0).toLocaleString()} bytes</div>
          <div><b>Installed Location:</b> <code style="color: var(--mint);">${data.installed_path || '.runtime/dolphin-user/GC/USA/Card A'}</code></div>
          <div class="divider" style="margin: 12px 0;"></div>
          <div><b>Status:</b> Ready for immediate boot in local Dolphin emulator instances.</div>
          <div><b>Unlocks:</b> All secret characters (Jigglypuff, Marth, Mewtwo, Luigi, etc.) & tournament stages active.</div>
        </div>
      `;
      container.appendChild(gcCard);
    }
  }

  // Render Recommendations
  function renderRecommendations(recomms) {
    const list = document.getElementById('diag-recomms-list');
    if (!list) return;
    list.innerHTML = '';

    recomms.forEach(text => {
      const li = document.createElement('li');
      li.textContent = text;
      list.appendChild(li);
    });
  }

  // Hook for app.js tab switching
  window.diagnosticTabActivated = () => {
    loadTargets();
  };

  window.selectDiagnosticTarget = async (path) => {
    if (diagnosticTargets.length === 0) {
      await loadTargets();
    }
    activeTarget = path;
    if (targetSelect) targetSelect.value = path;
    updateDropzoneLabel(path);
    executeGauntlet();
  };

  // Bootstrap when DOM ready
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
