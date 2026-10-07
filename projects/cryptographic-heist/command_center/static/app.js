const state = {
  registry: { commands: [] },
  group: "All",
  selected: null,
  activeJobId: null,
  jobs: [],
  replays: [],
  configs: [],
  curriculum: null,
  readiness: null,
  metrics: {},
};

const els = {
  tabs: document.getElementById("group-tabs"),
  search: document.getElementById("command-search"),
  commandList: document.getElementById("command-list"),
  argsLine: document.getElementById("args-line"),
  selectedCommand: document.getElementById("selected-command"),
  runSelected: document.getElementById("run-selected"),
  terminal: document.getElementById("terminal"),
  activeJobTitle: document.getElementById("active-job-title"),
  activeJobStatus: document.getElementById("active-job-status"),
  stopJob: document.getElementById("stop-job"),
  replayTable: document.getElementById("replay-table"),
  curriculumSummary: document.getElementById("curriculum-summary"),
  curriculumActions: document.getElementById("curriculum-actions"),
  configPicker: document.getElementById("config-picker"),
  configEditor: document.getElementById("config-editor"),
  commandCount: document.getElementById("command-count"),
  replayCount: document.getElementById("replay-count"),
  jobCount: document.getElementById("job-count"),
  jobList: document.getElementById("job-list"),
  metricCards: document.getElementById("metric-cards"),
  readinessBoard: document.getElementById("readiness-board"),
};

async function api(path, options = {}) {
  const res = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
  return res.json();
}

async function init() {
  bindNav();
  bindActions();
  await refreshAll();
  setInterval(refreshJobs, 1200);
  setInterval(refreshActiveJob, 800);
}

function bindNav() {
  document.querySelectorAll(".rail-btn").forEach((button) => {
    button.addEventListener("click", () => {
      document.querySelectorAll(".rail-btn").forEach((b) => b.classList.remove("active"));
      button.classList.add("active");
      const view = button.dataset.view;
      document.querySelectorAll("[data-panel]").forEach((panel) => {
        panel.classList.toggle("hidden", panel.dataset.panel !== view);
      });
    });
  });
}

function bindActions() {
  document.getElementById("refresh-all").addEventListener("click", refreshAll);
  document.getElementById("refresh-curriculum").addEventListener("click", refreshCurriculum);
  document.getElementById("refresh-replays").addEventListener("click", refreshReplays);
  document.getElementById("refresh-jobs").addEventListener("click", refreshJobs);
  els.search.addEventListener("input", renderCommands);
  els.argsLine.addEventListener("input", updateSelectedCommandText);
  els.runSelected.addEventListener("click", () => runCommand(state.selected?.id, els.argsLine.value));
  els.stopJob.addEventListener("click", stopActiveJob);
  els.configPicker.addEventListener("change", loadSelectedConfig);
  document.getElementById("save-config").addEventListener("click", saveSelectedConfig);
}

async function refreshAll() {
  const [registry] = await Promise.all([
    api("/api/commands"),
    refreshReadiness(),
    refreshCurriculum(),
    refreshReplays(),
    refreshConfigs(),
    refreshJobs(),
    refreshMetrics(),
  ]);
  state.registry = registry;
  state.selected = registry.commands[0] || null;
  els.argsLine.value = state.selected?.default_args || "";
  renderCurriculum();
  renderTabs();
  renderCommands();
  renderReadiness();
  updateCounts();
  updateSelectedCommandText();
}

async function refreshReadiness() {
  state.readiness = await api("/api/readiness");
  renderReadiness();
}

async function refreshCurriculum() {
  state.curriculum = await api("/api/curriculum_plan");
  renderCurriculum();
}

function renderReadiness() {
  const r = state.readiness;
  if (!r) {
    els.readinessBoard.innerHTML = "";
    return;
  }
  const acceptance = r.acceptance || {};
  const learned = r.learned_control || {};
  const learnedEvader = learned.evader_checkpoint_acceptance || {};
  const activeControl = learned.active_control_acceptance || {};
  const information = r.learned_information || {};
  const activeInfo = information.active_information || information.checkpoint_league || {};
  const fidelity = r.replay_fidelity || {};
  const physics = r.physics_validation || {};
  const env = r.env_validation || {};
  const spectator = r.spectator_validation || {};
  const operational = r.operational_validation || {};
  const evidenceBundle = r.evidence_bundle || {};
  const evidenceVerification = r.evidence_bundle_verification || {};
  const selfPlay = r.self_play || {};
  const curriculum = r.curriculum || {};
  const checkpoints = r.checkpoints || {};
  const replays = r.replays || {};
  const next = curriculum.next_action || {};
  const recommended = r.recommended_command || {};
  const topDiagnostic = (acceptance.top_diagnostics || [])[0];
  const nextCommand = commandById(recommended.command_id || next.command_id);
  const nextLabel = nextCommand?.label || recommended.command_id || next.command_id || "none";
  const nextSource = recommended.source || (next.command_id ? "curriculum" : "none");
  els.readinessBoard.innerHTML = `
    <article class="readiness-card readiness-wide">
      <div class="readiness-head">
        <span class="status ${readinessStatusClass(r.status)}">${escapeHtml(r.status)}</span>
        <strong>${escapeHtml(r.summary)}</strong>
      </div>
      <div class="readiness-grid">
        <span>Acceptance</span><code>${escapeHtml(acceptanceStatusLine(acceptance))}</code>
        <span>Evader</span><code>${escapeHtml(acceptanceStatusLine(learnedEvader))}</code>
        <span>Active control</span><code>${escapeHtml(acceptanceStatusLine(activeControl))}</code>
        <span>Info stack</span><code>${escapeHtml(informationStatusLine(activeInfo))}</code>
        <span>Replay proof</span><code>${escapeHtml(fidelityStatusLine(fidelity))}</code>
        <span>Physics</span><code>${escapeHtml(physicsStatusLine(physics))}</code>
        <span>MARL env</span><code>${escapeHtml(envStatusLine(env))}</code>
        <span>Spectator</span><code>${escapeHtml(spectatorStatusLine(spectator))}</code>
        <span>Operational</span><code>${escapeHtml(operationalStatusLine(operational))}</code>
        <span>Evidence</span><code>${escapeHtml(evidenceStatusLine(evidenceBundle, evidenceVerification))}</code>
        <span>Self-play</span><code>${escapeHtml(selfPlayStatusLine(selfPlay))}</code>
        <span>Milestone</span><code>${escapeHtml(milestoneStatusLine(acceptance))}</code>
        <span>Top issue</span><code>${escapeHtml(diagnosticLine(topDiagnostic))}</code>
      </div>
    </article>
    <article class="readiness-card">
      <div class="readiness-head">
        <span class="status running">next</span>
        <strong>${escapeHtml(nextLabel)}</strong>
      </div>
      <div class="readiness-grid">
        <span>Source</span><code>${escapeHtml(nextSource)}</code>
        <span>Lane</span><code>${escapeHtml(recommended.lane || next.lane || "n/a")}</code>
        <span>Plan</span><code>${escapeHtml(curriculum.status || "missing")} · ${escapeHtml(curriculum.diagnostic_count ?? "?")} diag</code>
      </div>
      ${recommended.command_id ? '<button id="run-next-readiness" class="run-btn readiness-run" title="Run recommended readiness action">▶</button>' : ""}
    </article>
    <article class="readiness-card">
      <div class="readiness-head">
        <span class="status ${checkpoints.missing ? "running" : "succeeded"}">models</span>
        <strong>${escapeHtml(checkpointLine(checkpoints))}</strong>
      </div>
      <div class="mini-list">${checkpointMiniList(checkpoints.items || [])}</div>
    </article>
    <article class="readiness-card">
      <div class="readiness-head">
        <span class="status ${replays.acceptance ? "succeeded" : "running"}">evidence</span>
        <strong>${escapeHtml(replayLine(replays))}</strong>
      </div>
      <div class="readiness-grid">
        <span>Latest</span><code>${escapeHtml(replays.latest?.name || "none")}</code>
        <span>Valid</span><code>${escapeHtml(replays.valid ?? 0)}</code>
        <span>Acceptance</span><code>${escapeHtml(replays.acceptance ?? 0)}</code>
        <span>Bundle</span><code>${escapeHtml(evidenceBundleLine(evidenceBundle))}</code>
        <span>Verify</span><code>${escapeHtml(evidenceVerificationLine(evidenceVerification))}</code>
      </div>
    </article>
  `;
  const runNext = document.getElementById("run-next-readiness");
  if (runNext) {
    runNext.addEventListener("click", async () => {
      if (recommended.source === "curriculum" && next.command_id === recommended.command_id) {
        const job = await api("/api/run_curriculum_action", {
          method: "POST",
          body: JSON.stringify({ index: 0 }),
        });
        state.activeJobId = job.id;
        renderActiveJob(job);
        await refreshJobs();
        return;
      }
      await runCommand(recommended.command_id, recommended.default_args || "");
    });
  }
}

function readinessStatusClass(status) {
  if (status === "operational") return "succeeded";
  if (
    status === "core_gate_failing" ||
    status === "needs_acceptance" ||
    status === "fidelity_gate_failing" ||
    status === "physics_gate_failing" ||
    status === "env_gate_failing" ||
    status === "spectator_gate_failing" ||
    status === "operational_gate_failing" ||
    status === "evidence_bundle_failing" ||
    status === "evidence_verification_failing" ||
    status === "self_play_needs_repair"
  ) return "failed";
  return "running";
}

function acceptanceStatusLine(acceptance) {
  if (!acceptance.exists) return "missing";
  return `${acceptance.status} · ${acceptance.required_checks_passed}/${acceptance.required_checks} required`;
}

function milestoneStatusLine(acceptance) {
  if (!acceptance.exists) return "missing";
  const total = acceptance.milestone_targets ?? 0;
  if (!total) return "none";
  return `${acceptance.milestone_targets_passed}/${total} full-plan targets`;
}

function informationStatusLine(summary) {
  if (!summary?.exists) return "missing";
  const score = typeof summary.overall_score === "number" ? summary.overall_score.toFixed(2) : "?";
  const lift = typeof summary.mean_deception_lift === "number" ? summary.mean_deception_lift.toFixed(1) : "?";
  return `${summary.status || "unknown"} · score ${score} · deception ${lift}`;
}

function fidelityStatusLine(summary) {
  if (!summary?.exists) return "missing";
  const state = summary.state_sha256_short || "?";
  const deterministic = summary.scripted_determinism_matched === undefined
    ? "det ?"
    : `det ${summary.scripted_determinism_matched ? "yes" : "no"}`;
  return `${summary.status || "unknown"} · ${state} · ${deterministic}`;
}

function physicsStatusLine(summary) {
  if (!summary?.exists) return "missing";
  const yaw = typeof summary.yaw_ratio === "number" ? summary.yaw_ratio.toFixed(2) : "?";
  const radius = typeof summary.radius_ratio === "number" ? summary.radius_ratio.toFixed(2) : "?";
  return `${summary.status || "unknown"} · ${summary.checks_passed ?? 0}/${summary.checks ?? 0} · yaw ${yaw} · radius ${radius}`;
}

function envStatusLine(summary) {
  if (!summary?.exists) return "missing";
  const checks = `${summary.checks_passed ?? 0}/${summary.checks ?? 0}`;
  const api = summary.pettingzoo_parallel_api === undefined
    ? "api ?"
    : `api ${summary.pettingzoo_parallel_api ? "yes" : "no"}`;
  return `${summary.status || "unknown"} · ${checks} · ${api}`;
}

function spectatorStatusLine(summary) {
  if (!summary?.exists) return "missing";
  const checks = `${summary.checks_passed ?? 0}/${summary.checks ?? 0}`;
  const audio = Array.isArray(summary.audio_buckets) ? summary.audio_buckets.join("/") : "?";
  return `${summary.status || "unknown"} · ${checks} · ${audio}`;
}

function operationalStatusLine(summary) {
  if (!summary?.exists) return "missing";
  const checks = `${summary.checks_passed ?? 0}/${summary.checks ?? 0}`;
  const failed = Array.isArray(summary.failed_checks) && summary.failed_checks.length
    ? summary.failed_checks[0]
    : "all";
  return `${summary.status || "unknown"} · ${checks} · ${failed}`;
}

function evidenceStatusLine(bundle, verification) {
  if (!bundle?.exists) return "bundle missing";
  if (!bundle.passed) return `bundle ${bundle.status || "failed"}`;
  if (!verification?.exists) return "verification missing";
  if (!verification.passed) return `verify ${verification.status || "failed"}`;
  return `passed · ${verification.checks_passed ?? 0}/${verification.checks ?? 0} checks`;
}

function evidenceBundleLine(summary) {
  if (!summary?.exists) return "missing";
  const replays = `${summary.active_replay_reports_nominal ?? 0}/${summary.active_replay_reports ?? 0} reports`;
  const checkpoints = `${summary.checkpoints_existing ?? 0}/${summary.checkpoints_expected ?? 0} ckpt`;
  return `${summary.status || "unknown"} · ${replays} · ${checkpoints}`;
}

function evidenceVerificationLine(summary) {
  if (!summary?.exists) return "missing";
  const checks = `${summary.checks_passed ?? 0}/${summary.checks ?? 0}`;
  const failures = summary.failure_count ? `${summary.failure_count} fail` : "clean";
  return `${summary.status || "unknown"} · ${checks} · ${failures}`;
}

function selfPlayStatusLine(summary) {
  if (!summary?.exists) return "missing";
  const required = `${summary.scenario_required_checks_passed ?? 0}/${summary.scenario_required_checks ?? 0}`;
  const reason = summary.reason || "unknown";
  return `${summary.status || "unknown"} · ${required} · ${reason}`;
}

function diagnosticLine(diagnostic) {
  if (!diagnostic) return "none";
  return `${diagnostic.scenario || "scenario"} · ${diagnostic.severity || "info"} · ${diagnostic.reason || "nominal"}`;
}

function checkpointLine(checkpoints) {
  const total = checkpoints.total ?? 0;
  const existing = checkpoints.existing ?? 0;
  return `${existing}/${total} expected checkpoints`;
}

function checkpointMiniList(items) {
  return items.slice(0, 5)
    .map((item) => `
      <div>
        <span class="dot ${item.exists ? "ok" : "missing"}"></span>
        <code>${escapeHtml(item.id)}</code>
      </div>
    `)
    .join("");
}

function replayLine(replays) {
  return `${replays.acceptance ?? 0} acceptance replays indexed`;
}

function renderCurriculum() {
  const payload = state.curriculum || {};
  const plan = payload.plan;
  if (!payload.exists || !plan) {
    els.curriculumSummary.innerHTML = `
      <div class="status-card">
        <span class="status idle">empty</span>
        <strong>No curriculum plan</strong>
        <code>${escapeHtml(payload.path || "logs/curriculum_plan.json")}</code>
      </div>
    `;
    els.curriculumActions.innerHTML = "";
    return;
  }

  els.curriculumSummary.innerHTML = `
    <div class="status-card">
      <span class="status ${plan.status === "nominal" ? "succeeded" : "running"}">${escapeHtml(plan.status)}</span>
      <strong>${escapeHtml(plan.summary)}</strong>
      <div class="job-meta">
        ${escapeHtml(payload.path)} · diagnostics ${escapeHtml(plan.diagnostic_count ?? 0)} · ${escapeHtml(payload.modified || "")}
      </div>
    </div>
  `;

  els.curriculumActions.innerHTML = "";
  (plan.actions || []).forEach((action, index) => {
    const card = document.createElement("article");
    card.className = "curriculum-card";
    const reasons = (action.diagnostic_reasons || []).map((reason) => `<span class="tag">${escapeHtml(reason)}</span>`).join("");
    const laneHistory = payload.history?.by_lane?.[action.lane] || {};
    const followups = (action.followup_command_ids || [])
      .map((id) => {
        const cmd = commandById(id);
        const label = cmd?.label || id;
        return `
          <button class="mini-btn followup-btn" data-followup="${escapeHtml(id)}" title="Run follow-up command">
            ${escapeHtml(label)}
          </button>
        `;
      })
      .join("");
    card.innerHTML = `
      <div class="curriculum-card-head">
        <div>
          <h3>${escapeHtml(action.lane)}</h3>
          <div class="job-meta">${escapeHtml(action.owner)} · ${escapeHtml(action.stage)} · priority ${escapeHtml(action.priority)}</div>
        </div>
        <button class="run-btn planned-run" title="Run planned action">▶</button>
      </div>
      <p>${escapeHtml(action.objective)}</p>
      <div class="command-desc">${escapeHtml(action.rationale)}</div>
      <div class="repair-row"><span>Command</span><code>${escapeHtml(action.command_id)}</code></div>
      <div class="repair-row"><span>Args</span><code>${escapeHtml(action.default_args)}</code></div>
      <div class="tag-row">${reasons}</div>
      <div class="followup-actions">
        <span>Follow-up</span>
        <div>${followups || '<span class="tag">none</span>'}</div>
      </div>
      ${renderRunbook(laneHistory)}
    `;
    card.querySelector(".planned-run").addEventListener("click", async () => {
      const job = await api("/api/run_curriculum_action", {
        method: "POST",
        body: JSON.stringify({ index }),
      });
      state.activeJobId = job.id;
      renderActiveJob(job);
      await refreshJobs();
    });
    card.querySelectorAll(".followup-btn").forEach((button) => {
      button.addEventListener("click", async () => {
        const job = await api("/api/run_curriculum_followup", {
          method: "POST",
          body: JSON.stringify({ index, command_id: button.dataset.followup }),
        });
        state.activeJobId = job.id;
        renderActiveJob(job);
        await refreshJobs();
      });
    });
    els.curriculumActions.appendChild(card);
  });
}

function commandById(id) {
  return (state.registry.commands || []).find((cmd) => cmd.id === id);
}

function renderRunbook(history) {
  const acceptance = history?.latest_acceptance;
  const repair = history?.latest_repair;
  const followup = history?.latest_followup;
  const trend = acceptance
    ? `${acceptance.diagnostic_trend || "unknown"} · ${acceptance.previous_diagnostic_count ?? "?"} → ${acceptance.diagnostic_count ?? "?"}`
    : "no acceptance history";
  return `
    <div class="runbook">
      <div class="runbook-title">Runbook</div>
      <div class="runbook-grid">
        <span>Trend</span><strong class="trend ${escapeHtml(acceptance?.diagnostic_trend || "unknown")}">${escapeHtml(trend)}</strong>
        <span>Repair</span><code>${escapeHtml(eventSummary(repair))}</code>
        <span>Follow-up</span><code>${escapeHtml(eventSummary(followup))}</code>
      </div>
    </div>
  `;
}

function eventSummary(event) {
  if (!event) return "none";
  const status = event.status || "unknown";
  const command = event.command_id || event.top_command_id || "unknown";
  const when = event.timestamp || "";
  return `${status} · ${command}${when ? ` · ${when}` : ""}`;
}

function renderTabs() {
  const groups = ["All", ...new Set(state.registry.commands.map((c) => c.group))];
  els.tabs.innerHTML = "";
  groups.forEach((group) => {
    const button = document.createElement("button");
    button.className = `tab ${group === state.group ? "active" : ""}`;
    button.textContent = group;
    button.addEventListener("click", () => {
      state.group = group;
      renderTabs();
      renderCommands();
    });
    els.tabs.appendChild(button);
  });
}

function renderCommands() {
  const q = els.search.value.trim().toLowerCase();
  const commands = state.registry.commands.filter((cmd) => {
    const groupOk = state.group === "All" || cmd.group === state.group;
    const text = `${cmd.label} ${cmd.description} ${(cmd.tags || []).join(" ")}`.toLowerCase();
    return groupOk && (!q || text.includes(q));
  });
  els.commandList.innerHTML = "";
  commands.forEach((cmd) => {
    const row = document.createElement("div");
    row.className = `command-row ${state.selected?.id === cmd.id ? "selected" : ""}`;
    row.innerHTML = `
      <button class="icon-btn" title="Run">▶</button>
      <div>
        <div class="command-title">
          <span>${escapeHtml(cmd.label)}</span>
          ${(cmd.tags || []).slice(0, 3).map((tag) => `<span class="tag">${escapeHtml(tag)}</span>`).join("")}
        </div>
        <div class="command-desc">${escapeHtml(cmd.description)}</div>
      </div>
      <code>${escapeHtml(cmd.base.join(" "))}</code>
    `;
    row.addEventListener("click", (event) => {
      state.selected = cmd;
      els.argsLine.value = cmd.default_args || "";
      renderCommands();
      updateSelectedCommandText();
      if (event.target.tagName === "BUTTON") runCommand(cmd.id, els.argsLine.value);
    });
    els.commandList.appendChild(row);
  });
}

function updateSelectedCommandText() {
  if (!state.selected) {
    els.selectedCommand.textContent = "";
    return;
  }
  els.selectedCommand.textContent = `${state.selected.base.join(" ")} ${els.argsLine.value}`.trim();
}

async function runCommand(id, args) {
  if (!id) return;
  const job = await api("/api/run", {
    method: "POST",
    body: JSON.stringify({ id, args }),
  });
  state.activeJobId = job.id;
  await refreshJobs();
  renderActiveJob(job);
}

async function refreshJobs() {
  state.jobs = await api("/api/jobs");
  renderJobs();
  await refreshMetrics();
  await refreshCurriculum();
  await refreshReadiness();
  updateCounts();
}

async function refreshActiveJob() {
  if (!state.activeJobId) return;
  try {
    const job = await api(`/api/jobs/${state.activeJobId}`);
    renderActiveJob(job);
    if (job.status !== "running") await refreshJobs();
  } catch {
    state.activeJobId = null;
  }
}

function renderJobs() {
  els.jobList.innerHTML = "";
  state.jobs.slice(0, 30).forEach((job) => {
    const row = document.createElement("div");
    row.className = "job-row";
    row.innerHTML = `
      <button class="icon-btn" title="Inspect">▸</button>
      <div>
        <div class="command-title">
          <span>${escapeHtml(job.label)}</span>
          <span class="status ${job.status}">${escapeHtml(job.status)}</span>
        </div>
        <div class="job-meta">${escapeHtml(job.argv.join(" "))}</div>
      </div>
      <span>${job.returncode ?? ""}</span>
    `;
    row.addEventListener("click", async () => {
      state.activeJobId = job.id;
      renderActiveJob(await api(`/api/jobs/${job.id}`));
    });
    els.jobList.appendChild(row);
  });
}

function renderActiveJob(job) {
  els.activeJobTitle.textContent = job.label || "Terminal";
  els.activeJobStatus.textContent = job.status || "idle";
  els.activeJobStatus.className = `status ${job.status || "idle"}`;
  els.terminal.textContent = job.log || "";
  els.terminal.scrollTop = els.terminal.scrollHeight;
}

async function stopActiveJob() {
  if (!state.activeJobId) return;
  await api(`/api/jobs/${state.activeJobId}/stop`, { method: "POST", body: "{}" });
  await refreshActiveJob();
}

async function refreshReplays() {
  state.replays = await api("/api/replays");
  renderReplays();
  await refreshMetrics();
  updateCounts();
}

async function refreshMetrics() {
  state.metrics = await api("/api/metrics");
  renderMetrics();
}

function renderMetrics() {
  const m = state.metrics || {};
  const cards = [
    ["Commands", m.commands ?? 0],
    ["Replays", m.replays ?? 0],
    ["Replay Time", `${Number(m.replay_duration ?? 0).toFixed(1)}s`],
    ["Captures", m.captures ?? 0],
    ["Waypoints", m.waypoints ?? 0],
    ["Confidence", `${Number((m.avg_confidence ?? 0) * 100).toFixed(0)}%`],
    ["Evader R", Number(m.avg_evader_reward ?? 0).toFixed(2)],
    ["Auth", Number(m.avg_pursuer_auth_penalty ?? 0).toFixed(2)],
    ["Decoder", `${Number((m.avg_decoder_accuracy ?? 0) * 100).toFixed(0)}%`],
    ["Entropy", Number(m.avg_radio_word_entropy ?? 0).toFixed(2)],
    ["Jams", m.jam_events ?? 0],
    ["Cipher", m.cipher_rotations ?? 0],
    ["Spoofed", m.spoofed_radio_events ?? 0],
    ["Running", m.jobs_running ?? 0],
  ];
  els.metricCards.innerHTML = cards
    .map(([label, value]) => `<div class="metric-card"><span>${escapeHtml(label)}</span><strong>${escapeHtml(value)}</strong></div>`)
    .join("");
}

function renderReplays() {
  els.replayTable.innerHTML = "";
  state.replays.forEach((replay) => {
    const tr = document.createElement("tr");
    const replayPath = replay.path || replay.name;
    const viewerUrl = replay.viewer_url || `/viewer3d/index.html?replay=${encodeURIComponent(replayPath)}`;
    tr.innerHTML = `
      <td>${escapeHtml(replay.name)}</td>
      <td>${replay.valid ? replay.frames : "bad"}</td>
      <td>${replay.valid ? `${Number(replay.duration).toFixed(2)}s` : ""}</td>
      <td>${replay.agents ?? ""}</td>
      <td>${replay.radio_events ?? ""}${replay.spoofed_radio_events ? ` / ${replay.spoofed_radio_events} spoof` : ""}</td>
      <td>${replay.valid ? replayMetricLine(replay) : escapeHtml(replay.error || "")}</td>
      <td>${escapeHtml(replay.modified)}</td>
      <td>
        <a class="mini-btn replay-link" href="${escapeHtml(viewerUrl)}" target="_blank" rel="noreferrer" title="Open cinematic web replay">3D</a>
        <button class="mini-btn" data-action="validate" data-replay="${escapeHtml(replayPath)}">✓</button>
        <button class="mini-btn" data-action="report" data-replay="${escapeHtml(replayPath)}">R</button>
        <button class="mini-btn" data-action="play" data-replay="${escapeHtml(replayPath)}">▶</button>
      </td>
    `;
    tr.querySelectorAll(".mini-btn").forEach((button) => {
      button.addEventListener("click", async (event) => {
        event.stopPropagation();
        const job = await api("/api/run_replay_action", {
          method: "POST",
          body: JSON.stringify({ action: button.dataset.action, replay: button.dataset.replay }),
        });
        state.activeJobId = job.id;
        renderActiveJob(job);
        await refreshJobs();
      });
    });
    els.replayTable.appendChild(tr);
  });
}

function replayMetricLine(replay) {
  const chase = `cap ${replay.captures} · wp ${replay.waypoints_hit} · conf ${(replay.avg_confidence * 100).toFixed(0)}%`;
  const reward = `R ${Number(replay.avg_evader_reward ?? 0).toFixed(2)} · auth ${Number(replay.avg_pursuer_auth_penalty ?? 0).toFixed(2)}`;
  const comms = `H ${Number(replay.radio_word_entropy ?? 0).toFixed(2)} · jam ${replay.jam_events ?? 0} · cipher ${replay.cipher_rotations ?? 0}`;
  return `${chase} · ${reward} · ${comms}`;
}

async function refreshConfigs() {
  state.configs = await api("/api/configs");
  els.configPicker.innerHTML = "";
  state.configs.forEach((cfg) => {
    const option = document.createElement("option");
    option.value = cfg.name;
    option.textContent = cfg.name;
    els.configPicker.appendChild(option);
  });
  if (state.configs.length) await loadSelectedConfig();
}

async function loadSelectedConfig() {
  const name = els.configPicker.value;
  if (!name) return;
  const cfg = await api(`/api/configs/${encodeURIComponent(name)}`);
  els.configEditor.value = cfg.text;
}

async function saveSelectedConfig() {
  const name = els.configPicker.value;
  if (!name) return;
  await api(`/api/configs/${encodeURIComponent(name)}`, {
    method: "POST",
    body: JSON.stringify({ text: els.configEditor.value }),
  });
  await refreshConfigs();
}

function updateCounts() {
  els.commandCount.textContent = `${state.registry.commands.length} commands`;
  els.replayCount.textContent = `${state.replays.length} replays`;
  els.jobCount.textContent = `${state.jobs.length} jobs`;
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

init().catch((err) => {
  els.terminal.textContent = `${err.stack || err}`;
});
