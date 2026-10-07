// DOM chrome: identity plate, meters, panels, toasts.
// IDs are the acceptance-test contract; layout/CSS is new.

const $ = (id) => document.getElementById(id);

let toastTimer = 0;

export function toast(message, ms = 3200) {
  const el = $("toast");
  if (!el) return;
  el.textContent = message;
  el.dataset.open = "true";
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => {
    el.dataset.open = "false";
  }, ms);
}

export function setConnection(state) {
  const el = $("connection-state");
  if (!el) return;
  el.dataset.state = state;
  el.textContent = String(state || "connecting").toUpperCase();
}

export function setLoading(hidden, title, detail, progress) {
  const card = $("loading-card");
  if (!card) return;
  card.dataset.hidden = hidden ? "true" : "false";
  if (title) $("loading-title").textContent = title;
  if (detail) $("loading-detail").textContent = detail;
  if (Number.isFinite(progress)) {
    $("loading-progress").style.width = `${Math.round(Math.max(0, Math.min(1, progress)) * 100)}%`;
  }
}

export function bindChrome({
  onCamera, onWeather, onPause, onStep, onReset, onSpeed, onScrub, onAudio, onPanel,
}) {
  $("plate-toggle")?.addEventListener("click", () => {
    const plate = $("identity-plate");
    const open = plate.dataset.expanded === "true";
    plate.dataset.expanded = open ? "false" : "true";
    $("plate-toggle").setAttribute("aria-expanded", open ? "false" : "true");
  });

  $("camera-mode")?.addEventListener("change", (e) => onCamera?.(e.target.value));
  $("weather-mode")?.addEventListener("change", (e) => onWeather?.(e.target.value));
  $("control-pause")?.addEventListener("click", () => onPause?.());
  $("control-step")?.addEventListener("click", () => onStep?.());
  $("control-reset")?.addEventListener("click", () => onReset?.());
  $("control-speed")?.addEventListener("change", (e) => onSpeed?.(Number(e.target.value)));
  $("control-scrub")?.addEventListener("change", (e) => onScrub?.(Number(e.target.value)));
  $("toggle-audio")?.addEventListener("click", () => onAudio?.());

  const panels = {
    controls: { btn: "toggle-controls", panel: "view-controls" },
    engineer: { btn: "toggle-engineer", panel: "engineer-overlay" },
    brain: { btn: "toggle-xray", panel: "brain-overlay" },
    replay: { btn: "toggle-replay", panel: "replay-overlay" },
  };

  function closeAll() {
    for (const key of Object.keys(panels)) {
      const { btn, panel } = panels[key];
      $(panel).dataset.open = "false";
      $(btn).setAttribute("aria-pressed", "false");
      $(btn).classList.remove("on");
    }
    $("mode-cinematic")?.classList.add("on");
    $("mode-cinematic")?.setAttribute("aria-pressed", "true");
    document.getElementById("app").dataset.panel = "none";
    onPanel?.("none");
  }

  function openPanel(key) {
    for (const name of Object.keys(panels)) {
      const { btn, panel } = panels[name];
      const active = name === key && $(panel).dataset.open !== "true";
      $(panel).dataset.open = active ? "true" : "false";
      $(btn).setAttribute("aria-pressed", active ? "true" : "false");
      $(btn).classList.toggle("on", active);
    }
    const any = Object.keys(panels).some((name) => $(panels[name].panel).dataset.open === "true");
    $("mode-cinematic")?.classList.toggle("on", !any);
    $("mode-cinematic")?.setAttribute("aria-pressed", any ? "false" : "true");
    document.getElementById("app").dataset.panel = any ? key : "none";
    onPanel?.(any ? key : "none");
  }

  $("mode-cinematic")?.addEventListener("click", closeAll);
  for (const key of Object.keys(panels)) {
    $(panels[key].btn)?.addEventListener("click", () => openPanel(key));
  }

  return { closeAll, openPanel };
}

export function updateIdentity(checkpoint) {
  if (!checkpoint) return;
  $("identity-car").textContent = String(checkpoint.car || "—").toUpperCase();
  $("identity-edition").textContent = checkpoint.edition || "—";
  $("identity-stage").textContent = checkpoint.stage || "—";
  $("identity-checkpoint").textContent = checkpoint.filename || checkpoint.checkpoint_id || "NO CHECKPOINT";
  $("identity-hash").textContent = checkpoint.file_sha256 || checkpoint.policy_sha256 || "HASH UNAVAILABLE";
  $("identity-classification").textContent = checkpoint.classification || "UNVERIFIED PLAYBACK";
  $("identity-drivetrain").textContent = checkpoint.drivetrain || "—";
  $("identity-observation").textContent = checkpoint.observation_layout || "—";
  $("identity-compatibility").textContent = checkpoint.compatibility_class || "—";
  const warnings = Array.isArray(checkpoint.warnings) ? checkpoint.warnings.filter(Boolean) : [];
  $("identity-warnings").textContent = warnings.length ? warnings.join(" · ") : "none";
}

function pad(n, w = 2) {
  return String(Math.trunc(Math.abs(n))).padStart(w, "0");
}

export function formatClock(seconds) {
  const s = Math.max(0, Number(seconds) || 0);
  const m = Math.floor(s / 60);
  const rem = s - m * 60;
  const whole = Math.floor(rem);
  const ms = Math.floor((rem - whole) * 1000);
  return `${pad(m)}:${pad(whole)}.${pad(ms, 3)}`;
}

export function updateFrame(frame, { shotLabel = "" } = {}) {
  if (!frame) return;
  $("telemetry-speed").textContent = pad(Math.round(frame.speedKmh), 3);
  $("telemetry-gear").textContent = frame.gear <= 0 ? "N" : String(frame.gear);
  $("telemetry-rpm").textContent = `${Math.round(frame.rpm)} RPM`;
  $("telemetry-progress").textContent = `${(frame.progress * 100).toFixed(1)}%`;
  const pace = frame.fable?.pace_ratio;
  $("telemetry-pace").textContent = Number.isFinite(pace) ? `${(pace * 100).toFixed(0)}%` : "—";
  const lap = frame.fable?.lap_completed_at;
  $("telemetry-lap").textContent = Number.isFinite(lap) ? formatClock(lap) : "—";
  $("camera-shot").textContent = shotLabel || "SIGNAL";

  $("telemetry-throttle").value = frame.throttle;
  $("telemetry-brake").value = frame.brake;
  $("telemetry-steer").value = frame.steer;
  $("telemetry-surface").textContent = frame.offTrack ? "OFF TRACK" : "TRACK";
  $("telemetry-boost").textContent = frame.boost ? frame.boost.toFixed(2) : "—";
  $("telemetry-soc").textContent = frame.hybridSoc ? `${(frame.hybridSoc * 100).toFixed(0)}%` : "—";
  $("telemetry-mgu").textContent = frame.mguFraction ? `${(frame.mguFraction * 100).toFixed(0)}%` : "—";
  $("telemetry-aero").textContent = frame.aeroEngaged ? "LOW DRAG" : "STANDARD";
  $("telemetry-fable").textContent = frame.fable?.valid === false
    ? String(frame.fable.termination_reason || "INVALID")
    : "VALID";

  const rows = $("wheel-rows");
  if (rows) {
    rows.innerHTML = frame.wheels.map((w) => (
      `<tr><td>${w.id}</td><td>${Math.round(w.load)}</td>`
      + `<td>${w.slipRatio.toFixed(2)}</td><td>${w.grip.toFixed(2)}</td></tr>`
    )).join("");
  }

  $("replay-clock").textContent = formatClock(frame.simTime);
  $("replay-episode").textContent = `EP ${frame.episode}`;
  const pauseBtn = $("control-pause");
  if (pauseBtn) pauseBtn.textContent = frame.playback.paused ? "Resume" : "Pause";
  const speed = $("control-speed");
  if (speed && document.activeElement !== speed) {
    const value = String(frame.playback.speed || 1);
    if ([...speed.options].some((o) => o.value === value)) speed.value = value;
  }

  updateBrain(frame);
}

export function updateBrain(frame) {
  const brain = frame?.brain;
  const action = brain?.action_mean || brain?.raw_action || frame?.observations?.action;
  const values = Array.isArray(action) ? action : [frame?.steer ?? 0, (frame?.throttle ?? 0) - (frame?.brake ?? 0), 0];
  $("brain-steer").value = values[0] ?? 0;
  $("brain-long").value = values[1] ?? 0;
  $("brain-gear").value = values[2] ?? 0;
  $("brain-critic").textContent = Number.isFinite(brain?.value) ? brain.value.toFixed(2) : "—";
  const std = brain?.action_std;
  $("brain-std").textContent = Array.isArray(std)
    ? std.map((v) => Number(v).toFixed(2)).join(" · ")
    : "—";

  paintVector($("brain-hidden"), brain?.hidden || brain?.activations || []);
  paintVector($("brain-obs"), frame?.observations?.vector || brain?.observation || []);
}

function paintVector(canvas, values) {
  if (!canvas || !canvas.getContext) return;
  const ctx = canvas.getContext("2d");
  const { width, height } = canvas;
  ctx.fillStyle = "#05080a";
  ctx.fillRect(0, 0, width, height);
  if (!Array.isArray(values) || !values.length) return;
  const n = Math.min(values.length, width);
  const bar = width / n;
  for (let i = 0; i < n; i += 1) {
    const v = Math.max(-1, Math.min(1, Number(values[i]) || 0));
    const h = Math.abs(v) * (height * 0.45);
    ctx.fillStyle = v >= 0 ? "#5ee1ff" : "#ff6a4d";
    ctx.fillRect(i * bar, height / 2 - (v >= 0 ? h : 0), Math.max(1, bar - 0.5), h || 1);
  }
}

export function recordReplayEvent(text, warn = false) {
  const log = $("replay-events");
  if (!log) return;
  const line = document.createElement("span");
  if (warn) line.className = "warn";
  const stamp = new Date().toLocaleTimeString();
  line.textContent = `${stamp}  ${text}`;
  log.prepend(line);
  while (log.childElementCount > 40) log.lastChild.remove();
}

export function setAudioButton(status) {
  const btn = $("toggle-audio");
  if (!btn) return;
  const live = status === "live";
  const buffering = status === "buffering";
  const pressed = live || buffering;
  btn.setAttribute("aria-pressed", pressed ? "true" : "false");
  btn.classList.toggle("degraded", status === "unavailable" || status === "offline" || status === "muted");
  if (status === "live") btn.textContent = "AUDIO LIVE";
  else if (status === "buffering") btn.textContent = "AUDIO BUFFERING";
  else if (status === "muted") btn.textContent = "AUDIO MUTED";
  else if (status === "unavailable" || status === "offline") btn.textContent = "AUDIO OFF";
  else btn.textContent = "AUDIO OFF";
}
