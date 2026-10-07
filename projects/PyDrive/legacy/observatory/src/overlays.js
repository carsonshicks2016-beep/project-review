const byId = (id) => document.getElementById(id);
const clamp = (value, min, max) => Math.min(max, Math.max(min, Number(value) || 0));

function clock(seconds) {
  const value = Math.max(0, Number(seconds) || 0);
  const minutes = Math.floor(value / 60);
  return `${String(minutes).padStart(2, "0")}:${(value % 60).toFixed(3).padStart(6, "0")}`;
}

export function updateIdentity(identity) {
  byId("identity-car").textContent = String(identity.car).toUpperCase();
  byId("identity-edition").textContent = identity.edition;
  byId("identity-stage").textContent = identity.stage;
  byId("identity-checkpoint").textContent = identity.checkpoint;
  const hash = byId("identity-hash");
  hash.textContent = identity.policyHash;
  hash.title = identity.policyHash;
  const classification = byId("identity-classification");
  classification.textContent = identity.classification;
  classification.dataset.classification = String(identity.classification).toLowerCase();
  byId("identity-drivetrain").textContent = identity.drivetrain;
  byId("identity-observation").textContent = identity.observationLayout;
  byId("identity-compatibility").textContent = identity.compatibility;
  const warnings = byId("identity-warnings");
  warnings.replaceChildren(...identity.warnings.map((warning) => {
    const item = document.createElement("span");
    item.textContent = warning;
    return item;
  }));
  warnings.dataset.hidden = "true";
  const warningToggle = byId("warning-toggle");
  byId("warning-count").textContent = String(identity.warnings.length);
  warningToggle.dataset.hidden = String(identity.warnings.length === 0);
  warningToggle.setAttribute("aria-expanded", "false");
  const inspector = byId("identity-inspector");
  const entries = [
    ["Checkpoint", identity.checkpoint],
    ["Classification", identity.classification],
    ["Drivetrain", identity.drivetrain],
    ["Observation layout", identity.observationLayout],
    ["Compatibility", identity.compatibility],
    ["Exact SHA-256", identity.policyHash],
  ];
  inspector.replaceChildren(...entries.map(([label, value]) => {
    const row = document.createElement("div");
    const heading = document.createElement("small");
    heading.textContent = label;
    const detail = document.createElement("code");
    detail.textContent = value || "—";
    detail.title = detail.textContent;
    row.append(heading, detail);
    return row;
  }));
}

export function updateFrame(frame) {
  const telemetry = frame.telemetry;
  byId("telemetry-speed").textContent = String(Math.round(telemetry.speedMps * 3.6)).padStart(3, "0");
  byId("telemetry-gear").textContent = telemetry.gear <= 0 ? "N" : String(Math.round(telemetry.gear));
  byId("telemetry-rpm").textContent = `${Math.round(telemetry.rpm).toString().padStart(4, "0")} rpm`;
  byId("telemetry-throttle").value = clamp(telemetry.throttle, 0, 1);
  byId("telemetry-brake").value = clamp(telemetry.brake, 0, 1);
  byId("telemetry-steer").value = clamp(telemetry.steer, -1, 1);
  byId("telemetry-pace").value = clamp(telemetry.paceRatio, 0, 1.3);
  byId("telemetry-stage").textContent = frame.identity.stage;
  byId("telemetry-lap").textContent = telemetry.lap ? String(telemetry.lap) : "—";
  byId("telemetry-progress").textContent = `${(telemetry.progress * 100).toFixed(2)}%`;
  byId("telemetry-delta").textContent = telemetry.delta == null ? "—" : `${telemetry.delta >= 0 ? "+" : ""}${telemetry.delta.toFixed(3)} s`;
  byId("telemetry-surface").textContent = telemetry.offTrack ? "OFF TRACK" : "TRACK";
  byId("telemetry-mgu").textContent = telemetry.mguPower == null ? "—" : `${Math.round(telemetry.mguPower / 1000)} kW`;
  byId("telemetry-soc").textContent = telemetry.hybridSoc == null ? "—" : `${(telemetry.hybridSoc * 100).toFixed(1)}%`;
  byId("telemetry-fable").textContent = telemetry.fableValid && telemetry.footprintValid ? "VALID" : "INVALID";
  byId("telemetry-boost").textContent = `${(telemetry.boost * 100).toFixed(0)}%`;
  byId("telemetry-aero").textContent = telemetry.activeAero ? `LOW DRAG ${(telemetry.activeAeroLowDrag * 100).toFixed(0)}%` : "BASELINE";
  const visualLength = Number(frame.vehicle.visualLength) || 0;
  byId("telemetry-collision").textContent = frame.vehicle.collisionLength
    ? `${frame.vehicle.collisionLength.toFixed(3)}m / visual ${visualLength.toFixed(3)}m`
    : "—";
  drawWheels(frame.vehicle.wheels);

  if (byId("brain-overlay").dataset.open === "true") {
    byId("brain-steer").textContent = signed(frame.brain.actions[0]);
    byId("brain-long").textContent = signed(frame.brain.actions[1]);
    byId("brain-gear").textContent = signed(frame.brain.actions[2]);
    byId("brain-critic").textContent = telemetry.criticValue == null ? "—" : signed(telemetry.criticValue);
    byId("brain-std").textContent = `Policy std ${frame.brain.policyStd.map((value) => value.toFixed(3)).join(" · ")}`;
    const age = frame.brain.predictionAgeSeconds == null
      ? "age unavailable"
      : `${frame.brain.predictionAgeSeconds.toFixed(2)}s old`;
    const prediction = `Projection ${frame.brain.predictionStatus || "unavailable"} · ${age}`;
    const note = frame.brain.note || (frame.brain.observations.length
      ? `${frame.brain.observations.length} normalized observations`
      : "Policy telemetry not present in this frame.");
    byId("brain-note").textContent = `${note} · ${prediction}`;
    drawObservationStrip(frame.brain.observations);
    drawActivityStrip(byId("brain-layer-one"), frame.brain.hiddenLayers[0] || [], 128);
    drawActivityStrip(byId("brain-layer-two"), frame.brain.hiddenLayers[1] || [], 128);
    drawObservationGroups(frame.brain.observationGroups);
    drawSensitivity(frame.brain.sensitivity);
  }

  byId("replay-clock").textContent = clock(frame.simTime || telemetry.lapTime);
  byId("replay-mode").textContent = frame.replay.live ? "LIVE" : "REPLAY";
  // The pause button is updated from the authoritative playback control
  // state in main.js. Render frames can intentionally lag for interpolation,
  // so they must not overwrite a newer pause/resume acknowledgement here.
}

function drawWheels(wheels) {
  const grid = byId("wheel-grid");
  grid.replaceChildren(...wheels.map((wheel) => {
    const node = document.createElement("div");
    node.textContent = `${wheel.id}${wheel.contact ? "" : " AIR"}\n${Math.round(wheel.loadN)}N\nμ ${wheel.grip.toFixed(2)}\nSR ${wheel.slipRatio.toFixed(2)} · SA ${(wheel.slipAngle * 57.2958).toFixed(1)}°`;
    return node;
  }));
}

function drawObservationGroups(groups) {
  const container = byId("brain-groups");
  container.replaceChildren(...groups.map((group) => {
    const node = document.createElement("span");
    const preview = group.samples.slice(0, 2).map((sample) => `${sample.label} ${sample.value.toFixed(2)}`).join(" · ");
    node.textContent = `${group.name}${preview ? ` · ${preview}` : ""}`;
    node.title = group.samples.map((sample) => `${sample.label}: ${sample.value.toFixed(4)}`).join("\n");
    return node;
  }));
}

function drawSensitivity(sensitivity) {
  const node = byId("brain-sensitivity");
  if (!sensitivity?.top) {
    node.textContent = "Sensitivity / not intent — unavailable";
    return;
  }
  const lines = Object.entries(sensitivity.top).map(([action, entries]) => {
    const top = Array.from(entries || []).slice(0, 3).map((entry) => `${entry.label} ${Number(entry.derivative).toFixed(2)}`).join(", ");
    return `${action}: ${top || "—"}`;
  });
  node.textContent = `Sensitivity / not intent · ${lines.join(" · ")}`;
}

function signed(value) {
  const number = Number(value) || 0;
  return `${number >= 0 ? "+" : ""}${number.toFixed(3)}`;
}

function drawObservationStrip(values) {
  drawActivityStrip(byId("brain-strip"), values, 64);
}

function drawActivityStrip(strip, values, limit) {
  const desired = Math.min(limit, values.length);
  while (strip.children.length < desired) strip.append(document.createElement("i"));
  while (strip.children.length > desired) strip.lastElementChild.remove();
  Array.from(strip.children).forEach((node, index) => {
    const value = clamp(values[index], -1, 1);
    node.style.setProperty("--value", Math.abs(value));
    node.dataset.sign = value >= 0 ? "positive" : "negative";
    node.title = `Observation ${index}: ${value.toFixed(4)}`;
  });
}

export function bindPanel(toggleId, panelId, defaultOpen = false) {
  const button = byId(toggleId);
  const panel = byId(panelId);
  let open = defaultOpen;
  const render = () => {
    panel.dataset.open = String(open);
    button.classList.toggle("active", open);
    button.setAttribute("aria-pressed", String(open));
  };
  const set = (value) => { open = Boolean(value); render(); };
  const toggle = () => { open = !open; render(); };
  button.addEventListener("click", toggle);
  document.querySelectorAll(`[data-toggle-panel="${panelId}"]`).forEach((control) => control.addEventListener("click", toggle));
  render();
  return { set, toggle, get open() { return open; } };
}

export function setConnection({ state, label }) {
  const node = byId("connection-state");
  node.dataset.state = state;
  node.textContent = label;
}

let toastTimer;
export function toast(message) {
  const node = byId("toast");
  node.textContent = message;
  node.dataset.open = "true";
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { node.dataset.open = "false"; }, 4200);
}

const replayEvents = [];
export function recordReplayEvent(payload) {
  const entries = Array.isArray(payload?.events) ? payload.events : [payload];
  entries.filter(Boolean).forEach((event) => {
    const kind = event.event || event.name || event.type || "event";
    const checkpoint = event.checkpoint?.filename || event.checkpoint?.checkpoint_id || event.checkpoint_id || "";
    const warning = event.warning || event.message || "";
    replayEvents.unshift(`${kind}${checkpoint ? ` · ${checkpoint}` : ""}${warning ? ` · ${warning}` : ""}`);
  });
  replayEvents.splice(4);
  const container = byId("replay-events");
  container.replaceChildren(...replayEvents.map((line) => {
    const span = document.createElement("span");
    span.textContent = line;
    span.title = line;
    return span;
  }));
}
