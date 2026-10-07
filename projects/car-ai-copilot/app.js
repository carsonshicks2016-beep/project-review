(function () {
  const state = {
    mode: "copilot",
    query: "What changed recently and what should I check first?",
    selectedPart: "brakes",
    selectedSound: "engineKnock",
    simulator: {
      running: true,
      training: true,
      goal: "lane",
      episode: 7,
      reward: 184,
      policy: 0.36,
      safety: 78,
      car: { x: 130, y: 300, angle: -0.03, speed: 2.25 },
      trace: [],
      obstacles: [
        { x: 520, y: 265, r: 18 },
        { x: 720, y: 332, r: 20 },
        { x: 900, y: 252, r: 17 },
      ],
    },
  };

  const sources = [
    { id: "manual", label: "Owner Manual", count: 42, mark: "M", detail: "Fluids, lights, intervals" },
    { id: "notes", label: "Repair Notes", count: 18, mark: "R", detail: "Shop work and DIY logs" },
    { id: "obd", label: "OBD-II Logs", count: 317, mark: "O", detail: "Codes and sensor snapshots" },
    { id: "drive", label: "GPS / Trips", count: 29, mark: "G", detail: "Routes and coach events" },
    { id: "audio", label: "Audio Captures", count: 8, mark: "A", detail: "Engine and brake samples" },
  ];

  const vehicleParts = [
    {
      id: "brakes",
      label: "Brake System",
      short: "B",
      risk: 82,
      trend: "+14",
      x: 28,
      y: 68,
      evidence: [
        "Brake squeal sample shows a 4.1 kHz band that matches pad wear signatures.",
        "Front-right pad note from 2,100 miles ago said 4 mm remaining.",
        "Two hard braking events appeared on the last mountain route.",
      ],
      action: "Inspect front pads and rotor surface before the next long drive.",
      signal: "High-frequency squeal plus pad age",
    },
    {
      id: "battery",
      label: "Battery / Charging",
      short: "V",
      risk: 68,
      trend: "+9",
      x: 20,
      y: 37,
      evidence: [
        "Cold start voltage dipped to 11.8 V on the last two starts.",
        "Battery install date is 44 months old.",
        "Alternator output stayed stable at 14.1 V once running.",
      ],
      action: "Run a load test soon; charging system currently looks stable.",
      signal: "Weak crank voltage",
    },
    {
      id: "engine",
      label: "Engine Temp / Misfire",
      short: "E",
      risk: 61,
      trend: "+6",
      x: 36,
      y: 47,
      evidence: [
        "OBD snapshot logged a pending P0302 cylinder 2 misfire.",
        "Coolant peaked at 219 F during traffic, above your normal 205 F pattern.",
        "Fuel trim on bank 1 drifted positive during idle.",
      ],
      action: "Check plug, coil, and coolant level; watch for repeat P0302 after warm starts.",
      signal: "Pending misfire and warmer idle",
    },
    {
      id: "fluids",
      label: "Fluids",
      short: "F",
      risk: 52,
      trend: "+3",
      x: 45,
      y: 61,
      evidence: [
        "Oil interval is at 5,400 miles since the last change.",
        "Coolant service note is older than the manual interval.",
        "No leak flags from parking-spot image notes.",
      ],
      action: "Plan oil service now and confirm coolant condition while the engine is cold.",
      signal: "Service interval pressure",
    },
    {
      id: "tires",
      label: "Tires / Alignment",
      short: "T",
      risk: 44,
      trend: "-4",
      x: 70,
      y: 72,
      evidence: [
        "Rear tires were rotated 900 miles ago.",
        "Left-front pressure dropped 3 psi over five days.",
        "No strong alignment pull in steering trace.",
      ],
      action: "Top off pressure and recheck left-front tire after 48 hours.",
      signal: "Slow pressure loss",
    },
    {
      id: "exhaust",
      label: "O2 / Exhaust",
      short: "X",
      risk: 37,
      trend: "-2",
      x: 72,
      y: 43,
      evidence: [
        "Downstream O2 signal is slow but inside expected range.",
        "No catalyst efficiency code present.",
        "Fuel economy improved 3 percent after the last air filter replacement.",
      ],
      action: "No immediate action; keep watching fuel trim after the misfire check.",
      signal: "Mild sensor aging",
    },
  ];

  const documents = [
    {
      id: "doc-001",
      type: "manual",
      title: "Owner Manual: Brake Pad Wear Indicator",
      date: "2026-05-21",
      tags: ["brakes", "manual", "squeal"],
      content:
        "The manual says a continuous high-pitched brake squeal can indicate the pad wear indicator is contacting the rotor. Inspect pad thickness and rotor condition before extended driving.",
    },
    {
      id: "doc-002",
      type: "repair",
      title: "Repair Note: Front Brake Inspection",
      date: "2026-04-18",
      tags: ["brakes", "front right", "pads"],
      content:
        "Front-right pad measured 4 mm remaining. Rotor face had light scoring. Shop recommended recheck within 2,000 to 3,000 miles, especially before mountain trips.",
    },
    {
      id: "doc-003",
      type: "obd",
      title: "OBD Snapshot: Pending P0302",
      date: "2026-06-12",
      tags: ["engine", "misfire", "temperature", "obd"],
      content:
        "Pending P0302 cylinder 2 misfire appeared after a warm restart. Coolant temperature peaked at 219 F in traffic, and short-term fuel trim rose positive during idle.",
    },
    {
      id: "doc-004",
      type: "sensor",
      title: "Battery Start Trend",
      date: "2026-06-13",
      tags: ["battery", "voltage", "charging"],
      content:
        "Cold crank voltage dipped to 11.8 V twice this week. Alternator output remained steady at 14.1 V after startup, suggesting the charging system is not the first suspect.",
    },
    {
      id: "doc-005",
      type: "audio",
      title: "Brake Squeal Audio Capture",
      date: "2026-06-13",
      tags: ["audio", "brakes", "spectrogram"],
      content:
        "Audio classifier found a strong narrow band around 4.1 kHz during low-speed braking. Confidence is highest for pad wear indicator or glazed pad surface.",
    },
    {
      id: "doc-006",
      type: "trip",
      title: "Driving Coach: Canyon Route",
      date: "2026-06-11",
      tags: ["driving", "braking", "cornering", "fuel"],
      content:
        "The canyon route had two hard braking events, three late-apex turns, and a 7 percent fuel efficiency drop versus your usual commute. Smooth throttle score improved on the return leg.",
    },
    {
      id: "doc-007",
      type: "maintenance",
      title: "Oil and Coolant Service",
      date: "2026-06-03",
      tags: ["fluids", "oil", "coolant", "maintenance"],
      content:
        "Oil has 5,400 miles since last service. Coolant age is past the conservative service window. No active leak note, but temperature trend makes coolant condition worth checking.",
    },
    {
      id: "doc-008",
      type: "sensor",
      title: "Tire Pressure Drift",
      date: "2026-06-10",
      tags: ["tires", "pressure", "alignment"],
      content:
        "Left-front tire pressure dropped 3 psi across five days. Steering trace does not show a strong alignment pull, so pressure or a slow leak is more likely than alignment.",
    },
  ];

  const warnings = [
    {
      level: "danger",
      title: "Brake wear likely",
      body: "Audio, prior pad measurement, and recent route stress all point at front brake inspection.",
    },
    {
      level: "watch",
      title: "Pending P0302 misfire",
      body: "Repeat after warm starts would raise engine risk quickly.",
    },
    {
      level: "watch",
      title: "Weak cold crank voltage",
      body: "Load test battery before colder weather or long trips.",
    },
  ];

  const timeline = [
    { date: "Today 8:42 AM", title: "Battery voltage dip", body: "Cold crank fell to 11.8 V, then alternator stabilized." },
    { date: "Today 8:38 AM", title: "Brake audio analyzed", body: "4.1 kHz squeal band increased brake risk score." },
    { date: "Jun 12", title: "Pending P0302 logged", body: "Cylinder 2 misfire after warm restart." },
    { date: "Jun 11", title: "Canyon drive reviewed", body: "Hard braking and late apex events detected." },
    { date: "Jun 10", title: "Tire drift noticed", body: "Left-front tire lost 3 psi over five days." },
  ];

  const coachMetrics = [
    { label: "Smooth Braking", value: 72, suffix: "/100", trend: "+5", note: "Better on return leg" },
    { label: "Corner Entry", value: 64, suffix: "/100", trend: "-8", note: "Late apex on tight turns" },
    { label: "Fuel Efficiency", value: 31.4, suffix: " mpg", trend: "-7%", note: "Traffic and elevation drag" },
    { label: "Safety Margin", value: 88, suffix: "/100", trend: "+2", note: "Good following distance" },
  ];

  const coachEvents = [
    "Ease into braking 0.4 seconds earlier on descending turns.",
    "Hold throttle steadier between 35 and 45 mph to reduce fuel spikes.",
    "Tire pressure drift may be affecting left-hand corner feel.",
    "Your best segment used lighter braking and a wider entry line.",
  ];

  const soundSamples = {
    engineKnock: {
      label: "Engine knock",
      subtitle: "Low-mid metallic tap during warm idle",
      seed: 2,
      confidence: [
        ["Loose heat shield", 0.71],
        ["Light knock / pre-ignition", 0.58],
        ["Injector tick", 0.44],
      ],
      note: "The model sees repeating mid-frequency pulses. Cross-check with fuel quality, heat shield movement, and misfire status.",
    },
    brakeSqueal: {
      label: "Brake squeal",
      subtitle: "High narrow band while stopping",
      seed: 7,
      confidence: [
        ["Pad wear indicator", 0.86],
        ["Glazed pad surface", 0.62],
        ["Rotor scoring", 0.48],
      ],
      note: "Strong narrow energy near 4 kHz is why brake risk is elevated on the health map.",
    },
    wheelBearing: {
      label: "Wheel bearing hum",
      subtitle: "Road-speed harmonic under load",
      seed: 12,
      confidence: [
        ["Wheel bearing", 0.74],
        ["Uneven tire wear", 0.52],
        ["Driveline resonance", 0.34],
      ],
      note: "The hum grows with simulated speed. Real capture would compare left and right turns.",
    },
  };

  const modeRoot = document.getElementById("modeRoot");
  const sourceList = document.getElementById("sourceList");
  const riskStack = document.getElementById("riskStack");
  const warningList = document.getElementById("warningList");
  const timelineList = document.getElementById("timelineList");

  let animationHandle = 0;
  let lastFrame = performance.now();

  function escapeHtml(value) {
    return String(value)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#039;");
  }

  function tokenize(text) {
    return text
      .toLowerCase()
      .replace(/[^a-z0-9\s-]/g, " ")
      .split(/\s+/)
      .filter((token) => token.length > 2);
  }

  function riskClass(score) {
    if (score >= 75) return "danger";
    if (score >= 50) return "watch";
    return "good";
  }

  function riskColor(score) {
    if (score >= 75) return "#ff5f56";
    if (score >= 50) return "#ffb23f";
    return "#63d471";
  }

  function retrieve(query) {
    const queryTokens = tokenize(query);
    const recentBoost = query.toLowerCase().includes("changed") || query.toLowerCase().includes("recent");
    const scored = documents.map((doc) => {
      const haystack = tokenize(`${doc.title} ${doc.tags.join(" ")} ${doc.content}`);
      const overlap = queryTokens.reduce((score, token) => score + haystack.filter((word) => word.includes(token)).length, 0);
      const recency = recentBoost ? Math.max(0, 12 - daysAgo(doc.date)) / 12 : 0;
      const warningBoost = doc.tags.some((tag) => ["brakes", "misfire", "battery"].includes(tag)) ? 0.8 : 0;
      return { doc, score: overlap + recency + warningBoost };
    });

    return scored.sort((a, b) => b.score - a.score).slice(0, 4);
  }

  function daysAgo(dateString) {
    const date = new Date(`${dateString}T12:00:00`);
    if (Number.isNaN(date.getTime())) return 8;
    const now = new Date("2026-06-13T12:00:00");
    return Math.max(0, Math.round((now - date) / 86400000));
  }

  function composeAnswer(query) {
    const matches = retrieve(query);
    const topParts = vehicleParts.slice().sort((a, b) => b.risk - a.risk).slice(0, 3);
    const q = query.toLowerCase();
    let lead =
      "The current local memory points first at brakes, then battery health, then a watch-level engine misfire pattern.";

    if (q.includes("brake")) {
      lead =
        "Brake risk is high because three signals agree: the prior pad measurement was low, the new audio capture has a wear-indicator band, and the canyon route added hard braking events.";
    } else if (q.includes("engine") || q.includes("misfire")) {
      lead =
        "The engine is not in emergency territory, but the pending P0302 matters because it appeared with warmer idle temperatures and positive fuel trim.";
    } else if (q.includes("changed") || q.includes("recent")) {
      lead =
        "The newest changes are the brake squeal classification today, the 11.8 V cold-crank battery dip today, and yesterday's pending P0302 misfire.";
    } else if (q.includes("maintenance") || q.includes("next")) {
      lead =
        "The practical next plan is brake inspection first, battery load test second, then oil and coolant service while checking for a repeat misfire.";
    }

    return {
      lead,
      matches,
      plan: [
        `Inspect ${topParts[0].label.toLowerCase()} because it is at ${topParts[0].risk}/100 risk.`,
        "Run a battery load test; alternator output looks steady, so the battery itself is the suspect.",
        "If P0302 returns, swap or inspect cylinder 2 plug and coil before chasing broader fuel issues.",
      ],
    };
  }

  function renderShellLists() {
    sourceList.innerHTML = sources
      .map(
        (source) => `
          <article class="source-item">
            <span class="source-mark">${source.mark}</span>
            <div>
              <strong>${source.label}</strong>
              <br />
              <small>${source.count} entries · ${source.detail}</small>
            </div>
          </article>
        `,
      )
      .join("");

    riskStack.innerHTML = vehicleParts
      .slice()
      .sort((a, b) => b.risk - a.risk)
      .slice(0, 4)
      .map(
        (part) => `
          <button class="risk-item" data-select-part="${part.id}">
            <div class="metric-heading">
              <strong>${part.label}</strong>
              <span class="pill ${riskClass(part.risk)}">${part.risk}</span>
            </div>
            <small>${part.signal} · ${part.trend}</small>
            <div class="risk-bar"><span class="risk-fill ${riskClass(part.risk)}" style="width:${part.risk}%"></span></div>
          </button>
        `,
      )
      .join("");

    warningList.innerHTML = warnings
      .map(
        (warning) => `
          <article class="warning-item ${warning.level === "danger" ? "danger" : ""}">
            <strong>${warning.title}</strong>
            <small>${warning.body}</small>
          </article>
        `,
      )
      .join("");

    timelineList.innerHTML = timeline
      .map(
        (event) => `
          <article class="timeline-item">
            <small>${event.date}</small>
            <strong>${event.title}</strong>
            <small>${event.body}</small>
          </article>
        `,
      )
      .join("");
  }

  function renderMode() {
    document.querySelectorAll(".mode-tab").forEach((button) => {
      button.classList.toggle("active", button.dataset.mode === state.mode);
    });

    if (state.mode === "copilot") renderCopilot();
    if (state.mode === "health") renderHealth();
    if (state.mode === "coach") renderCoach();
    if (state.mode === "sound") renderSound();
    if (state.mode === "simulator") renderSimulator();

    renderCanvases();
  }

  function renderCopilot() {
    const answer = composeAnswer(state.query);
    modeRoot.innerHTML = `
      <section class="hero-panel copilot-hero">
        <div class="assistant-console">
          <div>
            <p class="eyebrow">RAG-style local assistant</p>
            <h2>Ask the car what it knows</h2>
          </div>
          <form class="query-row" id="queryForm">
            <input id="queryInput" value="${escapeHtml(state.query)}" aria-label="Ask vehicle memory" />
            <button type="submit">Ask</button>
          </form>
          <div class="answer-card">
            <div class="answer-title">
              <div>
                <small>Answer generated from local demo memory</small>
                <h3>Copilot Readout</h3>
              </div>
              <span class="pill watch">3 linked systems</span>
            </div>
            <p>${escapeHtml(answer.lead)}</p>
            <div class="recommendation-list">
              ${answer.plan.map((item) => `<div class="action-item">${escapeHtml(item)}</div>`).join("")}
            </div>
          </div>
        </div>
        <div class="vehicle-card">
          <div class="vehicle-topline">
            <div>
              <p class="eyebrow">Live Risk Preview</p>
              <h2>Vehicle Health</h2>
            </div>
            <span class="pill watch">64 / 100</span>
          </div>
          ${renderCarModel(false)}
        </div>
      </section>

      <section class="copilot-grid">
        <div class="mode-panel">
          <div class="panel-heading">
            <div>
              <p class="eyebrow">Retrieved Context</p>
              <h2>Top Matches</h2>
            </div>
          </div>
          <div class="evidence-list">
            ${answer.matches
              .map(
                ({ doc }) => `
                  <article class="evidence-item">
                    <div class="metric-heading">
                      <strong>${doc.title}</strong>
                      <span class="pill">${doc.type}</span>
                    </div>
                    <small>${doc.date}</small>
                    <p class="muted">${doc.content}</p>
                    <div class="context-chips">${doc.tags.map((tag) => `<span class="chip">${tag}</span>`).join("")}</div>
                  </article>
                `,
              )
              .join("")}
          </div>
        </div>
        <div class="mode-panel">
          <div class="panel-heading">
            <div>
              <p class="eyebrow">Quick Prompts</p>
              <h2>Useful Questions</h2>
            </div>
          </div>
          <div class="quick-queries">
            ${[
              "What changed recently?",
              "Why is brake risk high?",
              "Summarize engine warnings",
              "What should I do before a road trip?",
              "What data is missing?",
            ]
              .map((query) => `<button data-query="${escapeHtml(query)}">${query}</button>`)
              .join("")}
          </div>
        </div>
      </section>
    `;
  }

  function renderCarModel(clickable) {
    return `
      <div class="car-3d" aria-label="3D car health map">
        <div class="car-shadow"></div>
        <div class="wheel fl"></div>
        <div class="wheel fr"></div>
        <div class="wheel rl"></div>
        <div class="wheel rr"></div>
        <div class="car-body"></div>
        <div class="car-hood"></div>
        <div class="car-cabin"></div>
        <div class="car-trunk"></div>
        ${vehicleParts
          .map((part) => {
            const active = part.id === state.selectedPart ? "active" : "";
            const attrs = clickable ? `button data-part-node="${part.id}"` : `span`;
            const close = clickable ? "button" : "span";
            return `
              <${attrs}
                class="part-node ${riskClass(part.risk)} ${active}"
                style="left:${part.x}%; top:${part.y}%"
                title="${part.label}: ${part.risk}/100"
              ><span>${part.short}</span></${close}>
            `;
          })
          .join("")}
      </div>
    `;
  }

  function renderHealth() {
    const part = vehicleParts.find((item) => item.id === state.selectedPart) || vehicleParts[0];
    const scoreColor = riskColor(part.risk);
    modeRoot.innerHTML = `
      <section class="health-layout">
        <div class="hero-panel vehicle-card">
          <div class="vehicle-topline">
            <div>
              <p class="eyebrow">3D Car Health Map</p>
              <h2>Click a glowing system</h2>
            </div>
            <span class="pill ${riskClass(part.risk)}">${part.label}</span>
          </div>
          ${renderCarModel(true)}
        </div>
        <aside class="mode-panel part-inspector">
          <div class="part-heading">
            <div>
              <p class="eyebrow">Selected System</p>
              <h2>${part.label}</h2>
            </div>
            <span class="pill ${riskClass(part.risk)}">Trend ${part.trend}</span>
          </div>
          <div class="part-score">
            <div class="score-dial" style="--score:${part.risk}; --score-color:${scoreColor}">
              <strong>${part.risk}</strong>
            </div>
            <div>
              <h3>${part.signal}</h3>
              <p class="muted">${part.action}</p>
            </div>
          </div>
          <div class="evidence-list">
            ${part.evidence.map((item) => `<div class="evidence-item">${item}</div>`).join("")}
          </div>
        </aside>
      </section>
      <section class="mode-panel">
        <div class="panel-heading">
          <div>
            <p class="eyebrow">All Systems</p>
            <h2>Risk Board</h2>
          </div>
        </div>
        <div class="part-list">
          ${vehicleParts
            .map(
              (item) => `
                <button class="part-row ${item.id === part.id ? "active" : ""}" data-select-part="${item.id}">
                  <span>
                    <strong>${item.label}</strong><br />
                    <small>${item.signal}</small>
                  </span>
                  <span class="pill ${riskClass(item.risk)}">${item.risk}</span>
                </button>
              `,
            )
            .join("")}
        </div>
      </section>
    `;
  }

  function renderCoach() {
    modeRoot.innerHTML = `
      <section class="coach-grid">
        <div class="hero-panel canvas-panel">
          <canvas id="coachCanvas" width="980" height="520" aria-label="Driving route heatmap"></canvas>
          <div class="canvas-overlay">
            <div>
              <p class="eyebrow">AI Driving Coach</p>
              <h2>Route behavior heatmap</h2>
            </div>
            <span class="pill good">29 trips learned</span>
          </div>
        </div>
        <aside class="mode-panel">
          <div class="panel-heading">
            <div>
              <p class="eyebrow">Trip Score</p>
              <h2>Latest Drive</h2>
            </div>
          </div>
          <div class="metric-grid">
            ${coachMetrics
              .map(
                (metric) => `
                  <article class="metric-card">
                    <small>${metric.label}</small>
                    <strong>${metric.value}${metric.suffix}</strong>
                    <span class="pill ${String(metric.trend).startsWith("-") ? "watch" : "good"}">${metric.trend}</span>
                    <small>${metric.note}</small>
                  </article>
                `,
              )
              .join("")}
          </div>
        </aside>
      </section>
      <section class="split-panels">
        <div class="mode-panel">
          <div class="panel-heading">
            <div>
              <p class="eyebrow">Coach Notes</p>
              <h2>Next Drive Focus</h2>
            </div>
          </div>
          <div class="recommendation-list">
            ${coachEvents.map((event) => `<div class="action-item">${event}</div>`).join("")}
          </div>
        </div>
        <div class="mode-panel">
          <div class="panel-heading">
            <div>
              <p class="eyebrow">Model Inputs</p>
              <h2>What It Learns From</h2>
            </div>
          </div>
          <div class="context-chips">
            <span class="chip">GPS trace</span>
            <span class="chip">Throttle position</span>
            <span class="chip">Brake pressure</span>
            <span class="chip">Steering angle</span>
            <span class="chip">Speed</span>
            <span class="chip">Fuel rate</span>
          </div>
        </div>
      </section>
    `;
  }

  function renderSound() {
    const sample = soundSamples[state.selectedSound];
    modeRoot.innerHTML = `
      <section class="sound-grid">
        <div class="hero-panel canvas-panel">
          <canvas id="soundCanvas" width="980" height="520" aria-label="Audio spectrogram"></canvas>
          <div class="canvas-overlay">
            <div>
              <p class="eyebrow">Car Sound Diagnosis</p>
              <h2>${sample.label}</h2>
              <small class="muted">${sample.subtitle}</small>
            </div>
            <span class="pill watch">Spectrogram</span>
          </div>
        </div>
        <aside class="mode-panel">
          <div class="panel-heading">
            <div>
              <p class="eyebrow">Audio Classifier</p>
              <h2>Probable Causes</h2>
            </div>
          </div>
          <div class="sample-tabs">
            ${Object.entries(soundSamples)
              .map(
                ([id, item]) => `
                  <button class="${id === state.selectedSound ? "active" : ""}" data-sound="${id}">${item.label}</button>
                `,
              )
              .join("")}
          </div>
          <div class="confidence-list">
            ${sample.confidence
              .map(
                ([label, value]) => `
                  <div class="confidence-row">
                    <div class="confidence-label">
                      <span>${label}</span>
                      <strong>${Math.round(value * 100)}%</strong>
                    </div>
                    <div class="confidence-bar"><span style="width:${Math.round(value * 100)}%"></span></div>
                  </div>
                `,
              )
              .join("")}
          </div>
          <p class="muted">${sample.note}</p>
          <label class="sound-upload-label">
            Import recording placeholder
            <input type="file" id="audioFile" accept="audio/*" />
          </label>
        </aside>
      </section>
    `;
  }

  function renderSimulator() {
    const sim = state.simulator;
    modeRoot.innerHTML = `
      <section class="sim-grid">
        <div class="hero-panel canvas-panel">
          <canvas id="simCanvas" width="980" height="560" aria-label="Mini self-driving simulator"></canvas>
          <div class="canvas-overlay">
            <div>
              <p class="eyebrow">Mini Self-Driving Simulator</p>
              <h2>Lane, obstacle, parking, and turn practice</h2>
            </div>
            <span class="pill ${sim.running ? "good" : "watch"}">${sim.running ? "Running" : "Paused"}</span>
          </div>
        </div>
        <aside class="mode-panel">
          <div class="panel-heading">
            <div>
              <p class="eyebrow">Training Controls</p>
              <h2>Policy Lab</h2>
            </div>
          </div>
          <div class="sim-controls">
            <button class="${sim.running ? "active" : ""}" data-sim-action="toggle">${sim.running ? "Pause" : "Run"}</button>
            <button class="${sim.training ? "active" : ""}" data-sim-action="training">${sim.training ? "Training On" : "Training Off"}</button>
            <button data-sim-action="reset">Reset</button>
          </div>
          <select id="simGoal" aria-label="Simulator goal">
            <option value="lane" ${sim.goal === "lane" ? "selected" : ""}>Lane following</option>
            <option value="obstacles" ${sim.goal === "obstacles" ? "selected" : ""}>Obstacle avoidance</option>
            <option value="parking" ${sim.goal === "parking" ? "selected" : ""}>Parking</option>
            <option value="turns" ${sim.goal === "turns" ? "selected" : ""}>Turns</option>
          </select>
          <div class="sim-stats">
            <div class="stat-row"><span>Episode</span><strong>${sim.episode}</strong></div>
            <div class="stat-row"><span>Reward</span><strong>${Math.round(sim.reward)}</strong></div>
            <div class="stat-row"><span>Policy confidence</span><strong>${Math.round(sim.policy * 100)}%</strong></div>
            <div class="stat-row"><span>Safety score</span><strong>${Math.round(sim.safety)}</strong></div>
          </div>
        </aside>
      </section>
    `;
  }

  function renderCanvases() {
    requestAnimationFrame(() => {
      const coachCanvas = document.getElementById("coachCanvas");
      if (coachCanvas) drawCoachCanvas(coachCanvas);

      const soundCanvas = document.getElementById("soundCanvas");
      if (soundCanvas) drawSoundCanvas(soundCanvas, performance.now());

      const simCanvas = document.getElementById("simCanvas");
      if (simCanvas) drawSimulatorCanvas(simCanvas, 16);
    });
  }

  function sizeCanvas(canvas) {
    const rect = canvas.getBoundingClientRect();
    const ratio = window.devicePixelRatio || 1;
    const width = Math.max(320, Math.round(rect.width * ratio));
    const height = Math.max(260, Math.round(rect.height * ratio));
    if (canvas.width !== width || canvas.height !== height) {
      canvas.width = width;
      canvas.height = height;
    }
    const context = canvas.getContext("2d");
    context.setTransform(ratio, 0, 0, ratio, 0, 0);
    return { context, width: rect.width, height: rect.height };
  }

  function drawCoachCanvas(canvas) {
    const { context: ctx, width, height } = sizeCanvas(canvas);
    ctx.clearRect(0, 0, width, height);
    const gradient = ctx.createLinearGradient(0, 0, width, height);
    gradient.addColorStop(0, "#101310");
    gradient.addColorStop(1, "#22251f");
    ctx.fillStyle = gradient;
    ctx.fillRect(0, 0, width, height);

    ctx.save();
    ctx.translate(width * 0.08, height * 0.54);
    ctx.lineCap = "round";
    ctx.lineJoin = "round";

    const points = [];
    for (let i = 0; i <= 140; i += 1) {
      const t = i / 140;
      const x = t * width * 0.82;
      const y = Math.sin(t * Math.PI * 2.2) * height * 0.16 + Math.sin(t * Math.PI * 5.1) * height * 0.05;
      points.push([x, y]);
    }

    ctx.strokeStyle = "rgba(242,245,242,0.14)";
    ctx.lineWidth = 44;
    drawPath(ctx, points);
    ctx.strokeStyle = "rgba(242,245,242,0.22)";
    ctx.setLineDash([16, 18]);
    ctx.lineWidth = 2;
    drawPath(ctx, points);
    ctx.setLineDash([]);

    points.forEach(([x, y], index) => {
      if (index % 7 !== 0) return;
      const intensity = Math.abs(Math.sin(index * 0.22));
      ctx.fillStyle = `rgba(${255}, ${95 + intensity * 80}, ${70}, ${0.08 + intensity * 0.16})`;
      ctx.beginPath();
      ctx.arc(x, y, 16 + intensity * 28, 0, Math.PI * 2);
      ctx.fill();
    });

    const events = [
      { t: 0.24, label: "Hard brake", color: "#ff5f56" },
      { t: 0.46, label: "Late apex", color: "#ffb23f" },
      { t: 0.69, label: "Smooth segment", color: "#63d471" },
      { t: 0.82, label: "Fuel spike", color: "#63d7e8" },
    ];
    events.forEach((event) => {
      const index = Math.floor(event.t * (points.length - 1));
      const [x, y] = points[index];
      ctx.fillStyle = event.color;
      ctx.beginPath();
      ctx.arc(x, y, 7, 0, Math.PI * 2);
      ctx.fill();
      ctx.fillStyle = "#f2f5f2";
      ctx.font = "12px Inter, sans-serif";
      ctx.fillText(event.label, x + 12, y - 12);
    });

    ctx.restore();
  }

  function drawPath(ctx, points) {
    ctx.beginPath();
    points.forEach(([x, y], index) => {
      if (index === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();
  }

  function drawSoundCanvas(canvas, now) {
    const { context: ctx, width, height } = sizeCanvas(canvas);
    const sample = soundSamples[state.selectedSound];
    ctx.clearRect(0, 0, width, height);
    ctx.fillStyle = "#090a09";
    ctx.fillRect(0, 0, width, height);

    const plotX = 34;
    const plotY = 72;
    const plotW = width - 64;
    const plotH = height - 112;
    const columns = 130;
    const rows = 70;
    const cellW = plotW / columns;
    const cellH = plotH / rows;
    const phase = now / 900 + sample.seed;

    for (let x = 0; x < columns; x += 1) {
      for (let y = 0; y < rows; y += 1) {
        const nx = x / columns;
        const ny = y / rows;
        let energy = 0.08 + Math.random() * 0.03;
        if (state.selectedSound === "brakeSqueal") {
          energy += Math.exp(-Math.pow(ny - 0.22, 2) / 0.0018) * (0.65 + Math.sin(nx * 44 + phase) * 0.08);
          energy += Math.exp(-Math.pow(ny - 0.28, 2) / 0.004) * 0.28;
        }
        if (state.selectedSound === "engineKnock") {
          energy += Math.exp(-Math.pow(ny - 0.58, 2) / 0.01) * (0.32 + Math.max(0, Math.sin(nx * 58 + phase)) * 0.38);
          energy += Math.exp(-Math.pow(ny - 0.73, 2) / 0.006) * 0.22;
        }
        if (state.selectedSound === "wheelBearing") {
          energy += Math.exp(-Math.pow(ny - (0.66 + Math.sin(nx * 10 + phase) * 0.04), 2) / 0.006) * 0.52;
          energy += Math.exp(-Math.pow(ny - 0.48, 2) / 0.014) * 0.24;
        }
        energy += Math.sin((nx + ny) * 18 + phase) * 0.03;
        const hue = 180 - Math.min(1, energy) * 125;
        const alpha = Math.min(0.95, 0.18 + energy);
        ctx.fillStyle = `hsla(${hue}, 95%, ${38 + energy * 28}%, ${alpha})`;
        ctx.fillRect(plotX + x * cellW, plotY + y * cellH, cellW + 0.5, cellH + 0.5);
      }
    }

    ctx.strokeStyle = "rgba(242,245,242,0.18)";
    ctx.strokeRect(plotX, plotY, plotW, plotH);
    ctx.fillStyle = "rgba(242,245,242,0.72)";
    ctx.font = "12px Inter, sans-serif";
    ctx.fillText("time", plotX + plotW - 34, plotY + plotH + 24);
    ctx.save();
    ctx.translate(plotX - 18, plotY + 36);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText("frequency", 0, 0);
    ctx.restore();
  }

  function resetSimulator() {
    state.simulator.episode += 1;
    state.simulator.reward = 80;
    state.simulator.safety = 78;
    state.simulator.car = { x: 130, y: 300, angle: -0.03, speed: 2.25 };
    state.simulator.trace = [];
  }

  function updateSimulator(dt) {
    const sim = state.simulator;
    if (!sim.running) return;

    const car = sim.car;
    const targetY = 280 + Math.sin(car.x * 0.014) * 38;
    const laneError = targetY - car.y;
    const learningGain = 0.012 + sim.policy * 0.04;
    const obstacleInfluence = sim.goal === "obstacles" ? nearestObstacleForce(car, sim.obstacles) : 0;
    const parkingPull = sim.goal === "parking" && car.x > 710 ? (410 - car.y) * 0.01 : 0;
    const turnBias = sim.goal === "turns" ? Math.sin(car.x * 0.025) * 0.022 : 0;

    car.angle += laneError * learningGain * dt + obstacleInfluence * dt + parkingPull * dt + turnBias * dt;
    car.angle *= 0.94;
    car.speed = 2.1 + sim.policy * 1.4;
    car.x += Math.cos(car.angle) * car.speed * dt * 4.6;
    car.y += Math.sin(car.angle) * car.speed * dt * 4.6;
    car.y += laneError * 0.006 * dt;

    if (car.x > 950) {
      car.x = 90;
      car.y = 300;
      sim.episode += 1;
      sim.reward += 28 + sim.policy * 40;
    }

    sim.trace.push({ x: car.x, y: car.y });
    if (sim.trace.length > 240) sim.trace.shift();

    const errorPenalty = Math.min(38, Math.abs(laneError) * 0.3);
    const collisionRisk = Math.max(0, 1 - distanceToNearest(car, sim.obstacles) / 80) * 35;
    sim.safety = Math.max(4, Math.min(100, 98 - errorPenalty - collisionRisk + sim.policy * 10));
    if (sim.training) sim.policy = Math.min(0.96, sim.policy + 0.0009 * dt * (1 + sim.safety / 100));
    sim.reward += (sim.safety / 120 - 0.2) * dt;
  }

  function nearestObstacleForce(car, obstacles) {
    let force = 0;
    obstacles.forEach((obstacle) => {
      const dx = obstacle.x - car.x;
      const dy = obstacle.y - car.y;
      const dist = Math.hypot(dx, dy);
      if (dx > 0 && dx < 170 && dist < 145) {
        force += (dy > 0 ? -1 : 1) * (1 - dist / 145) * 0.08;
      }
    });
    return force;
  }

  function distanceToNearest(car, obstacles) {
    return obstacles.reduce((best, obstacle) => Math.min(best, Math.hypot(obstacle.x - car.x, obstacle.y - car.y)), Infinity);
  }

  function drawSimulatorCanvas(canvas, dt) {
    const { context: ctx, width, height } = sizeCanvas(canvas);
    const sim = state.simulator;
    updateSimulator(dt / 16);

    ctx.clearRect(0, 0, width, height);
    ctx.fillStyle = "#0a0c0a";
    ctx.fillRect(0, 0, width, height);

    const roadY = height * 0.52;
    ctx.fillStyle = "#242823";
    roundedRect(ctx, 36, roadY - 118, width - 72, 236, 48);
    ctx.fill();

    ctx.strokeStyle = "rgba(242,245,242,0.2)";
    ctx.setLineDash([18, 18]);
    ctx.lineWidth = 3;
    ctx.beginPath();
    for (let x = 54; x < width - 54; x += 6) {
      const y = roadY + Math.sin(x * 0.014) * 38;
      if (x === 54) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();
    ctx.setLineDash([]);

    if (sim.goal === "parking") {
      ctx.strokeStyle = "rgba(99,215,232,0.7)";
      ctx.lineWidth = 3;
      roundedRect(ctx, width - 220, roadY + 70, 110, 72, 8);
      ctx.stroke();
      ctx.fillStyle = "rgba(99,215,232,0.12)";
      ctx.fill();
    }

    sim.obstacles.forEach((obstacle, index) => {
      const sx = scaleSimX(obstacle.x, width);
      const sy = scaleSimY(obstacle.y, height);
      ctx.fillStyle = index % 2 === 0 ? "#ffb23f" : "#ff5f56";
      ctx.beginPath();
      ctx.arc(sx, sy, obstacle.r, 0, Math.PI * 2);
      ctx.fill();
      ctx.fillStyle = "rgba(0,0,0,0.28)";
      ctx.beginPath();
      ctx.arc(sx - 4, sy - 4, obstacle.r * 0.5, 0, Math.PI * 2);
      ctx.fill();
    });

    ctx.strokeStyle = "rgba(99,215,232,0.34)";
    ctx.lineWidth = 4;
    ctx.beginPath();
    sim.trace.forEach((point, index) => {
      const x = scaleSimX(point.x, width);
      const y = scaleSimY(point.y, height);
      if (index === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();

    drawSimCar(ctx, scaleSimX(sim.car.x, width), scaleSimY(sim.car.y, height), sim.car.angle, sim.safety);

    ctx.fillStyle = "rgba(242,245,242,0.78)";
    ctx.font = "12px Inter, sans-serif";
    ctx.fillText(`Goal: ${sim.goal}`, 22, height - 28);
    ctx.fillText(`Reward: ${Math.round(sim.reward)}  Policy: ${Math.round(sim.policy * 100)}%`, 22, height - 10);
  }

  function scaleSimX(x, width) {
    return (x / 1000) * width;
  }

  function scaleSimY(y, height) {
    return (y / 560) * height;
  }

  function drawSimCar(ctx, x, y, angle, safety) {
    ctx.save();
    ctx.translate(x, y);
    ctx.rotate(angle);
    ctx.fillStyle = safety > 70 ? "#63d471" : safety > 45 ? "#ffb23f" : "#ff5f56";
    roundedRect(ctx, -18, -10, 36, 20, 6);
    ctx.fill();
    ctx.fillStyle = "rgba(8,9,8,0.72)";
    roundedRect(ctx, 0, -7, 12, 14, 4);
    ctx.fill();
    ctx.fillStyle = "#f2f5f2";
    ctx.fillRect(12, -6, 8, 4);
    ctx.fillRect(12, 2, 8, 4);
    ctx.restore();
  }

  function roundedRect(ctx, x, y, width, height, radius) {
    const r = Math.min(radius, width / 2, height / 2);
    ctx.beginPath();
    ctx.moveTo(x + r, y);
    ctx.arcTo(x + width, y, x + width, y + height, r);
    ctx.arcTo(x + width, y + height, x, y + height, r);
    ctx.arcTo(x, y + height, x, y, r);
    ctx.arcTo(x, y, x + width, y, r);
    ctx.closePath();
  }

  function animationLoop(now) {
    const dt = Math.min(34, now - lastFrame);
    lastFrame = now;

    const soundCanvas = document.getElementById("soundCanvas");
    if (soundCanvas) drawSoundCanvas(soundCanvas, now);

    const simCanvas = document.getElementById("simCanvas");
    if (simCanvas) drawSimulatorCanvas(simCanvas, dt);

    animationHandle = requestAnimationFrame(animationLoop);
  }

  function bindEvents() {
    document.querySelectorAll(".mode-tab").forEach((button) => {
      button.addEventListener("click", () => {
        state.mode = button.dataset.mode;
        renderMode();
      });
    });

    document.addEventListener("click", (event) => {
      const queryButton = event.target.closest("[data-query]");
      if (queryButton) {
        state.query = queryButton.dataset.query;
        renderMode();
        return;
      }

      const partButton = event.target.closest("[data-select-part], [data-part-node]");
      if (partButton) {
        state.selectedPart = partButton.dataset.selectPart || partButton.dataset.partNode;
        state.mode = "health";
        renderMode();
        return;
      }

      const soundButton = event.target.closest("[data-sound]");
      if (soundButton) {
        state.selectedSound = soundButton.dataset.sound;
        renderMode();
        return;
      }

      const simButton = event.target.closest("[data-sim-action]");
      if (simButton) {
        const action = simButton.dataset.simAction;
        if (action === "toggle") state.simulator.running = !state.simulator.running;
        if (action === "training") state.simulator.training = !state.simulator.training;
        if (action === "reset") resetSimulator();
        renderMode();
      }
    });

    document.addEventListener("submit", (event) => {
      if (event.target.id === "queryForm") {
        event.preventDefault();
        const input = document.getElementById("queryInput");
        state.query = input.value.trim() || "What should I check first?";
        renderMode();
      }
    });

    document.addEventListener("change", (event) => {
      if (event.target.id === "simGoal") {
        state.simulator.goal = event.target.value;
        resetSimulator();
        renderMode();
      }
      if (event.target.id === "audioFile" && event.target.files && event.target.files[0]) {
        const name = event.target.files[0].name.toLowerCase();
        if (name.includes("brake")) state.selectedSound = "brakeSqueal";
        else if (name.includes("bearing") || name.includes("wheel")) state.selectedSound = "wheelBearing";
        else state.selectedSound = "engineKnock";
        renderMode();
      }
    });

    window.addEventListener("resize", () => renderCanvases());
  }

  function init() {
    renderShellLists();
    renderMode();
    bindEvents();
    animationHandle = requestAnimationFrame(animationLoop);
  }

  init();
})();
