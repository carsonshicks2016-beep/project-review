(function () {
  const state = {
    awake: false,
    scanMode: "delta",
    wakePhrase: localStorage.getItem("lr4WakePhrase") || "hey whitey",
    lastTick: performance.now(),
    t: 0,
    values: {},
    llm: {
      provider: localStorage.getItem("lr4LlmProvider") || "mock",
      endpoint: localStorage.getItem("lr4LlmEndpoint") || "http://127.0.0.1:11434",
      model: localStorage.getItem("lr4LlmModel") || "llama3.2:1b",
      status: "Mock ready",
      statusTone: "quiet",
      responseLine: "",
      responseSubline: "",
      lastResponse: "No LLM brief requested yet.",
      busy: false,
    },
    session: [
      {
        level: "info",
        title: "Prototype booted",
        body: "Mock OBD, Bluetooth, IMU, and voice layers are active.",
        time: "standby",
      },
    ],
  };

  const vehicle = {
    name: "2015 Land Rover LR4 HSE",
    engine: "3.0L supercharged V6",
    drivetrain: "Full-time 4WD, low range capable",
    install: "Hidden Pi unit under dash, OBD adapter at diagnostic port, IMU fixed to chassis-aligned bracket.",
  };

  const hardware = [
    { icon: "Pi", title: "Raspberry Pi brain", detail: "Runs local collector, web UI, voice wake, TTS, and SQLite memory." },
    { icon: "BT", title: "Bluetooth OBD-II adapter", detail: "Streams standard PIDs; enhanced LR data can be added later." },
    { icon: "IMU", title: "6-DoF accelerometer/gyro", detail: "Measures brake dive, acceleration, cornering, pitch, and road harshness." },
    { icon: "Mic", title: "Cabin microphone + speaker", detail: "Silent by default; listens for the configurable wake phrase locally." },
    { icon: "DB", title: "Local memory", detail: "Stores drive sessions, deltas, alerts, and service notes without cloud dependency." },
  ];

  const obdSignals = [
    { id: "rpm", label: "Engine RPM", unit: "rpm", baseline: 725, warning: 1050, danger: 1400, kind: "generic" },
    { id: "coolant", label: "Coolant Temp", unit: "F", baseline: 204, warning: 218, danger: 230, kind: "generic" },
    { id: "iat", label: "Intake Air Temp", unit: "F", baseline: 96, warning: 128, danger: 148, kind: "generic" },
    { id: "voltage", label: "Battery Voltage", unit: "V", baseline: 12.6, warning: 12.1, danger: 11.8, lowerBad: true, kind: "adapter" },
    { id: "stft", label: "Short Fuel Trim", unit: "%", baseline: 1.8, warning: 8, danger: 12, abs: true, kind: "generic" },
    { id: "ltft", label: "Long Fuel Trim", unit: "%", baseline: 2.4, warning: 8, danger: 12, abs: true, kind: "generic" },
    { id: "maf", label: "MAF Airflow", unit: "g/s", baseline: 4.9, warning: 8.5, danger: 11, kind: "generic" },
    { id: "throttle", label: "Throttle Angle", unit: "%", baseline: 14, warning: 38, danger: 54, kind: "generic" },
  ];

  const motionSignals = [
    { id: "accel", label: "Launch Accel", unit: "g", baseline: 0.08, warning: 0.32, danger: 0.48 },
    { id: "brake", label: "Brake Decel", unit: "g", baseline: 0.12, warning: 0.38, danger: 0.56 },
    { id: "corner", label: "Cornering Load", unit: "g", baseline: 0.14, warning: 0.42, danger: 0.6 },
    { id: "pitch", label: "Pitch Delta", unit: "deg", baseline: 1.2, warning: 4.5, danger: 6.5 },
    { id: "roll", label: "Roll Delta", unit: "deg", baseline: 1.6, warning: 5.2, danger: 7.5 },
    { id: "roughness", label: "Road Harshness", unit: "idx", baseline: 18, warning: 48, danger: 68 },
  ];

  const watchItems = [
    {
      title: "Cooling delta",
      score: 62,
      tag: "watch",
      body: "LR4 baseline is normal, but rising coolant and intake heat after idle should be watched.",
    },
    {
      title: "Battery reserve",
      score: 58,
      tag: "watch",
      body: "Voltage sag during start is a useful early warning for accessory-heavy cabin setups.",
    },
    {
      title: "Air suspension behavior",
      score: 46,
      tag: "good",
      body: "IMU pitch/roll can flag height or damping changes before the dash complains.",
    },
    {
      title: "Fuel trim drift",
      score: 54,
      tag: "watch",
      body: "Bank trim deltas can hint at intake leaks, dirty MAF data, or aging oxygen sensors.",
    },
  ];

  const wakeButton = document.getElementById("wakeButton");
  const speakButton = document.getElementById("speakButton");
  const sleepButton = document.getElementById("sleepButton");
  const wakePhraseInput = document.getElementById("wakePhraseInput");
  const wakeStateDot = document.getElementById("wakeStateDot");
  const wakeStateLabel = document.getElementById("wakeStateLabel");
  const wakeStateDetail = document.getElementById("wakeStateDetail");
  const wakeOrb = document.getElementById("wakeOrb");
  const valetLine = document.getElementById("valetLine");
  const valetSubline = document.getElementById("valetSubline");
  const hardwareStack = document.getElementById("hardwareStack");
  const deltaGrid = document.getElementById("deltaGrid");
  const motionGrid = document.getElementById("motionGrid");
  const watchList = document.getElementById("watchList");
  const sessionLog = document.getElementById("sessionLog");
  const motionScore = document.getElementById("motionScore");
  const llmStatus = document.getElementById("llmStatus");
  const llmProvider = document.getElementById("llmProvider");
  const llmEndpoint = document.getElementById("llmEndpoint");
  const llmModel = document.getElementById("llmModel");
  const llmHealthButton = document.getElementById("llmHealthButton");
  const llmBriefButton = document.getElementById("llmBriefButton");
  const llmLastResponse = document.getElementById("llmLastResponse");
  const canvas = document.getElementById("mainCanvas");

  function clamp(value, min, max) {
    return Math.max(min, Math.min(max, value));
  }

  function formatValue(value, unit) {
    if (unit === "rpm") return `${Math.round(value)} ${unit}`;
    if (unit === "V") return `${value.toFixed(2)} ${unit}`;
    if (unit === "%") return `${value.toFixed(1)}${unit}`;
    if (unit === "g") return `${value.toFixed(2)} ${unit}`;
    if (unit === "deg") return `${value.toFixed(1)} ${unit}`;
    return `${Math.round(value)} ${unit}`;
  }

  function riskFor(signal, value) {
    const compareValue = signal.abs ? Math.abs(value) : value;
    if (signal.lowerBad) {
      if (compareValue <= signal.danger) return "danger";
      if (compareValue <= signal.warning) return "watch";
      return "good";
    }
    if (compareValue >= signal.danger) return "danger";
    if (compareValue >= signal.warning) return "watch";
    return "good";
  }

  function riskPercent(signal, value) {
    const compareValue = signal.abs ? Math.abs(value) : value;
    if (signal.lowerBad) {
      return clamp(((signal.baseline - compareValue) / (signal.baseline - signal.danger)) * 100, 6, 100);
    }
    return clamp(((compareValue - signal.baseline) / (signal.danger - signal.baseline)) * 100, 6, 100);
  }

  function tickValues() {
    const t = state.t;
    state.values = {
      rpm: 728 + Math.sin(t * 0.9) * 38 + (state.awake ? 18 : 0),
      coolant: 204 + Math.sin(t * 0.18) * 7 + (state.awake ? 5 : 1),
      iat: 98 + Math.sin(t * 0.31 + 1.2) * 12 + (state.awake ? 8 : 2),
      voltage: 12.55 + Math.sin(t * 0.24 + 2) * 0.18 - (state.awake ? 0.08 : 0),
      stft: 2.2 + Math.sin(t * 0.68) * 3.5 + (state.awake ? 1.4 : 0),
      ltft: 2.6 + Math.sin(t * 0.16 + 1) * 2.4,
      maf: 5 + Math.sin(t * 0.7) * 1.4 + (state.awake ? 0.8 : 0),
      throttle: 14 + Math.max(0, Math.sin(t * 0.5 - 0.8)) * 18,
      accel: Math.abs(Math.sin(t * 0.7)) * 0.22 + (state.awake ? 0.05 : 0),
      brake: Math.abs(Math.sin(t * 0.5 + 1.6)) * 0.34,
      corner: Math.abs(Math.sin(t * 0.44 + 0.3)) * 0.38,
      pitch: 1.4 + Math.sin(t * 0.85) * 2.4,
      roll: 1.8 + Math.sin(t * 0.63 + 1.3) * 2.8,
      roughness: 19 + Math.max(0, Math.sin(t * 1.2 - 0.4)) * 37,
    };
  }

  function valetText() {
    const hotSignals = obdSignals
      .filter((signal) => riskFor(signal, state.values[signal.id]) !== "good")
      .map((signal) => signal.label.toLowerCase());
    const focus = hotSignals.length ? hotSignals.slice(0, 3).join(", ") : "coolant temperature, battery voltage, and fuel trims";

    if (!state.awake) {
      return {
        line: "Good evening, Carson. I am standing by.",
        subline: "The cabin remains quiet until the wake command is detected.",
      };
    }

    if (state.llm.responseLine) {
      return {
        line: state.llm.responseLine,
        subline: state.llm.responseSubline || "Local LLM brief generated from the current LR4 sensor packet.",
      };
    }

    return {
      line: `Hello, Carson. Checking OBD log and delta values for ${focus}.`,
      subline: "Bluetooth OBD and chassis motion streams are being compared against your LR4 baseline.",
    };
  }

  function setAwake(nextAwake, reason) {
    state.awake = nextAwake;
    if (nextAwake) {
      state.llm.responseLine = "";
      state.llm.responseSubline = "";
      addLog("awake", "Wake command accepted", reason || "Valet mode is now listening and scanning.");
      requestValetBrief({ announce: true });
    } else {
      state.llm.responseLine = "";
      state.llm.responseSubline = "";
      addLog("info", "Returned to silent cabin mode", "Voice output is muted until the next wake command.");
    }
    renderStatic();
  }

  function addLog(level, title, body) {
    const time = new Date().toLocaleTimeString([], { hour: "numeric", minute: "2-digit", second: "2-digit" });
    state.session.unshift({ level, title, body, time });
    state.session = state.session.slice(0, 9);
    renderSessionLog();
  }

  function speakCurrentLine() {
    if (!("speechSynthesis" in window)) {
      addLog("warn", "Browser voice unavailable", "Speech synthesis is not exposed in this browser.");
      return;
    }
    const { line } = valetText();
    const utterance = new SpeechSynthesisUtterance(line);
    const voices = window.speechSynthesis.getVoices();
    const britishVoice =
      voices.find((voice) => voice.lang === "en-GB" && /Daniel|Arthur|George|Oliver|UK|British/i.test(voice.name)) ||
      voices.find((voice) => voice.lang === "en-GB") ||
      voices.find((voice) => voice.lang.startsWith("en"));
    if (britishVoice) utterance.voice = britishVoice;
    utterance.lang = britishVoice?.lang || "en-GB";
    utterance.rate = 0.86;
    utterance.pitch = 0.82;
    window.speechSynthesis.cancel();
    window.speechSynthesis.speak(utterance);
  }

  function renderStatic() {
    wakePhraseInput.value = state.wakePhrase;
    const text = valetText();
    valetLine.textContent = text.line;
    valetSubline.textContent = text.subline;
    wakeOrb.classList.toggle("awake", state.awake);
    wakeStateDot.className = `state-dot ${state.awake ? "scan" : "standby"}`;
    wakeStateLabel.textContent = state.awake ? "Valet mode awake" : "Silent cabin mode";
    wakeStateDetail.textContent = state.awake ? "Scanning LR4 deltas" : "Waiting for wake command";

    hardwareStack.innerHTML = hardware
      .map(
        (item) => `
          <article class="hardware-card">
            <span class="hardware-icon">${item.icon}</span>
            <div>
              <strong>${item.title}</strong><br />
              <small>${item.detail}</small>
            </div>
          </article>
        `,
      )
      .join("");

    watchList.innerHTML = watchItems
      .map(
        (item) => `
          <article class="watch-card">
            <div class="watch-topline">
              <strong>${item.title}</strong>
              <span class="pill ${item.tag}">${item.score}</span>
            </div>
            <small>${item.body}</small>
            <div class="meter"><span class="${item.tag}" style="width:${item.score}%"></span></div>
          </article>
        `,
      )
      .join("");

    document.querySelectorAll(".mini-tab").forEach((button) => {
      button.classList.toggle("active", button.dataset.scan === state.scanMode);
    });

    llmProvider.value = state.llm.provider;
    llmEndpoint.value = state.llm.endpoint;
    llmModel.value = state.llm.model;
    llmStatus.textContent = state.llm.busy ? "Thinking" : state.llm.status;
    llmStatus.className = `pill ${state.llm.busy ? "watch" : state.llm.statusTone}`;
    llmLastResponse.textContent = state.llm.lastResponse;

    renderSessionLog();
    renderSignals();
  }

  function renderSessionLog() {
    sessionLog.innerHTML = state.session
      .map(
        (entry) => `
          <article class="log-card ${entry.level}">
            <small>${entry.time}</small>
            <strong>${entry.title}</strong><br />
            <small>${entry.body}</small>
          </article>
        `,
      )
      .join("");
  }

  function currentSensorPacket() {
    return {
      vehicle,
      timestamp: new Date().toISOString(),
      obd: obdSignals.map((signal) => {
        const value = state.values[signal.id];
        return {
          id: signal.id,
          label: signal.label,
          value: Number(value.toFixed(signal.unit === "V" ? 2 : 1)),
          unit: signal.unit,
          baseline: signal.baseline,
          risk: riskFor(signal, value),
          delta: Number((value - signal.baseline).toFixed(2)),
        };
      }),
      motion: motionSignals.map((signal) => {
        const value = state.values[signal.id];
        return {
          id: signal.id,
          label: signal.label,
          value: Number(value.toFixed(signal.unit === "g" ? 2 : 1)),
          unit: signal.unit,
          baseline: signal.baseline,
          risk: riskFor(signal, value),
          delta: Number((value - signal.baseline).toFixed(2)),
        };
      }),
      watchlist: watchItems.map((item) => ({ title: item.title, score: item.score, status: item.tag })),
    };
  }

  function llmSystemPrompt() {
    return [
      "You are the local valet copilot for Carson's 2015 Land Rover LR4 HSE.",
      "You run on a Raspberry Pi in the vehicle and must be concise, calm, and safety-minded.",
      "Use a refined British valet tone without imitating any copyrighted character.",
      "Do not claim a repair is certain. Speak in probabilities and next checks.",
      "Do not recommend clearing codes or sending vehicle commands.",
      "Return one spoken sentence under 32 words, followed by one short dashboard note after 'NOTE:'.",
    ].join(" ");
  }

  function llmUserPrompt() {
    return `Wake phrase accepted. Generate a valet brief from this LR4 packet:\n${JSON.stringify(currentSensorPacket(), null, 2)}`;
  }

  function endpointBase() {
    return state.llm.endpoint.replace(/\/+$/, "");
  }

  function defaultEndpointFor(provider) {
    if (provider === "proxy") return "/api/llm/brief";
    if (provider === "llamacpp") return "http://127.0.0.1:8080";
    if (provider === "ollama") return "http://127.0.0.1:11434";
    return "mock://pi-local";
  }

  function proxyHealthEndpoint() {
    if (!state.llm.endpoint.includes("/llm/brief")) return "/api/health";
    return state.llm.endpoint.replace(/\/llm\/brief$/, "/health");
  }

  function parseValetResponse(text) {
    const clean = text.replace(/\s+/g, " ").trim();
    const [linePart, notePart] = clean.split(/\bNOTE:\s*/i);
    return {
      line: (linePart || clean || fallbackLlmBrief()).replace(/^["']|["']$/g, "").trim(),
      note: notePart || "Local LLM used current OBD and IMU deltas for the brief.",
    };
  }

  function fallbackLlmBrief() {
    const hotObd = obdSignals
      .map((signal) => ({ signal, value: state.values[signal.id], risk: riskFor(signal, state.values[signal.id]) }))
      .filter((item) => item.risk !== "good")
      .slice(0, 3);
    const focus = hotObd.length
      ? hotObd.map((item) => item.signal.label.toLowerCase()).join(", ")
      : "coolant, voltage, and fuel trims";
    return `Hello, Carson. I am reviewing ${focus}; the LR4 is within mock limits, with a few deltas worth watching. NOTE: Mock Pi LLM response.`;
  }

  async function healthCheckLlm() {
    state.llm.busy = true;
    renderStatic();
    try {
      if (state.llm.provider === "mock") {
        state.llm.status = "Mock ready";
        state.llm.statusTone = "quiet";
        addLog("info", "LLM health check", "Mock Pi LLM is ready for local testing.");
        return;
      }

      let url = `${endpointBase()}/health`;
      if (state.llm.provider === "ollama") url = `${endpointBase()}/api/tags`;
      if (state.llm.provider === "proxy") url = proxyHealthEndpoint();
      const response = await fetch(url, { method: "GET" });
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      state.llm.status = state.llm.provider === "proxy" ? "Proxy online" : "LLM online";
      state.llm.statusTone = "good";
      addLog("info", "LLM health check", `${state.llm.provider} endpoint is reachable.`);
    } catch (error) {
      state.llm.status = "Offline fallback";
      state.llm.statusTone = "watch";
      addLog("warn", "LLM health check failed", `${error.message}. The mock fallback will stay active.`);
    } finally {
      state.llm.busy = false;
      renderStatic();
    }
  }

  async function requestValetBrief(options = {}) {
    if (state.llm.busy) return;
    state.llm.busy = true;
    state.llm.status = "Thinking";
    state.llm.statusTone = "watch";
    renderStatic();

    try {
      let text;
      let proxyFallback = false;
      if (state.llm.provider === "mock") {
        await new Promise((resolve) => window.setTimeout(resolve, 280));
        text = fallbackLlmBrief();
      } else if (state.llm.provider === "proxy") {
        const result = await requestProxyBrief();
        text = result.text;
        proxyFallback = result.fallback;
      } else if (state.llm.provider === "ollama") {
        text = await requestOllamaBrief();
      } else {
        text = await requestLlamaCppBrief();
      }

      const parsed = parseValetResponse(text);
      state.llm.responseLine = parsed.line;
      state.llm.responseSubline = parsed.note;
      state.llm.lastResponse = `${parsed.line} ${parsed.note}`;
      state.llm.status = proxyFallback ? "Proxy fallback" : state.llm.provider === "mock" ? "Mock ready" : "LLM online";
      state.llm.statusTone = proxyFallback ? "watch" : state.llm.provider === "mock" ? "quiet" : "good";
      addLog("awake", "LLM valet brief", parsed.note);
      if (options.announce) speakCurrentLine();
    } catch (error) {
      const parsed = parseValetResponse(fallbackLlmBrief());
      state.llm.responseLine = parsed.line;
      state.llm.responseSubline = `${error.message}. Using local fallback.`;
      state.llm.lastResponse = `${parsed.line} ${state.llm.responseSubline}`;
      state.llm.status = "Fallback active";
      state.llm.statusTone = "watch";
      addLog("warn", "LLM brief failed", `${error.message}. Used deterministic fallback.`);
      if (options.announce) speakCurrentLine();
    } finally {
      state.llm.busy = false;
      renderStatic();
    }
  }

  async function requestOllamaBrief() {
    const response = await fetch(`${endpointBase()}/api/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model: state.llm.model,
        stream: false,
        options: { temperature: 0.2, num_predict: 90 },
        messages: [
          { role: "system", content: llmSystemPrompt() },
          { role: "user", content: llmUserPrompt() },
        ],
      }),
    });
    if (!response.ok) throw new Error(`Ollama HTTP ${response.status}`);
    const data = await response.json();
    return data.message?.content || "";
  }

  async function requestProxyBrief() {
    const response = await fetch(state.llm.endpoint, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model: state.llm.model,
        system: llmSystemPrompt(),
        prompt: llmUserPrompt(),
        packet: currentSensorPacket(),
      }),
    });
    if (!response.ok) throw new Error(`Pi proxy HTTP ${response.status}`);
    const data = await response.json();
    return {
      text: data.text || `${data.line || ""} NOTE: ${data.note || ""}`,
      fallback: Boolean(data.fallback),
    };
  }

  async function requestLlamaCppBrief() {
    const response = await fetch(`${endpointBase()}/v1/chat/completions`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model: state.llm.model,
        temperature: 0.2,
        max_tokens: 90,
        messages: [
          { role: "system", content: llmSystemPrompt() },
          { role: "user", content: llmUserPrompt() },
        ],
      }),
    });
    if (!response.ok) throw new Error(`llama.cpp HTTP ${response.status}`);
    const data = await response.json();
    return data.choices?.[0]?.message?.content || "";
  }

  function renderSignals() {
    const warningCount = obdSignals.filter((signal) => riskFor(signal, state.values[signal.id]) !== "good").length;
    document.getElementById("obdStatus").textContent = warningCount ? `${warningCount} deltas watched` : "Stable mock stream";
    document.getElementById("obdStatus").className = `pill ${warningCount >= 3 ? "danger" : warningCount ? "watch" : "good"}`;

    deltaGrid.innerHTML = obdSignals
      .map((signal) => {
        const value = state.values[signal.id];
        const risk = riskFor(signal, value);
        const pct = riskPercent(signal, value);
        const delta = value - signal.baseline;
        const prefix = delta >= 0 ? "+" : "";
        return `
          <article class="delta-card">
            <div class="delta-topline">
              <strong>${signal.label}</strong>
              <span class="pill ${risk}">${formatValue(value, signal.unit)}</span>
            </div>
            <small>${signal.kind} PID · delta ${prefix}${formatValue(delta, signal.unit)}</small>
            <div class="meter"><span class="${risk}" style="width:${pct}%"></span></div>
          </article>
        `;
      })
      .join("");

    const motionRiskScore = Math.round(
      motionSignals.reduce((sum, signal) => sum + riskPercent(signal, state.values[signal.id]), 0) / motionSignals.length,
    );
    motionScore.textContent = `Motion ${motionRiskScore}`;
    motionScore.className = `pill ${motionRiskScore > 62 ? "danger" : motionRiskScore > 38 ? "watch" : "good"}`;

    motionGrid.innerHTML = motionSignals
      .map((signal) => {
        const value = state.values[signal.id];
        const risk = riskFor(signal, value);
        const pct = riskPercent(signal, value);
        const delta = value - signal.baseline;
        const prefix = delta >= 0 ? "+" : "";
        return `
          <article class="motion-card">
            <div class="motion-topline">
              <strong>${signal.label}</strong>
              <span class="pill ${risk}">${formatValue(value, signal.unit)}</span>
            </div>
            <small>baseline delta ${prefix}${formatValue(delta, signal.unit)}</small>
            <div class="meter"><span class="${risk}" style="width:${pct}%"></span></div>
          </article>
        `;
      })
      .join("");
  }

  function sizeCanvas() {
    const rect = canvas.getBoundingClientRect();
    const ratio = window.devicePixelRatio || 1;
    const width = Math.max(320, Math.round(rect.width * ratio));
    const height = Math.max(300, Math.round(rect.height * ratio));
    if (canvas.width !== width || canvas.height !== height) {
      canvas.width = width;
      canvas.height = height;
    }
    const ctx = canvas.getContext("2d");
    ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
    return { ctx, width: rect.width, height: rect.height };
  }

  function draw() {
    const { ctx, width, height } = sizeCanvas();
    ctx.clearRect(0, 0, width, height);
    const bg = ctx.createLinearGradient(0, 0, width, height);
    bg.addColorStop(0, "#0b0d0d");
    bg.addColorStop(0.55, "#17201e");
    bg.addColorStop(1, "#090a0a");
    ctx.fillStyle = bg;
    ctx.fillRect(0, 0, width, height);

    if (state.scanMode === "delta") drawDeltaMap(ctx, width, height);
    if (state.scanMode === "drive") drawDriveFeel(ctx, width, height);
    if (state.scanMode === "install") drawInstallPlan(ctx, width, height);
  }

  function drawDeltaMap(ctx, width, height) {
    const cx = width * 0.5;
    const cy = height * 0.55;
    const carW = Math.min(width * 0.72, 620);
    const carH = carW * 0.42;

    ctx.save();
    ctx.translate(cx, cy);
    ctx.fillStyle = "rgba(0,0,0,0.34)";
    roundedRect(ctx, -carW * 0.5, -carH * 0.5 + carH * 0.56, carW, carH * 0.32, 36);
    ctx.fill();

    ctx.fillStyle = "#1d2422";
    ctx.strokeStyle = "rgba(238,242,237,0.22)";
    ctx.lineWidth = 2;
    roundedRect(ctx, -carW * 0.5, -carH * 0.36, carW, carH * 0.72, 46);
    ctx.fill();
    ctx.stroke();

    ctx.fillStyle = "rgba(103,216,238,0.16)";
    roundedRect(ctx, -carW * 0.16, -carH * 0.32, carW * 0.32, carH * 0.64, 18);
    ctx.fill();

    ctx.fillStyle = "#101413";
    roundedRect(ctx, -carW * 0.45, -carH * 0.24, carW * 0.18, carH * 0.48, 18);
    ctx.fill();
    roundedRect(ctx, carW * 0.27, -carH * 0.24, carW * 0.18, carH * 0.48, 18);
    ctx.fill();

    const nodes = [
      { id: "coolant", x: -0.33, y: -0.08, label: "Cooling" },
      { id: "iat", x: -0.18, y: -0.16, label: "Intake" },
      { id: "voltage", x: -0.38, y: 0.14, label: "Battery" },
      { id: "stft", x: 0.02, y: -0.05, label: "Fuel Trim" },
      { id: "maf", x: -0.08, y: 0.15, label: "MAF" },
      { id: "throttle", x: 0.24, y: 0.08, label: "Throttle" },
    ];

    nodes.forEach((node) => {
      const signal = obdSignals.find((item) => item.id === node.id);
      const value = state.values[node.id];
      const risk = riskFor(signal, value);
      const color = risk === "danger" ? "#ff675f" : risk === "watch" ? "#f8b84e" : "#62d38e";
      const x = node.x * carW;
      const y = node.y * carH;
      ctx.fillStyle = color;
      ctx.shadowColor = color;
      ctx.shadowBlur = state.awake ? 26 : 12;
      ctx.beginPath();
      ctx.arc(x, y, 12 + riskPercent(signal, value) * 0.08, 0, Math.PI * 2);
      ctx.fill();
      ctx.shadowBlur = 0;
      ctx.fillStyle = "#eef2ed";
      ctx.font = "12px Inter, sans-serif";
      ctx.fillText(node.label, x + 18, y + 4);
    });

    drawWheels(ctx, carW, carH);
    ctx.restore();

    drawTelemetryStrip(ctx, width, height, "Bluetooth OBD stream", [
      `RPM ${Math.round(state.values.rpm)}`,
      `Coolant ${Math.round(state.values.coolant)} F`,
      `Voltage ${state.values.voltage.toFixed(2)} V`,
      `Fuel trim ${state.values.stft.toFixed(1)}%`,
    ]);
  }

  function drawWheels(ctx, carW, carH) {
    const wheels = [
      [-0.37, -0.42],
      [0.37, -0.42],
      [-0.37, 0.42],
      [0.37, 0.42],
    ];
    wheels.forEach(([x, y]) => {
      ctx.fillStyle = "#070808";
      ctx.strokeStyle = "#39423f";
      ctx.lineWidth = 8;
      ctx.beginPath();
      ctx.ellipse(x * carW, y * carH, carW * 0.08, carH * 0.18, 0, 0, Math.PI * 2);
      ctx.fill();
      ctx.stroke();
    });
  }

  function drawDriveFeel(ctx, width, height) {
    const roadY = height * 0.6;
    ctx.fillStyle = "#202825";
    roundedRect(ctx, width * 0.08, roadY - 88, width * 0.84, 176, 48);
    ctx.fill();

    ctx.strokeStyle = "rgba(238,242,237,0.22)";
    ctx.setLineDash([18, 18]);
    ctx.lineWidth = 3;
    ctx.beginPath();
    for (let x = width * 0.1; x <= width * 0.9; x += 8) {
      const y = roadY + Math.sin(x * 0.014 + state.t) * 22;
      if (x === width * 0.1) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();
    ctx.setLineDash([]);

    const carX = width * 0.32 + Math.sin(state.t * 0.45) * width * 0.18;
    const carY = roadY + Math.sin(carX * 0.014 + state.t) * 22;
    ctx.save();
    ctx.translate(carX, carY);
    ctx.rotate(Math.sin(state.t * 0.52) * 0.08);
    ctx.fillStyle = "#67d8ee";
    roundedRect(ctx, -34, -15, 68, 30, 8);
    ctx.fill();
    ctx.fillStyle = "#0b0d0d";
    roundedRect(ctx, -10, -11, 20, 22, 5);
    ctx.fill();
    ctx.restore();

    const waves = [
      { value: state.values.brake, label: "Brake dive", color: "#f8b84e", y: height * 0.28 },
      { value: state.values.corner, label: "Corner load", color: "#67d8ee", y: height * 0.37 },
      { value: state.values.roughness / 100, label: "Road harshness", color: "#ff675f", y: height * 0.46 },
    ];
    waves.forEach((wave) => {
      ctx.strokeStyle = wave.color;
      ctx.lineWidth = 2;
      ctx.beginPath();
      for (let i = 0; i < 160; i += 1) {
        const x = width * 0.08 + (i / 159) * width * 0.84;
        const y = wave.y + Math.sin(i * 0.18 + state.t * 1.4) * 18 * wave.value;
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.stroke();
      ctx.fillStyle = "rgba(238,242,237,0.76)";
      ctx.font = "12px Inter, sans-serif";
      ctx.fillText(wave.label, width * 0.08, wave.y - 14);
    });
  }

  function drawInstallPlan(ctx, width, height) {
    const cards = [
      { title: "1. OBD adapter", body: "Pairs over Bluetooth and streams generic PIDs first.", x: 0.12, y: 0.32 },
      { title: "2. Pi collector", body: "Stores logs and hosts this dashboard on local Wi-Fi.", x: 0.42, y: 0.2 },
      { title: "3. IMU module", body: "Bolted level to a rigid cabin point for repeatable deltas.", x: 0.42, y: 0.54 },
      { title: "4. Voice I/O", body: "Mic wake gate plus speaker for short valet responses.", x: 0.72, y: 0.34 },
    ];

    ctx.strokeStyle = "rgba(103,216,238,0.38)";
    ctx.lineWidth = 2;
    [[0, 1], [1, 3], [2, 3], [0, 2]].forEach(([a, b]) => {
      const from = cards[a];
      const to = cards[b];
      ctx.beginPath();
      ctx.moveTo(from.x * width + 92, from.y * height + 38);
      ctx.lineTo(to.x * width + 92, to.y * height + 38);
      ctx.stroke();
    });

    cards.forEach((card) => {
      const x = card.x * width;
      const y = card.y * height;
      ctx.fillStyle = "rgba(238,242,237,0.07)";
      ctx.strokeStyle = "rgba(238,242,237,0.16)";
      roundedRect(ctx, x, y, 190, 84, 8);
      ctx.fill();
      ctx.stroke();
      ctx.fillStyle = "#eef2ed";
      ctx.font = "700 13px Inter, sans-serif";
      ctx.fillText(card.title, x + 12, y + 25);
      ctx.font = "12px Inter, sans-serif";
      wrapText(ctx, card.body, x + 12, y + 48, 164, 16);
    });

    drawTelemetryStrip(ctx, width, height, "No car required yet", [
      vehicle.engine,
      "Mock Bluetooth",
      "Mock IMU",
      "Local-only UI",
    ]);
  }

  function drawTelemetryStrip(ctx, width, height, title, items) {
    ctx.fillStyle = "rgba(6,7,7,0.62)";
    roundedRect(ctx, width * 0.05, height - 72, width * 0.9, 46, 8);
    ctx.fill();
    ctx.fillStyle = "#99a39d";
    ctx.font = "11px Inter, sans-serif";
    ctx.fillText(title.toUpperCase(), width * 0.07, height - 48);
    ctx.fillStyle = "#eef2ed";
    ctx.font = "13px Inter, sans-serif";
    ctx.fillText(items.join("  |  "), width * 0.07, height - 28);
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

  function wrapText(ctx, text, x, y, maxWidth, lineHeight) {
    const words = text.split(" ");
    let line = "";
    words.forEach((word) => {
      const next = `${line}${word} `;
      if (ctx.measureText(next).width > maxWidth && line) {
        ctx.fillText(line, x, y);
        line = `${word} `;
        y += lineHeight;
      } else {
        line = next;
      }
    });
    ctx.fillText(line, x, y);
  }

  function loop(now) {
    const dt = Math.min(0.05, (now - state.lastTick) / 1000);
    state.lastTick = now;
    state.t += dt;
    tickValues();
    renderSignals();
    const text = valetText();
    valetLine.textContent = text.line;
    valetSubline.textContent = text.subline;
    draw();
    requestAnimationFrame(loop);
  }

  function bindEvents() {
    wakeButton.addEventListener("click", () => {
      setAwake(true, `Heard "${state.wakePhrase}".`);
    });

    speakButton.addEventListener("click", speakCurrentLine);

    sleepButton.addEventListener("click", () => {
      setAwake(false);
    });

    wakePhraseInput.addEventListener("change", () => {
      state.wakePhrase = wakePhraseInput.value.trim() || "hey rover";
      localStorage.setItem("lr4WakePhrase", state.wakePhrase);
      addLog("info", "Wake phrase updated", `New prototype phrase: "${state.wakePhrase}".`);
      renderStatic();
    });

    llmProvider.addEventListener("change", () => {
      state.llm.provider = llmProvider.value;
      state.llm.endpoint = defaultEndpointFor(state.llm.provider);
      state.llm.model = state.llm.provider === "llamacpp" ? "local-lr4-q4" : "llama3.2:1b";
      state.llm.status = state.llm.provider === "mock" ? "Mock ready" : "Not checked";
      state.llm.statusTone = state.llm.provider === "mock" ? "quiet" : "watch";
      localStorage.setItem("lr4LlmProvider", state.llm.provider);
      localStorage.setItem("lr4LlmEndpoint", state.llm.endpoint);
      localStorage.setItem("lr4LlmModel", state.llm.model);
      addLog("info", "LLM provider changed", `Using ${llmProvider.options[llmProvider.selectedIndex].text}.`);
      renderStatic();
    });

    llmEndpoint.addEventListener("change", () => {
      state.llm.endpoint = llmEndpoint.value.trim() || defaultEndpointFor(state.llm.provider);
      localStorage.setItem("lr4LlmEndpoint", state.llm.endpoint);
      addLog("info", "LLM endpoint updated", state.llm.endpoint);
      renderStatic();
    });

    llmModel.addEventListener("change", () => {
      state.llm.model = llmModel.value.trim() || "llama3.2:1b";
      localStorage.setItem("lr4LlmModel", state.llm.model);
      addLog("info", "LLM model updated", state.llm.model);
      renderStatic();
    });

    llmHealthButton.addEventListener("click", healthCheckLlm);
    llmBriefButton.addEventListener("click", () => requestValetBrief({ announce: state.awake }));

    document.querySelectorAll(".mini-tab").forEach((button) => {
      button.addEventListener("click", () => {
        state.scanMode = button.dataset.scan;
        addLog("info", "Display mode changed", `Showing ${button.textContent}.`);
        renderStatic();
        draw();
      });
    });

    window.addEventListener("resize", draw);
  }

  function init() {
    tickValues();
    renderStatic();
    bindEvents();
    requestAnimationFrame(loop);
  }

  init();
})();
