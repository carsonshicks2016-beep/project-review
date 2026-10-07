#!/usr/bin/env node
/**
 * Cross-browser acceptance for the clean-room Fable Observatory client.
 *
 * Starts its own Command Center on an ephemeral loopback port, creates one
 * browser-owned session at a time, and terminates only that child server.
 */
import { spawn, spawnSync } from "node:child_process";
import fs from "node:fs";
import net from "node:net";
import path from "node:path";
import process from "node:process";
import { fileURLToPath } from "node:url";

import { chromium, webkit } from "playwright";


const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const SERVER = path.join(ROOT, "command-center", "server.py");
const PLAYWRIGHT_BIN = path.join(ROOT, "node_modules", ".bin", "playwright");
const QA_ROOT = path.join(ROOT, "runtime", "observatory-qa");
const timestamp = new Date().toISOString().replace(/[-:]/g, "").replace(/\.\d+Z$/, "Z");
const QA_DIR = path.join(QA_ROOT, timestamp);
const SCREENSHOT_DIR = path.join(QA_DIR, "screenshots");
const VIEWPORT = Object.freeze({ width: 2560, height: 1440 });
const PRINCIPAL_CAMERAS = ["broadcast", "chase", "roof", "trackside", "drone", "free"];
const VISUAL_CONDITIONS = [
  { weather: "clear-day", light: "clear-day" },
  { weather: "wet-dusk", light: "wet-dusk" },
  { weather: "rain-day", light: "rain-day" },
  { weather: "rain-night", light: "rain-night" },
];
const CAMERA_WEATHER_MATRIX = PRINCIPAL_CAMERAS.flatMap((camera) =>
  VISUAL_CONDITIONS.map((preset) => ({ camera, ...preset })));

fs.mkdirSync(SCREENSHOT_DIR, { recursive: true });

const report = {
  schema: "fable-observatory-playwright-qa-v1",
  started_at: new Date().toISOString(),
  qa_directory: QA_DIR,
  viewport: { ...VIEWPORT, device_scale_factor: 1 },
  checks: [],
  diagnostics: [],
  browsers: {},
  screenshots: [],
  performance: {},
};


function gate(name, condition, detail = "") {
  report.checks.push({ name, pass: Boolean(condition), detail });
  const status = condition ? "PASS" : "FAIL";
  console.log(`  [${status}] ${name}${detail ? ` — ${detail}` : ""}`);
  if (!condition) throw new Error(`gate failed: ${name}${detail ? ` (${detail})` : ""}`);
}


function diagnostic(name, targetMet, detail = "") {
  report.diagnostics.push({ name, target_met: Boolean(targetMet), detail });
  console.log(`  [INFO] ${name}${detail ? ` — ${detail}` : ""}`);
}


function browserExecutableReady(browserType) {
  try {
    return fs.existsSync(browserType.executablePath());
  } catch {
    return false;
  }
}


function ensureBrowserBinaries() {
  if (!browserExecutableReady(chromium)) {
    throw new Error(`Chromium browser binary is missing: ${chromium.executablePath()}. It was not installed because this validator is only authorized to install missing WebKit.`);
  }
  if (!browserExecutableReady(webkit)) {
    console.log("Playwright WebKit is missing; installing only the WebKit browser binary...");
    const installed = spawnSync(PLAYWRIGHT_BIN, ["install", "webkit"], {
      cwd: ROOT,
      env: process.env,
      stdio: "inherit",
    });
    if (installed.status !== 0 || !browserExecutableReady(webkit)) {
      throw new Error(`Playwright WebKit installation failed with status ${installed.status}`);
    }
    report.webkit_installed = true;
  } else {
    report.webkit_installed = false;
  }
}


function allocatePort() {
  return new Promise((resolve, reject) => {
    const server = net.createServer();
    server.unref();
    server.once("error", reject);
    server.listen(0, "127.0.0.1", () => {
      const address = server.address();
      const port = typeof address === "object" && address ? address.port : 0;
      server.close((error) => error ? reject(error) : resolve(port));
    });
  });
}


function startCommandCenter(port) {
  const output = [];
  const child = spawn(process.env.PYTHON || "python3", [SERVER], {
    cwd: ROOT,
    detached: false,
    env: {
      ...process.env,
      PORT: String(port),
      PYTHONUNBUFFERED: "1",
      SDL_VIDEODRIVER: "dummy",
      SDL_AUDIODRIVER: "dummy",
      NO_PROXY: "127.0.0.1,localhost",
      no_proxy: "127.0.0.1,localhost",
    },
    stdio: ["ignore", "pipe", "pipe"],
  });
  const collect = (chunk) => {
    output.push(String(chunk));
    if (output.length > 400) output.splice(0, output.length - 400);
  };
  child.stdout.on("data", collect);
  child.stderr.on("data", collect);
  return { child, output };
}


async function stopCommandCenter(child) {
  if (!child || child.exitCode != null) return;
  child.kill("SIGTERM");
  const exited = await Promise.race([
    new Promise((resolve) => child.once("exit", () => resolve(true))),
    new Promise((resolve) => setTimeout(() => resolve(false), 8000)),
  ]);
  if (!exited && child.exitCode == null) {
    child.kill("SIGKILL");
    await new Promise((resolve) => child.once("exit", resolve));
  }
}


async function waitForCommandCenter(baseUrl, child, output, timeoutMs = 30000) {
  const deadline = Date.now() + timeoutMs;
  let lastError = "not contacted";
  while (Date.now() < deadline) {
    if (child.exitCode != null) {
      throw new Error(`isolated Command Center exited ${child.exitCode}\n${output.join("").slice(-8000)}`);
    }
    try {
      const response = await fetch(`${baseUrl}/api/observatory/catalog`);
      if (response.ok) return response.json();
      lastError = `HTTP ${response.status}: ${(await response.text()).slice(0, 300)}`;
    } catch (error) {
      lastError = error.message;
    }
    await new Promise((resolve) => setTimeout(resolve, 100));
  }
  throw new Error(`isolated Command Center did not become ready: ${lastError}`);
}


async function poll(predicate, { timeoutMs = 10000, intervalMs = 100, label = "condition" } = {}) {
  const deadline = Date.now() + timeoutMs;
  let last;
  while (Date.now() < deadline) {
    try {
      last = await predicate();
      if (last) return last;
    } catch (error) {
      last = error;
    }
    await new Promise((resolve) => setTimeout(resolve, intervalMs));
  }
  throw new Error(`timed out waiting for ${label}; last=${last instanceof Error ? last.message : JSON.stringify(last)}`);
}


function webSocketCaptureInit() {
  const NativeWebSocket = window.WebSocket;
  const sockets = [];
  const messages = [];
  const controls = [];
  Object.defineProperty(window, "__observatoryQaSockets", {
    configurable: false,
    enumerable: false,
    value: sockets,
  });
  Object.defineProperty(window, "__observatoryQaMessages", {
    configurable: false,
    enumerable: false,
    value: messages,
  });
  Object.defineProperty(window, "__observatoryQaControls", {
    configurable: false,
    enumerable: false,
    value: controls,
  });
  function QaWebSocket(url, protocols) {
    const socket = protocols === undefined
      ? new NativeWebSocket(url)
      : new NativeWebSocket(url, protocols);
    sockets.push(socket);
    if (String(url).includes("/api/observatory/ws/")) {
      socket.addEventListener("message", (event) => {
        if (typeof event.data !== "string") return;
        try { messages.push(JSON.parse(event.data)); }
        catch { messages.push({ type: "unparseable", data: event.data }); }
        if (messages.length > 200) messages.splice(0, messages.length - 200);
      });
      const nativeSend = socket.send.bind(socket);
      socket.send = (payload) => {
        try { controls.push(JSON.parse(payload)); }
        catch { controls.push({ type: "unparseable", data: String(payload) }); }
        return nativeSend(payload);
      };
    }
    return socket;
  }
  QaWebSocket.prototype = NativeWebSocket.prototype;
  for (const key of ["CONNECTING", "OPEN", "CLOSING", "CLOSED"]) {
    Object.defineProperty(QaWebSocket, key, { value: NativeWebSocket[key] });
  }
  window.WebSocket = QaWebSocket;
}


async function identitySnapshot(page) {
  return page.evaluate(() => {
    const text = (id) => document.getElementById(id)?.textContent?.trim() || "";
    return {
      car: text("identity-car"),
      edition: text("identity-edition"),
      stage: text("identity-stage"),
      checkpoint: text("identity-checkpoint"),
      hash: text("identity-hash"),
      classification: text("identity-classification"),
      drivetrain: text("identity-drivetrain"),
      observation: text("identity-observation"),
      compatibility: text("identity-compatibility"),
      warnings: text("identity-warnings"),
    };
  });
}


function identityIsResolved(identity, editionId, expectedCar) {
  return identity.car === expectedCar
    && identity.edition === editionId
    && identity.checkpoint.endsWith(".pt")
    && /^[a-f0-9]{64}$/i.test(identity.hash)
    && identity.classification
    && identity.classification !== "UNVERIFIED PLAYBACK"
    && identity.drivetrain !== "—"
    && identity.observation !== "—"
    && identity.compatibility !== "—";
}


async function toggleOverlay(page, buttonSelector, panelSelector, name) {
  const before = await page.locator(panelSelector).getAttribute("data-open");
  await page.locator(buttonSelector).click();
  await page.waitForFunction(
    ({ selector, previous }) => document.querySelector(selector)?.dataset.open !== previous,
    { selector: panelSelector, previous: before },
  );
  const changed = await page.locator(panelSelector).getAttribute("data-open");
  await page.locator(buttonSelector).click();
  await page.waitForFunction(
    ({ selector, expected }) => document.querySelector(selector)?.dataset.open === expected,
    { selector: panelSelector, expected: before },
  );
  gate(`${name} overlay toggles`, changed !== before && ["true", "false"].includes(changed));
}


async function measureRaf(page, sampleCount = 240) {
  const sample = await page.evaluate((count) => new Promise((resolve) => {
    const values = [];
    const qualityLevels = new Set();
    let previous = 0;
    function sample(now) {
      qualityLevels.add(document.getElementById("app")?.dataset.quality || "unknown");
      if (previous) values.push(now - previous);
      previous = now;
      if (values.length >= count) {
        const canvas = document.getElementById("stage");
        const rect = canvas.getBoundingClientRect();
        resolve({
          values,
          qualityLevels: [...qualityLevels],
          activeQuality: document.getElementById("app")?.dataset.quality || "unknown",
          backingWidth: canvas.width,
          backingHeight: canvas.height,
          renderScale: rect.width > 0 ? canvas.width / rect.width : 0,
        });
      }
      else requestAnimationFrame(sample);
    }
    requestAnimationFrame(sample);
  }), sampleCount);
  const deltas = sample.values;
  const sorted = [...deltas].sort((a, b) => a - b);
  const percentile = (q) => sorted[Math.min(sorted.length - 1, Math.ceil(sorted.length * q) - 1)];
  return {
    samples: deltas.length,
    mean_ms: deltas.reduce((sum, value) => sum + value, 0) / deltas.length,
    p50_ms: percentile(0.50),
    p95_ms: percentile(0.95),
    p99_ms: percentile(0.99),
    maximum_ms: sorted.at(-1),
    quality_levels: sample.qualityLevels,
    active_quality: sample.activeQuality,
    backing_resolution: `${sample.backingWidth}x${sample.backingHeight}`,
    render_scale: sample.renderScale,
  };
}


async function releaseAndVerify(baseUrl, page, sessionId) {
  await page.evaluate(() => {
    const event = typeof PageTransitionEvent === "function"
      ? new PageTransitionEvent("pagehide", { persisted: false })
      : new Event("pagehide");
    window.dispatchEvent(event);
  });
  await poll(async () => {
    const response = await fetch(`${baseUrl}/api/observatory/catalog`);
    const payload = await response.json();
    return payload.sessions?.active === 0;
  }, { timeoutMs: 10000, label: "pagehide session release" });
  const response = await fetch(`${baseUrl}/api/observatory/sessions/${encodeURIComponent(sessionId)}`, {
    method: "DELETE",
  });
  const payload = await response.json();
  gate("pagehide releases its browser session", response.ok && payload.released === false);
}


async function exerciseAudio(page, browserName) {
  const clockBefore = await page.locator("#replay-clock").textContent();
  await page.locator("#toggle-audio").click();
  await page.waitForFunction(() => {
    const button = document.getElementById("toggle-audio");
    const toast = document.getElementById("toast");
    if (button?.getAttribute("aria-pressed") === "true") return true;
    if (toast?.dataset.open !== "true") return false;
    const label = button?.textContent?.trim().toLowerCase() || "";
    const notice = toast.textContent?.trim().toLowerCase() || "";
    return label.includes("audio buffering")
      || notice.includes("audio unavailable")
      || notice.includes("audio muted")
      || notice.includes("audio stream offline");
  }, null, { timeout: 10000 });
  await page.waitForFunction(
    (previous) => document.getElementById("connection-state")?.dataset.state === "live"
      && document.getElementById("replay-clock")?.textContent !== previous,
    clockBefore,
    { timeout: 10000 },
  );
  const outcome = await page.evaluate(() => ({
    button: document.getElementById("toggle-audio")?.textContent?.trim() || "",
    pressed: document.getElementById("toggle-audio")?.getAttribute("aria-pressed") || "false",
    toast: document.getElementById("toast")?.dataset.open === "true"
      ? document.getElementById("toast")?.textContent?.trim() || ""
      : "",
    telemetry: document.getElementById("connection-state")?.dataset.state || "",
  }));
  gate(`${browserName} audio control degrades gracefully without stopping telemetry`,
    outcome.telemetry === "live"
      && (/(live|buffering|muted)/i.test(outcome.button)
        || /(unavailable|muted|offline)/i.test(outcome.toast)),
    `${outcome.button}${outcome.toast ? `; ${outcome.toast}` : ""}`);
  if (outcome.pressed === "true") {
    await page.locator("#toggle-audio").click();
    await page.waitForFunction(() => document.getElementById("toggle-audio")?.getAttribute("aria-pressed") === "false", null, { timeout: 10000 });
  }
  return outcome;
}


async function runScenario({ browserName, browser, baseUrl, editionId, carQuery, expectedCar, captureVisuals = true }) {
  const context = await browser.newContext({
    viewport: VIEWPORT,
    deviceScaleFactor: 1,
    locale: "en-US",
  });
  await context.addInitScript(webSocketCaptureInit);
  const page = await context.newPage();
  const consoleErrors = [];
  const pageErrors = [];
  page.on("console", (message) => {
    if (message.type() === "error") consoleErrors.push(message.text());
  });
  page.on("pageerror", (error) => pageErrors.push(error.stack || error.message));

  let sessionId = "";
  const scenario = {
    browser: browserName,
    edition: editionId,
    deliberately_wrong_car_query: carQuery,
    car: expectedCar,
    identity: null,
    reconnect: false,
    released: false,
    screenshots: [],
    console_errors: [],
    page_errors: [],
  };
  try {
    const sessionResponse = page.waitForResponse(
      (response) => response.url().endsWith("/api/observatory/sessions")
        && response.request().method() === "POST",
      { timeout: 30000 },
    );
    const url = `${baseUrl}/observatory/?edition=${encodeURIComponent(editionId)}&car=${encodeURIComponent(carQuery)}&checkpoint=active-best&mode=replay`;
    await page.goto(url, { waitUntil: "domcontentloaded", timeout: 30000 });
    const response = await sessionResponse;
    const responseBody = await response.json();
    gate(`${browserName}/${editionId} creates a session`, response.status() === 201,
      `HTTP ${response.status()}`);
    sessionId = String(responseBody.session_id || "");
    gate(`${browserName}/${editionId} session carries exact track/checkpoint identity`,
      Boolean(sessionId)
      && /^[a-f0-9]{64}$/i.test(String(responseBody.track_hash || ""))
      && responseBody.hello?.checkpoint?.policy_sha256);

    await page.waitForFunction(() => document.getElementById("loading-card")?.dataset.hidden === "true", null, { timeout: 30000 });
    await page.waitForFunction(() => document.getElementById("connection-state")?.dataset.state === "live", null, { timeout: 30000 });
    await page.waitForFunction(
      ({ edition, car }) => {
        const value = (id) => document.getElementById(id)?.textContent?.trim() || "";
        return value("identity-edition") === edition
          && value("identity-car") === car
          && /^[a-f0-9]{64}$/i.test(value("identity-hash"));
      },
      { edition: editionId, car: expectedCar },
      { timeout: 30000 },
    );

    const initialClock = await page.locator("#replay-clock").textContent();
    await page.waitForFunction(
      (previous) => document.getElementById("replay-clock")?.textContent !== previous,
      initialClock,
      { timeout: 10000 },
    );
    const canvas = await page.locator("#stage").evaluate((node) => {
      const rect = node.getBoundingClientRect();
      const gl = node.getContext("webgl2") || node.getContext("webgl");
      return { width: rect.width, height: rect.height, backingWidth: node.width,
        backingHeight: node.height, webgl: Boolean(gl),
        renderScale: rect.width > 0 ? node.width / rect.width : 0,
        quality: document.getElementById("app")?.dataset.quality || "unknown" };
    });
    scenario.canvas = canvas;
    gate(`${browserName}/${editionId} reaches a live WebGL canvas`,
      canvas.webgl && Math.round(canvas.width) === VIEWPORT.width && Math.round(canvas.height) === VIEWPORT.height,
      `CSS ${canvas.width}x${canvas.height}, backing ${canvas.backingWidth}x${canvas.backingHeight}, scale ${canvas.renderScale.toFixed(2)}, quality ${canvas.quality}`);

    const identity = await identitySnapshot(page);
    gate(`${browserName}/${editionId} keeps complete provenance visible`,
      identityIsResolved(identity, editionId, expectedCar),
      `${identity.checkpoint} / ${identity.classification}`);
    gate(`${browserName}/${editionId} ignores the opposite car query and uses registry identity`,
      identity.car === expectedCar && identity.car.toLowerCase() !== carQuery.toLowerCase(),
      `query=${carQuery}, resolved=${identity.car}`);
    if (editionId === "919") {
      gate(`${browserName}/919 shows its permanent legacy authority warning`,
        identity.warnings.toLowerCase().includes("not faithful-v2"));
    }
    scenario.identity = identity;

    await toggleOverlay(page, "#toggle-xray", "#brain-overlay", `${browserName}/${editionId} Brain X-Ray`);
    await toggleOverlay(page, "#toggle-engineer", "#engineer-overlay", `${browserName}/${editionId} Engineer`);
    await toggleOverlay(page, "#toggle-replay", "#replay-overlay", `${browserName}/${editionId} Replay`);

    // Cinematic is intentionally the clean default after the visual revamp.
    // Open Replay explicitly before exercising transport controls instead of
    // coupling the transport contract to a permanently visible panel.
    if (await page.locator("#replay-overlay").getAttribute("data-open") !== "true") {
      await page.locator("#toggle-replay").click();
      await page.waitForFunction(() => document.getElementById("replay-overlay")?.dataset.open === "true");
    }

    const preResumeClock = await page.locator("#replay-clock").textContent();
    await page.locator("#control-pause").click();
    await page.waitForFunction(() => window.__observatoryQaControls.some((entry) => entry.type === "pause"), null, { timeout: 10000 });
    await page.waitForFunction(() => window.__observatoryQaMessages.some((entry) => entry.type === "event" && entry.event === "paused"), null, { timeout: 10000 });
    await page.waitForFunction(() => document.getElementById("replay-events")?.textContent?.toLowerCase().includes("paused"), null, { timeout: 10000 });
    await page.waitForFunction(() => document.getElementById("control-pause")?.textContent?.trim() === "Resume", null, { timeout: 10000 });
    await page.locator("#control-pause").click();
    await page.waitForFunction(() => window.__observatoryQaControls.some((entry) => entry.type === "resume"), null, { timeout: 10000 });
    await page.waitForFunction(
      (previous) => document.getElementById("replay-clock")?.textContent !== previous
        && document.getElementById("control-pause")?.textContent?.trim() === "Pause",
      preResumeClock,
      { timeout: 10000 },
    );
    gate(`${browserName}/${editionId} pause immediately exposes resume and returns live`, true);

    await page.locator("#control-speed").selectOption("2");
    await page.waitForFunction(() => document.getElementById("replay-events")?.textContent?.includes("speed_changed"), null, { timeout: 10000 });
    gate(`${browserName}/${editionId} accepts 2x playback`,
      await page.locator("#control-speed").inputValue() === "2");

    await page.locator("#control-reset").click();
    await page.waitForFunction(() => document.getElementById("replay-events")?.textContent?.toLowerCase().includes("reset"), null, { timeout: 10000 });
    gate(`${browserName}/${editionId} reset reaches the replay timeline`, true);

    await page.waitForFunction(() => document.getElementById("connection-state")?.dataset.state === "live"
      && document.getElementById("control-pause")?.textContent?.trim() === "Pause", null, { timeout: 10000 });

    if (editionId === "787b") {
      scenario.audio = await exerciseAudio(page, browserName);
    }

    const socketCount = await page.evaluate(() => window.__observatoryQaSockets.length);
    await page.evaluate(() => {
      const sockets = window.__observatoryQaSockets;
      const socket = [...sockets].reverse().find((entry) => String(entry.url).includes("/api/observatory/ws/"));
      if (!socket) throw new Error("telemetry socket was not captured");
      socket.close(4001, "observatory acceptance reconnect");
    });
    await page.waitForFunction(
      (count) => window.__observatoryQaSockets.length > count,
      socketCount,
      { timeout: 20000 },
    );
    await page.waitForFunction(() => document.getElementById("connection-state")?.dataset.state === "live", null, { timeout: 20000 });
    scenario.reconnect = true;
    gate(`${browserName}/${editionId} disconnects and reconnects its telemetry socket`, true);

    const persistent = await identitySnapshot(page);
    gate(`${browserName}/${editionId} provenance persists through controls/reconnect`,
      persistent.car === identity.car
      && persistent.edition === identity.edition
      && persistent.checkpoint === identity.checkpoint
      && persistent.hash === identity.hash
      && persistent.compatibility === identity.compatibility);

    if (browserName === "chromium" && captureVisuals) {
      const performance = await measureRaf(page);
      report.performance[editionId] = performance;
      scenario.performance = performance;
      // Headless Chromium advertises SwiftShader in this environment.  Keep
      // its number as a diagnostic while the separate headed Metal scenario
      // below owns the current-Mac performance acceptance gate.
      diagnostic(`headless chromium/${editionId} SwiftShader rAF diagnostic`,
        performance.p95_ms < 20,
        `p95 ${performance.p95_ms.toFixed(3)} ms, mean ${performance.mean_ms.toFixed(3)} ms, quality ${performance.active_quality}, backing ${performance.backing_resolution}, scale ${performance.render_scale.toFixed(2)}`);

      // The revamp makes camera controls a focused drawer and keeps Cinematic
      // clean by default. Exercise the real drawer for every change, then use
      // representative Engineer and Brain captures without leaving controls
      // permanently over the world.
      for (let index = 0; index < CAMERA_WEATHER_MATRIX.length; index += 1) {
        const combo = CAMERA_WEATHER_MATRIX[index];
        if (await page.locator("#view-controls").getAttribute("data-open") !== "true") {
          await page.locator("#toggle-controls").click();
          await page.waitForFunction(() => document.getElementById("view-controls")?.dataset.open === "true");
        }
        await page.locator("#camera-mode").selectOption(combo.camera);
        await page.locator("#weather-mode").selectOption(combo.weather);
        if (index === 8) {
          await page.locator("#toggle-xray").click();
        } else if (index === 12) {
          await page.locator("#toggle-engineer").click();
        } else {
          await page.locator("#mode-cinematic").click();
        }
        await page.waitForFunction(
          ({ camera, weather }) => document.getElementById("camera-mode")?.value === camera
            && document.getElementById("weather-mode")?.value === weather
            && document.getElementById("app")?.dataset.weather === weather,
          combo,
        );
        await page.waitForTimeout(combo.camera === "trackside" ? 900 : 550);
        const filename = `${editionId}-${combo.light}-${combo.camera}.png`;
        const screenshotPath = path.join(SCREENSHOT_DIR, filename);
        await page.screenshot({ path: screenshotPath, fullPage: false });
        scenario.screenshots.push(screenshotPath);
        report.screenshots.push(screenshotPath);
      }
      gate(`chromium/${editionId} captures all 6 cameras across all 4 visual conditions`,
        scenario.screenshots.length === PRINCIPAL_CAMERAS.length * VISUAL_CONDITIONS.length,
        `${scenario.screenshots.length} screenshots`);
    }

    await releaseAndVerify(baseUrl, page, sessionId);
    await page.waitForTimeout(500);
    scenario.console_errors = [...consoleErrors];
    scenario.page_errors = [...pageErrors];
    gate(`${browserName}/${editionId} has zero page or console errors through reconnect and release`,
      pageErrors.length === 0 && consoleErrors.length === 0,
      [...pageErrors, ...consoleErrors].join(" | "));
    scenario.released = true;
    sessionId = "";
    return scenario;
  } finally {
    if (sessionId) {
      try {
        await fetch(`${baseUrl}/api/observatory/sessions/${encodeURIComponent(sessionId)}`, { method: "DELETE" });
      } catch { /* isolated server cleanup is the final fallback */ }
    }
    await context.close();
  }
}


async function runBrowser(browserName, browserType, baseUrl, { captureVisuals = true } = {}) {
  const launchOptions = { headless: true };
  if (browserName === "chromium") {
    launchOptions.args = ["--enable-webgl", "--ignore-gpu-blocklist"];
  }
  const browser = await browserType.launch(launchOptions);
  const browserReport = { version: browser.version(), executable: browserType.executablePath(), scenarios: [] };
  report.browsers[browserName] = browserReport;
  try {
    browserReport.scenarios.push(await runScenario({
      browserName, browser, baseUrl,
      editionId: "787b", carQuery: "porsche_919evo", expectedCar: "MAZDA787B",
      captureVisuals,
    }));
    browserReport.scenarios.push(await runScenario({
      browserName, browser, baseUrl,
      editionId: "919", carQuery: "mazda787b", expectedCar: "PORSCHE_919EVO",
      captureVisuals,
    }));
  } finally {
    await browser.close();
  }
}


async function runChromiumAudioPreflight(baseUrl) {
  const browser = await chromium.launch({
    headless: true,
    args: ["--enable-webgl", "--ignore-gpu-blocklist"],
  });
  const context = await browser.newContext({
    viewport: VIEWPORT,
    deviceScaleFactor: 1,
    locale: "en-US",
  });
  const page = await context.newPage();
  const consoleErrors = [];
  const pageErrors = [];
  page.on("console", (message) => {
    if (message.type() === "error") consoleErrors.push(message.text());
  });
  page.on("pageerror", (error) => pageErrors.push(error.stack || error.message));
  let sessionId = "";
  try {
    const sessionResponse = page.waitForResponse(
      (response) => response.url().endsWith("/api/observatory/sessions")
        && response.request().method() === "POST",
      { timeout: 30000 },
    );
    await page.goto(`${baseUrl}/observatory/?edition=787b&car=porsche_919evo&checkpoint=active-best&mode=replay`, {
      waitUntil: "domcontentloaded",
      timeout: 30000,
    });
    const response = await sessionResponse;
    const body = await response.json();
    sessionId = String(body.session_id || "");
    gate("Chromium Audio preflight creates a session",
      response.status() === 201 && Boolean(sessionId), `HTTP ${response.status()}`);
    await page.waitForFunction(() => document.getElementById("loading-card")?.dataset.hidden === "true", null, { timeout: 30000 });
    await page.waitForFunction(() => document.getElementById("connection-state")?.dataset.state === "live", null, { timeout: 30000 });
    const outcome = await exerciseAudio(page, "Chromium preflight");
    await releaseAndVerify(baseUrl, page, sessionId);
    await page.waitForTimeout(500);
    gate("Chromium Audio enable/disable and release complete with zero browser socket or page errors",
      pageErrors.length === 0 && consoleErrors.length === 0,
      [...pageErrors, ...consoleErrors].join(" | "));
    report.audio_preflight = {
      outcome,
      console_errors: [...consoleErrors],
      page_errors: [...pageErrors],
    };
    sessionId = "";
  } finally {
    if (sessionId) {
      try {
        await fetch(`${baseUrl}/api/observatory/sessions/${encodeURIComponent(sessionId)}`, { method: "DELETE" });
      } catch { /* isolated server cleanup remains the final fallback */ }
    }
    await context.close();
    await browser.close();
  }
}


async function runHardwarePerformance(baseUrl) {
  const browser = await chromium.launch({
    headless: false,
    args: [
      "--enable-webgl",
      "--ignore-gpu-blocklist",
      "--use-angle=metal",
      `--window-size=${VIEWPORT.width},${VIEWPORT.height}`,
    ],
  });
  const context = await browser.newContext({
    viewport: VIEWPORT,
    deviceScaleFactor: 1,
    locale: "en-US",
  });
  const page = await context.newPage();
  const consoleErrors = [];
  const pageErrors = [];
  page.on("console", (message) => {
    if (message.type() === "error") consoleErrors.push(message.text());
  });
  page.on("pageerror", (error) => pageErrors.push(error.stack || error.message));
  let sessionId = "";
  const hardware = {
    browser_version: browser.version(),
    executable: chromium.executablePath(),
    headed: true,
    viewport: `${VIEWPORT.width}x${VIEWPORT.height}`,
  };
  report.browsers.chromium_hardware = hardware;
  try {
    const sessionResponse = page.waitForResponse(
      (response) => response.url().endsWith("/api/observatory/sessions")
        && response.request().method() === "POST",
      { timeout: 30000 },
    );
    await page.goto(`${baseUrl}/observatory/?edition=787b&car=porsche_919evo&checkpoint=active-best&mode=replay`, {
      waitUntil: "domcontentloaded",
      timeout: 30000,
    });
    const response = await sessionResponse;
    const body = await response.json();
    sessionId = String(body.session_id || "");
    gate("headed Chromium creates the hardware performance session",
      response.status() === 201 && Boolean(sessionId), `HTTP ${response.status()}`);
    await page.waitForFunction(() => document.getElementById("loading-card")?.dataset.hidden === "true", null, { timeout: 30000 });
    await page.waitForFunction(() => document.getElementById("connection-state")?.dataset.state === "live", null, { timeout: 30000 });
    await page.waitForFunction(() => {
      const car = document.getElementById("identity-car")?.textContent?.trim();
      const hash = document.getElementById("identity-hash")?.textContent?.trim() || "";
      return car === "MAZDA787B" && /^[a-f0-9]{64}$/i.test(hash);
    }, null, { timeout: 30000 });
    await page.waitForTimeout(1500);

    const renderer = await page.locator("#stage").evaluate((canvas) => {
      const gl = canvas.getContext("webgl2") || canvas.getContext("webgl");
      const extension = gl?.getExtension("WEBGL_debug_renderer_info");
      const rect = canvas.getBoundingClientRect();
      return {
        webgl: Boolean(gl),
        css_width: rect.width,
        css_height: rect.height,
        backing_width: canvas.width,
        backing_height: canvas.height,
        render_scale: rect.width > 0 ? canvas.width / rect.width : 0,
        quality: document.getElementById("app")?.dataset.quality || "unknown",
        vendor: extension ? gl.getParameter(extension.UNMASKED_VENDOR_WEBGL) : gl?.getParameter(gl.VENDOR) || "unknown",
        renderer: extension ? gl.getParameter(extension.UNMASKED_RENDERER_WEBGL) : gl?.getParameter(gl.RENDERER) || "unknown",
        version: gl?.getParameter(gl.VERSION) || "unknown",
        shading_language: gl?.getParameter(gl.SHADING_LANGUAGE_VERSION) || "unknown",
      };
    });
    hardware.renderer = renderer;
    gate("headed Chromium uses a 2560x1440 live WebGL viewport",
      renderer.webgl
        && Math.round(renderer.css_width) === VIEWPORT.width
        && Math.round(renderer.css_height) === VIEWPORT.height,
      `CSS ${renderer.css_width}x${renderer.css_height}, backing ${renderer.backing_width}x${renderer.backing_height}, quality ${renderer.quality}`);
    gate("headed Chromium is hardware accelerated rather than SwiftShader",
      !/(swiftshader|software|llvmpipe)/i.test(`${renderer.vendor} ${renderer.renderer}`),
      `${renderer.vendor} / ${renderer.renderer}`);

    const performance = await measureRaf(page, 360);
    hardware.performance = performance;
    report.hardware_performance = performance;
    gate("headed hardware Chromium rAF p95 stays below 20 ms",
      performance.p95_ms < 20,
      `p95 ${performance.p95_ms.toFixed(3)} ms, mean ${performance.mean_ms.toFixed(3)} ms, quality ${performance.active_quality}, backing ${performance.backing_resolution}, scale ${performance.render_scale.toFixed(2)}`);
    await releaseAndVerify(baseUrl, page, sessionId);
    await page.waitForTimeout(500);
    gate("headed hardware performance run has zero page or console errors through release",
      pageErrors.length === 0 && consoleErrors.length === 0,
      [...pageErrors, ...consoleErrors].join(" | "));
    sessionId = "";
  } finally {
    if (sessionId) {
      try {
        await fetch(`${baseUrl}/api/observatory/sessions/${encodeURIComponent(sessionId)}`, { method: "DELETE" });
      } catch { /* isolated server cleanup remains the final fallback */ }
    }
    await context.close();
    await browser.close();
  }
}


async function main() {
  ensureBrowserBinaries();
  const hardwareTailOnly = process.env.OBSERVATORY_HARDWARE_TAIL_ONLY === "1";
  const functionalTailOnly = process.env.OBSERVATORY_FUNCTIONAL_TAIL_ONLY === "1";
  if (hardwareTailOnly && functionalTailOnly) {
    throw new Error("choose only one Observatory tail mode");
  }
  report.mode = hardwareTailOnly
    ? "headed-hardware-tail"
    : (functionalTailOnly ? "cross-browser-functional-tail" : "full");
  gate("clean-room Observatory build exists", fs.existsSync(path.join(ROOT, "observatory", "dist", "index.html")));
  const port = await allocatePort();
  const baseUrl = `http://127.0.0.1:${port}`;
  report.base_url = baseUrl;
  const { child, output } = startCommandCenter(port);
  try {
    console.log(`== isolated Observatory acceptance at ${baseUrl} ==`);
    const catalog = await waitForCommandCenter(baseUrl, child, output);
    gate("isolated server exposes both Observatory editions",
      catalog.editions?.some((item) => item.id === "787b")
      && catalog.editions?.some((item) => item.id === "919"));

    if (!hardwareTailOnly) {
      console.log("== Chromium Audio enable/disable preflight ==");
      await runChromiumAudioPreflight(baseUrl);
      console.log("== Chromium: 787B + 919 ==");
      await runBrowser("chromium", chromium, baseUrl, { captureVisuals: !functionalTailOnly });
      console.log("== WebKit: 787B + 919 ==");
      await runBrowser("webkit", webkit, baseUrl, { captureVisuals: false });
    }
    if (!functionalTailOnly) {
      console.log("== Headed Chromium hardware performance ==");
      await runHardwarePerformance(baseUrl);
    }

    const finalCatalog = await (await fetch(`${baseUrl}/api/observatory/catalog`)).json();
    gate("all browser-created sessions are released", finalCatalog.sessions?.active === 0,
      `active=${finalCatalog.sessions?.active}`);
    if (!hardwareTailOnly && !functionalTailOnly) {
      gate("Chromium visual QA set contains 2 cars x 6 cameras x 4 visual conditions",
        report.screenshots.length === 2 * PRINCIPAL_CAMERAS.length * VISUAL_CONDITIONS.length
          && report.screenshots.every((filename) => fs.existsSync(filename)),
        `${report.screenshots.length} screenshots`);
    }
    const failedChecks = report.checks.filter((check) => !check.pass);
    report.status = failedChecks.length ? "failed" : "passed";
    report.failed_checks = failedChecks;
    report.finished_at = new Date().toISOString();
    const resultPath = path.join(QA_DIR, failedChecks.length ? "results.failed.json" : "results.json");
    fs.writeFileSync(resultPath, `${JSON.stringify(report, null, 2)}\n`);
    if (failedChecks.length) {
      console.error(`Observatory Playwright acceptance completed with ${failedChecks.length} failed gate(s)\n  Results: ${resultPath}\n  Screenshots: ${SCREENSHOT_DIR}`);
      process.exitCode = 1;
    } else {
      console.log(`Observatory Playwright acceptance OK\n  Results: ${resultPath}\n  Screenshots: ${SCREENSHOT_DIR}`);
    }
  } catch (error) {
    report.status = "failed";
    report.finished_at = new Date().toISOString();
    report.error = error.stack || error.message;
    report.server_output_tail = output.join("").slice(-12000);
    const failurePath = path.join(QA_DIR, "results.failed.json");
    fs.writeFileSync(failurePath, `${JSON.stringify(report, null, 2)}\n`);
    console.error(`\n--- isolated Command Center output (tail) ---\n${report.server_output_tail}`);
    console.error(`Failure report: ${failurePath}`);
    throw error;
  } finally {
    await stopCommandCenter(child);
  }
}


main().catch((error) => {
  console.error(error.stack || error);
  process.exitCode = 1;
});
