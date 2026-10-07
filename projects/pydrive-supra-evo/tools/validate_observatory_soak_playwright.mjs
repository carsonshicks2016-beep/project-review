#!/usr/bin/env node
/**
 * Fifteen-minute single-session stability soak for the Fable Observatory.
 *
 * The validator owns an isolated Command Center on an ephemeral loopback port
 * and one headed Chromium page.  It never discovers, signals, or reuses the
 * live dashboard/trainer.  Audio is disabled to keep the test focused on the
 * authoritative 1x telemetry/render path and to minimize interference with a
 * concurrent training run.
 */
import { spawn, spawnSync } from "node:child_process";
import fs from "node:fs";
import net from "node:net";
import path from "node:path";
import process from "node:process";
import { fileURLToPath } from "node:url";

import { chromium } from "playwright";


const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const SERVER = path.join(ROOT, "command-center", "server.py");
const QA_ROOT = path.join(ROOT, "runtime", "observatory-qa");
const stamp = new Date().toISOString().replace(/[-:]/g, "").replace(/\.\d+Z$/, "Z");
const QA_DIR = path.join(QA_ROOT, `soak-${stamp}`);
const RESULT_PATH = path.join(QA_DIR, "results.json");
const TARGET_SECONDS = Number(process.env.OBSERVATORY_SOAK_SECONDS || "900");
const SAMPLE_SECONDS = 30;
const MIB = 1024 * 1024;

if (!Number.isFinite(TARGET_SECONDS) || TARGET_SECONDS < 60) {
  throw new Error("OBSERVATORY_SOAK_SECONDS must be a finite value of at least 60 seconds");
}

fs.mkdirSync(QA_DIR, { recursive: true });

const report = {
  schema: "fable-observatory-soak-v1",
  started_at: new Date().toISOString(),
  qa_directory: QA_DIR,
  result_path: RESULT_PATH,
  isolation: {
    live_dashboard_port_excluded: 8770,
    server_scope: "isolated child process only",
    chromium: "headed ANGLE Metal",
    session_count: 1,
    edition: "919",
    checkpoint: "active-best",
    mode: "replay",
    seed: 7,
    speed: 1,
    audio: false,
  },
  target_duration_s: TARGET_SECONDS,
  sample_interval_s: SAMPLE_SECONDS,
  thresholds: {
    minimum_wall_duration_s: TARGET_SECONDS,
    minimum_frame_rate_hz: 10,
    minimum_sim_to_wall_ratio: 0.90,
    monotonic_violations: 0,
    maximum_server_rss_bytes: 1536 * MIB,
    maximum_server_rss_growth_bytes: 128 * MIB,
    maximum_server_rss_end_growth_bytes: 64 * MIB,
    maximum_js_heap_growth_bytes: 192 * MIB,
    maximum_js_heap_end_growth_bytes: 96 * MIB,
    maximum_js_heap_limit_fraction: 0.80,
    page_errors: 0,
    console_errors: 0,
    active_sessions_during_soak: 1,
    active_sessions_after_release: 0,
  },
  checks: [],
  samples: [],
  browser: {},
  telemetry: {},
  resources: {},
  page_errors: [],
  console_errors: [],
};


function print(line) {
  process.stdout.write(`${line}\n`);
}


function gate(name, condition, detail = "") {
  const pass = Boolean(condition);
  report.checks.push({ name, pass, detail });
  print(`  [${pass ? "PASS" : "FAIL"}] ${name}${detail ? ` — ${detail}` : ""}`);
  if (!pass) throw new Error(`gate failed: ${name}${detail ? ` (${detail})` : ""}`);
}


function allocatePort() {
  return new Promise((resolve, reject) => {
    const socket = net.createServer();
    socket.unref();
    socket.once("error", reject);
    socket.listen(0, "127.0.0.1", () => {
      const address = socket.address();
      const port = typeof address === "object" && address ? address.port : 0;
      socket.close((error) => error ? reject(error) : resolve(port));
    });
  });
}


async function isolatedPort() {
  for (let attempt = 0; attempt < 10; attempt += 1) {
    const port = await allocatePort();
    if (port && port !== 8770) return port;
  }
  throw new Error("could not allocate an ephemeral port distinct from live port 8770");
}


function startCommandCenter(port) {
  const output = [];
  let outputBytes = 0;
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
    const value = String(chunk);
    output.push(value);
    outputBytes += Buffer.byteLength(value);
    while (outputBytes > 128 * 1024 && output.length > 1) {
      outputBytes -= Buffer.byteLength(output.shift());
    }
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


async function catalog(baseUrl) {
  const response = await fetch(`${baseUrl}/api/observatory/catalog`);
  if (!response.ok) throw new Error(`catalog returned HTTP ${response.status}`);
  return response.json();
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


function processRssBytes(pid) {
  const result = spawnSync("/bin/ps", ["-o", "rss=", "-p", String(pid)], {
    encoding: "utf8",
    timeout: 5000,
  });
  if (result.status !== 0) return null;
  const kib = Number(String(result.stdout || "").trim());
  return Number.isFinite(kib) && kib > 0 ? kib * 1024 : null;
}


function median(values) {
  const sorted = values.filter(Number.isFinite).sort((a, b) => a - b);
  if (!sorted.length) return null;
  const middle = Math.floor(sorted.length / 2);
  return sorted.length % 2 ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) / 2;
}


function fmtMib(bytes) {
  return Number.isFinite(bytes) ? `${(bytes / MIB).toFixed(1)} MiB` : "unavailable";
}


function soakCaptureInit() {
  const NativeWebSocket = window.WebSocket;
  const stats = {
    telemetrySockets: 0,
    openTelemetrySockets: 0,
    closeEvents: 0,
    frameCount: 0,
    firstSequence: null,
    lastSequence: null,
    firstSimTime: null,
    lastSimTime: null,
    monotonicViolations: 0,
    sequenceGapCount: 0,
    maximumSequenceGap: 0,
    parseErrors: 0,
    transportErrors: [],
  };
  Object.defineProperty(window, "__observatorySoak", {
    configurable: false,
    enumerable: false,
    value: stats,
  });
  function SoakWebSocket(url, protocols) {
    const socket = protocols === undefined
      ? new NativeWebSocket(url)
      : new NativeWebSocket(url, protocols);
    if (String(url).includes("/api/observatory/ws/")) {
      stats.telemetrySockets += 1;
      socket.addEventListener("open", () => { stats.openTelemetrySockets += 1; });
      socket.addEventListener("close", () => {
        stats.openTelemetrySockets = Math.max(0, stats.openTelemetrySockets - 1);
        stats.closeEvents += 1;
      });
      socket.addEventListener("error", () => stats.transportErrors.push("websocket error"));
      socket.addEventListener("message", (event) => {
        if (typeof event.data !== "string") return;
        let message;
        try { message = JSON.parse(event.data); }
        catch {
          stats.parseErrors += 1;
          return;
        }
        if (message?.type === "error") {
          stats.transportErrors.push(String(message.message || message.code || "telemetry error"));
        }
        if (message?.type !== "frame") return;
        const frame = message.frame && typeof message.frame === "object" ? message.frame : message;
        const sequence = Number(frame.sequence);
        const simTime = Number(frame.sim_time);
        if (!Number.isFinite(sequence) || !Number.isFinite(simTime)) {
          stats.parseErrors += 1;
          return;
        }
        if (stats.lastSequence != null) {
          if (sequence <= stats.lastSequence || simTime <= stats.lastSimTime) {
            stats.monotonicViolations += 1;
          }
          const gap = sequence - stats.lastSequence;
          if (gap > 1) {
            stats.sequenceGapCount += 1;
            stats.maximumSequenceGap = Math.max(stats.maximumSequenceGap, gap);
          }
        } else {
          stats.firstSequence = sequence;
          stats.firstSimTime = simTime;
        }
        stats.lastSequence = sequence;
        stats.lastSimTime = simTime;
        stats.frameCount += 1;
      });
    }
    return socket;
  }
  SoakWebSocket.prototype = NativeWebSocket.prototype;
  for (const key of ["CONNECTING", "OPEN", "CLOSING", "CLOSED"]) {
    Object.defineProperty(SoakWebSocket, key, { value: NativeWebSocket[key] });
  }
  window.WebSocket = SoakWebSocket;
}


async function pageSample(page) {
  return page.evaluate(() => {
    const stats = window.__observatorySoak || {};
    const canvas = document.getElementById("stage");
    const rect = canvas?.getBoundingClientRect();
    const memory = performance.memory;
    return {
      telemetry: { ...stats, transportErrors: [...(stats.transportErrors || [])] },
      connection: document.getElementById("connection-state")?.dataset.state || "missing",
      replayClock: document.getElementById("replay-clock")?.textContent?.trim() || "",
      speed: document.getElementById("control-speed")?.value || "",
      policyHash: document.getElementById("identity-hash")?.textContent?.trim() || "",
      edition: document.getElementById("identity-edition")?.textContent?.trim() || "",
      car: document.getElementById("identity-car")?.textContent?.trim() || "",
      quality: document.getElementById("app")?.dataset.quality || "unknown",
      canvas: canvas && rect ? {
        cssWidth: rect.width,
        cssHeight: rect.height,
        backingWidth: canvas.width,
        backingHeight: canvas.height,
        renderScale: rect.width > 0 ? canvas.width / rect.width : 0,
      } : null,
      jsHeap: memory ? {
        used: Number(memory.usedJSHeapSize),
        total: Number(memory.totalJSHeapSize),
        limit: Number(memory.jsHeapSizeLimit),
      } : null,
    };
  });
}


async function rendererInfo(page) {
  return page.evaluate(() => {
    const canvas = document.getElementById("stage");
    const gl = canvas?.getContext("webgl2") || canvas?.getContext("webgl");
    if (!gl) return { webgl: false, renderer: "", vendor: "" };
    const extension = gl.getExtension("WEBGL_debug_renderer_info");
    return {
      webgl: true,
      renderer: extension ? String(gl.getParameter(extension.UNMASKED_RENDERER_WEBGL)) : "masked",
      vendor: extension ? String(gl.getParameter(extension.UNMASKED_VENDOR_WEBGL)) : "masked",
      version: String(gl.getParameter(gl.VERSION)),
    };
  });
}


function summarizeResources(samples) {
  const rssValues = samples.map((sample) => sample.serverRssBytes).filter(Number.isFinite);
  const heapSamples = samples.filter((sample) => Number.isFinite(sample.jsHeap?.used));
  const baselineRssWindow = samples.filter((sample) => sample.elapsedS >= 30 && sample.elapsedS <= 90)
    .map((sample) => sample.serverRssBytes);
  const endRssWindow = samples.slice(-3).map((sample) => sample.serverRssBytes);
  const baselineHeapWindow = heapSamples.filter((sample) => sample.elapsedS >= 30 && sample.elapsedS <= 90)
    .map((sample) => sample.jsHeap.used);
  const endHeapWindow = heapSamples.slice(-3).map((sample) => sample.jsHeap.used);
  const rssBaseline = median(baselineRssWindow.length ? baselineRssWindow : rssValues.slice(0, 3));
  const rssEnd = median(endRssWindow);
  const heapValues = heapSamples.map((sample) => sample.jsHeap.used);
  const heapBaseline = median(baselineHeapWindow.length ? baselineHeapWindow : heapValues.slice(0, 3));
  const heapEnd = median(endHeapWindow);
  return {
    server_rss: {
      available: rssValues.length > 0,
      baseline_bytes: rssBaseline,
      end_bytes: rssEnd,
      maximum_bytes: rssValues.length ? Math.max(...rssValues) : null,
      maximum_growth_bytes: rssValues.length && rssBaseline != null ? Math.max(...rssValues) - rssBaseline : null,
      end_growth_bytes: rssEnd != null && rssBaseline != null ? rssEnd - rssBaseline : null,
    },
    js_heap: {
      available: heapValues.length > 0,
      baseline_bytes: heapBaseline,
      end_bytes: heapEnd,
      maximum_bytes: heapValues.length ? Math.max(...heapValues) : null,
      maximum_growth_bytes: heapValues.length && heapBaseline != null ? Math.max(...heapValues) - heapBaseline : null,
      end_growth_bytes: heapEnd != null && heapBaseline != null ? heapEnd - heapBaseline : null,
      limit_bytes: heapSamples.length ? heapSamples.at(-1).jsHeap.limit : null,
    },
  };
}


async function run() {
  const port = await isolatedPort();
  gate("isolated server does not use live dashboard port 8770", port !== 8770, `port ${port}`);
  const baseUrl = `http://127.0.0.1:${port}`;
  report.isolation.base_url = baseUrl;
  const { child, output } = startCommandCenter(port);
  report.isolation.server_pid = child.pid;
  let browser;
  let context;
  let page;
  let sessionId = "";
  let soakStartedMs = 0;
  try {
    const initialCatalog = await waitForCommandCenter(baseUrl, child, output);
    gate("isolated catalog starts with zero sessions", initialCatalog.sessions?.active === 0,
      `active=${initialCatalog.sessions?.active}`);

    browser = await chromium.launch({
      headless: false,
      args: [
        "--enable-webgl",
        "--ignore-gpu-blocklist",
        "--use-angle=metal",
        "--enable-precise-memory-info",
        "--window-size=1280,720",
      ],
    });
    report.browser.version = browser.version();
    report.browser.executable = chromium.executablePath();
    report.browser.headed = true;
    report.browser.viewport = "1280x720";
    context = await browser.newContext({
      viewport: { width: 1280, height: 720 },
      deviceScaleFactor: 1,
      locale: "en-US",
    });
    await context.addInitScript(soakCaptureInit);
    page = await context.newPage();
    page.on("console", (message) => {
      if (message.type() === "error") report.console_errors.push(message.text());
    });
    page.on("pageerror", (error) => report.page_errors.push(error.stack || error.message));

    await page.route("**/api/observatory/sessions", async (route) => {
      const request = route.request();
      if (request.method() !== "POST") return route.continue();
      const payload = JSON.parse(request.postData() || "{}");
      return route.continue({
        postData: JSON.stringify({ ...payload, seed: 7, audio: false }),
        headers: { ...request.headers(), "content-type": "application/json" },
      });
    });

    const sessionResponse = page.waitForResponse(
      (response) => response.url().endsWith("/api/observatory/sessions")
        && response.request().method() === "POST",
      { timeout: 30000 },
    );
    await page.goto(`${baseUrl}/observatory/?edition=919&car=mazda787b&checkpoint=active-best&mode=replay`, {
      waitUntil: "domcontentloaded",
      timeout: 30000,
    });
    const response = await sessionResponse;
    const body = await response.json();
    sessionId = String(body.session_id || "");
    report.session = {
      id: sessionId,
      status: response.status(),
      checkpoint: body.resolved_checkpoint || null,
      track_hash: body.track_hash || "",
      audio_socket: body.audio_socket ?? null,
    };
    gate("one browser session is created with audio disabled",
      response.status() === 201 && Boolean(sessionId) && body.audio_socket == null,
      `HTTP ${response.status()}, ${sessionId.slice(0, 8)}`);

    await page.waitForFunction(() => document.getElementById("loading-card")?.dataset.hidden === "true", null, { timeout: 30000 });
    await page.waitForFunction(() => document.getElementById("connection-state")?.dataset.state === "live", null, { timeout: 30000 });
    await page.waitForFunction(() => window.__observatorySoak?.frameCount >= 10, null, { timeout: 30000 });
    await page.locator("#control-speed").selectOption("1");
    await page.waitForFunction(() => document.getElementById("control-speed")?.value === "1", null, { timeout: 10000 });

    const renderer = await rendererInfo(page);
    report.browser.webgl = renderer;
    gate("headed Chromium uses a hardware WebGL renderer",
      renderer.webgl && /metal|apple/i.test(`${renderer.renderer} ${renderer.vendor}`)
        && !/swiftshader|software/i.test(renderer.renderer),
      `${renderer.vendor} / ${renderer.renderer}`);

    const starting = await pageSample(page);
    report.identity = {
      edition: starting.edition,
      car: starting.car,
      policy_hash: starting.policyHash,
    };
    gate("soak resolves the 919 registry identity and immutable policy hash",
      starting.edition === "919"
        && starting.car === "PORSCHE_919EVO"
        && /^[a-f0-9]{64}$/i.test(starting.policyHash),
      `${starting.edition}/${starting.car}/${starting.policyHash.slice(0, 12)}`);
    gate("playback is explicitly fixed at 1x", starting.speed === "1");

    const oneSession = await catalog(baseUrl);
    gate("exactly one isolated session is active before the soak",
      oneSession.sessions?.active === 1, `active=${oneSession.sessions?.active}`);

    soakStartedMs = Date.now();
    print(`SOAK_START duration=${TARGET_SECONDS}s sample=${SAMPLE_SECONDS}s server_pid=${child.pid} port=${port}`);
    let nextSampleS = 0;
    while ((Date.now() - soakStartedMs) / 1000 < TARGET_SECONDS) {
      const elapsedS = (Date.now() - soakStartedMs) / 1000;
      const waitMs = Math.max(0, Math.min(1000, (nextSampleS - elapsedS) * 1000));
      if (waitMs > 0) await page.waitForTimeout(waitMs);
      const nowS = (Date.now() - soakStartedMs) / 1000;
      if (nowS + 0.001 < nextSampleS) continue;
      if (child.exitCode != null) throw new Error(`isolated server exited during soak with ${child.exitCode}`);
      const browserSample = await pageSample(page);
      const sampleCatalog = await catalog(baseUrl);
      const rss = processRssBytes(child.pid);
      const sample = {
        elapsedS: Number(nowS.toFixed(3)),
        wallTimestamp: new Date().toISOString(),
        frameCount: browserSample.telemetry.frameCount,
        firstSequence: browserSample.telemetry.firstSequence,
        lastSequence: browserSample.telemetry.lastSequence,
        firstSimTime: browserSample.telemetry.firstSimTime,
        lastSimTime: browserSample.telemetry.lastSimTime,
        monotonicViolations: browserSample.telemetry.monotonicViolations,
        sequenceGapCount: browserSample.telemetry.sequenceGapCount,
        maximumSequenceGap: browserSample.telemetry.maximumSequenceGap,
        parseErrors: browserSample.telemetry.parseErrors,
        transportErrors: browserSample.telemetry.transportErrors,
        openTelemetrySockets: browserSample.telemetry.openTelemetrySockets,
        connection: browserSample.connection,
        replayClock: browserSample.replayClock,
        speed: browserSample.speed,
        policyHash: browserSample.policyHash,
        activeSessions: sampleCatalog.sessions?.active,
        serverRssBytes: rss,
        jsHeap: browserSample.jsHeap,
        quality: browserSample.quality,
        canvas: browserSample.canvas,
        serverAlive: child.exitCode == null,
        pageErrors: report.page_errors.length,
        consoleErrors: report.console_errors.length,
      };
      report.samples.push(sample);
      print(`PROGRESS elapsed=${sample.elapsedS.toFixed(0)}s frames=${sample.frameCount} sim=${Number(sample.lastSimTime || 0).toFixed(2)}s rss=${fmtMib(rss)} heap=${fmtMib(sample.jsHeap?.used)} link=${sample.connection} sessions=${sample.activeSessions} errors=${sample.pageErrors + sample.consoleErrors}`);
      nextSampleS += SAMPLE_SECONDS;
      if (nextSampleS > TARGET_SECONDS) nextSampleS = TARGET_SECONDS;
      if (nowS >= TARGET_SECONDS) break;
    }

    const wallDurationS = (Date.now() - soakStartedMs) / 1000;
    const final = await pageSample(page);
    const finalCatalog = await catalog(baseUrl);
    const firstSimTime = Number(final.telemetry.firstSimTime);
    const lastSimTime = Number(final.telemetry.lastSimTime);
    const simAdvanceS = lastSimTime - firstSimTime;
    const frameRateHz = final.telemetry.frameCount / wallDurationS;
    const simToWall = simAdvanceS / wallDurationS;
    report.telemetry = {
      ...final.telemetry,
      wall_duration_s: wallDurationS,
      sim_advance_s: simAdvanceS,
      frame_rate_hz: frameRateHz,
      sim_to_wall_ratio: simToWall,
      final_connection: final.connection,
      final_speed: final.speed,
      final_policy_hash: final.policyHash,
      active_sessions_before_release: finalCatalog.sessions?.active,
    };
    report.resources = summarizeResources(report.samples);

    gate("soak runs for the full requested wall duration",
      wallDurationS >= TARGET_SECONDS,
      `${wallDurationS.toFixed(3)}s / ${TARGET_SECONDS}s`);
    gate("telemetry sustains at least the conservative 10 Hz floor",
      frameRateHz >= report.thresholds.minimum_frame_rate_hz,
      `${final.telemetry.frameCount} frames, ${frameRateHz.toFixed(2)} Hz`);
    gate("1x simulation advances at least 90% of wall time",
      simToWall >= report.thresholds.minimum_sim_to_wall_ratio,
      `${simAdvanceS.toFixed(3)}s sim / ${wallDurationS.toFixed(3)}s wall (${simToWall.toFixed(4)}x)`);
    gate("all sampled frames have strictly monotonic sequence and simulation time",
      final.telemetry.monotonicViolations === 0,
      `violations=${final.telemetry.monotonicViolations}`);
    gate("telemetry remains parseable and transport-error free",
      final.telemetry.parseErrors === 0 && final.telemetry.transportErrors.length === 0,
      `parse=${final.telemetry.parseErrors}, transport=${final.telemetry.transportErrors.join(" | ")}`);
    gate("one telemetry socket remains live without reconnect",
      final.telemetry.telemetrySockets === 1
        && final.telemetry.openTelemetrySockets === 1
        && final.telemetry.closeEvents === 0
        && final.connection === "live",
      `created=${final.telemetry.telemetrySockets}, open=${final.telemetry.openTelemetrySockets}, closes=${final.telemetry.closeEvents}`);
    gate("every sample keeps exactly one server session, live link, 1x speed, and stable policy",
      report.samples.every((sample) => sample.activeSessions === 1
        && sample.connection === "live"
        && sample.speed === "1"
        && sample.serverAlive
        && sample.policyHash === report.identity.policy_hash),
      `${report.samples.length} samples`);
    gate("browser records zero page and console errors during the soak",
      report.page_errors.length === 0 && report.console_errors.length === 0,
      [...report.page_errors, ...report.console_errors].join(" | "));

    const rss = report.resources.server_rss;
    gate("server RSS remains available and below the absolute ceiling",
      rss.available && rss.maximum_bytes <= report.thresholds.maximum_server_rss_bytes,
      `baseline=${fmtMib(rss.baseline_bytes)}, max=${fmtMib(rss.maximum_bytes)}, end=${fmtMib(rss.end_bytes)}`);
    gate("server RSS has no material peak or end-of-run growth",
      rss.maximum_growth_bytes <= report.thresholds.maximum_server_rss_growth_bytes
        && rss.end_growth_bytes <= report.thresholds.maximum_server_rss_end_growth_bytes,
      `peak growth=${fmtMib(rss.maximum_growth_bytes)}, end growth=${fmtMib(rss.end_growth_bytes)}`);

    const heap = report.resources.js_heap;
    if (heap.available) {
      gate("JS heap has no material peak or end-of-run growth",
        heap.maximum_growth_bytes <= report.thresholds.maximum_js_heap_growth_bytes
          && heap.end_growth_bytes <= report.thresholds.maximum_js_heap_end_growth_bytes,
        `baseline=${fmtMib(heap.baseline_bytes)}, peak growth=${fmtMib(heap.maximum_growth_bytes)}, end growth=${fmtMib(heap.end_growth_bytes)}`);
      gate("JS heap stays below 80% of its reported limit",
        heap.maximum_bytes <= heap.limit_bytes * report.thresholds.maximum_js_heap_limit_fraction,
        `max=${fmtMib(heap.maximum_bytes)}, limit=${fmtMib(heap.limit_bytes)}`);
    } else {
      report.checks.push({ name: "JS heap sampling", pass: true, detail: "performance.memory unavailable; non-gating by contract" });
      print("  [PASS] JS heap sampling — performance.memory unavailable; non-gating by contract");
    }

    await page.evaluate(() => {
      const event = typeof PageTransitionEvent === "function"
        ? new PageTransitionEvent("pagehide", { persisted: false })
        : new Event("pagehide");
      window.dispatchEvent(event);
    });
    await poll(async () => (await catalog(baseUrl)).sessions?.active === 0, {
      timeoutMs: 10000,
      label: "pagehide session release",
    });
    const releasedCatalog = await catalog(baseUrl);
    report.release = {
      method: "pagehide",
      active_sessions: releasedCatalog.sessions?.active,
      released_at: new Date().toISOString(),
    };
    gate("pagehide cleanly releases the soak session",
      releasedCatalog.sessions?.active === 0,
      `active=${releasedCatalog.sessions?.active}`);
    sessionId = "";
  } finally {
    if (sessionId) {
      try {
        await fetch(`${baseUrl}/api/observatory/sessions/${encodeURIComponent(sessionId)}`, { method: "DELETE" });
      } catch { /* isolated child shutdown is the final fallback */ }
    }
    if (context) await context.close().catch(() => {});
    if (browser) await browser.close().catch(() => {});
    report.server_log_tail = output.join("").slice(-32768);
    await stopCommandCenter(child);
    report.isolation.server_exit_code = child.exitCode;
  }
}


try {
  await run();
  report.passed = report.checks.every((check) => check.pass);
} catch (error) {
  report.passed = false;
  report.error = error.stack || error.message || String(error);
  print(`SOAK_FAILED ${error.message || error}`);
} finally {
  report.finished_at = new Date().toISOString();
  fs.writeFileSync(RESULT_PATH, `${JSON.stringify(report, null, 2)}\n`);
  print(`SOAK_REPORT ${RESULT_PATH}`);
}

if (!report.passed) process.exitCode = 1;
