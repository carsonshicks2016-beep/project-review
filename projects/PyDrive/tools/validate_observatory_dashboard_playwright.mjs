#!/usr/bin/env node
/** Browser wiring smoke for Command Center -> Fable Observatory launches. */
import { spawn } from "node:child_process";
import fs from "node:fs";
import net from "node:net";
import path from "node:path";
import process from "node:process";
import { fileURLToPath } from "node:url";
import { chromium } from "playwright";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const SERVER = path.join(ROOT, "command-center", "server.py");
const stamp = new Date().toISOString().replace(/[-:]/g, "").replace(/\.\d+Z$/, "Z");
const QA_DIR = path.join(ROOT, "runtime", "observatory-qa", `dashboard-${stamp}`);
fs.mkdirSync(QA_DIR, { recursive: true });

const report = {
  schema: "fable-observatory-dashboard-playwright-v1",
  started_at: new Date().toISOString(),
  checks: [],
  opened_urls: [],
};

function gate(name, condition, detail = "") {
  report.checks.push({ name, pass: Boolean(condition), detail });
  console.log(`  [${condition ? "PASS" : "FAIL"}] ${name}${detail ? ` — ${detail}` : ""}`);
  if (!condition) throw new Error(`gate failed: ${name}${detail ? ` (${detail})` : ""}`);
}

function allocatePort() {
  return new Promise((resolve, reject) => {
    const socket = net.createServer();
    socket.unref();
    socket.once("error", reject);
    socket.listen(0, "127.0.0.1", () => {
      const address = socket.address();
      socket.close((error) => error ? reject(error) : resolve(address.port));
    });
  });
}

function startServer(port) {
  const output = [];
  const child = spawn(process.env.PYTHON || "python3", [SERVER], {
    cwd: ROOT,
    env: { ...process.env, PORT: String(port), PYTHONUNBUFFERED: "1" },
    stdio: ["ignore", "pipe", "pipe"],
  });
  const collect = (chunk) => {
    output.push(String(chunk));
    if (output.length > 300) output.splice(0, output.length - 300);
  };
  child.stdout.on("data", collect);
  child.stderr.on("data", collect);
  return { child, output };
}

async function stopServer(child) {
  if (!child || child.exitCode != null) return;
  child.kill("SIGTERM");
  const stopped = await Promise.race([
    new Promise((resolve) => child.once("exit", () => resolve(true))),
    new Promise((resolve) => setTimeout(() => resolve(false), 5000)),
  ]);
  if (!stopped && child.exitCode == null) child.kill("SIGKILL");
}

async function waitReady(baseUrl, child, output) {
  const deadline = Date.now() + 30000;
  while (Date.now() < deadline) {
    if (child.exitCode != null) throw new Error(`server exited ${child.exitCode}\n${output.join("")}`);
    try {
      const response = await fetch(`${baseUrl}/api/observatory/catalog`);
      if (response.ok) return response.json();
    } catch { /* startup */ }
    await new Promise((resolve) => setTimeout(resolve, 100));
  }
  throw new Error("isolated Command Center did not become ready");
}

function editionRecords(catalog) {
  return Array.isArray(catalog.editions)
    ? catalog.editions
    : Object.entries(catalog.editions || {}).map(([id, value]) => ({ id, ...value }));
}

function checkpointRecords(catalog) {
  return editionRecords(catalog).flatMap((edition) => {
    const records = edition.checkpoints || edition.state?.checkpoints || [];
    return records.map((checkpoint) => ({ ...checkpoint, edition: edition.id }));
  });
}

function checkpointName(record) {
  return String(record?.name || record?.checkpoint || record?.file || "");
}

function parseLaunch(baseUrl, raw) {
  const url = new URL(raw, baseUrl);
  return {
    raw,
    pathname: url.pathname,
    edition: url.searchParams.get("edition"),
    checkpoint: url.searchParams.get("checkpoint"),
    mode: url.searchParams.get("mode"),
  };
}

async function clickLaunch(page, baseUrl, selector) {
  const count = await page.evaluate(() => window.__observatoryDashboardQaOpened.length);
  await page.locator(selector).click();
  await page.waitForFunction(
    (previous) => window.__observatoryDashboardQaOpened.length > previous,
    count,
    { timeout: 10000 },
  );
  const raw = await page.evaluate(() => window.__observatoryDashboardQaOpened.at(-1).url);
  const launch = parseLaunch(baseUrl, raw);
  report.opened_urls.push(launch);
  return launch;
}

function gateLaunch(name, launch, { edition, checkpoint, mode }) {
  gate(name,
    launch.pathname === "/observatory/"
      && launch.edition === edition
      && launch.checkpoint === checkpoint
      && launch.mode === mode,
    launch.raw);
}

async function selectSegment(page, selector, editionId, index) {
  await page.locator(`${selector} button`).nth(index).click();
  await page.waitForFunction(
    ({ selector: control, edition }) => document.querySelector(control)?.dataset.val === edition,
    { selector, edition: editionId },
  );
}

async function run() {
  const port = await allocatePort();
  const baseUrl = `http://127.0.0.1:${port}`;
  report.base_url = baseUrl;
  const { child, output } = startServer(port);
  let browser;
  try {
    const catalog = await waitReady(baseUrl, child, output);
    const editions = editionRecords(catalog);
    gate("server catalogue exposes both Fable editions",
      editions.some((edition) => edition.id === "787b")
        && editions.some((edition) => edition.id === "919"));

    const checkpointResponse = await fetch(`${baseUrl}/api/checkpoints`);
    const checkpoints = await checkpointResponse.json();
    gate("checkpoint browser API is available", checkpointResponse.ok && Array.isArray(checkpoints));

    browser = await chromium.launch({ headless: true });
    const context = await browser.newContext({ viewport: { width: 1440, height: 1000 } });
    await context.addInitScript(() => {
      Object.defineProperty(window, "__observatoryDashboardQaOpened", {
        value: [], configurable: false, enumerable: false,
      });
      window.open = (url, target, features) => {
        window.__observatoryDashboardQaOpened.push({ url: String(url), target, features });
        return null;
      };
    });
    const page = await context.newPage();
    const consoleErrors = [];
    const pageErrors = [];
    page.on("console", (message) => {
      if (message.type() === "error") consoleErrors.push(message.text());
    });
    page.on("pageerror", (error) => pageErrors.push(error.stack || error.message));
    await page.goto(`${baseUrl}/`, { waitUntil: "domcontentloaded", timeout: 30000 });
    await page.waitForFunction(() => document.querySelectorAll("#fb-edition button").length >= 2, null, { timeout: 30000 });
    const v2Route = await fetch(`${baseUrl}/observatory-v2/`);
    gate("dashboard exposes no dead Observatory V2 controls or route",
      await page.locator("#fb-watch-best-3d-v2, #fb-watch-selected-3d-v2, #obs-open-v2-replay, #obs-open-v2-follow").count() === 0
        && v2Route.status === 404,
      `route HTTP ${v2Route.status}`);

    await page.locator('.tab[data-tab="fable"]').click();
    await page.waitForFunction(() => document.getElementById("tab-fable")?.classList.contains("active"));
    for (let index = 0; index < editions.length; index += 1) {
      const edition = editions[index];
      await selectSegment(page, "#fb-edition", edition.id, index);
      const launch = await clickLaunch(page, baseUrl, "#fb-watch-best-3d");
      gateLaunch(`Fable Watch Best 3D uses catalogue edition ${edition.id}`, launch, {
        edition: edition.id, checkpoint: "active-best", mode: "replay",
      });
    }

    await page.locator('.tab[data-tab="observatory"]').click();
    await page.waitForFunction(() => document.getElementById("tab-observatory")?.classList.contains("active"));
    await page.waitForFunction(() => document.querySelectorAll("#obs-edition button").length >= 2, null, { timeout: 30000 });
    for (let index = 0; index < editions.length; index += 1) {
      const edition = editions[index];
      await selectSegment(page, "#obs-edition", edition.id, index);
      gateLaunch(`Observatory hub Replay uses catalogue edition ${edition.id}`,
        await clickLaunch(page, baseUrl, "#obs-open-replay"), {
          edition: edition.id, checkpoint: "active-best", mode: "replay",
        });
      gateLaunch(`Observatory hub Follow uses catalogue edition ${edition.id}`,
        await clickLaunch(page, baseUrl, "#obs-open-follow"), {
          edition: edition.id, checkpoint: "active-best", mode: "follow-active-best",
        });
    }

    await page.locator('.tab[data-tab="checkpoints"]').click();
    await page.waitForFunction(() => document.querySelectorAll("#ck-list .ckrow").length > 0, null, { timeout: 30000 });
    const rows = await page.locator("#ck-list .ckrow").evaluateAll((nodes) => nodes.map((row) => ({
      name: (row.querySelector(".ck-name")?.textContent || "").replace(/\s+★\s*$/, "").trim(),
      view3d: [...row.querySelectorAll("button")].some((button) => button.textContent.trim() === "View in 3D"),
    })));
    const byName = new Map(rows.map((row) => [row.name, row]));
    const records = checkpointRecords(catalog);
    const playableNames = new Set(records
      .filter((record) => record.playable !== false && record.classification !== "incompatible")
      .map(checkpointName)
      .filter(Boolean));
    const playable = records.find((record) => {
      const name = checkpointName(record);
      return name && byName.has(name)
        && record.playable !== false && record.classification !== "incompatible";
    });
    gate("Garage gives a validated Fable checkpoint View in 3D",
      Boolean(playable && byName.get(checkpointName(playable))?.view3d), checkpointName(playable));

    const knownNames = new Set(records.map(checkpointName).filter(Boolean));
    const incompatible = records.find((record) => {
      const name = checkpointName(record);
      return name && byName.has(name)
        && !playableNames.has(name)
        && (record.playable === false || record.classification === "incompatible");
    });
    if (incompatible) {
      gate("Garage withholds View in 3D from an incompatible Fable checkpoint",
        !byName.get(checkpointName(incompatible))?.view3d, checkpointName(incompatible));
    }
    const nonFable = checkpoints.find((checkpoint) => checkpoint.name?.endsWith(".pt")
      && byName.has(checkpoint.name) && !knownNames.has(checkpoint.name));
    gate("Garage withholds View in 3D from a non-Fable checkpoint",
      Boolean(nonFable && !byName.get(nonFable.name)?.view3d), nonFable?.name || "none found");

    const validatedRow = page.locator("#ck-list .ckrow").filter({ hasText: checkpointName(playable) }).first();
    const validatedLaunch = await clickLaunch(page, baseUrl,
      `#ck-list .ckrow:has(.ck-name:text-is("${checkpointName(playable)}")) button:text-is("View in 3D")`)
      .catch(async () => {
        const count = await page.evaluate(() => window.__observatoryDashboardQaOpened.length);
        await validatedRow.getByRole("button", { name: "View in 3D", exact: true }).click();
        await page.waitForFunction((previous) => window.__observatoryDashboardQaOpened.length > previous, count);
        const raw = await page.evaluate(() => window.__observatoryDashboardQaOpened.at(-1).url);
        const parsed = parseLaunch(baseUrl, raw);
        report.opened_urls.push(parsed);
        return parsed;
      });
    gateLaunch("Garage View in 3D opens the validated checkpoint replay", validatedLaunch, {
      edition: playable.edition, checkpoint: checkpointName(playable), mode: "replay",
    });

    gate("dashboard smoke has zero page or console errors",
      pageErrors.length === 0 && consoleErrors.length === 0,
      [...pageErrors, ...consoleErrors].join(" | "));
    report.browser_version = browser.version();
    report.catalog_editions = editions.map((edition) => edition.id);
    report.validated_checkpoint = checkpointName(playable);
    report.incompatible_checkpoint = checkpointName(incompatible);
    report.non_fable_checkpoint = nonFable?.name || "";
    await context.close();

    report.status = "passed";
    report.finished_at = new Date().toISOString();
    const resultPath = path.join(QA_DIR, "results.json");
    fs.writeFileSync(resultPath, `${JSON.stringify(report, null, 2)}\n`);
    console.log(`Dashboard Observatory Playwright smoke OK\n  Results: ${resultPath}`);
  } catch (error) {
    report.status = "failed";
    report.finished_at = new Date().toISOString();
    report.error = error.stack || error.message;
    report.server_output_tail = output.join("").slice(-10000);
    const resultPath = path.join(QA_DIR, "results.failed.json");
    fs.writeFileSync(resultPath, `${JSON.stringify(report, null, 2)}\n`);
    console.error(`Dashboard smoke failure: ${resultPath}`);
    throw error;
  } finally {
    await browser?.close().catch(() => {});
    await stopServer(child);
  }
}

run().catch((error) => {
  console.error(error.stack || error);
  process.exitCode = 1;
});
