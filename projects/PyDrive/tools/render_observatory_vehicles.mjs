#!/usr/bin/env node
/** Render close, repeatable QA views for both clean-room Observatory cars. */
import { spawn } from "node:child_process";
import fs from "node:fs";
import net from "node:net";
import path from "node:path";
import process from "node:process";
import { fileURLToPath } from "node:url";
import { chromium } from "playwright";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const stamp = new Date().toISOString().replace(/[-:]/g, "").replace(/\.\d+Z$/, "Z");
const output = path.join(ROOT, "runtime", "observatory-vehicle-qa", stamp);
const views = ["front", "front3q", "side", "rear3q", "rear", "top"];
const cars = ["mazda787b", "porsche_919evo"];
fs.mkdirSync(output, { recursive: true });

function port() {
  return new Promise((resolve, reject) => {
    const server = net.createServer();
    server.listen(0, "127.0.0.1", () => {
      const value = server.address().port;
      server.close((error) => error ? reject(error) : resolve(value));
    });
  });
}

const httpPort = await port();
const server = spawn("python3", ["-m", "http.server", String(httpPort), "--bind", "127.0.0.1"], {
  cwd: ROOT,
  stdio: ["ignore", "ignore", "pipe"],
});
let stderr = "";
server.stderr.on("data", (chunk) => { stderr += String(chunk); });

try {
  const browser = await chromium.launch({ headless: true });
  const page = await browser.newPage({ viewport: { width: 1600, height: 1000 }, deviceScaleFactor: 1 });
  for (const car of cars) {
    for (const view of views) {
      const url = `http://127.0.0.1:${httpPort}/tools/preview_observatory_vehicle.html?car=${car}&view=${view}`;
      await page.goto(url, { waitUntil: "networkidle", timeout: 30000 });
      await page.waitForFunction(() => window.vehicleQa?.ready === true, null, { timeout: 30000 });
      await page.screenshot({ path: path.join(output, `${car}-${view}.png`) });
    }
  }
  await browser.close();
  fs.writeFileSync(path.join(output, "results.json"), `${JSON.stringify({ status: "passed", output, cars, views }, null, 2)}\n`);
  console.log(output);
} finally {
  if (server.exitCode == null) server.kill("SIGTERM");
  if (stderr && server.exitCode && server.exitCode !== 0) process.stderr.write(stderr);
}
