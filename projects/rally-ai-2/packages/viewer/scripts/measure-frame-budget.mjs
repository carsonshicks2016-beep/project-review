/**
 * Measure rolling mean + p95 frame time at each visual-benchmark route.
 * Viewport matches capture-benchmarks.mjs (1184×757). Budget: 16.6 ms p95.
 *
 * Routes stay paused (as they do for captures) so the number matches the
 * fixed-viewpoint cost, not free-running playback.
 *
 *   npm run frame-budget --prefix packages/viewer
 */

import { chromium } from "playwright";
import { createServer } from "vite";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const VIEWER = dirname(dirname(fileURLToPath(import.meta.url)));
const VIEWPORT = { width: 1184, height: 757 };
const BUDGET_MS = 16.6;
const SETTLE_MS = 500;

const server = await createServer({
  root: VIEWER,
  configFile: join(VIEWER, "vite.config.ts"),
  server: { port: 5276, strictPort: true },
  logLevel: "warn",
});
await server.listen();
const browser = await chromium.launch();
const page = await browser.newPage({
  viewport: VIEWPORT,
  deviceScaleFactor: 1,
  baseURL: `http://localhost:5276`,
});
page.on("pageerror", (err) => console.error("PAGEERROR", err.message));
page.on("console", (msg) => {
  if (msg.type() === "error") console.error("CONSOLE", msg.text());
});

try {
  await page.goto("/", { waitUntil: "load" });
  const benchmarks = await page.evaluate(async () => {
    const m = await import("/src/benchmarks.ts");
    return m.VISUAL_BENCHMARKS.map((b) => b.id);
  });

  console.log("| route | mean (ms) | p95 (ms) | samples |");
  console.log("|---|---:|---:|---:|");
  let worstP95 = 0;
  for (const id of benchmarks) {
    await page.goto(`/?benchmark=${id}`, { waitUntil: "load" });
    await page.waitForFunction(() => Boolean(window.watch?.frameBudget), null, {
      timeout: 20000,
    });
    await page.waitForTimeout(SETTLE_MS);
    await page.evaluate(() => {
      window.watch.playback.playing = false;
      window.watch.frameBudget.reset();
    });
    await page.waitForFunction(
      () => window.watch.frameBudget.count >= 120,
      null,
      { timeout: 30000 },
    );
    const stats = await page.evaluate(() => {
      const fb = window.watch.frameBudget;
      return { mean: fb.mean(), p95: fb.p95(), count: fb.count };
    });
    worstP95 = Math.max(worstP95, stats.p95);
    console.log(
      `| \`${id}\` | ${stats.mean.toFixed(2)} | ${stats.p95.toFixed(2)} | ${stats.count} |`,
    );
  }
  console.log(`\nworst p95: ${worstP95.toFixed(2)} ms (budget ${BUDGET_MS})`);
  if (worstP95 > BUDGET_MS) process.exitCode = 1;
} finally {
  await browser.close();
  await server.close();
}
