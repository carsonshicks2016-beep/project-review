/**
 * Measure the procedural audio update's cost share of the 60 Hz frame budget.
 *
 * Plays each shipped replay for real (audio unlocked, transport running) and
 * reads `RallyAudio.costSample` — an EMA of the main-thread `update()` wall
 * time. Synthesis itself runs on the browser's audio rendering thread through
 * native nodes (oscillators, buffer loops, biquads, gains); the only JS the
 * audio adds per frame is this AudioParam automation, which is what's timed.
 *
 *   npm run audio-cost --prefix packages/viewer
 */

import { chromium } from "playwright";
import { createServer } from "vite";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const VIEWER = dirname(dirname(fileURLToPath(import.meta.url)));
const VIEWPORT = { width: 1184, height: 757 };
const REPLAYS = ["demo", "proving_ground"];
/** Seconds of playback to average over. */
const MEASURE_S = 10;
/** Fail if the audio update's EMA exceeds this share of 16.67 ms. */
const SHARE_BUDGET = 0.05;

// Watching is off so a source edit landing mid-measure cannot full-reload the
// page and wipe the dev handle halfway through a run.
const server = await createServer({
  root: VIEWER,
  configFile: join(VIEWER, "vite.config.ts"),
  server: {
    port: 5277,
    strictPort: true,
    watch: null,
    hmr: false,
  },
  logLevel: "warn",
});
await server.listen();
// No user gesture exists headless; let the AudioContext start so the full
// update path runs instead of early-returning while locked.
const browser = await chromium.launch({
  args: ["--autoplay-policy=no-user-gesture-required", "--mute-audio"],
});
const page = await browser.newPage({
  viewport: VIEWPORT,
  deviceScaleFactor: 1,
  baseURL: `http://localhost:5277`,
});
page.on("pageerror", (err) => console.error("PAGEERROR", err.message));
page.on("crash", () => console.error("PAGE CRASHED"));
page.on("console", (msg) => {
  if (msg.type() === "error") console.error("CONSOLE", msg.text());
});

try {
  console.log("| replay | audio ema (ms) | share of 16.67 ms | samples |");
  console.log("|---|---:|---:|---:|");
  let worstShare = 0;
  for (const id of REPLAYS) {
    await page.goto(`/?r=${id}`, { waitUntil: "load" });
    await page.waitForFunction(() => Boolean(window.watch?.audio), null, {
      timeout: 20000,
    });
    await page.evaluate(async () => {
      await window.watch.audio.unlock();
      window.watch.playback.seek(0);
      window.watch.playback.playing = true;
    });
    // Null-safe: a dev-server reload mid-wait resets window.watch, which must
    // read as "keep polling", not a predicate crash.
    await page.waitForFunction(
      (t) => (window.watch?.playback?.time ?? 0) >= t,
      MEASURE_S,
      { timeout: 60000 },
    );
    const cost = await page.evaluate(() => window.watch.audio.costSample);
    if (cost.samples === 0) {
      console.log(`| \`${id}\` | n/a | n/a | 0 |`);
      console.error(`  ${id}: AudioContext never unlocked — no cost samples`);
      process.exitCode = 1;
      continue;
    }
    worstShare = Math.max(worstShare, cost.frameShare);
    console.log(
      `| \`${id}\` | ${cost.emaMs.toFixed(3)} | ${(cost.frameShare * 100).toFixed(2)}% | ${cost.samples} |`,
    );
  }
  console.log(
    `\nworst audio share: ${(worstShare * 100).toFixed(2)}% ` +
      `(budget ${(SHARE_BUDGET * 100).toFixed(0)}%)`,
  );
  if (worstShare > SHARE_BUDGET) process.exitCode = 1;
} finally {
  await browser.close();
  await server.close();
}
