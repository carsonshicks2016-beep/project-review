/**
 * Captures every fixed visual benchmark to docs/visual-benchmarks/.
 *
 * The benchmark list is read out of the running dev server rather than
 * duplicated here, so `src/benchmarks.ts` stays the single source of the
 * replay, timestamp and camera each capture uses.
 *
 * Determinism: benchmark routes pause playback, so particles, tracks and car
 * effects are frozen at their prewarmed state. The chase camera still smooths
 * on real frame time, so each route is given a fixed settle period before the
 * shot. Run with --verify to capture twice and fail on any pixel difference.
 */

import { mkdir, readdir, unlink, writeFile } from "node:fs/promises";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

import { chromium } from "playwright";
import { createServer } from "vite";

const VIEWER = dirname(dirname(fileURLToPath(import.meta.url)));
const OUT = join(VIEWER, "..", "..", "docs", "visual-benchmarks");

/** The viewport every checked-in capture uses. Changing it invalidates them all. */
const VIEWPORT = { width: 1184, height: 757 };

/** Frames of camera smoothing to burn before the shot. */
const SETTLE_MS = 1500;

const verify = process.argv.includes("--verify");

async function shoot(page, id) {
  await page.goto(`/?benchmark=${id}`, { waitUntil: "load" });
  await page.waitForFunction(() => Boolean(window.watch), null, {
    timeout: 15000,
  });
  await page.waitForTimeout(SETTLE_MS);
  return page.screenshot({ type: "png" });
}

async function main() {
  const server = await createServer({
    root: VIEWER,
    configFile: join(VIEWER, "vite.config.ts"),
    server: { port: 5273, strictPort: true },
    logLevel: "warn",
  });
  await server.listen();
  const base = `http://localhost:${server.config.server.port}`;

  const browser = await chromium.launch();
  const page = await browser.newPage({
    viewport: VIEWPORT,
    deviceScaleFactor: 1,
    baseURL: base,
  });

  const failures = [];
  try {
    await page.goto("/", { waitUntil: "load" });
    const benchmarks = await page.evaluate(async () => {
      const m = await import("/src/benchmarks.ts");
      return m.VISUAL_BENCHMARKS.map((b) => ({ id: b.id, purpose: b.purpose }));
    });

    await mkdir(OUT, { recursive: true });
    const expected = new Set(benchmarks.map((b) => `${b.id}.png`));
    for (const name of await readdir(OUT)) {
      if (name.endsWith(".png") && !expected.has(name)) {
        await unlink(join(OUT, name));
        console.log(`removed stale ${name}`);
      }
    }

    for (const benchmark of benchmarks) {
      const png = await shoot(page, benchmark.id);
      await writeFile(join(OUT, `${benchmark.id}.png`), png);
      let note = `${png.length} bytes`;
      if (verify) {
        const again = await shoot(page, benchmark.id);
        const same = again.equals(png);
        if (!same) failures.push(benchmark.id);
        note += same ? ", reproducible" : ", NOT REPRODUCIBLE";
      }
      console.log(`${benchmark.id}.png — ${note}`);
    }
  } finally {
    await browser.close();
    await server.close();
  }

  if (failures.length) {
    console.error(`\nnon-deterministic captures: ${failures.join(", ")}`);
    process.exitCode = 1;
  }
}

await main();
