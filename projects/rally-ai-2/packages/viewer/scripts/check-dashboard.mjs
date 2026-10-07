/**
 * Assert the dashboard is fully wired: every section has a panel, every panel
 * has a file, and the control-room proxy still points somewhere real.
 *
 * `tsc` already catches a missing import. What it cannot catch is a section
 * added to `SectionId` that nobody put in the nav, or a nav item with no route
 * behind it — both compile fine and both show up as a dead link. This script
 * is the cheap structural guard for that class of drift.
 *
 * Deliberately dependency-free and browser-free so it can run in a pre-commit
 * hook or CI in well under a second.
 *
 *   npm run check:dashboard --prefix packages/viewer
 */

import { readFileSync, existsSync } from "node:fs";
import { dirname, join, relative } from "node:path";
import { fileURLToPath } from "node:url";

const VIEWER = dirname(dirname(fileURLToPath(import.meta.url)));
const DASH = join(VIEWER, "src", "dashboard");
const PANELS = join(DASH, "panels");

/** Files the shell cannot boot without. */
const CORE_FILES = [
  "types.ts",
  "nav.ts",
  "router.ts",
  "shell.ts",
  "registry.ts",
  "DashboardApp.ts",
  "panels/index.ts",
];

const failures = [];
const notes = [];

function fail(message) {
  failures.push(message);
}

function read(path) {
  return readFileSync(path, "utf8");
}

function rel(path) {
  return relative(VIEWER, path);
}

// ---------------------------------------------------------------- core files

for (const name of CORE_FILES) {
  const path = join(DASH, name);
  if (!existsSync(path)) fail(`missing shell file: ${rel(path)}`);
}

if (failures.length > 0) {
  report();
}

// ------------------------------------------------------------------ sections

/**
 * `SectionId` is the source of truth for what routes exist. Parsing it beats
 * hardcoding the list here, which would just be a second thing to forget.
 */
function parseSectionIds(source) {
  const block = source.match(/export type SectionId\s*=([\s\S]*?);/);
  if (!block) return [];
  return [...block[1].matchAll(/"([a-z][a-z0-9_-]*)"/g)].map((m) => m[1]);
}

const sections = parseSectionIds(read(join(DASH, "types.ts")));
if (sections.length === 0) {
  fail("could not parse SectionId union from src/dashboard/types.ts");
  report();
}
notes.push(`${sections.length} sections: ${sections.join(", ")}`);

// ----------------------------------------------------------------- nav cover

const navSource = read(join(DASH, "nav.ts"));
const navIds = [...navSource.matchAll(/\{\s*id:\s*"([a-z0-9_-]+)"/g)].map(
  (m) => m[1],
);

for (const id of sections) {
  if (!navIds.includes(id)) fail(`section "${id}" is not in NAV_SECTIONS`);
}
for (const id of navIds) {
  if (!sections.includes(id)) fail(`nav item "${id}" is not a SectionId`);
}

// ------------------------------------------------------------- panel wiring

const barrelSource = read(join(PANELS, "index.ts"));

/** `overview: mountOverview,` inside the PANEL_MOUNTS table. */
const mountTable = barrelSource.match(
  /PANEL_MOUNTS[^=]*=\s*\{([\s\S]*?)\n\};/,
);
if (!mountTable) {
  fail("could not parse PANEL_MOUNTS from src/dashboard/panels/index.ts");
  report();
}

const wiring = new Map(
  [...mountTable[1].matchAll(/^\s*([a-z0-9_-]+):\s*([A-Za-z0-9_]+)\s*,/gm)].map(
    (m) => [m[1], m[2]],
  ),
);

/** `import { mountOverview } from "./overview";` → mountOverview → overview.ts */
const importedFrom = new Map(
  [
    ...barrelSource.matchAll(
      /import\s*\{\s*([A-Za-z0-9_]+)\s*\}\s*from\s*"\.\/([A-Za-z0-9_-]+)"/g,
    ),
  ].map((m) => [m[1], m[2]]),
);

for (const id of sections) {
  const mount = wiring.get(id);
  if (!mount) {
    fail(`section "${id}" has no entry in PANEL_MOUNTS`);
    continue;
  }

  const moduleName = importedFrom.get(mount);
  if (!moduleName) {
    fail(`PANEL_MOUNTS."${id}" uses ${mount}, which the barrel never imports`);
    continue;
  }

  const panelPath = join(PANELS, `${moduleName}.ts`);
  if (!existsSync(panelPath)) {
    fail(`section "${id}" → missing panel file ${rel(panelPath)}`);
    continue;
  }

  const panelSource = read(panelPath);
  const exported = new RegExp(
    `export\\s+(?:async\\s+)?(?:function|const|let)\\s+${mount}\\b`,
  ).test(panelSource);
  if (!exported) {
    fail(`${rel(panelPath)} does not export ${mount}`);
    continue;
  }

  notes.push(`  #/${id.padEnd(12)} → panels/${moduleName}.ts (${mount})`);
}

// ------------------------------------------------------------- barrel loaded

/**
 * The table above is worthless if nobody imports it. This is the one link in
 * the chain `tsc` cannot see: dropping the import still compiles, and every
 * section silently degrades to the fallback stub at runtime.
 */
const appSource = read(join(DASH, "DashboardApp.ts"));
if (!/import\(\s*["']\.\/panels["']\s*\)|from\s+["']\.\/panels["']/.test(appSource)) {
  fail("DashboardApp.ts never imports ./panels — no panel would register");
} else if (!/registerAllPanels\s*\(/.test(appSource)) {
  fail("DashboardApp.ts imports ./panels but never calls registerAllPanels()");
}

// --------------------------------------------------------------- vite proxy

const viteConfig = read(join(VIEWER, "vite.config.ts"));
if (!/["']\/api["']\s*:/.test(viteConfig)) {
  fail("vite.config.ts no longer proxies /api");
} else if (!/127\.0\.0\.1:8765|localhost:8765/.test(viteConfig)) {
  fail("vite.config.ts /api proxy does not target the control room on :8765");
} else if (!/\bws\s*:\s*true/.test(viteConfig)) {
  // Metrics and live frames are WebSockets; without this the panels connect
  // to the Vite dev server itself and hang with no error.
  fail("vite.config.ts /api proxy is missing `ws: true`");
} else {
  notes.push("vite proxy: /api → http://127.0.0.1:8765 (ws enabled)");
}

report();

function report() {
  if (failures.length === 0) {
    console.log("dashboard wiring OK");
    for (const note of notes) console.log(note);
    process.exit(0);
  }
  console.error(`dashboard wiring FAILED (${failures.length})`);
  for (const message of failures) console.error(`  - ${message}`);
  process.exit(1);
}
