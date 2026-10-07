/**
 * Electron main process (M6 integration).
 *
 *  - Create the BrowserWindow that hosts the Three.js renderer (Agent D),
 *    wired to a preload bridge for IPC.
 *  - Best-effort spawn + supervise the Swift audio helper (Agent B); the
 *    renderer connects to its WebSocket directly for live reactivity.
 *  - When SPOTIFY_CLIENT_ID is set, host the Spotify service (Agent A), turn
 *    each track change into a WorldSpec via the generation core (Agent C), and
 *    push it to the renderer. With no credentials the renderer self-runs the
 *    demo world.
 */
import { app, BrowserWindow } from "electron";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import { spawn, type ChildProcess } from "node:child_process";
import { existsSync, readFileSync } from "node:fs";
import type { WorldSpec } from "../contracts";

const __dirname = dirname(fileURLToPath(import.meta.url));

/**
 * Minimal .env loader. electron-vite does not inject .env into the main
 * process at runtime, so we read it ourselves from the working directory.
 * Existing process.env values win (so shell overrides still work).
 */
function loadDotenv(): void {
  const envPath = join(process.cwd(), ".env");
  if (!existsSync(envPath)) return;
  for (const raw of readFileSync(envPath, "utf8").split("\n")) {
    const eq = raw.indexOf("=");
    if (eq < 0) continue;
    const key = raw.slice(0, eq).trim();
    if (!key || key.startsWith("#")) continue;
    let val = raw.slice(eq + 1).trim();
    const q = val[0];
    if (val.length >= 2 && (q === '"' || q === "'") && val.at(-1) === q) {
      val = val.slice(1, -1);
    }
    if (process.env[key] === undefined) process.env[key] = val;
  }
}

let win: BrowserWindow | null = null;
let helper: ChildProcess | null = null;
let stopSpotify: (() => void) | null = null;
let latestWorld: WorldSpec | null = null;
let rendererReady = false;

function sendWorld(spec: WorldSpec): void {
  latestWorld = spec;
  if (win && rendererReady) win.webContents.send("sonic:world", spec);
}

function sendStatus(mode: "spotify" | "demo", message?: string): void {
  if (win && rendererReady) win.webContents.send("sonic:status", { mode, message });
}

function createWindow(): void {
  win = new BrowserWindow({
    width: 1280,
    height: 800,
    backgroundColor: "#000000",
    show: false,
    webPreferences: {
      contextIsolation: true,
      nodeIntegration: false,
      preload: join(__dirname, "../preload/preload.cjs"),
    },
  });

  win.once("ready-to-show", () => {
    win?.show();
    console.log("[main] window ready — showing");
  });

  // Flush any world generated before the renderer finished loading.
  win.webContents.on("did-finish-load", () => {
    rendererReady = true;
    console.log("[main] renderer finished loading (demo world live)");
    if (latestWorld) sendWorld(latestWorld);
  });

  const devUrl = process.env["ELECTRON_RENDERER_URL"];
  if (devUrl) {
    void win.loadURL(devUrl);
  } else {
    void win.loadFile(join(__dirname, "../renderer/index.html"));
  }
}

/** Spawn the Swift audio helper if it's been built; otherwise skip silently. */
function startHelper(): void {
  // Visuals-only escape hatch (used for headless launch checks): skip the
  // native capture helper so no Screen Recording prompt appears.
  if (process.env["SONIC_NO_HELPER"]) {
    console.log("[main] SONIC_NO_HELPER set — skipping audio helper (visuals self-animate).");
    return;
  }
  const candidates = [
    join(process.cwd(), "helper/.build/release/AudioHelper"),
    join(process.cwd(), "helper/.build/debug/AudioHelper"),
  ];
  const bin = candidates.find((p) => existsSync(p));
  if (!bin) {
    console.log(
      "[main] audio helper not built — run `npm run helper:build`. Visuals will self-animate.",
    );
    return;
  }
  helper = spawn(bin, [], { stdio: ["ignore", "inherit", "inherit"] });
  helper.on("error", (e) => console.warn("[main] audio helper failed to start:", e.message));
  helper.on("exit", (code) => {
    if (code) console.log(`[main] audio helper exited with code ${code}`);
    helper = null;
  });
}

/** Start Spotify → generation → renderer, or fall back to demo mode. */
async function startMusicPipeline(): Promise<void> {
  if (!process.env["SPOTIFY_CLIENT_ID"]) {
    sendStatus("demo", "Set SPOTIFY_CLIENT_ID (see .env.example) to drive worlds from your music.");
    return;
  }
  try {
    // Loaded lazily so demo mode never pulls in the Spotify/auth stack.
    const { createSpotifyService } = await import("../spotify/spotifyService");
    const { generateWorld } = await import("../generation/generationCore");
    const service = createSpotifyService((ctx) => {
      try {
        sendWorld(generateWorld(ctx));
      } catch (err) {
        console.error("[main] world generation failed:", err);
      }
    });
    stopSpotify = () => service.stop();
    await service.start();
    sendStatus("spotify");
    console.log("[main] Spotify pipeline running.");
  } catch (err) {
    console.error("[main] Spotify pipeline failed; staying in demo mode:", err);
    sendStatus("demo", err instanceof Error ? err.message : String(err));
  }
}

app.whenReady().then(() => {
  loadDotenv();
  createWindow();
  startHelper();
  void startMusicPipeline();
  app.on("activate", () => {
    if (BrowserWindow.getAllWindows().length === 0) createWindow();
  });
});

app.on("window-all-closed", () => {
  if (process.platform !== "darwin") app.quit();
});

app.on("will-quit", () => {
  stopSpotify?.();
  helper?.kill();
});
