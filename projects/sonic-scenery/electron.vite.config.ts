import { resolve } from "node:path";
import { defineConfig, externalizeDepsPlugin } from "electron-vite";

/**
 * electron-vite configuration (M6).
 *
 * Three processes:
 *  - main:     Electron main process (src/main/main.ts) → dist/main
 *  - preload:  contextBridge IPC bridge (src/preload/preload.ts) → dist/preload/preload.cjs
 *  - renderer: the Three.js scene (src/renderer/index.html) → dist/renderer
 *
 * main + preload externalize node_modules (so heavy deps like node-vibrant are
 * required from disk, not bundled); the renderer bundles three for the browser.
 */
export default defineConfig({
  main: {
    plugins: [externalizeDepsPlugin()],
    build: {
      outDir: "dist/main",
      lib: {
        entry: resolve(__dirname, "src/main/main.ts"),
      },
      rollupOptions: {
        output: { format: "es" },
      },
    },
  },
  preload: {
    plugins: [externalizeDepsPlugin()],
    build: {
      outDir: "dist/preload",
      lib: {
        entry: resolve(__dirname, "src/preload/preload.ts"),
      },
      rollupOptions: {
        output: { format: "cjs", entryFileNames: "preload.cjs" },
      },
    },
  },
  renderer: {
    root: resolve(__dirname, "src/renderer"),
    build: {
      outDir: resolve(__dirname, "dist/renderer"),
      rollupOptions: {
        input: {
          index: resolve(__dirname, "src/renderer/index.html"),
        },
      },
    },
  },
});
