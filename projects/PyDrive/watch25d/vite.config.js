import { defineConfig } from "vite";

// Served by Command Center at /watch25d/ — relative base so dist works
// behind that prefix with zero network deps (same pattern as Observatory).
export default defineConfig({
  base: "./",
  build: {
    target: "es2022",
    sourcemap: true,
    assetsInlineLimit: 0,
  },
  server: {
    strictPort: true,
    port: 5174,
  },
});
