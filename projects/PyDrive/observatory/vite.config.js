import { defineConfig } from "vite";

// Served by Command Center at /observatory/ — every asset reference must be
// relative so the dist works behind that prefix with zero network deps.
export default defineConfig({
  base: "./",
  build: {
    target: "es2022",
    sourcemap: true,
    assetsInlineLimit: 0,
  },
  server: {
    strictPort: true,
    port: 5173,
  },
});
