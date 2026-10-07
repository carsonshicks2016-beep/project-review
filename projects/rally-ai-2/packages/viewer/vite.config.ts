import { defineConfig } from "vite";

export default defineConfig({
  server: {
    port: 5173,
    proxy: {
      // Control room (Phase F) on :8765 — metrics WS + live/drive streams.
      "/api": {
        target: "http://127.0.0.1:8765",
        changeOrigin: true,
        ws: true,
      },
    },
  },
  build: { target: "es2022", outDir: "dist" },
});
