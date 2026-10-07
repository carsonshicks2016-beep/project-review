import { defineConfig } from "vite";

export default defineConfig({
  base: "./",
  build: {
    target: "es2022",
    sourcemap: true,
    assetsInlineLimit: 0,
    rollupOptions: {
      output: {
        manualChunks(id) {
          return id.includes("/node_modules/three/") ? "three" : undefined;
        },
      },
    },
  },
  server: {
    strictPort: true,
    port: 5173,
  },
});
