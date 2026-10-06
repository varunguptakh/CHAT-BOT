import fs from "node:fs";
import path from "node:path";
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

function apiTarget() {
  if (process.env.VITE_API_TARGET) return process.env.VITE_API_TARGET;
  const marker = path.resolve("..", ".api_url");
  if (fs.existsSync(marker)) return fs.readFileSync(marker, "utf8").trim();
  return "http://127.0.0.1:8000";
}

export default defineConfig({
  plugins: [react()],
  server: {
    host: "127.0.0.1",
    port: 5173,
    proxy: {
      "/api": {
        target: apiTarget(),
        changeOrigin: true,
      },
    },
  },
});
