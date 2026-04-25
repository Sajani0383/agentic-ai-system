import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/state": "http://127.0.0.1:8000",
      "/step": "http://127.0.0.1:8000",
      "/reset": "http://127.0.0.1:8000",
      "/scenario": "http://127.0.0.1:8000",
      "/llm": "http://127.0.0.1:8000"
    }
  }
});
