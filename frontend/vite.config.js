import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// base "./" lets the built site run from any folder or sub-path; routing uses the URL hash, so no server rewrites are needed.
export default defineConfig({ base: "./", plugins: [react()] });
