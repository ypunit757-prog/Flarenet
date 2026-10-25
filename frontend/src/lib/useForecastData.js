import { useCallback, useEffect, useRef, useState } from "react";
import demo from "../assets/demo-data.json";

const KEY = "flarenet-src";
const readSrc = () => { try { return localStorage.getItem(KEY) === "demo" ? "demo" : "latest"; } catch { return "latest"; } };

// Where the real payload comes from: the Render API when VITE_API_URL is set, otherwise a data.json next to the site.
const API = (import.meta.env.VITE_API_URL || "").replace(/\/+$/, "");
const LATEST_URL = API ? `${API}/api/forecast` : `${import.meta.env.BASE_URL}data.json`;

/**
 * Loads the exported payload. "latest" reads it from LATEST_URL; if that fails the hook falls back to the bundled
 * demo and tells the caller through onFallback.
 */
export function useForecastData(onFallback) {
  const [src, setSrcState] = useState(readSrc);
  const [state, setState] = useState({ data: null, loading: true, error: null, updated: null });
  const seq = useRef(0);

  const load = useCallback(async (source) => {
    const my = ++seq.current;
    setState((s) => ({ ...s, loading: true, error: null }));
    let data = demo, used = "demo";
    if (source === "latest") {
      try {
        const r = await fetch(LATEST_URL, { cache: "no-store" });
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        data = await r.json();
        used = "latest";
      } catch {
        onFallback?.();
      }
    }
    if (my !== seq.current) return;
    setState({ data, loading: false, error: null, updated: Date.now() });
    if (used !== source) { setSrcState(used); }
  }, [onFallback]);

  useEffect(() => { load(src); /* eslint-disable-next-line react-hooks/exhaustive-deps */ }, [src]);

  const setSrc = (s) => { try { localStorage.setItem(KEY, s); } catch { /* ignore */ } setSrcState(s); };
  return { ...state, src, setSrc, reload: () => load(src) };
}
