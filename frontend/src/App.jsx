import { useCallback, useEffect, useState } from "react";
import { NavLink, Route, Routes, useLocation } from "react-router-dom";
import { TimeProvider, ToastProvider, useTime, useToast } from "./lib/context.jsx";
import { useForecastData } from "./lib/useForecastData.js";
import { dur } from "./lib/format.js";
import Forecast from "./views/Forecast.jsx";
import Results from "./views/Results.jsx";
import Analyze from "./views/Analyze.jsx";
import About from "./views/About.jsx";

const Icon = ({ d }) => (
  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">{d}</svg>
);
const ICONS = {
  pulse: <path d="M3 12h4l2-6 4 12 2-6h6" />,
  cpu: <><rect x="6" y="6" width="12" height="12" rx="2" /><path d="M9 2v4M15 2v4M9 18v4M15 18v4M2 9h4M2 15h4M18 9h4M18 15h4" /></>,
  upload: <path d="M12 16V4M7 9l5-5 5 5M4 20h16" />,
  info: <><circle cx="12" cy="12" r="9" /><path d="M12 11v6M12 7.5v.01" /></>,
  refresh: <path d="M20 11a8 8 0 1 0-2.3 5.7M20 4v7h-7" />,
  moon: <path d="M20 14.5A8 8 0 0 1 9.5 4 8 8 0 1 0 20 14.5z" />,
  sun: <><circle cx="12" cy="12" r="4" /><path d="M12 2v2M12 20v2M2 12h2M20 12h2M5 5l1.4 1.4M17.6 17.6L19 19M5 19l1.4-1.4M17.6 6.4L19 5" /></>,
};
const ROUTES = [
  { path: "/", title: "Forecast", sub: "Solar flare nowcast and 6-hour M/X outlook", icon: "pulse" },
  { path: "/results", title: "Results", sub: "What was measured, and what was not", icon: "cpu" },
  { path: "/analyze", title: "Analyze", sub: "Look at your own X-ray file", icon: "upload" },
  { path: "/about", title: "About", sub: "Method, flare classes and limits", icon: "info" },
];

function Shell() {
  const toast = useToast();
  const { tz, setTz } = useTime();
  const { pathname } = useLocation();
  const [theme, setTheme] = useState(() => document.documentElement.dataset.theme || "dark");
  const onFallback = useCallback(() => toast("Could not reach the forecast data, so the demo is shown."), [toast]);
  const store = useForecastData(onFallback);
  const route = ROUTES.find((r) => r.path === pathname) || ROUTES[0];
  const onForecast = route.path === "/";
  const D = store.data;

  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    try { localStorage.setItem("flarenet-theme", theme); } catch { /* ignore */ }
  }, [theme]);
  useEffect(() => { document.title = `${route.title} · FlareNet`; window.scrollTo(0, 0); }, [route.title]);
  useEffect(() => {
    const onKey = (e) => {
      if (e.metaKey || e.ctrlKey || e.altKey || /^(INPUT|TEXTAREA|SELECT)$/.test(e.target.tagName)) return;
      if (e.key === "t") setTheme((t) => (t === "dark" ? "light" : "dark"));
      if (e.key === "r" && onForecast) store.reload();
    };
    addEventListener("keydown", onKey);
    return () => removeEventListener("keydown", onKey);
  }, [onForecast, store]);

  return (
    <>
      <a className="skip" href="#view">Skip to content</a>
      <div className="app">
        <aside>
          <NavLink className="brand" to="/" aria-label="FlareNet home">
            <svg viewBox="0 0 40 40" aria-hidden="true"><rect width="40" height="40" rx="11" fill="#161e2e" /><circle cx="20" cy="20" r="7" fill="#ffb224" /><g stroke="#ffb224" strokeWidth="2.4" strokeLinecap="round"><path d="M20 6v4M20 30v4M6 20h4M30 20h4M10 10l2.8 2.8M27.2 27.2L30 30M10 30l2.8-2.8M27.2 12.8L30 10" /></g></svg>
            <span><b>FlareNet</b><small>Solar flare forecasting</small></span>
          </NavLink>
          <nav aria-label="Main">
            {ROUTES.map((r) => (
              <NavLink key={r.path} to={r.path} end>
                <Icon d={ICONS[r.icon]} /><span>{r.title}</span>
              </NavLink>
            ))}
          </nav>
          <div className="sfoot" aria-live="polite">
            {D ? <><strong>{D.demo ? "Demo data" : D.model.name || "FlareNet model"}</strong>{D.demo ? "simulated, not the real model" : `window ${D.model.window_min} min · horizon ${dur(D.model.horizon_min)}`}</> : <strong>Loading…</strong>}
          </div>
        </aside>

        <div className="main">
          <header className="topbar">
            <h1>{route.title}<small>{route.sub}</small></h1>
            <div className="actions">
              {onForecast && store.updated && !store.loading && <span className="muted num" style={{ fontSize: 12 }}>Loaded {new Date(store.updated).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}</span>}
              {onForecast && (
                <div className="seg" role="radiogroup" aria-label="Data source">
                  <button role="radio" aria-checked={store.src === "latest"} onClick={() => store.setSrc("latest")}>Latest</button>
                  <button role="radio" aria-checked={store.src === "demo"} onClick={() => store.setSrc("demo")}>Demo</button>
                </div>
              )}
              <div className="seg" role="group" aria-label="Time zone">
                <button aria-pressed={tz === "UTC"} onClick={() => setTz("UTC")}>UTC</button>
                <button aria-pressed={tz !== "UTC"} onClick={() => setTz("local")}>Local</button>
              </div>
              {onForecast && (
                <button className={`btn icon ${store.loading ? "spin" : ""}`} onClick={store.reload} aria-label="Reload data" title="Reload (R)"><Icon d={ICONS.refresh} /></button>
              )}
              <button className="btn icon" onClick={() => setTheme(theme === "dark" ? "light" : "dark")} aria-label="Switch theme" title="Theme (T)"><Icon d={theme === "dark" ? ICONS.sun : ICONS.moon} /></button>
            </div>
          </header>
          <main id="view" tabIndex={-1}>
            <Routes>
              <Route path="/" element={<Forecast store={store} />} />
              <Route path="/results" element={<Results />} />
              <Route path="/analyze" element={<Analyze />} />
              <Route path="/about" element={<About store={store} />} />
              <Route path="*" element={<Forecast store={store} />} />
            </Routes>
          </main>
        </div>
      </div>
    </>
  );
}

export default function App() {
  return (
    <TimeProvider>
      <ToastProvider>
        <Shell />
      </ToastProvider>
    </TimeProvider>
  );
}
