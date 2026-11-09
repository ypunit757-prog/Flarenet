import { useCallback, useEffect, useMemo, useState } from "react";
import { NavLink, Route, Routes, useLocation } from "react-router-dom";
import { TimeProvider, ToastProvider, useTime, useToast } from "./lib/context.jsx";
import { useForecastData } from "./lib/useForecastData.js";
import { summarize } from "./lib/summarize.js";
import Forecast from "./views/Forecast.jsx";
import Results from "./views/Results.jsx";
import Analyze from "./views/Analyze.jsx";
import About from "./views/About.jsx";

const ROUTES = [
  { path: "/", title: "Forecast" },
  { path: "/results", title: "Results" },
  { path: "/analyze", title: "Analyze" },
  { path: "/about", title: "About" },
];
const STATE_WORD = { alert: "Alert", watch: "Watch", quiet: "Quiet", demo: "Demo", expired: "Out of date" };

/** UTC time to the second, as on a ground-station display. Not announced to screen readers. */
function Clock() {
  const [now, setNow] = useState(() => new Date());
  useEffect(() => { const id = setInterval(() => setNow(new Date()), 1000); return () => clearInterval(id); }, []);
  return <span className="clock num" aria-hidden="true"><span>UTC</span> {now.toISOString().slice(11, 19)}</span>;
}

function Shell() {
  const toast = useToast();
  const { tz, setTz } = useTime();
  const { pathname } = useLocation();
  const [theme, setTheme] = useState(() => document.documentElement.dataset.theme || "dark");
  const onFallback = useCallback(() => toast("Could not reach the forecast data, so the demo is shown."), [toast]);
  const store = useForecastData(onFallback);
  const route = ROUTES.find((r) => r.path === pathname) || ROUTES[0];
  const onForecast = route.path === "/";
  const state = useMemo(() => { try { return store.data ? summarize(store.data).state : null; } catch { return null; } }, [store.data]);

  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    try { localStorage.setItem("flarenet-theme", theme); } catch { /* ignore */ }
  }, [theme]);
  // the tab title carries the live state, so a pinned tab still tells you something
  useEffect(() => { document.title = `${onForecast && state ? STATE_WORD[state] : route.title} · FlareNet`; }, [route.title, onForecast, state]);
  useEffect(() => { window.scrollTo(0, 0); }, [route.path]);
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
      <header className="top">
        <div className="wrap">
          <NavLink className="mark" to="/" aria-label="FlareNet, forecast">
            <svg viewBox="0 0 30 22" fill="none" stroke="currentColor" strokeWidth="2.6" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true"><path d="M1.5 18H9.5L14 3.5L18.5 18H28.5" /></svg>
            <b>FlareNet</b>
          </NavLink>
          <nav className="tabs" aria-label="Main">
            {ROUTES.map((r) => <NavLink key={r.path} to={r.path} end>{r.title}</NavLink>)}
          </nav>
          <div className="tools">
            <Clock />
            <button className="tool" onClick={() => setTz(tz === "UTC" ? "local" : "UTC")} title="Switch between UTC and your local time" aria-label={`Times are in ${tz === "UTC" ? "UTC" : "your local time"}. Switch.`}>{tz === "UTC" ? "UTC" : "Local time"}</button>
            <button className="tool" onClick={() => setTheme(theme === "dark" ? "light" : "dark")} title="Light or dark theme (T)" aria-label={`Switch to the ${theme === "dark" ? "light" : "dark"} theme`}>
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" aria-hidden="true"><circle cx="12" cy="12" r="8.5" /><path d="M12 3.5a8.5 8.5 0 0 1 0 17z" fill="currentColor" stroke="none" /></svg>
            </button>
          </div>
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
      <footer className="foot">
        <div className="wrap">
          <p>FlareNet is a research prototype. For official space weather alerts use <a href="https://www.swpc.noaa.gov/" target="_blank" rel="noopener noreferrer">NOAA SWPC</a>. <a href="https://github.com/ypunit757-prog/Flarenet" target="_blank" rel="noopener noreferrer">Source code on GitHub</a>.</p>
        </div>
      </footer>
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
