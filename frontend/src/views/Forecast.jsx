import { useEffect, useMemo, useState } from "react";
import Banner from "../components/Banner.jsx";
import Chart from "../components/Chart.jsx";
import Chip from "../components/Chip.jsx";
import Kpis from "../components/Kpis.jsx";
import { dur, iso, isMX, pct } from "../lib/format.js";
import { useTime } from "../lib/context.jsx";

function Skeleton() {
  return (
    <div className="stack">
      <div className="grid g4">
        {[0, 1, 2, 3].map((i) => (
          <div className="card" key={i}><div className="skel" style={{ height: 14, width: "55%" }} /><div className="skel" style={{ height: 44, width: "70%", marginTop: 12 }} /></div>
        ))}
      </div>
      <div className="card"><div className="skel" style={{ height: 330 }} /></div>
    </div>
  );
}

export default function Forecast({ store }) {
  const { data: D, loading, error, reload } = store;
  const { tz, fmtDT } = useTime();
  const [range, setRange] = useState("7d");
  const [zoom, setZoom] = useState(null);
  const [selected, setSelected] = useState(null);
  const [minCls, setMinCls] = useState("M");

  const end = D ? iso(D.period.end) : 0;
  const hours = D ? { "6h": 6, "24h": 24, "3d": 72, "7d": 168, all: D.period.days * 24 } : {};
  const avail = D ? Object.keys(hours).filter((k) => k === "all" || hours[k] <= D.period.days * 24 + 1) : [];
  const activeRange = avail.includes(range) ? range : avail.includes("7d") ? "7d" : "all";
  const dom = zoom || [end - (hours[activeRange] || 0) * 3600, end];

  useEffect(() => { setZoom(null); setSelected(null); }, [D]);

  const mx = useMemo(() => (D ? D.flares.filter(isMX) : []), [D]);
  const rows = useMemo(() => {
    if (!D) return [];
    return D.flares
      .filter((f) => (minCls === "X" ? f.class[0] === "X" : isMX(f)) && iso(f.peak) >= dom[0] && iso(f.peak) <= dom[1])
      .sort((a, b) => iso(b.peak) - iso(a.peak));
  }, [D, minCls, dom[0], dom[1]]); // eslint-disable-line react-hooks/exhaustive-deps

  if (loading && !D) return <Skeleton />;
  if (error || !D) {
    return (
      <div className="card empty">
        <h3>Could not load data</h3><p>{String(error?.message || "")}</p>
        <button className="btn primary" onClick={reload}>Try again</button>
      </div>
    );
  }

  const scored = rows.filter((f) => f.forecast !== "NO COVERAGE");
  const warned = scored.filter((f) => f.forecast === "WARNED").length;
  const alertRate = D.forecast.alert.reduce((a, b) => a + b, 0) / Math.max(1, D.forecast.alert.length);
  const hoursOld = (Date.now() / 1000 - end) / 3600;

  const pick = (f) => {
    const t = iso(f.peak);
    setSelected(f.peak);
    setZoom([t - 3 * 3600, t + 3 * 3600]);
  };

  return (
    <div className="stack">
      {D.demo && (
        <Banner kind="warn"><b>Demo data.</b> The flux is simulated and the forecast is a simple hand-written rule, not the FlareNet model. Connect the API (<code>VITE_API_URL</code>) or put a <code>data.json</code> from <code>scripts/export_web.py</code> in <code>frontend/public/</code> to see real model output.</Banner>
      )}
      {!D.demo && hoursOld > 3 && <Banner kind="warn"><b>This data is old.</b> Re-run the export to refresh it.</Banner>}
      <Banner>Research prototype, not an operational service. For official space weather alerts use <a href="https://www.swpc.noaa.gov/" target="_blank" rel="noopener noreferrer">NOAA SWPC</a>.</Banner>

      <Kpis D={D} />

      <section className="card" aria-labelledby="fx">
        <div className="card-head">
          <div>
            <h2 className="card-title" id="fx">X-ray flux and flare risk</h2>
            <p className="card-sub">{D.source.name} · {fmtDT(dom[0])} to {fmtDT(dom[1])}{zoom ? " · zoomed, double-click to reset" : " · drag to zoom"}</p>
          </div>
          <div className="seg sm" role="group" aria-label="Time range">
            {avail.map((k) => (
              <button key={k} aria-pressed={!zoom && k === activeRange} onClick={() => { setRange(k); setZoom(null); setSelected(null); }}>{k === "all" ? "All" : k}</button>
            ))}
          </div>
        </div>
        <div className="clabel"><span>Soft X-ray flux · flare classes</span><span className="muted">W/m² · {tz === "UTC" ? "UTC" : "local time"}</span></div>
        <Chart D={D} dom={dom} flares={mx} highlight={selected} onZoom={(z) => { setZoom(z); setSelected(null); }} />
        <div className="clabel" style={{ marginTop: 2 }}>
          <span>Chance of an M/X flare within the next {Math.round(D.model.horizon_min / 60)} h (model score)</span><span className="muted">shaded = alert</span>
        </div>
        <div className="row" style={{ gap: 16, fontSize: 12.5, color: "var(--tx2)", marginTop: 6 }}>
          <span><i className="dot" style={{ background: "var(--mc)" }} />flare, warned beforehand</span>
          <span><i className="dot" style={{ background: "var(--s1)", border: "2px solid var(--mc)", width: 11, height: 11 }} />flare, no alert beforehand</span>
        </div>
      </section>

      <section className="card" aria-labelledby="eh">
        <div className="card-head">
          <div>
            <h2 className="card-title" id="eh">Flares and what the model said first</h2>
            <p className="card-sub">Click a row to zoom the chart to that flare. "Warned" means an alert was raised in the 6 hours before it started.</p>
          </div>
          <div className="seg sm">
            <button aria-pressed={minCls === "M"} onClick={() => setMinCls("M")}>M and X</button>
            <button aria-pressed={minCls === "X"} onClick={() => setMinCls("X")}>X only</button>
          </div>
        </div>
        <p style={{ margin: "0 0 10px" }}>
          {rows.length
            ? <><b>{warned} of {scored.length}</b> {minCls === "X" ? "X-class" : "M and X"} flares in view had an alert beforehand. The model was in alert for {Math.round(alertRate * 100)}% of all hours, so read the two numbers together: a model that is always in alert warns before every flare.</>
            : "No flares of this class in view."}
        </p>
        <div className="wrapx">
          {rows.length > 0 && (
            <table>
              <thead><tr><th>Peak</th><th>Class</th><th>Forecast beforehand</th><th>Highest score</th></tr></thead>
              <tbody>
                {rows.map((f) => (
                  <tr key={f.peak} className={`click ${selected === f.peak ? "sel" : ""}`} tabIndex={0} onClick={() => pick(f)} onKeyDown={(e) => e.key === "Enter" && pick(f)}>
                    <td className="mono">{fmtDT(iso(f.peak))}</td>
                    <td><Chip cls={f.class} /></td>
                    <td>
                      {f.forecast === "WARNED" ? <span className="out W">Warned {dur(f.lead_min)} ahead</span>
                        : f.forecast === "MISSED" ? <span className="out Mi">Missed</span>
                        : <span className="out N">No forecast before it</span>}
                    </td>
                    <td>{f.max_prob == null ? "–" : <><span className="bar"><i style={{ width: `${f.max_prob * 100}%` }} /></span><span className="mono">{pct(f.max_prob)}</span></>}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>
      </section>

      <details className="card">
        <summary>How to read this page</summary>
        <div className="prose">
          <p><b>Risk</b> is a model score for an M or X flare starting in the next {Math.round(D.model.horizon_min / 60)} hours. At or above the <b>threshold</b> (dashed line) the page shows an alert. The threshold was chosen on validation data to maximise TSS. The score is <b>not</b> a calibrated probability.</p>
          <p>Drag on a chart to zoom, double-click to reset. Flare classes follow the GOES scale on 0.1–0.8 nm flux.</p>
        </div>
      </details>
    </div>
  );
}
