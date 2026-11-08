import { useEffect, useMemo, useRef, useState } from "react";
import Note from "../components/Note.jsx";
import Chart from "../components/Chart.jsx";
import ClassTag from "../components/ClassTag.jsx";
import Facts from "../components/Facts.jsx";
import Status from "../components/Status.jsx";
import { ago, dur, iso, isMX, pct } from "../lib/format.js";
import { summarize } from "../lib/summarize.js";
import { useTime } from "../lib/context.jsx";

function Plain({ word, say, children }) {
  return (
    <section className="band" data-state="wait">
      <div className="wrap" style={{ gridTemplateColumns: "1fr" }}>
        <div role="status"><p className="word">{word}</p><p className="say">{say}</p>{children}</div>
      </div>
    </section>
  );
}

export default function Forecast({ store }) {
  const { data: D, loading, error, reload, src, setSrc, updated } = store;
  const { tz, fmtDT, fmtClock } = useTime();
  const [range, setRange] = useState("7d");
  const [zoom, setZoom] = useState(null);
  const [selected, setSelected] = useState(null);
  const [minCls, setMinCls] = useState("M");
  const chartBox = useRef(null);

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

  if (loading && !D) return <><h1 className="sr">Forecast</h1><Plain word="Loading" say="Fetching the latest forecast." /></>;
  if (error || !D) {
    return (
      <><h1 className="sr">Forecast</h1>
        <Plain word="No data" say="The forecast could not be loaded.">
          {error?.message && <p className="fine">{String(error.message)}</p>}
          <div className="btn-row"><button className="band-btn" onClick={reload}>Try again</button></div>
        </Plain></>
    );
  }

  const s = summarize(D);
  const h = Math.round(D.model.horizon_min / 60);
  const scored = rows.filter((f) => f.forecast !== "NO COVERAGE");
  const warned = scored.filter((f) => f.forecast === "WARNED").length;
  const alertRate = D.forecast.alert.reduce((a, b) => a + b, 0) / Math.max(1, D.forecast.alert.length);

  const pick = (f) => {
    const t = iso(f.peak);
    setSelected(f.peak);
    setZoom([t - 3 * 3600, t + 3 * 3600]);
    const el = chartBox.current; // bring the chart back into view when the list is far below it
    if (el && el.getBoundingClientRect().top < 0) el.scrollIntoView({ block: "start", behavior: "smooth" });
  };

  return (
    <>
      <h1 className="sr">Forecast</h1>
      <Status D={D} s={s} />
      <div className="wrap page">
        <div className="notes" style={{ display: "grid", gap: 12 }}>
          {D.demo && (
            <Note kind="warn"><b>You are looking at demo data.</b> The flux is simulated and the forecast is a simple hand-written rule, not the FlareNet model. Set <code>VITE_API_URL</code> to your API, or put a <code>data.json</code> from <code>scripts/export_web.py</code> in <code>frontend/public/</code>, to see real model output.</Note>
          )}
          <Note>Research prototype, not an operational service. For official space weather alerts use <a href="https://www.swpc.noaa.gov/" target="_blank" rel="noopener noreferrer">NOAA SWPC</a>.</Note>
          <p className="dataline">
            <span className="txt">
              {D.demo ? "Simulated data" : <>Data from {D.source.name}</>}, ending {fmtDT(end)}{s.hoursOld >= 0 && !D.demo ? ` (${ago(s.hoursOld)} ago)` : ""}.
              {updated && <> Loaded at {fmtClock(updated / 1000)}.</>}
            </span>
            <button className="link" onClick={reload}>Reload</button>
            <button className="link" onClick={() => setSrc(src === "demo" ? "latest" : "demo")}>{src === "demo" ? "Show the latest data" : "Show demo data"}</button>
          </p>
        </div>

        <Facts s={s} />

        <section aria-labelledby="fx" ref={chartBox}>
          <div className="sec-head">
            <div>
              <h2 id="fx">X-ray flux and flare score</h2>
              <p className="sec-sub">{fmtDT(dom[0])} to {fmtDT(dom[1])}, {tz === "UTC" ? "UTC" : "local time"}. {zoom ? "Zoomed in." : "Drag across the chart to zoom."}</p>
            </div>
            <div className="range" role="group" aria-label="Time range">
              {avail.map((k) => (
                <button key={k} aria-pressed={!zoom && k === activeRange} onClick={() => { setRange(k); setZoom(null); setSelected(null); }}>{k === "all" ? "All" : k}</button>
              ))}
              {zoom && <button className="reset" onClick={() => { setZoom(null); setSelected(null); }}>Show full range</button>}
            </div>
          </div>
          <Chart D={D} dom={dom} flares={mx} highlight={selected} onZoom={(z) => { setZoom(z); setSelected(null); }} nowLabel animate />
          <div className="key">
            <span><i className="solid" />Flare with an alert before it</span>
            <span><i />Flare with no alert before it</span>
            <span><i className="shade" />Model in alert</span>
          </div>
        </section>

        <section aria-labelledby="eh">
          <div className="sec-head">
            <div>
              <h2 id="eh">Flares and what the model said first</h2>
              <p className="sec-sub">Select a flare to zoom the chart to it. Warned means an alert was raised in the {h} hours before it started. The bar and number are the highest score in that time.</p>
            </div>
            <div className="range" role="group" aria-label="Flare classes to list">
              <button aria-pressed={minCls === "M"} onClick={() => setMinCls("M")}>M and X</button>
              <button aria-pressed={minCls === "X"} onClick={() => setMinCls("X")}>X only</button>
            </div>
          </div>
          <p className="lead">
            {rows.length
              ? <><b>{warned} of {scored.length}</b> {minCls === "X" ? "X-class" : "M and X"} flares in view had an alert beforehand. The model was in alert for {Math.round(alertRate * 100)}% of all hours, so read the two numbers together: a model that is always in alert warns before every flare.</>
              : <span className="empty">No flares of this class in view. Widen the time range above.</span>}
          </p>
          {rows.length > 0 && (
            <ul className="log">
              {rows.map((f) => (
                <li key={f.peak}>
                  <button type="button" aria-pressed={selected === f.peak} onClick={() => pick(f)}>
                    <ClassTag cls={f.class} />
                    <span className="when num">{fmtDT(iso(f.peak))}</span>
                    <span className={`out ${f.forecast === "WARNED" ? "hit" : f.forecast === "MISSED" ? "miss" : "none"}`}>
                      {f.forecast === "WARNED" ? `Warned ${dur(f.lead_min)} ahead` : f.forecast === "MISSED" ? "Missed: no alert before it" : "No forecast before it"}
                    </span>
                    <span className="sc num" aria-label={f.max_prob == null ? "no score" : `highest score ${pct(f.max_prob)}`}>
                      {f.max_prob == null ? "–" : <><i aria-hidden="true"><b style={{ width: `${f.max_prob * 100}%` }} /></i>{pct(f.max_prob)}</>}
                    </span>
                  </button>
                </li>
              ))}
            </ul>
          )}
        </section>

        <details className="more">
          <summary>How to read this page</summary>
          <div className="prose">
            <p><b>The score</b> is the model's estimate that an M or X flare starts in the next {h} hours. At or above the <b>alert line</b> (dashed) the page shows an alert. The line was chosen on validation data to maximise TSS.{" "}
              {D.model.calibrated
                ? <>The score was calibrated on validation data (Platt scaling), so it is meant to track how often an M or X flare followed such a score. Check the reliability plot in the run report before quoting it as a probability on a new period.{D.model.stack && D.model.stack.used && <> The score blends the neural network with a tree model on the last {D.model.stack.history_hours} hours of X-ray history (weights {D.model.stack.weights.network} and {D.model.stack.weights.tree_model} on the two log-odds). Forecasts earlier than {D.model.stack.history_hours} hours into the data, {D.model.stack.network_only_rows} of {D.model.stack.rows} here, use the network alone.</>}</>
                : <>The score is <b>not</b> a calibrated probability.</>}
              {D.model.stack && D.model.stack.used === false && <> The tree-model blend is switched off ({D.model.stack.reason}), so the network alone is shown.</>}
              {D.model.alert_levels && Object.keys(D.model.alert_levels).length > 1 && (
                <> Alert lines fixed on validation data: {Object.entries(D.model.alert_levels).map(([k, v]) => `${k} ${pct(v)}`).join(", ")}. A higher line means fewer false alarms and fewer flares warned.</>
              )}</p>
            <p>The trace changes colour with the flare class it reaches: blue for B, green for C, amber for M, red for X. Classes follow the GOES scale on 0.1 to 0.8 nm flux. Drag across a chart to zoom.</p>
          </div>
        </details>
      </div>
    </>
  );
}
