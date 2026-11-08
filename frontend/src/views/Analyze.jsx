import { useMemo, useState } from "react";
import Note from "../components/Note.jsx";
import Chart from "../components/Chart.jsx";
import ClassTag from "../components/ClassTag.jsx";
import { classOf, iso } from "../lib/format.js";
import { useTime } from "../lib/context.jsx";

const TIME_COLS = ["time_tag", "time", "timestamp", "datetime", "date", "utc", "t"];
const FLUX_COLS = ["xrsb", "xrsb_flux", "flux", "sxr", "long", "b_avg"];
const MAX_BYTES = 25 * 1024 * 1024;

/** Parse a flux CSV and find M/X flares with a simple threshold rule. Returns { error } or { D, flares, rows }. */
async function parseFile(file) {
  if (file.size > MAX_BYTES) return { error: "That file is larger than 25 MB." };
  const lines = (await file.text()).split(/\r?\n/).filter(Boolean);
  if (lines.length < 3) return { error: "The file has fewer than three rows." };
  const cols = lines[0].split(",").map((c) => c.trim().replace(/^"|"$/g, ""));
  const ti = cols.findIndex((c) => TIME_COLS.includes(c.toLowerCase())), fi = cols.findIndex((c) => FLUX_COLS.includes(c.toLowerCase()));
  if (ti < 0 || fi < 0) return { error: `Could not find a time column and a flux column in: ${cols.join(", ")}. Rename them to time_tag and xrsb.` };
  const t = [], fl = [];
  for (let i = 1; i < lines.length; i++) {
    const p = lines[i].split(","), tv = p[ti]?.trim() ?? "";
    const ts = /^\d+(\.\d+)?$/.test(tv) ? +tv : Date.parse(/Z|[+-]\d\d:?\d\d$/.test(tv) ? tv : tv + "Z") / 1000;
    const v = parseFloat(p[fi]);
    if (isFinite(ts)) { t.push(ts); fl.push(isFinite(v) && v > 0 ? v : NaN); }
  }
  if (t.length < 3) return { error: "No readable rows." };
  const flares = [];
  for (let k = 0; k < fl.length;) {
    if (fl[k] >= 1e-5) {
      let j = k, pk = k;
      while (j < fl.length && (fl[j] >= 5e-6 || t[j] - t[k] < 600)) { if (fl[j] > fl[pk]) pk = j; j++; }
      flares.push({ peak: new Date(t[pk] * 1000).toISOString(), class: classOf(fl[pk]) });
      k = j;
    } else k++;
  }
  const D = {
    series: { t, flux: fl }, forecast: null, flares, source: { name: file.name, goes_calibrated: true },
    period: { end: new Date(t[t.length - 1] * 1000).toISOString(), days: (t[t.length - 1] - t[0]) / 86400 },
  };
  return { D, flares, rows: t.length };
}

export default function Analyze() {
  const { fmtDT } = useTime();
  const [res, setRes] = useState(null);
  const [over, setOver] = useState(false);
  const [zoom, setZoom] = useState(null);

  const take = async (file) => { if (file) { setZoom(null); setRes(await parseFile(file)); } };
  const dom = useMemo(() => (res?.D ? [res.D.series.t[0], res.D.series.t[res.D.series.t.length - 1]] : null), [res]);

  return (
    <div className="wrap">
      <div className="head">
        <h1 className="title">Look at your own X-ray file</h1>
        <p className="lede">Drop a CSV with a time column and the 0.1 to 0.8 nm flux in W/m². It stays in your browser.</p>
      </div>
      <div className="page">
        <Note>This page lists the M and X flares it finds with a plain threshold rule. It does not run the FlareNet model. To forecast from a file, run <code>scripts/export_web.py</code> on it.</Note>
        <label className={`drop ${over ? "over" : ""}`} onDragOver={(e) => { e.preventDefault(); setOver(true); }} onDragLeave={() => setOver(false)}
          onDrop={(e) => { e.preventDefault(); setOver(false); take(e.dataTransfer.files[0]); }}>
          <strong>Drop a CSV here, or click to choose one</strong>
          <span className="muted">Up to 25 MB.</span>
          <input type="file" accept=".csv,text/csv,text/plain" className="sr" onChange={(e) => take(e.target.files[0])} />
        </label>
        {res?.error && <Note kind="bad"><b>That file could not be read.</b> {res.error}</Note>}
        {res?.D && (
          <section aria-label="Result">
            <div className="sec-head">
              <div>
                <h2 className="h2">{res.flares.length} {res.flares.length === 1 ? "flare" : "flares"} of M class or above</h2>
                <p className="sec-sub">{res.rows.toLocaleString()} rows, {fmtDT(dom[0])} to {fmtDT(dom[1])}. Drag across the chart to zoom.</p>
              </div>
              {zoom && <div className="range"><button className="reset" onClick={() => setZoom(null)}>Show full range</button></div>}
            </div>
            <Chart D={res.D} dom={zoom || dom} flares={res.flares} onZoom={setZoom} />
            {res.flares.length > 0 && (
              <ul className="log" style={{ marginTop: 24 }}>
                {res.flares.map((f) => (
                  <li key={f.peak}><div style={{ display: "flex", gap: 24, padding: "11px 0", alignItems: "center" }}><ClassTag cls={f.class} /><span className="when num">{fmtDT(iso(f.peak))}</span></div></li>
                ))}
              </ul>
            )}
            <div style={{ marginTop: 24 }}><Note>Assumes GOES-calibrated flux in W/m². For other instruments the class labels are not meaningful.</Note></div>
          </section>
        )}
        <details className="more">
          <summary>Which file format works?</summary>
          <div className="prose">
            <p>Any CSV with a header row. Time as ISO text in UTC, or as Unix seconds. Columns named <code>xrsb</code>, <code>flux</code> or <code>sxr</code> are picked up automatically.</p>
            <pre className="sample">{`time_tag,xrsa,xrsb\n2026-10-10T00:00:00Z,1.2e-8,6.1e-7\n2026-10-10T00:01:00Z,1.2e-8,6.2e-7`}</pre>
          </div>
        </details>
      </div>
    </div>
  );
}
