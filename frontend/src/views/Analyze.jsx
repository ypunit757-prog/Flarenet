import { useMemo, useState } from "react";
import Banner from "../components/Banner.jsx";
import Chart from "../components/Chart.jsx";
import Chip from "../components/Chip.jsx";
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
    <div className="stack" style={{ maxWidth: 920 }}>
      <section className="card">
        <div className="card-head">
          <div>
            <h2 className="card-title">Look at your own X-ray file</h2>
            <p className="card-sub">Drop a CSV with a time column and the 0.1–0.8 nm flux in W/m². Everything runs in your browser and the file is not uploaded. This page lists the flares it finds; it does not run the FlareNet model. To forecast from a file, run <code>scripts/export_web.py</code> on it.</p>
          </div>
        </div>
        <label className={`dropzone ${over ? "over" : ""}`} onDragOver={(e) => { e.preventDefault(); setOver(true); }} onDragLeave={() => setOver(false)}
          onDrop={(e) => { e.preventDefault(); setOver(false); take(e.dataTransfer.files[0]); }}>
          <strong>Drop a CSV here or click to browse</strong>
          <span className="muted">Up to 25 MB</span>
          <input type="file" accept=".csv,text/csv,text/plain" hidden onChange={(e) => take(e.target.files[0])} />
        </label>
        {res?.error && <div style={{ marginTop: 16 }}><Banner kind="bad">{res.error}</Banner></div>}
        {res?.D && (
          <div style={{ marginTop: 16 }}>
            <p><b>{res.rows.toLocaleString()}</b> rows · {fmtDT(dom[0])} to {fmtDT(dom[1])} · <b>{res.flares.length}</b> {res.flares.length === 1 ? "flare" : "flares"} of M class or above</p>
            <Chart D={res.D} dom={zoom || dom} flares={res.flares} onZoom={setZoom} />
            {res.flares.length > 0 && (
              <div className="wrapx">
                <table>
                  <thead><tr><th>Peak</th><th>Class</th></tr></thead>
                  <tbody>{res.flares.map((f) => <tr key={f.peak}><td className="mono">{fmtDT(iso(f.peak))}</td><td><Chip cls={f.class} /></td></tr>)}</tbody>
                </table>
              </div>
            )}
            <Banner>Assumes GOES-calibrated flux in W/m². For other instruments the class labels are not meaningful.</Banner>
          </div>
        )}
      </section>
      <details className="card">
        <summary>What format works?</summary>
        <div className="prose">
          <p>Any CSV with a header. Time as ISO text (UTC) or Unix seconds. Columns named <code>xrsb</code>, <code>flux</code> or <code>sxr</code> are picked up automatically.</p>
          <pre style={{ background: "var(--s2)", padding: 10, borderRadius: 8, overflow: "auto", font: "12px var(--m)" }}>{`time_tag,xrsa,xrsb\n2026-10-10T00:00:00Z,1.2e-8,6.1e-7\n2026-10-10T00:01:00Z,1.2e-8,6.2e-7`}</pre>
        </div>
      </details>
    </div>
  );
}
