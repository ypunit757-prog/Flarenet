import { Plain } from "../components/Status.jsx";
import Note from "../components/Note.jsx";
import { ago, iso, isMX } from "../lib/format.js";
import { useTime } from "../lib/context.jsx";

const num = (n) => Number(n).toLocaleString("en-GB");

function quality(D) {
  const f = D.series.flux, t = D.series.t;
  const valid = f.filter((x) => typeof x === "number" && isFinite(x)).length;
  let gaps = 0, longest = 0, run = 0;
  for (let i = 0; i < f.length; i++) {
    if (typeof f[i] === "number" && isFinite(f[i])) { if (run) gaps++; longest = Math.max(longest, run); run = 0; } else run++;
  }
  if (run) { gaps++; longest = Math.max(longest, run); }
  return { points: f.length, valid, share: f.length ? valid / f.length : 0, gaps, longest, step: D.series.step_min || Math.round((t[1] - t[0]) / 60) || 10 };
}

export default function Dataset({ store }) {
  const { data: D, loading } = store;
  const { fmtDT } = useTime();
  if (loading && !D) return <><h1 className="sr">Dataset</h1><Plain word="Loading" say="Fetching the latest forecast." /></>;
  if (!D) return <><h1 className="sr">Dataset</h1><Plain word="No data" say="The forecast could not be loaded." /></>;
  const q = quality(D);
  const hoursOld = (Date.now() / 1000 - iso(D.period.end)) / 3600;
  const mx = D.flares.filter(isMX).length;
  const src = D.source;
  return (
    <div className="wrap">
      <div className="head">
        <h1 className="title">The data behind it</h1>
        <p className="lede">Where the X-ray readings come from, how they are cleaned, and what the model was trained and tested on.</p>
      </div>
      <div className="page">
        {D.demo && <Note kind="warn">The page is showing simulated demo data, so the live figures below describe the demo, not a real instrument.</Note>}
        <section className="panel" aria-labelledby="d-live">
          <h2 className="h2" id="d-live">The data on screen now</h2>
          <dl className="facts">
            <div><dt>Source</dt><dd>{src.name} <span className="muted">({src.kind})</span></dd></div>
            <div><dt>Quantity</dt><dd>X-ray flux, 0.1 to 0.8 nm, in {src.units || "W/m²"}</dd></div>
            <div><dt>Period</dt><dd className="num">{fmtDT(iso(D.period.start))} to {fmtDT(iso(D.period.end))} ({D.period.days} days)</dd></div>
            <div><dt>Newest reading</dt><dd>{D.demo ? "simulated" : `${ago(Math.max(0, hoursOld))} old`}</dd></div>
            <div><dt>Readings</dt><dd className="num">{num(q.valid)} of {num(q.points)} present ({(q.share * 100).toFixed(1)}%), one every {q.step} min</dd></div>
            <div><dt>Gaps</dt><dd className="num">{q.gaps ? `${q.gaps} gap${q.gaps === 1 ? "" : "s"}, the longest ${q.longest * q.step} min` : "none"}</dd></div>
            <div><dt>Flares found</dt><dd className="num">{D.flare_class_min === "C" ? `${num(D.flares.length)} of class C or stronger, ${mx} of them M or X` : `${num(mx)} of class M or stronger (weaker flares are not listed)`}</dd></div>
            <div><dt>Flare classes</dt><dd>{src.goes_calibrated ? "From the GOES scale, as published" : "Calibrated against GOES, so approximate"}</dd></div>
            <div><dt>Model input</dt><dd>the last {D.model.window_min} minutes; it forecasts the next {Math.round(D.model.horizon_min / 60)} hours</dd></div>
            <div><dt>Exported</dt><dd className="num">{D.generated || "n/a"}</dd></div>
          </dl>
        </section>
        <section className="panel prose" aria-labelledby="d-train">
          <h2 className="h2" id="d-train">What the model learned from</h2>
          <div className="wrapx">
            <table>
              <thead><tr><th>Use</th><th>Data</th><th>Period</th></tr></thead>
              <tbody>
                <tr><td>Training</td><td>GOES-16 X-ray flux, 1-minute</td><td>the earliest 78% of the record, from March 2017</td></tr>
                <tr><td>Calibration and alert lines</td><td>GOES-16, the next 12%</td><td>a later year (2023 to 2024)</td></tr>
                <tr><td>Weaker check</td><td>GOES-16, the latest 10%</td><td>mid-2024 to April 2025</td></tr>
                <tr><td>Final test</td><td>GOES-18, a different satellite</td><td>April 2025 to September 2026</td></tr>
                <tr><td>Live forecast</td><td>NOAA SWPC GOES X-ray feed</td><td>last 7 days, refreshed every 30 minutes</td></tr>
              </tbody>
            </table>
          </div>
          <p className="fine">The split is by time, never random, with a gap between the parts, so each test period comes after the data the model learned from. The final test is also a different satellite. See Results for the scores.</p>
        </section>
        <section className="panel prose" aria-labelledby="d-pipe">
          <h2 className="h2" id="d-pipe">How the data is prepared</h2>
          <ol>
            <li><b>Read.</b> The public NOAA SWPC and NASA/NOAA GOES X-ray files. Values that are not positive numbers are treated as missing.</li>
            <li><b>Clean.</b> Missing readings stay missing; they are never filled with made-up values. A window with too many missing readings is not scored.</li>
            <li><b>Label.</b> The flare class comes from the peak flux using the NOAA scale. Flare times come from an NOAA-style rule on the flux and can differ from NOAA's list by minutes.</li>
            <li><b>Window.</b> The flux is cut into {D.model.window_min}-minute windows. A window's label is whether an M or X flare started in the next {Math.round(D.model.horizon_min / 60)} hours.</li>
            <li><b>Scale.</b> Flux is put on a log scale, and each input is normalised with the mean and spread of the training windows only.</li>
          </ol>
          <h3 className="h3">Limits of the data</h3>
          <ul>
            <li>The 6-hour forecast uses only X-ray flux. No sunspot, magnetic-field or imaging data goes into it. (The separate daily forecaster also reads NOAA's sunspot-region reports.)</li>
            <li>Flares are rare: M or X flares are a small share of all windows, so a model that always says "no flare" scores high accuracy. That is why the Results page reports TSS, AUC and false alarms, not accuracy alone.</li>
            <li>Instruments other than GOES, including ISRO Aditya-L1, are supported by the code but not tested yet.</li>
          </ul>
        </section>
      </div>
    </div>
  );
}
