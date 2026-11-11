import { useMemo, useState } from "react";
import Note from "../components/Note.jsx";
import ClassTag from "../components/ClassTag.jsx";
import { Plain } from "../components/Status.jsx";
import Matrix from "../components/Matrix.jsx";
import { daily, flareRows, forwardSummary, warnedShare } from "../lib/history.js";
import { classOf, dur, iso, pct } from "../lib/format.js";
import { apiUrl } from "../lib/remote.js";
import { useRemote } from "../lib/useRemote.js";
import { useTime } from "../lib/context.jsx";

const WORD = { WARNED: "warned in time", MISSED: "not warned", "NO COVERAGE": "no coverage" };

function Days({ rows, floor }) {
  const showC = floor === "C";
  const top = Math.max(1, ...rows.map((r) => r.C + r.M + r.X));
  return (
    <div className="wrapx">
      <table className="days">
        <caption className="sr">Flares per day, by class, and the day's highest X-ray flux</caption>
        <thead><tr><th>Day (UTC)</th><th>Flares {floor} or stronger</th>{showC && <th className="r">C</th>}<th className="r">M</th><th className="r">X</th><th className="r">Highest</th></tr></thead>
        <tbody>
          {rows.map((r) => (
            <tr key={r.day}>
              <td className="num">{new Date(r.day * 1000).toISOString().slice(5, 10)}</td>
              <td aria-hidden="true">
                <span className="stack">
                  {["C", "M", "X"].map((k) => r[k] > 0 && <i key={k} className={`d-${k}`} style={{ width: `${(r[k] / top) * 100}%` }} />)}
                </span>
              </td>
              {showC && <td className="r num">{r.C}</td>}<td className="r num">{r.M}</td><td className="r num">{r.X}</td>
              <td className="r num">{r.maxFlux > 0 ? classOf(r.maxFlux) : "–"}{r.valid < r.points * 0.9 && <span className="muted" title="some readings missing"> *</span>}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function Forward() {
  const r = useRemote(apiUrl("forward-log?limit=120"));
  const { fmt } = useTime();
  const sum = useMemo(() => (r.status === "ok" ? forwardSummary(r.data.rows) : null), [r]);
  return (
    <section className="panel" aria-labelledby="h-fw">
      <h2 className="h2" id="h-fw">Daily forward test</h2>
      <p className="sub">Each day's forecast of "an M flare or stronger tomorrow" is recorded before the day starts and checked afterwards, so it cannot be adjusted to fit. This is the daily forecaster, separate from the 6-hour forecast above.</p>
      {r.status === "off" && <Note>This copy of the site was built without an API (VITE_API_URL), so the forward-test record is not available here.</Note>}
      {r.status === "loading" && <p className="fine" role="status">Loading the forward test.</p>}
      {r.status === "error" && <Note kind="warn">The forward test could not be loaded ({r.error?.message}). <button className="link" onClick={r.reload}>Try again</button></Note>}
      {sum && sum.days === 0 && <p className="fine">No days are recorded yet.</p>}
      {sum && sum.days > 0 && (
        <>
          <p className="fine"><b>{sum.scored}</b> of {sum.days} days are scored; {sum.pending} still waiting for their day to end. {sum.scored < 30 ? "That is too few days to judge skill: wait for more before reading anything into it." : ""}</p>
          {sum.scored > 0 && <Matrix cm={[[sum.tn, sum.fp], [sum.fn, sum.tp]]} label="Daily forward test: alert against whether an M flare happened" />}
          <div className="wrapx">
            <table>
              <thead><tr><th>Day (UTC)</th><th className="r">Score</th><th className="pad">Alert</th><th>Result</th></tr></thead>
              <tbody>
                {r.data.rows.slice(0, 14).map((x) => (
                  <tr key={x.day}>
                    <td className="num">{x.day}</td>
                    <td className="r num">{x.p == null ? "–" : pct(x.p)}</td>
                    <td>{x.alert == null ? "–" : x.alert ? "yes" : "no"}</td>
                    <td>{!x.resolved ? "pending" : x.flare == null ? "no coverage" : x.flare ? "M flare occurred" : "no M flare"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      )}
    </section>
  );
}

export default function History({ store }) {
  const { data: D, loading } = store;
  const { fmtDT } = useTime();
  const [min, setMin] = useState("M");
  const rows = useMemo(() => (D ? daily(D) : []), [D]);
  const flares = useMemo(() => (D ? flareRows(D, min) : []), [D, min]);
  if (loading && !D) return <><h1 className="sr">History</h1><Plain word="Loading" say="Fetching the latest forecast." /></>;
  if (!D) return <><h1 className="sr">History</h1><Plain word="No data" say="The forecast could not be loaded." /></>;
  const w = warnedShare(D);
  const floor = D.flare_class_min === "C" ? "C" : "M";   // flares below the floor are not in the payload
  return (
    <div className="wrap">
      <div className="head">
        <h1 className="title">Recent activity</h1>
        <p className="lede">The last {Math.round(D.period.days)} days of X-ray flares, and how the forecast did on each.</p>
      </div>
      <div className="page">
        {D.demo && <Note kind="warn">This is simulated demo data. The flares and forecasts below are not real.</Note>}
        <section className="panel" aria-labelledby="h-days">
          <h2 className="h2" id="h-days">Flares by day</h2>
          <Days rows={rows} floor={floor} />
          <p className="fine">Flare classes come from the peak of the 0.1 to 0.8 nm X-ray flux. An asterisk marks a day with more than 10% of readings missing. The first and last days may be partial.</p>
        </section>
        <section className="panel" aria-labelledby="h-fl">
          <h2 className="h2" id="h-fl">Flares and the forecast</h2>
          {w.total > 0 && <p className="sub">{w.warned} of {w.total} M or X flares in this period had an alert before them (the alert had to come before the flare started). A week is too short to say how good the forecast is; see Results for the measured skill.</p>}
          <div className="choice" role="group" aria-label="Smallest class shown">
            {["C", "M", "X"].filter((c) => floor === "C" || c !== "C").map((c) => <button key={c} aria-pressed={min === c} onClick={() => setMin(c)}>{c} and stronger</button>)}
          </div>
          {flares.length === 0 ? <p className="fine">No flares of this class in the period.</p> : (
            <div className="wrapx">
              <table>
                <thead><tr><th>Peak (time)</th><th>Class</th><th>Forecast</th><th className="r">Warning</th></tr></thead>
                <tbody>
                  {flares.map((f) => (
                    <tr key={f.peak}>
                      <td className="num">{fmtDT(iso(f.peak))}</td>
                      <td><ClassTag cls={f.class} /></td>
                      <td>{/^[MX]/.test(f.class) ? WORD[f.forecast] || f.forecast : <span className="muted">not forecast (below M)</span>}</td>
                      <td className="r num">{f.forecast === "WARNED" && f.lead_min != null ? dur(f.lead_min) : "–"}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </section>
        <Forward />
      </div>
    </div>
  );
}
