import Gauge from "./Gauge.jsx";
import Chip from "./Chip.jsx";
import { classOf, dur, iso, pct } from "../lib/format.js";
import { summarize } from "../lib/summarize.js";
import { useTime } from "../lib/context.jsx";

export default function Kpis({ D }) {
  const { fmtDT, fmtClock } = useTime();
  const s = summarize(D), h = Math.round(D.model.horizon_min / 60);
  const age = (Date.now() / 1000 - s.end) / 3600, tot = Math.max(1, s.ev.length);
  const stale = !D.demo && age > 3;
  return (
    <div className="grid g4">
      <section className={`card kpi ${s.alert ? "alerting" : ""}`} aria-labelledby="k1">
        <div className="kpi-label" id="k1">Flare risk · next {h} h</div>
        <div className="risk">
          <Gauge p={s.p} thr={s.thr} alert={s.alert} />
          <div>
            {s.alert ? <span className="badge bad"><span className="pulse" />ALERT</span>
              : s.p > 0.75 * s.thr ? <span className="badge warn">WATCH</span>
              : <span className="badge ok">No alert</span>}
            <div className="kpi-meta">Threshold <b className="num">{pct(s.thr)}</b></div>
            <div className="kpi-meta">{s.alert && s.since ? <>In alert since <b>{fmtClock(s.since)}</b></> : <>Forecast at <b>{fmtClock(s.ft)}</b></>}</div>
          </div>
        </div>
      </section>

      <section className="card kpi" aria-labelledby="k2">
        <div className="kpi-label" id="k2">Current X-ray level</div>
        <div className="kpi-val">{s.gc ? s.curClass : "–"}</div>
        <div className="kpi-meta num">{isFinite(s.cur) ? s.cur.toExponential(1) + " W/m²" : "no data"}</div>
        <div className="kpi-foot">
          {s.gc && <Chip cls={s.curClass} />}{" "}
          <span className="muted" style={{ fontSize: 12.5 }}>
            {stale ? `data is ${age > 48 ? Math.round(age / 24) + " days" : Math.round(age) + " h"} old` : "up to " + fmtClock(s.end)}
          </span>
        </div>
      </section>

      <section className="card kpi" aria-labelledby="k3">
        <div className="kpi-label" id="k3">Peak · last 24 h</div>
        <div className="kpi-val">{s.pk && s.gc ? classOf(s.pk) : "–"}</div>
        <div className="kpi-meta num">{s.pk ? s.pk.toExponential(1) + " W/m²" : "no data"}</div>
        <div className="kpi-foot muted" style={{ fontSize: 12.5 }}>
          {s.pkT ? "at " + fmtDT(s.pkT) : ""}
          {s.last && <><br />Last M/X: <Chip cls={s.last.class} /> {dur((s.end - iso(s.last.peak)) / 60)} before the data ends</>}
        </div>
      </section>

      <section className="card kpi" aria-labelledby="k4">
        <div className="kpi-label" id="k4">M/X flares · last 24 h</div>
        <div className="kpi-val">{s.ev.length}<small>detected</small></div>
        <div className="minibar" aria-hidden="true">
          {s.ev.length > 0 && <><span style={{ width: `${(s.by.M / tot) * 100}%`, background: "var(--mc)" }} /><span style={{ width: `${(s.by.X / tot) * 100}%`, background: "var(--x)" }} /></>}
        </div>
        <div className="legend-row">
          <span><i className="dot" style={{ background: "var(--mc)" }} />M <b className="num">{s.by.M}</b></span>
          <span><i className="dot" style={{ background: "var(--x)" }} />X <b className="num">{s.by.X}</b></span>
        </div>
        <div className="kpi-meta muted">{s.inAlert} h in alert state</div>
      </section>
    </div>
  );
}
