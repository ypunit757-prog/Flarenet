import ClassTag from "./ClassTag.jsx";
import Ladder from "./Ladder.jsx";
import Meter from "./Meter.jsx";
import { classOf, dur, iso, letterOf, pct } from "../lib/format.js";
import { useTime } from "../lib/context.jsx";

/** The three readings beside the chart: the class now on its gauge, the score against the alert line, the last 24 hours. */
export default function Readouts({ D, s }) {
  const { fmtDT, fmtClock } = useTime();
  const h = Math.round(D.model.horizon_min / 60);
  const flux = (f) => (isFinite(f) && f > 0 ? f.toExponential(1) + " W/m²" : "no data");
  const cap = s.expired
    ? <>The last forecast, made at {fmtClock(s.ft)}. It no longer covers the next {h} h.</>
    : s.alert
      ? <>Above the alert line of <b>{pct(s.thr)}</b>.{s.since ? <> In alert since {fmtClock(s.since)}.</> : null}</>
      : <>Below the alert line of <b>{pct(s.thr)}</b>. Forecast made at {fmtClock(s.ft)}.</>;
  return (
    <aside className="rail" data-state={s.state} aria-label="Current readings">
      <section className="panel now" aria-labelledby="rd-now">
        <header className="ph"><h2 id="rd-now">X-ray level now</h2><span>to {fmtClock(s.end)}</span></header>
        <div className="now-body">
          {s.gc && <Ladder flux={s.cur} peak={s.pk} />}
          <div className="read">
            {s.gc
              ? <p className={`big cls ${letterOf(s.curClass)}`}>{s.curClass}</p>
              : <p className="big" style={{ fontSize: "2.6rem" }}>{isFinite(s.cur) ? s.cur.toExponential(1) : "–"}</p>}
            <p className="sub num">{flux(s.cur)}</p>
            {s.gc && s.pk > 0 && <p className="sub2"><i aria-hidden="true" />24 h peak <b className={`cls ${letterOf(classOf(s.pk))}`} style={{ fontSize: "1.25rem" }}>{classOf(s.pk)}</b></p>}
          </div>
        </div>
      </section>

      <section className="panel score" aria-labelledby="rd-score">
        <header className="ph"><h2 id="rd-score">Flare score</h2><span>M or X, next {h} h</span></header>
        <p className="big num" style={{ fontFamily: "var(--display)" }}>{pct(s.p)}</p>
        <Meter p={s.p} thr={s.thr} />
        <p className="cap">{cap}</p>
      </section>

      <section className="panel day" aria-labelledby="rd-day">
        <header className="ph"><h2 id="rd-day">Last 24 hours</h2><span>to {fmtClock(s.end)}</span></header>
        <dl>
          <div className="row">
            <dt>Strongest<small>{s.pkT ? fmtDT(s.pkT) : "no data"}</small></dt>
            <dd>{s.pk && s.gc ? <ClassTag cls={classOf(s.pk)} /> : "–"}</dd>
          </div>
          <div className="row">
            <dt>M and X flares<small>{s.ev.length ? `${s.by.M} M, ${s.by.X} X` : "none"}{s.last ? `; last one ${dur((s.end - iso(s.last.peak)) / 60)} before the data ends` : ""}</small></dt>
            <dd>{s.ev.length}</dd>
          </div>
          <div className="row">
            <dt>Hours in alert</dt>
            <dd>{s.inAlert}<small>of 24</small></dd>
          </div>
        </dl>
      </section>
    </aside>
  );
}
