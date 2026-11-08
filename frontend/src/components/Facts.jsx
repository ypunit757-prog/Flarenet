import ClassTag from "./ClassTag.jsx";
import { classOf, dur, iso } from "../lib/format.js";
import { useTime } from "../lib/context.jsx";

/** Four plain readings under the band. The caption sits under each value. */
export default function Facts({ s }) {
  const { fmtDT, fmtClock } = useTime();
  const flux = (f) => (isFinite(f) && f > 0 ? f.toExponential(1) + " W/m²" : "no data");
  return (
    <dl className="facts">
      <div className="fact">
        <dt>X-ray level now<small>up to {fmtClock(s.end)}</small></dt>
        <dd>{s.gc ? <ClassTag cls={s.curClass} /> : "–"}<small className="num">{flux(s.cur)}</small></dd>
      </div>
      <div className="fact">
        <dt>Strongest in the last 24 hours<small>{s.pkT ? `at ${fmtDT(s.pkT)}` : "no data"}</small></dt>
        <dd>{s.pk && s.gc ? <ClassTag cls={classOf(s.pk)} /> : "–"}<small className="num">{s.pk ? flux(s.pk) : ""}</small></dd>
      </div>
      <div className="fact">
        <dt>M and X flares in the last 24 hours<small>{s.ev.length ? `${s.by.M} M, ${s.by.X} X` : "none"}{s.last ? `. The last one was ${dur((s.end - iso(s.last.peak)) / 60)} before the data ends.` : ""}</small></dt>
        <dd className="num">{s.ev.length}</dd>
      </div>
      <div className="fact">
        <dt>Hours in alert, last 24 hours<small>of 24</small></dt>
        <dd className="num">{s.inAlert}</dd>
      </div>
    </dl>
  );
}
