import Ruler from "./Ruler.jsx";
import { ago, pct } from "../lib/format.js";
import { useTime } from "../lib/context.jsx";

/** The band at the top of the forecast page: the state in one word, one sentence, and the score on a ruler. */
export default function Status({ D, s }) {
  const { fmtClock } = useTime();
  const h = Math.round(D.model.horizon_min / 60), hrs = `${h} ${h === 1 ? "hour" : "hours"}`;
  const copy = {
    alert: ["Alert", `An M or X flare may start in the next ${hrs}.`],
    watch: ["Watch", `The score is close to the alert line for the next ${hrs}.`],
    quiet: ["Quiet", `No alert for the next ${hrs}.`],
    demo: ["Demo", "This data is simulated. It is not the FlareNet model."],
    expired: ["Out of date", `The latest forecast is ${ago(s.hoursOld)} old, so it no longer covers the next ${hrs}.`],
  }[s.state];
  return (
    <section className="band" data-state={s.state} aria-labelledby="state-word">
      <div className="wrap">
        <div role="status">
          <p className={`word ${copy[0].length > 6 ? "long" : ""}`} id="state-word">{copy[0]}</p>
          <p className="say">{copy[1]}</p>
          <p className="fine">
            {s.state === "demo"
              ? <>The score and alert line below come from a simple hand-written rule.</>
              : s.expired
                ? <>The last score was <b>{pct(s.p)}</b> against an alert line of <b>{pct(s.thr)}</b>, made at {fmtClock(s.ft)}{s.alert ? ", when the model was in alert" : ""}. The page updates when the next forecast is published.</>
                : <>Score <b>{pct(s.p)}</b> against an alert line of <b>{pct(s.thr)}</b>. {s.alert && s.since ? <>In alert since {fmtClock(s.since)}.</> : <>Forecast made at {fmtClock(s.ft)}.</>}</>}
          </p>
          {s.stale && !s.expired && <p className="fine"><b>This forecast is {ago(s.hoursOld)} old</b>, so it may be out of date.</p>}
        </div>
        <Ruler p={s.p} thr={s.thr} />
      </div>
    </section>
  );
}
