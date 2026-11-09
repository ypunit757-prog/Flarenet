import { ago, pct } from "../lib/format.js";
import { useTime } from "../lib/context.jsx";

/** Loading and error states use the same strip, so the page never jumps. */
export function Plain({ word, say, children }) {
  return (
    <section className="strip" data-state="wait">
      <div className="wrap">
        <div className="strip-main" role="status">
          <p className="word"><i className="dot" aria-hidden="true" />{word}</p>
          <div className="say-box"><p className="say">{say}</p>{children}</div>
        </div>
      </div>
    </section>
  );
}

/** The slim strip under the top bar: the state in one word, one sentence, and the numbers behind it. */
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
  const fine = s.state === "demo"
    ? <>The score and alert line come from a simple hand-written rule.</>
    : s.expired
      ? <>The last score was <b>{pct(s.p)}</b> against an alert line of <b>{pct(s.thr)}</b>, made at {fmtClock(s.ft)}{s.alert ? ", when the model was in alert" : ""}. The page updates when the next forecast is published.</>
      : s.stale
        ? <><b>This forecast is {ago(s.hoursOld)} old</b>, so it may be out of date.</>
        : s.alert && s.since ? <>In alert since {fmtClock(s.since)}.</> : <>Forecast made at {fmtClock(s.ft)}.</>;
  return (
    <section className="strip" data-state={s.state} aria-labelledby="state-word">
      <div className="wrap">
        <div className="strip-main" role="status">
          <p className="word" id="state-word"><i className="dot" aria-hidden="true" />{copy[0]}</p>
          <div className="say-box">
            <p className="say">{copy[1]}</p>
            <p className="fine">{fine}</p>
          </div>
        </div>
        <dl className="strip-stats">
          <div><dt>Score</dt><dd>{pct(s.p)}</dd></div>
          <div><dt>Alert line</dt><dd>{pct(s.thr)}</dd></div>
          {s.hoursOld >= 0 && !D.demo && <div><dt>Forecast age</dt><dd>{ago(s.hoursOld)}</dd></div>}
        </dl>
      </div>
    </section>
  );
}
