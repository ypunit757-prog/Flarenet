import { useMemo, useState } from "react";
import Note from "../components/Note.jsx";
import { Plain } from "../components/Status.jsx";
import ClassTag from "../components/ClassTag.jsx";
import { LEVELS, LEVEL_TEXT, OBSERVED_CLASSES, episodes, thresholdFor } from "../lib/alerts.js";
import { pct } from "../lib/format.js";
import { apiUrl } from "../lib/remote.js";
import { useRemote } from "../lib/useRemote.js";
import { useTime } from "../lib/context.jsx";

const KIND = { predicted_risk: ["Forecast", "A model forecast. No flare has been observed."], observed_flare: ["Observed", "A flare seen in the live GOES X-ray reading. Not an official NOAA record."] };
const OUT = { hit: "flare followed", "false alarm": "no flare followed", pending: "still open" };

function Settings({ alerts, D }) {
  const { settings: s, setSettings, permission, enable, test, problem } = alerts;
  const line = thresholdFor(D, s);
  const levels = D?.model?.alert_levels || {};
  const on = s.enabled && permission === "granted";
  return (
    <section className="panel" aria-labelledby="a-set">
      <h2 className="h2" id="a-set">Alerts in this browser</h2>
      <p className="sub">
        {on ? "On. " : "Off. "}A notification appears when the forecast crosses your alert line, and when a flare is seen in the X-ray reading.
        This works only while a FlareNet tab is open; a web page cannot run in the background. For alerts with the page closed use the e-mail alerts.
      </p>
      {permission === "unsupported" && <Note kind="warn">This browser does not support notifications. Alerts will show as a message on the page instead.</Note>}
      {permission === "denied" && <Note kind="warn">Notifications are blocked for this site. Allow them in the browser's site settings, then turn alerts on again.</Note>}
      {problem && <Note kind="warn">Alerts are paused: {problem}. No alert is ever raised from old, simulated or broken data.</Note>}
      <div className="btn-row">
        {on
          ? <button className="btn" onClick={() => setSettings({ enabled: false })}>Turn alerts off</button>
          : <button className="btn" onClick={enable}>Turn alerts on</button>}
        <button className="btn" onClick={test}>Send a test notification</button>
      </div>
      <form className="form" onSubmit={(e) => e.preventDefault()}>
        <fieldset>
          <legend>Alert line</legend>
          {LEVELS.map((l) => (
            <label key={l} className="opt">
              <input type="radio" name="level" checked={!s.custom && s.level === l} onChange={() => setSettings({ level: l, custom: false })} />
              <span>{LEVEL_TEXT[l]}{typeof levels[l] === "number" && <> <b className="num">({pct(levels[l])})</b></>}</span>
            </label>
          ))}
          <label className="opt">
            <input type="radio" name="level" checked={s.custom} onChange={() => setSettings({ custom: true })} />
            <span>Custom score</span>
          </label>
          <label className="field">
            <span>Alert when the score reaches <b className="num">{pct(s.threshold)}</b></span>
            <input type="range" min="0.05" max="0.95" step="0.01" value={s.threshold} disabled={!s.custom} onChange={(e) => setSettings({ threshold: Number(e.target.value), custom: true })} />
          </label>
          <p className="fine">In force now: <b>{line.name}</b>, {line.value != null ? pct(line.value) : "unknown"}. A lower line catches more flares and raises more false alarms.</p>
        </fieldset>
        <fieldset>
          <legend>Repeats</legend>
          <label className="field">
            <span>Wait at least <b className="num">{s.cooldownMin}</b> minutes between forecast alerts</span>
            <input type="range" min="0" max="720" step="30" value={s.cooldownMin} onChange={(e) => setSettings({ cooldownMin: Number(e.target.value) })} />
          </label>
          <p className="fine">One alert is sent per episode of high risk, however long it lasts. The wait also stops a score hovering around the line from alerting again and again.</p>
        </fieldset>
        <fieldset>
          <legend>Observed flares</legend>
          <label className="opt"><input type="checkbox" checked={s.observed} onChange={(e) => setSettings({ observed: e.target.checked })} /><span>Also notify when a flare is seen in the X-ray reading</span></label>
          <label className="field row">
            <span>Smallest class</span>
            <select value={s.observedMin} disabled={!s.observed} onChange={(e) => setSettings({ observedMin: e.target.value })}>
              {OBSERVED_CLASSES.map((c) => <option key={c} value={c}>{c} and stronger</option>)}
            </select>
          </label>
        </fieldset>
      </form>
      <Note>FlareNet forecasts "M or stronger". It has no separate forecast for X-class flares, so a forecast alert means an M or X flare may follow, not that an X flare will.</Note>
    </section>
  );
}

/** The list of server alert events. A "back below the line" record is never announced by design, so it is not flagged as unsent. */
export function AlertFeed({ events }) {
  const { fmtDT } = useTime();
  return (
    <ul className="feed">
      {events.map((e) => {
        const [name, hint] = KIND[e.kind] || [e.kind, ""];
        return (
          <li key={e.id}>
            <span className={`kind ${e.kind}`} title={hint}>{name}</span>
            <span className="num when">{fmtDT(Date.parse(e.time) / 1000)}</span>
            <span className="what">
              {e.kind === "observed_flare" && <><ClassTag cls={e.class} /> flare peaked{e.forecast ? `, forecast ${String(e.forecast).toLowerCase()}` : ""}</>}
              {e.kind === "predicted_risk" && e.type === "started" && <>Risk raised: score {pct(e.p)} against a line of {pct(e.threshold)} ({e.level})</>}
              {e.kind === "predicted_risk" && e.type === "ended" && <>Risk back below the line ({pct(e.p)})</>}
              {e.notified === false && e.type !== "ended" && <i className="muted"> Logged, not announced{e.reason ? `: ${e.reason}` : ""}.</i>}
            </span>
          </li>
        );
      })}
    </ul>
  );
}

function Server() {
  const [kind, setKind] = useState("");
  const url = apiUrl(`alerts?limit=50${kind ? `&kind=${kind}` : ""}`);
  const r = useRemote(url);
  return (
    <section className="panel" aria-labelledby="a-srv">
      <h2 className="h2" id="a-srv">Alert history</h2>
      <p className="sub">Every alert the server decided, whether or not anyone was e-mailed. Newest first.</p>
      <div className="choice" role="group" aria-label="Show">
        {[["", "All"], ["predicted_risk", "Forecast"], ["observed_flare", "Observed"]].map(([k, t]) => (
          <button key={k} aria-pressed={kind === k} onClick={() => setKind(k)}>{t}</button>
        ))}
      </div>
      {r.status === "off" && <Note>The alert history is kept by the API, and this copy of the site was built without one (VITE_API_URL). The episodes in the forecast on screen are listed below.</Note>}
      {r.status === "loading" && <p className="fine" role="status">Loading the alert history.</p>}
      {r.status === "error" && <Note kind="warn">The alert history could not be loaded ({r.error?.message}). <button className="link" onClick={r.reload}>Try again</button></Note>}
      {r.status === "ok" && r.data.events.length === 0 && <p className="fine">No alerts have been recorded yet. That is normal while the forecast stays quiet.</p>}
      {r.status === "ok" && r.data.events.length > 0 && (
        <AlertFeed events={r.data.events} />
      )}
      {r.status === "ok" && <p className="fine">{r.data.total} {r.data.total === 1 ? "event" : "events"} in all{r.data.updated ? `, updated ${String(r.data.updated).slice(0, 16).replace("T", " ")} UTC` : ""}.</p>}
    </section>
  );
}

export default function Alerts({ store, alerts }) {
  const { data: D, loading } = store;
  const { fmtDT } = useTime();
  const line = D ? thresholdFor(D, alerts.settings) : null;
  const eps = useMemo(() => (D && line && line.value ? episodes(D, line.value) : []), [D, line && line.value]); // eslint-disable-line react-hooks/exhaustive-deps
  const stepMin = D && D.forecast.t.length > 1 ? Math.max(1, Math.round((D.forecast.t[1] - D.forecast.t[0]) / 60)) : 10;
  if (loading && !D) return <><h1 className="sr">Alerts</h1><Plain word="Loading" say="Fetching the latest forecast." /></>;
  return (
    <div className="wrap">
      <div className="head">
        <h1 className="title">Alerts</h1>
        <p className="lede">When the forecast crosses a line you choose, and when a strong flare is seen. Forecasts and observations are kept apart.</p>
      </div>
      <div className="page">
        {D && <Settings alerts={alerts} D={D} />}
        <Server />
        {D && (
          <section className="panel" aria-labelledby="a-ep">
            <h2 className="h2" id="a-ep">Episodes in the forecast on screen</h2>
            <p className="sub">Each run of forecast times at or above {line && line.value ? pct(line.value) : "the line"} in the last {Math.round(D.period.days)} days, and whether an M or X flare followed.{D.demo ? " This is simulated demo data." : ""}</p>
            {eps.length === 0 ? <p className="fine">The score never reached the alert line in this period.</p> : (
              <div className="wrapx">
                <table className="eps">
                  <thead><tr><th>Started</th><th className="r">Length</th><th className="r">Highest score</th><th className="pad">Outcome</th></tr></thead>
                  <tbody>{eps.map((e) => (
                    <tr key={e.start}>
                      <td className="num">{fmtDT(e.start)}</td>
                      <td className="r num">{Math.max(stepMin, Math.round((e.end - e.start) / 60) + stepMin)} min</td>
                      <td className="r num">{pct(e.peakP)}</td>
                      <td className="pad"><span className={`outc ${e.outcome.replace(" ", "-")}`}>{OUT[e.outcome]}{e.flare ? ` (${e.flare})` : ""}</span></td>
                    </tr>
                  ))}</tbody>
                </table>
              </div>
            )}
          </section>
        )}
        <section className="panel prose" aria-labelledby="a-rules">
          <h2 className="h2" id="a-rules">How alerts are decided</h2>
          <ul>
            <li><b>No alert from bad data.</b> Simulated data, a forecast more than 3 hours old, or a missing or invalid score never raises an alert.</li>
            <li><b>One per episode.</b> A long spell above the line is one alert, not one every 10 minutes.</li>
            <li><b>Forecast or observed.</b> "Forecast" means the model thinks a flare may start. "Observed" means a flare was seen in the live X-ray reading, from NOAA SWPC data. Neither is an official NOAA alert.</li>
            <li><b>E-mail and history</b> are decided on the server every 30 minutes with the same rules, so the browser and the e-mail agree.</li>
          </ul>
        </section>
      </div>
    </div>
  );
}
