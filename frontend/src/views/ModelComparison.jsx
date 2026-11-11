import { useState } from "react";
import Note from "../components/Note.jsx";
import Matrix from "../components/Matrix.jsx";
import CurveChart from "../components/CurveChart.jsx";
import { COLUMNS, best, cell, datasetNames, defaultDataset, pairedText, valid } from "../lib/compare.js";
import { staticUrl } from "../lib/remote.js";
import { useRemote } from "../lib/useRemote.js";

const SHORT = { cnn_bilstm: "CNN-BiLSTM", lstm: "LSTM", random_forest: "Random Forest", xgboost: "XGBoost", gradient_boosting: "Grad. boosting", stack: "Stack" };

export function ComparisonBody({ rep }) {
  const names = datasetNames(rep);
  const [ds, setDs] = useState(() => defaultDataset(rep));
  const [model, setModel] = useState("cnn_bilstm");
  const d = rep.datasets[ds] || rep.datasets[names[0]];
  const models = Object.keys(d.metrics);
  const shown = models.includes(model) ? model : models[0];
  const curves = (key) => models.map((m) => ({ name: SHORT[m] || m, points: d.curves?.[m]?.[key] || [] }));
  const prev = d.metrics[shown]?.prevalence;
  return (
    <>
      <p className="sub">
        {rep.train ? <>Trained on {Number(rep.train.windows).toLocaleString("en-GB")} windows. </> : null}
        Scored on <b>{Number(d.windows).toLocaleString("en-GB")}</b> windows, <b>{Number(d.flare_windows).toLocaleString("en-GB")}</b> followed by an M/X flare within {Math.round(rep.horizon_min / 60)} h
        {d.period?.start ? ` (${String(d.period.start).slice(0, 10)} to ${String(d.period.end).slice(0, 10)})` : ""}.
        Every model's alert line was chosen on a separate validation set, not on these windows.
      </p>
      {names.length > 1 && (
        <label className="field row"><span>Test set</span>
          <select value={ds} onChange={(e) => setDs(e.target.value)}>{names.map((n) => <option key={n} value={n}>{n}</option>)}</select>
        </label>
      )}
      <div className="wrapx">
        <table className="cmp">
          <caption className="sr">Scores of each model on {ds}</caption>
          <thead>
            <tr><th>Model</th>{COLUMNS.map(([k, t, hint]) => <th key={k} className="r" title={hint}>{t}</th>)}</tr>
          </thead>
          <tbody>
            {models.map((m) => (
              <tr key={m}>
                <th scope="row">{rep.models?.[m]?.label || m}{rep.models?.[m]?.family && <span className="muted"> {rep.models[m].family}</span>}</th>
                {COLUMNS.map(([k]) => <td key={k} className="r num">{cell(d.metrics[m][k])}{best(models, d.metrics, k).includes(m) ? <b className="top" title="highest on this test set (lowest for Brier); see the intervals below before calling it better"> ●</b> : null}</td>)}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="fine">● marks the best value in a column on this one test set. It is not proof of a better model: see the intervals below. {typeof prev === "number" && <>Positive windows are {cell(prev * 100, 1)}% of the total, so a model that never alerts has accuracy near {cell((1 - prev) * 100, 1)}%.</>}</p>
      <div className="two">
        <div><h3 className="h3">ROC curve</h3><CurveChart kind="roc" curves={curves("roc")} label="ROC curves of each model" /></div>
        <div><h3 className="h3">Precision-recall curve</h3><CurveChart kind="pr" curves={curves("pr")} base={prev} label="Precision-recall curves of each model" /></div>
      </div>
      <h3 className="h3">Confusion matrix</h3>
      <div className="choice" role="group" aria-label="Model">
        {models.map((m) => <button key={m} aria-pressed={shown === m} onClick={() => setModel(m)}>{SHORT[m] || m}</button>)}
      </div>
      <Matrix cm={d.metrics[shown].confusion} label={`Confusion matrix of ${SHORT[shown] || shown}`} />
      <p className="fine">Alert line used: {cell(d.metrics[shown].threshold, 2)} (picked on validation data).</p>
      {Object.keys(d.paired_vs_cnn_bilstm || {}).length > 0 && (
        <>
          <h3 className="h3">Difference from the CNN-BiLSTM</h3>
          <p className="fine">Each model minus the CNN-BiLSTM on the same windows, with a 95% interval from resampling 3-day blocks. If the interval contains zero, the data do not show a difference.</p>
          <div className="wrapx">
            <table>
              <thead><tr><th>Model</th><th className="r">ROC-AUC</th><th className="r">TSS</th><th className="r">Brier (lower is better)</th></tr></thead>
              <tbody>
                {Object.entries(d.paired_vs_cnn_bilstm).map(([m, p]) => (
                  <tr key={m}>
                    <th scope="row">{SHORT[m] || m}</th>
                    {["AUC", "TSS", "Brier"].map((k) => { const r = pairedText(p[k], k); return <td key={k} className="r num">{r.text}<span className="muted"> {r.reading}</span></td>; })}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      )}
      {Object.keys(rep.skipped || {}).length > 0 && <Note>Not run: {Object.entries(rep.skipped).map(([k, v]) => `${SHORT[k] || k} (${v})`).join("; ")}.</Note>}
    </>
  );
}

export default function ModelComparison() {
  const r = useRemote(staticUrl("model_comparison.json"));
  return (
    <section className="panel" aria-labelledby="mc">
      <h2 className="h2" id="mc">Model comparison</h2>
      {r.status === "loading" && <p className="fine" role="status">Loading the comparison.</p>}
      {(r.status === "error" || (r.status === "ok" && !valid(r.data))) && (
        <Note>
          The comparison of the CNN-BiLSTM with an LSTM, a Random Forest and XGBoost has not been run on real data yet, and no numbers are shown until it has.
          Run <code>uv run python scripts/compare_models.py -c configs/noaa_all_chrono.yaml</code> on your computer, then copy
          {" "}<code>runs/&lt;run&gt;_compare/model_comparison.json</code> to <code>frontend/public/model_comparison.json</code>.
        </Note>
      )}
      {r.status === "ok" && valid(r.data) && <ComparisonBody rep={r.data} />}
    </section>
  );
}
