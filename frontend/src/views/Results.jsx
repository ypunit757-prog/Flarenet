import ModelComparison from "./ModelComparison.jsx";
import { CHECKS, VALIDATION } from "../lib/validation.js";

const LABEL = { ok: "Shown", bad: "Not met", warn: "Pending" };

export default function Results() {
  return (
    <div className="wrap">
      <div className="head">
        <h1 className="title">What has been shown, and what has not</h1>
        <p className="lede">Every figure here comes from runs of this project on data the model never trained on. Nothing is listed that was not measured.</p>
      </div>
      <div className="page">
        <ul className="ledger" aria-label="Summary of claims">
          {CHECKS.map(([k, a, b]) => (
            <li key={a}>
              <span className={`verdict ${k}`}>{LABEL[k]}</span>
              <b>{a}</b>
              <span className="why">{b}</span>
            </li>
          ))}
        </ul>
        <ModelComparison />
        <div>
          {VALIDATION.map((v) => (
            <section className="result" key={v.key} aria-labelledby={`r-${v.key}`}>
              <div>
                <h2 id={`r-${v.key}`}>{v.title}</h2>
                <p className="where">{v.where}</p>
                {v.tag && <span className={`status ${v.tag[1]}`}>{v.tag[0]}</span>}
              </div>
              <div>
                {v.big && (
                  <dl className="figs">
                    {v.big.map(([n, l]) => <div key={l}><dt>{l}</dt><dd>{n}</dd></div>)}
                  </dl>
                )}
                {v.table && (
                  <div className="wrapx">
                    <table>
                      <thead><tr>{v.table.head.map((h, i) => <th key={h} className={i ? "r" : undefined}>{h}</th>)}</tr></thead>
                      <tbody>{v.table.rows.map((r) => <tr key={r[0]}>{r.map((c, i) => <td key={i} className={i ? "r num" : undefined}>{c}</td>)}</tr>)}</tbody>
                    </table>
                  </div>
                )}
                <ul>{v.list.map((l) => <li key={l}>{l}</li>)}</ul>
              </div>
            </section>
          ))}
        </div>
      </div>
    </div>
  );
}
