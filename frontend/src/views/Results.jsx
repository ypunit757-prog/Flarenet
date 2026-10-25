import Banner from "../components/Banner.jsx";
import { CHECKS, VALIDATION } from "../lib/validation.js";

const LABEL = { ok: "SHOWN", bad: "NOT MET", warn: "PENDING" };

export default function Results() {
  return (
    <div className="stack">
      <Banner>Measured on data the model never trained on. Every figure comes from runs of this project, and nothing is shown that was not measured.</Banner>
      <section className="card">
        <div className="card-head"><div><h2 className="card-title">Where the project stands</h2><p className="card-sub">The honest summary, including what has not been shown.</p></div></div>
        {CHECKS.map(([k, a, b]) => (
          <div className="status-row" key={a}>
            <span className={`badge ${k}`} style={{ margin: 0, minWidth: 70, justifyContent: "center" }}>{LABEL[k]}</span>
            <b>{a}</b><span className="t">{b}</span>
          </div>
        ))}
      </section>
      <div className="grid gauto">
        {VALIDATION.map((v) => (
          <article className="card" key={v.key}>
            <h3>{v.title} {v.tag && <span className={`badge ${v.tag[1]}`} style={{ margin: "0 0 0 6px", fontSize: 10.5 }}>{v.tag[0]}</span>}</h3>
            <p className="card-sub" style={{ marginBottom: 10 }}>{v.where}</p>
            {v.big && <div className="big">{v.big.map(([n, l]) => <div key={l}><b>{n}</b><span>{l}</span></div>)}</div>}
            {v.table && (
              <table style={{ marginTop: 4 }}>
                <thead><tr>{v.table.head.map((h, i) => <th key={i} style={i ? { textAlign: "right" } : undefined}>{h}</th>)}</tr></thead>
                <tbody>{v.table.rows.map((r) => <tr key={r[0]}>{r.map((c, i) => <td key={i} className={i ? "mono" : undefined} style={i ? { textAlign: "right" } : undefined}>{c}</td>)}</tr>)}</tbody>
              </table>
            )}
            <ul className="l">{v.list.map((l) => <li key={l}>{l}</li>)}</ul>
          </article>
        ))}
      </div>
    </div>
  );
}
