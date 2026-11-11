import Note from "../components/Note.jsx";
import { Plain } from "../components/Status.jsx";
import { Diverging, Sizes } from "../components/Bars.jsx";
import { explanation, globalRows, signed, split, strength, whyNone } from "../lib/explain.js";
import { pct } from "../lib/format.js";
import { useTime } from "../lib/context.jsx";

function Global({ g }) {
  const trees = globalRows(g, "trees"), net = globalRows(g, "network");
  if (!trees.length && !net.length) return null;
  const rows = (list) => list.map((r) => ({ id: r.id, label: r.label, value: Math.max(0, r.auc_drop), note: `±${(r.auc_drop_std ?? 0).toFixed(3)}` }));
  return (
    <section className="panel" aria-labelledby="x-global">
      <h2 className="h2" id="x-global">What the models rely on overall</h2>
      <p className="sub">
        {g.dataset ? `Measured on ${g.dataset}` : "Measured on a held-out dataset"}
        {g.windows ? `: ${Number(g.windows).toLocaleString("en-GB")} windows` : ""}
        {g.flare_windows ? `, ${Number(g.flare_windows).toLocaleString("en-GB")} followed by an M/X flare` : ""}
        {g.period?.start ? ` (${String(g.period.start).slice(0, 10)} to ${String(g.period.end).slice(0, 10)})` : ""}.
        The bar is how much ROC-AUC the model loses when that group of inputs is shuffled between windows.
      </p>
      <div className="two">
        {net.length > 0 && <div><h3 className="h3">Neural network <span className="muted num">AUC {g.network.baseline_auc}</span></h3><Sizes items={rows(net)} label="Network input groups by importance" /></div>}
        {trees.length > 0 && <div><h3 className="h3">Tree model <span className="muted num">AUC {g.trees.baseline_auc}</span></h3><Sizes items={rows(trees)} label="Tree model input groups by importance" /></div>}
      </div>
      <p className="fine">A loss near zero means the model does not use that group, or another group carries the same information. Strongly related groups can share credit. This describes the models, not the Sun.</p>
    </section>
  );
}

export default function Explain({ store }) {
  const { data: D, loading } = store;
  const { fmtDT } = useTime();
  if (loading && !D) return <><h1 className="sr">Explainable AI</h1><Plain word="Loading" say="Fetching the latest forecast." /></>;
  if (!D) return <><h1 className="sr">Explainable AI</h1><Plain word="No data" say="The forecast could not be loaded, so there is nothing to explain." /></>;

  const e = explanation(D);
  const { up, down } = e ? split(e.groups) : { up: [], down: [] };
  const hasGlobal = D.explain && D.explain.global;
  const time = e && e.time ? Date.parse(e.time) / 1000 : null;

  return (
    <div className="wrap">
      <div className="head">
        <h1 className="title">Why this score?</h1>
        <p className="lede">Which parts of the X-ray record pushed today's forecast up or down, and what the models rely on in general.</p>
      </div>
      <div className="page">
        {!e && <Note kind="warn">{whyNone(D)}</Note>}
        {e && (
          <section className="panel" aria-labelledby="x-now">
            <h2 className="h2" id="x-now">The latest forecast</h2>
            <p className="sub">{time ? `Made for ${fmtDT(time)}. ` : ""}The score is <b>{pct(e.p)}</b>. A typical window, with no unusual activity, would score about <b>{pct(e.p_base)}</b>. The bars show how each group of inputs moved the score away from that.</p>
            {e.consistent === false && <Note kind="warn">The parts below do not add up to the score on the page (rounding or a changed model), so treat them as a rough guide.</Note>}
            <dl className="figs small">
              {e.models.map((m) => <div key={m.id}><dt>{m.label}</dt><dd>{pct(m.p)}</dd></div>)}
              <div><dt>Combined score</dt><dd>{pct(e.p)}</dd></div>
            </dl>
            {e.models.length > 1 && <p className="fine">The combined score is a weighted mix of the two models (weights {e.models.map((m) => `${m.label.split(" (")[0].toLowerCase()} ${m.weight}`).join(", ")}).</p>}
            <Diverging
              label="Contribution of each input group to the forecast, in log-odds"
              items={[...up.slice(0, 6), ...down.slice(0, 6)].sort((a, b) => Math.abs(b.effect) - Math.abs(a.effect)).map((g) => ({
                id: `${g.model}-${g.id}`, label: g.label, value: g.effect,
                detail: [g.model === "trees" ? "tree model" : "network", strength(g.effect), g.detail].filter(Boolean).join(" · "),
              }))}
            />
            <p className="fine">
              Effects are in log-odds: {signed(0.1)} raises the chance a little, {signed(-0.1)} lowers it a little. Raised: {up.length} {up.length === 1 ? "group" : "groups"}; lowered: {down.length}.
              Only the {Math.min(12, e.groups.length)} largest are listed.
            </p>
          </section>
        )}
        {hasGlobal ? <Global g={D.explain.global} /> : (
          <Note>What the models rely on overall has not been added to this site yet. Run <code>scripts/explain_model.py</code> on your computer and commit <code>models/explain_global.json</code>; the next forecast refresh shows it here.</Note>
        )}
        <section className="panel prose" aria-labelledby="x-how">
          <h2 className="h2" id="x-how">How to read this</h2>
          <ul>
            <li><b>Method.</b> {e ? e.method : "Shapley values over groups of inputs"}. Each group of inputs is switched, in every possible order, from a typical value to its real value, and the average change in the score is its contribution. The contributions add up exactly to the difference from the typical score.</li>
            <li><b>Not a cause.</b> A contribution says what moved the model's number. It does not say that the input caused a flare.</li>
            <li><b>Still a probability.</b> A high score means flares followed windows like this more often, not that one will happen.</li>
            <li><b>Inputs.</b> The 6-hour model sees only the GOES X-ray flux. It has no sunspot or magnetic-field data, so the explanation can never mention them.</li>
          </ul>
        </section>
      </div>
    </div>
  );
}
