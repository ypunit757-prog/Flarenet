const W = 360, H = 300, L = 44, R = 12, T = 12, B = 38;
// Distinct by dash and marker as well as colour, so the lines can be told apart without colour vision.
const STYLE = [
  { c: "var(--ink)", d: "" }, { c: "var(--c-B)", d: "6 3" }, { c: "var(--c-C)", d: "2 3" },
  { c: "var(--c-M)", d: "8 3 2 3" }, { c: "var(--c-X)", d: "1 4" }, { c: "var(--ink-2)", d: "12 3 2 3 2 3" },
];

/**
 * ROC or precision-recall curves, one line per model. kind "roc": x is false-alarm rate, y is hit rate, with the chance
 * diagonal. kind "pr": x is recall, y is precision, with the base-rate line. curves: [{ name, points: [[x, y], ...] }].
 */
export default function CurveChart({ kind, curves, base, label }) {
  const xs = (v) => L + v * (W - L - R), ys = (v) => T + (1 - v) * (H - T - B);
  const ticks = [0, 0.25, 0.5, 0.75, 1];
  const path = (pts) => pts.map(([x, y], i) => `${i ? "L" : "M"}${xs(x).toFixed(1)} ${ys(y).toFixed(1)}`).join("");
  const has = curves.filter((c) => c.points && c.points.length > 1);
  return (
    <figure className="curve">
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={label}>
        {ticks.map((v) => (
          <g key={v}>
            <line x1={L} x2={W - R} y1={ys(v)} y2={ys(v)} className="cv-grid" />
            <line y1={T} y2={H - B} x1={xs(v)} x2={xs(v)} className="cv-grid" />
            <text x={L - 6} y={ys(v) + 4} textAnchor="end" className="cv-tick">{v}</text>
            <text x={xs(v)} y={H - B + 15} textAnchor="middle" className="cv-tick">{v}</text>
          </g>
        ))}
        {kind === "roc"
          ? <line x1={xs(0)} y1={ys(0)} x2={xs(1)} y2={ys(1)} className="cv-ref" />
          : typeof base === "number" && <line x1={xs(0)} x2={xs(1)} y1={ys(base)} y2={ys(base)} className="cv-ref" />}
        {has.map((c, i) => <path key={c.name} d={path(c.points)} fill="none" stroke={STYLE[i % STYLE.length].c} strokeDasharray={STYLE[i % STYLE.length].d} strokeWidth="2" strokeLinejoin="round" />)}
        <text x={(L + W - R) / 2} y={H - 4} textAnchor="middle" className="cv-axis">{kind === "roc" ? "False-alarm rate" : "Recall (share of flares caught)"}</text>
        <text transform={`translate(11 ${(T + H - B) / 2}) rotate(-90)`} textAnchor="middle" className="cv-axis">{kind === "roc" ? "Hit rate" : "Precision"}</text>
      </svg>
      <figcaption>
        <ul className="legend">
          {has.map((c, i) => (
            <li key={c.name}><svg width="30" height="8" aria-hidden="true"><line x1="0" x2="30" y1="4" y2="4" stroke={STYLE[i % STYLE.length].c} strokeDasharray={STYLE[i % STYLE.length].d} strokeWidth="2" /></svg>{c.name}</li>
          ))}
          <li className="ref"><svg width="30" height="8" aria-hidden="true"><line x1="0" x2="30" y1="4" y2="4" className="cv-ref" /></svg>{kind === "roc" ? "chance" : "base rate"}</li>
        </ul>
      </figcaption>
    </figure>
  );
}
