import { pct } from "../lib/format.js";

const R = 46, C = 2 * Math.PI * R, SWEEP = 0.75, ROT = 135;

/** 270-degree probability gauge with a tick at the alert threshold. */
export default function Gauge({ p, thr, alert }) {
  const dash = C * SWEEP, off = dash * (1 - Math.max(0, Math.min(1, p)));
  const color = alert ? "var(--bad)" : p > 0.75 * thr ? "var(--warn)" : "var(--ok)";
  const a = ((ROT + 270 * thr) * Math.PI) / 180;
  const pt = (r) => [(60 + r * Math.cos(a)).toFixed(2), (60 + r * Math.sin(a)).toFixed(2)];
  const [x1, y1] = pt(R - 8), [x2, y2] = pt(R + 7);
  return (
    <svg className="gauge" viewBox="0 0 120 120" role="img" aria-label={`Forecast score ${pct(p)}, alert threshold ${pct(thr)}`}>
      <circle className="track" cx="60" cy="60" r={R} fill="none" strokeWidth="9" strokeLinecap="round" strokeDasharray={`${dash} ${C}`} transform={`rotate(${ROT} 60 60)`} />
      <circle className="arc" cx="60" cy="60" r={R} fill="none" strokeWidth="9" strokeLinecap="round" stroke={color} strokeDasharray={`${dash} ${C}`} strokeDashoffset={off} transform={`rotate(${ROT} 60 60)`} />
      <line className="tick" x1={x1} y1={y1} x2={x2} y2={y2} strokeLinecap="round" />
      <text x="60" y="63" textAnchor="middle">{Math.round(p * 100)}<tspan className="unit" fontSize="13" dy="-6">%</tspan></text>
      <text className="unit" x="60" y="82" textAnchor="middle" style={{ fontSize: "9.5px" }}>M/X SCORE</text>
    </svg>
  );
}
