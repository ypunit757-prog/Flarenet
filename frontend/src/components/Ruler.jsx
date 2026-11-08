import { useEffect, useState } from "react";
import { pct } from "../lib/format.js";

const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));

/** The score on a 0 to 100% ruler, with the alert line marked. The bar grows in once when the page opens. */
export default function Ruler({ p, thr }) {
  const [on, setOn] = useState(false);
  useEffect(() => { const id = setTimeout(() => setOn(true), 120); return () => clearTimeout(id); }, []);
  const at = (v) => `${clamp(v, 0, 1) * 100}%`;
  return (
    <div className="ruler" role="img" aria-label={`Score ${pct(p)} on a scale from 0 to 100 percent. The alert line is at ${pct(thr)}.`}>
      <span className="ruler-tag ruler-score" style={{ left: `${clamp(p * 100, 5, 95)}%` }}>{pct(p)}</span>
      <div className="ruler-track">
        <span className="ruler-fill" style={{ width: on ? at(p) : 0 }} />
        <span className="ruler-line" style={{ left: at(thr) }} />
      </div>
      <span className="ruler-tag ruler-alert" style={{ left: `${clamp(thr * 100, 14, 86)}%` }}>alert line {pct(thr)}</span>
      <div className="ruler-ends" aria-hidden="true">
        <span style={thr < 0.14 ? { visibility: "hidden" } : undefined}>0</span>
        <span style={thr > 0.86 ? { visibility: "hidden" } : undefined}>100%</span>
      </div>
    </div>
  );
}
