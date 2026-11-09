import { useEffect, useState } from "react";
import { pct } from "../lib/format.js";

const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));

/** The score on a 0 to 100% bar with the alert line as a tick. The bar fills once when the page opens. */
export default function Meter({ p, thr }) {
  const [on, setOn] = useState(false);
  useEffect(() => { const id = setTimeout(() => setOn(true), 150); return () => clearTimeout(id); }, []);
  return (
    <div className="meter" role="img" aria-label={`Score ${pct(p)} on a scale from 0 to 100 percent. The alert line is at ${pct(thr)}.`}>
      <div className="meter-track">
        <span className="meter-fill" style={{ width: on ? `${clamp(p, 0, 1) * 100}%` : 0 }} />
        <span className="meter-line" style={{ left: `${clamp(thr, 0, 1) * 100}%` }}><em style={{ transform: `translateX(-${clamp(thr * 100, 12, 88)}%)`, left: 0 }}>line {pct(thr)}</em></span>
      </div>
      <div className="meter-ends" aria-hidden="true"><span>0</span><span>100%</span></div>
    </div>
  );
}
