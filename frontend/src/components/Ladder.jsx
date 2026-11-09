import { useEffect, useState } from "react";
import { classOf, ladderPos } from "../lib/format.js";

const CLASSES = ["X", "M", "C", "B", "A"];
const EDGES = [["1e-4", 0.8], ["1e-5", 0.6], ["1e-6", 0.4], ["1e-7", 0.2]];

/**
 * The A to X gauge: a vertical log scale where each class is a fifth of the height. The bar fills up to the current
 * flux, a line marks it, and a hollow diamond marks the strongest reading of the last 24 hours.
 */
export default function Ladder({ flux, peak }) {
  const [on, setOn] = useState(false);
  useEffect(() => { const id = setTimeout(() => setOn(true), 150); return () => clearTimeout(id); }, []);
  const pos = ladderPos(flux), pk = ladderPos(peak);
  const segs = () => CLASSES.map((L) => <i key={L} className={`seg ${L}`} />);
  return (
    <div className="ladder" role="img" aria-label={`X-ray level ${classOf(flux)} on the A to X scale. The strongest in the last 24 hours was ${classOf(peak)}.`}>
      <div className="ladder-letters" aria-hidden="true">{CLASSES.map((L) => <span key={L} className={L}>{L}</span>)}</div>
      <div className="ladder-bar" aria-hidden="true">
        <div className="dim">{segs()}</div>
        <div className="lit" style={{ clipPath: `inset(${on ? (1 - pos) * 100 : 100}% 0 0 0)` }}>{segs()}</div>
        {peak > 0 && <span className="pk" style={{ bottom: `${pk * 100}%` }} />}
        <span className="cur" style={{ bottom: `${on ? pos * 100 : 0}%` }} />
      </div>
      <div className="ladder-edges" aria-hidden="true">{EDGES.map(([l, p]) => <span key={l} style={{ bottom: `${p * 100}%` }}>{l}</span>)}</div>
    </div>
  );
}
