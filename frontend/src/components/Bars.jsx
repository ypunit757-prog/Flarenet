import { signed } from "../lib/explain.js";

/**
 * Horizontal bars for contributions to a score. Bars grow right from the centre line when a group raised the forecast and
 * left when it lowered it; the sign is also written (+ or minus) so colour is never the only signal.
 * items: [{ id, label, value, detail? }]. `unit` is shown after the value.
 */
export function Diverging({ items, unit = "", label }) {
  const max = Math.max(0.01, ...items.map((i) => Math.abs(i.value)));
  return (
    <ul className="bars" aria-label={label}>
      {items.map((i) => {
        const w = (Math.abs(i.value) / max) * 50;
        return (
          <li key={i.id}>
            <div className="bars-text"><b>{i.label}</b>{i.detail && <span>{i.detail}</span>}</div>
            <div className="bars-track" aria-hidden="true">
              <i className={`bars-fill ${i.value >= 0 ? "up" : "down"}`} style={i.value >= 0 ? { left: "50%", width: `${w}%` } : { right: "50%", width: `${w}%` }} />
            </div>
            <span className="bars-val num">{signed(i.value)}{unit}</span>
          </li>
        );
      })}
    </ul>
  );
}

/** Plain bars from zero, for sizes such as "AUC lost when shuffled". items: [{ id, label, value, note? }]. */
export function Sizes({ items, digits = 3, label }) {
  const max = Math.max(1e-9, ...items.map((i) => i.value));
  return (
    <ul className="bars sizes" aria-label={label}>
      {items.map((i) => (
        <li key={i.id}>
          <div className="bars-text"><b>{i.label}</b>{i.note && <span>{i.note}</span>}</div>
          <div className="bars-track" aria-hidden="true"><i className="bars-fill size" style={{ left: 0, width: `${Math.max(0, (i.value / max) * 100)}%` }} /></div>
          <span className="bars-val num">{i.value.toFixed(digits)}</span>
        </li>
      ))}
    </ul>
  );
}
