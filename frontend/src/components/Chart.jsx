import { useEffect, useMemo, useRef, useState } from "react";
import { classOf, dur, iso, pct } from "../lib/format.js";
import { useTime } from "../lib/context.jsx";

const CLASS_LABEL = { "-8": "A", "-7": "B", "-6": "C", "-5": "M", "-4": "X" };

// index of the first element >= v in a sorted array
function lowerBound(a, v) {
  let lo = 0, hi = a.length;
  while (lo < hi) { const m = (lo + hi) >> 1; if (a[m] < v) lo = m + 1; else hi = m; }
  return lo;
}

/**
 * Two linked panels sharing one time axis: X-ray flux (log scale, flare-class bands) and, when D.forecast exists,
 * the model score with its alert threshold. Drag to zoom (onZoom([t0,t1])), double-click to reset (onZoom(null)).
 */
export default function Chart({ D, dom, flares, highlight, onZoom }) {
  const { tz, fmt, fmtDT } = useTime();
  const host = useRef(null);
  const [W, setW] = useState(900);
  const [hover, setHover] = useState(null); // { t, left }
  const [sel, setSel] = useState(null); // { a, b } in px
  const drag = useRef(null);

  useEffect(() => {
    const el = host.current;
    const measure = () => setW(Math.max(320, el.clientWidth));
    measure();
    const ro = new ResizeObserver(measure);
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  const S = D.series, F = D.forecast, gc = D.source.goes_calibrated;
  const [t0, t1] = dom, span = t1 - t0;
  const small = W < 560, H1 = small ? 200 : 250, H2 = F ? 96 : 0, gap = F ? 28 : 0;
  const mL = 50, mR = 12, mT = 8, mB = 30, Ht = mT + H1 + gap + H2 + mB, inner = W - mL - mR;
  const x = (t) => mL + ((t - t0) / span) * inner;

  const g = useMemo(() => {
    let lo = -8, hi = -3;
    if (!gc) {
      const v = S.flux.filter((q) => q > 0).map(Math.log10).sort((a, b) => a - b);
      lo = Math.floor(v[Math.floor(v.length * 0.01)]);
      hi = Math.ceil(v[v.length - 1]);
    }
    const yF = (f) => mT + (1 - (Math.log10(Math.max(f, 1e-30)) - lo) / (hi - lo)) * H1;
    const yP = (p) => mT + H1 + gap + (1 - p) * H2;

    // flux line
    let flux = "", pen = false;
    for (let k = lowerBound(S.t, t0 - 600); k < S.t.length && S.t[k] <= t1 + 600; k++) {
      const f = S.flux[k];
      if (!isFinite(f) || f <= 0) { pen = false; continue; }
      flux += (pen ? "L" : "M") + x(S.t[k]).toFixed(1) + " " + yF(f).toFixed(1);
      pen = true;
    }
    // alert bands + score step line
    const alerts = [];
    let prob = "", first = true, fx = null, lx = 0;
    if (F) {
      for (let k = 0; k < F.t.length; k++) {
        if (F.t[k] + 3600 < t0 || F.t[k] > t1) continue;
        const a = x(Math.max(F.t[k], t0)), b = x(Math.min(F.t[k] + 3600, t1)), yy = yP(F.p[k]).toFixed(1);
        if (F.alert[k]) alerts.push([a, Math.max(0.5, b - a)]);
        if (fx === null) fx = a;
        prob += (first ? "M" : "L") + a.toFixed(1) + " " + yy + "L" + b.toFixed(1) + " " + yy;
        first = false; lx = b;
      }
    }
    // horizontal grid + labels
    const rows = [];
    for (let e = Math.ceil(lo); e <= hi; e++) rows.push({ y: yF(10 ** e), label: (gc && CLASS_LABEL[e] ? CLASS_LABEL[e] + " " : "") + "1e" + e });
    // x ticks, at least 72 px apart
    const step = span <= 8 * 3600 ? 3600 : span <= 36 * 3600 ? 4 * 3600 : span <= 5 * 86400 ? 12 * 3600 : 86400;
    const d0 = new Date(t0 * 1000);
    if (tz === "UTC") d0.setUTCMinutes(0, 0, 0); else d0.setMinutes(0, 0, 0);
    const ticks = []; let lastX = -1e9;
    for (let t = d0.getTime() / 1000; t <= t1; t += 3600) {
      if (t < t0) continue;
      const e = new Date(t * 1000), h = tz === "UTC" ? e.getUTCHours() : e.getHours();
      if (step >= 4 * 3600 && step < 86400 && h % (step / 3600) !== 0) continue;
      if (step === 86400 && h !== 0) continue;
      const xx = x(t); if (xx - lastX < 72) continue; lastX = xx;
      ticks.push({ x: xx, label: step === 86400 || (h === 0 && span > 24 * 3600) ? fmt(t, { day: "2-digit", month: "short" }) : fmt(t, { hour: "2-digit", minute: "2-digit" }) });
    }
    // flare markers sit on the flux peak near the flare
    const marks = [];
    for (const f of flares) {
      const tp = iso(f.peak); if (tp < t0 || tp > t1) continue;
      let best = 0;
      for (let k = lowerBound(S.t, tp - 600); k < S.t.length && S.t[k] <= tp + 600; k++) if (isFinite(S.flux[k])) best = Math.max(best, S.flux[k]);
      if (best) marks.push({ f, cx: x(tp), cy: yF(best) });
    }
    return { flux, prob, fx, lx, alerts, rows, ticks, marks, yP };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [D, t0, t1, W, tz, flares]);

  const toT = (ev) => {
    const r = ev.currentTarget.ownerSVGElement.getBoundingClientRect(), px = ((ev.clientX - r.left) / r.width) * W;
    return { px, t: t0 + ((px - mL) / inner) * span };
  };

  function onMove(ev) {
    const { px, t } = toT(ev);
    if (drag.current && Math.abs(px - drag.current.px) > 4) setSel({ a: Math.min(px, drag.current.px), b: Math.max(px, drag.current.px) });
    setHover({ t, left: (px / W) * host.current.clientWidth });
  }

  // tooltip content for the hovered time
  let tip = null;
  if (hover) {
    let k = Math.min(S.t.length - 1, lowerBound(S.t, hover.t));
    if (k > 0 && Math.abs(S.t[k - 1] - hover.t) < Math.abs(S.t[k] - hover.t)) k--;
    let fi = -1;
    if (F) fi = lowerBound(F.t, hover.t + 1) - 1;
    const fl = S.flux[k], near = flares.find((f) => Math.abs(iso(f.peak) - hover.t) < Math.max(900, span / 80));
    tip = (
      <div className="tip" style={{ left: Math.min(Math.max(0, hover.left + 12), Math.max(0, W - 190)) }}>
        <b>{fmtDT(S.t[k])}</b><br />flux {isFinite(fl) ? (gc ? classOf(fl) + " · " : "") + fl.toExponential(1) : "no data"}
        {fi >= 0 && <><br />score {pct(F.p[fi])} · {F.alert[fi] ? <b>alert</b> : "no alert"}</>}
        {near && <><br />flare {near.class}{near.forecast ? " · " + (near.forecast === "WARNED" ? "warned " + dur(near.lead_min) + " ahead" : near.forecast === "MISSED" ? "not warned" : "no forecast before it") : ""}</>}
      </div>
    );
  }

  const crossX = hover ? x(hover.t) : null;
  const thr = F ? D.model.threshold : 0;
  return (
    <div className="chart" ref={host}>
      <svg viewBox={`0 0 ${W} ${Ht}`} role="img" aria-label={`X-ray flux${F ? " and forecast score" : ""}`}>
        <clipPath id="cp"><rect x={mL} y="0" width={inner} height={Ht} /></clipPath>
        {gc && <>
          <rect x={mL} y={mT} width={inner} height={Math.max(0, bandY(1e-4, D, gc, H1, mT) - mT)} fill="var(--band-x)" />
          <rect x={mL} y={bandY(1e-4, D, gc, H1, mT)} width={inner} height={bandY(1e-5, D, gc, H1, mT) - bandY(1e-4, D, gc, H1, mT)} fill="var(--band-m)" />
        </>}
        {g.alerts.map(([a, w], i) => (
          <g key={i}>
            <rect x={a} y={mT} width={w} height={H1} fill="var(--alertband)" />
            <rect x={a} y={mT + H1 + gap} width={w} height={H2} fill="var(--alertband)" />
          </g>
        ))}
        {g.rows.map((r, i) => (
          <g key={i}><line className="g" x1={mL} x2={W - mR} y1={r.y} y2={r.y} /><text x={mL - 8} y={r.y + 4} textAnchor="end">{r.label}</text></g>
        ))}
        {g.ticks.map((t, i) => (
          <g key={i}><line className="g" x1={t.x} x2={t.x} y1={mT} y2={mT + H1 + gap + H2} opacity=".5" /><text x={t.x} y={Ht - 10} textAnchor="middle">{t.label}</text></g>
        ))}
        <path className="flux" d={g.flux} clipPath="url(#cp)" />
        {F && <>
          <line className="g" x1={mL} x2={W - mR} y1={g.yP(0)} y2={g.yP(0)} /><line className="g" x1={mL} x2={W - mR} y1={g.yP(1)} y2={g.yP(1)} />
          <text x={mL - 8} y={g.yP(1) + 4} textAnchor="end">100%</text><text x={mL - 8} y={g.yP(0) + 4} textAnchor="end">0</text>
          <text x={mL - 8} y={g.yP(thr) + 4} textAnchor="end">{pct(thr)}</text>
          <line className="thr" x1={mL} x2={W - mR} y1={g.yP(thr)} y2={g.yP(thr)} />
          {g.prob && <><path className="pf" d={`${g.prob}L${g.lx.toFixed(1)} ${g.yP(0)}L${g.fx.toFixed(1)} ${g.yP(0)}Z`} /><path className="prob" d={g.prob} /></>}
        </>}
        {g.marks.map(({ f, cx, cy }) => {
          const col = f.class[0] === "X" ? "var(--x)" : "var(--mc)", warned = f.forecast === "WARNED";
          return (
            <g key={f.peak}>
              {warned ? <circle cx={cx} cy={cy} r="5.5" fill={col} stroke="var(--s1)" strokeWidth="1.5" /> : <circle cx={cx} cy={cy} r="5.5" fill="var(--s1)" stroke={col} strokeWidth="2" />}
              {highlight === f.peak && <circle className="hl" cx={cx} cy={cy} r="11" />}
            </g>
          );
        })}
        {crossX !== null && <line y1={mT} y2={mT + H1 + gap + H2} x1={crossX} x2={crossX} stroke="var(--tx)" opacity=".5" />}
        {sel && <rect className="sel" x={sel.a} y={mT} width={sel.b - sel.a} height={H1 + gap + H2} />}
        <rect x={mL} y={mT} width={inner} height={H1 + gap + H2} fill="transparent" style={{ cursor: "crosshair" }}
          onPointerDown={(ev) => { drag.current = toT(ev); try { ev.currentTarget.setPointerCapture(ev.pointerId); } catch { /* ignore */ } }}
          onPointerMove={onMove}
          onPointerUp={(ev) => {
            const a = drag.current; drag.current = null; setSel(null);
            if (!a) return; const { px, t } = toT(ev);
            if (Math.abs(px - a.px) > 8) onZoom([Math.min(t, a.t), Math.max(t, a.t)]);
          }}
          onPointerLeave={() => setHover(null)}
          onDoubleClick={() => onZoom(null)} />
      </svg>
      {tip}
    </div>
  );
}

// y position of a flux level, mirroring the panel's scale (A..X range when GOES-calibrated)
function bandY(f, D, gc, H1, mT) {
  const lo = -8, hi = -3;
  return mT + (1 - (Math.log10(f) - lo) / (hi - lo)) * H1;
}
