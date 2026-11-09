import { useEffect, useId, useMemo, useRef, useState } from "react";
import { classOf, dur, iso, pct } from "../lib/format.js";
import { useTime } from "../lib/context.jsx";

const LETTER = { "-8": "A", "-7": "B", "-6": "C", "-5": "M", "-4": "X" };
const TEXT = { A: "var(--t-A)", B: "var(--t-B)", C: "var(--t-C)", M: "var(--t-M)", X: "var(--t-X)" };

// index of the first element >= v in a sorted array
function lowerBound(a, v) {
  let lo = 0, hi = a.length;
  while (lo < hi) { const m = (lo + hi) >> 1; if (a[m] < v) lo = m + 1; else hi = m; }
  return lo;
}

/**
 * Two panels on one time axis: the X-ray trace (log scale, drawn in the colour of the flare class it reaches) and, when
 * D.forecast exists, the model score with its alert line. Drag to zoom (onZoom([t0,t1])); onZoom(null) resets.
 * nowLabel marks the latest reading on the trace; animate draws the trace in once.
 */
export default function Chart({ D, dom, flares, highlight, onZoom, nowLabel = false, animate = false }) {
  const { tz, fmt, fmtDT } = useTime();
  const gid = "g" + useId().replace(/[^a-zA-Z0-9]/g, "");
  const host = useRef(null);
  const [W, setW] = useState(900);
  const [hover, setHover] = useState(null); // { t, left }
  const [sel, setSel] = useState(null); // { a, b } in px
  const [drawn, setDrawn] = useState(!animate);
  const drag = useRef(null);

  useEffect(() => {
    const el = host.current;
    const measure = () => setW(Math.max(320, el.clientWidth));
    measure();
    const ro = new ResizeObserver(measure);
    ro.observe(el);
    return () => ro.disconnect();
  }, []);
  useEffect(() => {
    if (drawn) return undefined;
    const id = setTimeout(() => setDrawn(true), 1500);
    return () => clearTimeout(id);
  }, [drawn]);

  const S = D.series, F = D.forecast, gc = D.source.goes_calibrated;
  const [t0, t1] = dom, span = t1 - t0;
  const small = W < 560, H1 = small ? 210 : 270, H2 = F ? (small ? 84 : 104) : 0, gap = F ? 30 : 0;
  const side = W >= 700; // wide: label the latest reading and the alert line in a right margin
  const mL = small ? 54 : 66, mR = side ? 86 : 14, mT = 10, mB = 30, Ht = mT + H1 + gap + H2 + mB, inner = W - mL - mR;
  const x = (t) => mL + ((t - t0) / span) * inner;
  const p0 = mT + H1 + gap; // top of the score panel

  const g = useMemo(() => {
    let lo = -8, hi = -3;
    if (!gc) {
      const v = S.flux.filter((q) => q > 0).map(Math.log10).sort((a, b) => a - b);
      lo = Math.floor(v[Math.floor(v.length * 0.01)]);
      hi = Math.ceil(v[v.length - 1]);
    }
    const yF = (f) => mT + (1 - (Math.log10(Math.max(f, 1e-30)) - lo) / (hi - lo)) * H1;
    const yP = (p) => p0 + (1 - p) * H2;

    // flux trace, broken at gaps
    let flux = "", pen = false;
    for (let k = lowerBound(S.t, t0 - 600); k < S.t.length && S.t[k] <= t1 + 600; k++) {
      const f = S.flux[k];
      if (!isFinite(f) || f <= 0) { pen = false; continue; }
      flux += (pen ? "L" : "M") + x(S.t[k]).toFixed(1) + " " + yF(f).toFixed(1);
      pen = true;
    }
    // class colours along the vertical axis (hard stops at each class boundary)
    const off = (e) => (e - lo) / (hi - lo);
    const stops = gc ? [[0, "A"], [off(-7), "A"], [off(-7), "B"], [off(-6), "B"], [off(-6), "C"], [off(-5), "C"], [off(-5), "M"], [off(-4), "M"], [off(-4), "X"], [1, "X"]] : [];

    // alert bands and the score as a step line
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
    const bands = gc ? [
      { y: mT, h: yF(1e-4) - mT, fill: "var(--band-x)" },
      { y: yF(1e-4), h: yF(1e-5) - yF(1e-4), fill: "var(--band-m)" },
    ] : [];
    const rows = [];
    for (let e = Math.ceil(lo); e <= hi; e++) rows.push({ y: yF(10 ** e), label: "1e" + e, cut: gc && e >= -7 && e <= -4 });
    const letters = gc ? [-8, -7, -6, -5, -4].map((e) => ({ y: yF(10 ** (e + 0.5)) + 4.5, L: LETTER[e] })) : [];

    // time ticks, at least 76 px apart
    const step = span <= 8 * 3600 ? 3600 : span <= 36 * 3600 ? 4 * 3600 : span <= 5 * 86400 ? 12 * 3600 : 86400;
    const d0 = new Date(t0 * 1000);
    if (tz === "UTC") d0.setUTCMinutes(0, 0, 0); else d0.setMinutes(0, 0, 0);
    const ticks = []; let lastX = -1e9;
    for (let t = d0.getTime() / 1000; t <= t1; t += 3600) {
      if (t < t0) continue;
      const e = new Date(t * 1000), h = tz === "UTC" ? e.getUTCHours() : e.getHours();
      if (step >= 4 * 3600 && step < 86400 && h % (step / 3600) !== 0) continue;
      if (step === 86400 && h !== 0) continue;
      const xx = x(t); if (xx - lastX < 76) continue; lastX = xx;
      ticks.push({ x: xx, label: step === 86400 || (h === 0 && span > 24 * 3600) ? fmt(t, { day: "2-digit", month: "short" }) : fmt(t, { hour: "2-digit", minute: "2-digit" }) });
    }
    // flare markers sit on the trace peak near each flare
    const marks = [];
    for (const f of flares) {
      const tp = iso(f.peak); if (tp < t0 || tp > t1) continue;
      let best = 0;
      for (let k = lowerBound(S.t, tp - 600); k < S.t.length && S.t[k] <= tp + 600; k++) if (isFinite(S.flux[k])) best = Math.max(best, S.flux[k]);
      if (best) marks.push({ f, cx: x(tp), cy: yF(best), label: f.class[0] === "X" || span <= 2 * 86400 });
    }
    // latest reading
    let now = null;
    if (nowLabel) {
      for (let k = Math.min(S.t.length - 1, lowerBound(S.t, t1 + 1)); k >= 0 && S.t[k] >= t0; k--) {
        if (isFinite(S.flux[k]) && S.flux[k] > 0) {
          if (S.t[k] >= t1 - 2 * 3600) now = { cx: x(S.t[k]), cy: yF(S.flux[k]), text: "now " + (gc ? classOf(S.flux[k]) : S.flux[k].toExponential(1)) };
          break;
        }
      }
    }
    return { flux, stops, alerts, prob, fx, lx, bands, rows, letters, ticks, marks, now, yP, yF };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [D, t0, t1, W, tz, flares, nowLabel]);

  const toT = (ev) => {
    const r = ev.currentTarget.ownerSVGElement.getBoundingClientRect(), px = ((ev.clientX - r.left) / r.width) * W;
    return { px, t: t0 + ((px - mL) / inner) * span };
  };
  function onMove(ev) {
    const { px, t } = toT(ev);
    if (drag.current && Math.abs(px - drag.current.px) > 4) setSel({ a: Math.min(px, drag.current.px), b: Math.max(px, drag.current.px) });
    setHover({ t, left: (px / W) * host.current.clientWidth });
  }

  // tooltip for the hovered time
  let tip = null;
  if (hover) {
    let k = Math.min(S.t.length - 1, lowerBound(S.t, hover.t));
    if (k > 0 && Math.abs(S.t[k - 1] - hover.t) < Math.abs(S.t[k] - hover.t)) k--;
    let fi = -1;
    if (F) fi = lowerBound(F.t, hover.t + 1) - 1;
    const fl = S.flux[k], near = flares.find((f) => Math.abs(iso(f.peak) - hover.t) < Math.max(900, span / 80));
    tip = (
      <div className="tip" style={{ left: Math.min(Math.max(0, hover.left + 12), Math.max(0, W - 200)) }}>
        <b>{fmtDT(S.t[k])}</b><br />X-ray {isFinite(fl) ? (gc ? classOf(fl) + ", " : "") + fl.toExponential(1) + " W/m²" : "no data"}
        {fi >= 0 && fi < F.p.length && <><br />Score {pct(F.p[fi])}, {F.alert[fi] ? <b>alert</b> : "no alert"}</>}
        {near && <><br />Flare {near.class}{near.forecast ? ": " + (near.forecast === "WARNED" ? "warned " + dur(near.lead_min) + " ahead" : near.forecast === "MISSED" ? "no alert before it" : "no forecast before it") : ""}</>}
      </div>
    );
  }

  const crossX = hover ? x(hover.t) : null;
  const thr = F ? D.model.threshold : 0;
  const stroke = gc ? `url(#${gid}f)` : "var(--ink)";
  const panelH = H1 + gap + H2;
  return (
    <div className="chart" ref={host}>
      <svg viewBox={`0 0 ${W} ${Ht}`} role="img" aria-label={`X-ray trace${F ? " and forecast score" : ""} for the selected period. The same flares are listed in text below.`}>
        <defs>
          <clipPath id={`${gid}c`}><rect x={mL} y="0" width={inner} height={Ht} /></clipPath>
          {gc && (
            <linearGradient id={`${gid}f`} gradientUnits="userSpaceOnUse" x1="0" x2="0" y1={mT + H1} y2={mT}>
              {g.stops.map(([o, c], i) => <stop key={i} offset={o} stopColor={`var(--c-${c})`} />)}
            </linearGradient>
          )}
        </defs>
        {g.bands.map((b, i) => <rect key={i} x={mL} y={b.y} width={inner} height={Math.max(0, b.h)} fill={b.fill} />)}
        {g.alerts.map(([a, w], i) => (
          <g key={i}>
            <rect x={a} y={mT} width={w} height={H1} fill="var(--alert-shade)" />
            {F && <rect x={a} y={p0} width={w} height={H2} fill="var(--alert-shade)" />}
          </g>
        ))}
        {g.rows.map((r, i) => (
          <g key={i}><line className={`g ${r.cut ? "cut" : ""}`} x1={mL} x2={W - mR} y1={r.y} y2={r.y} /><text x={mL - 8} y={r.y + 4} textAnchor="end">{r.label}</text></g>
        ))}
        {g.letters.map((l) => <text key={l.L} className="cls-l" x={10} y={l.y} fill={TEXT[l.L]} style={{ fill: TEXT[l.L] }}>{l.L}</text>)}
        {g.ticks.map((t, i) => (
          <g key={i}><line className="g" x1={t.x} x2={t.x} y1={mT} y2={mT + panelH} opacity=".55" /><text x={t.x > W - 30 ? W - 2 : t.x} y={Ht - 10} textAnchor={t.x > W - 30 ? "end" : "middle"}>{t.label}</text></g>
        ))}
        <path className="trace glow" d={g.flux} stroke={stroke} clipPath={`url(#${gid}c)`} />
        <path className={`trace ${drawn ? "" : "draw"}`} d={g.flux} pathLength="1" stroke={stroke} clipPath={`url(#${gid}c)`} />
        {F && <>
          <line className="g" x1={mL} x2={W - mR} y1={g.yP(0)} y2={g.yP(0)} /><line className="g" x1={mL} x2={W - mR} y1={g.yP(1)} y2={g.yP(1)} />
          <text x={mL - 8} y={g.yP(1) + 4} textAnchor="end">100%</text><text x={mL - 8} y={g.yP(0) + 4} textAnchor="end">0</text>
          {g.prob && <><path className="scorefill" d={`${g.prob}L${g.lx.toFixed(1)} ${g.yP(0)}L${g.fx.toFixed(1)} ${g.yP(0)}Z`} /><path className="score" d={g.prob} /></>}
          <line className="thr" x1={mL} x2={W - mR} y1={g.yP(thr)} y2={g.yP(thr)} />
          {side
            ? <text className="thr-tag" x={W - mR + 12} y={g.yP(thr) - 2}>alert line<tspan x={W - mR + 12} dy="14">{pct(thr)}</tspan></text>
            : <text className="thr-tag" x={mL + 6} y={g.yP(thr) - 6} style={{ paintOrder: "stroke", stroke: "var(--halo)", strokeWidth: 4 }}>alert line {pct(thr)}</text>}
          <text x={mL + 6} y={p0 + 15} style={{ paintOrder: "stroke", stroke: "var(--halo)", strokeWidth: 4 }}>Chance of an M or X flare within {Math.round(D.model.horizon_min / 60)} h</text>
        </>}
        <text x={mL + 6} y={mT + 15} style={{ paintOrder: "stroke", stroke: "var(--halo)", strokeWidth: 4 }}>X-ray flux, W/m²</text>
        {g.marks.map(({ f, cx, cy, label }) => {
          const col = f.class[0] === "X" ? "var(--c-X)" : "var(--c-M)", warned = f.forecast === "WARNED";
          return (
            <g key={f.peak}>
              {warned ? <circle cx={cx} cy={cy} r="5.5" fill="var(--ink)" stroke="var(--halo)" strokeWidth="1.5" /> : <circle cx={cx} cy={cy} r="5.5" fill="var(--panel)" stroke="var(--ink)" strokeWidth="2" />}
              <circle cx={cx} cy={cy} r="1.8" fill={col} />
              {label && <text className="flag" x={cx} y={Math.max(mT + 30, cy - 10)} textAnchor="middle" style={{ fill: f.class[0] === "X" ? "var(--t-X)" : "var(--t-M)", paintOrder: "stroke", stroke: "var(--halo)", strokeWidth: 4 }}>{f.class}</text>}
              {highlight === f.peak && <circle className="ring" cx={cx} cy={cy} r="12" />}
            </g>
          );
        })}
        {g.now && <g>
          <circle className="now-ring" cx={g.now.cx} cy={g.now.cy} r="5" />
          <circle className="now-dot" cx={g.now.cx} cy={g.now.cy} r="3.5" />
          {side
            ? <><line x1={g.now.cx + 4} x2={W - mR + 8} y1={g.now.cy} y2={g.now.cy} stroke="var(--ink)" strokeWidth="1" /><text className="now-tag" x={W - mR + 12} y={g.now.cy + 4.5}>{g.now.text}</text></>
            : null /* phone: the trace is too dense to label; the first fact on the page gives the same reading */}
        </g>}
        {crossX !== null && <line y1={mT} y2={mT + panelH} x1={crossX} x2={crossX} stroke="var(--ink)" opacity=".5" />}
        {sel && <rect className="sel" x={sel.a} y={mT} width={sel.b - sel.a} height={panelH} />}
        <rect x={mL} y={mT} width={inner} height={panelH} fill="transparent" style={{ cursor: "crosshair" }}
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
