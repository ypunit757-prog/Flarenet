// Numbers for the History page, computed from the exported payload (about a week of data) and the daily forward log.

import { iso, isMX } from "./format.js";

const DAY = 86400;

/** One row per UTC day in the payload: flares by class and the day's highest flux. */
export function daily(D) {
  const t = D.series.t, f = D.series.flux;
  const days = new Map();
  const row = (u) => {
    const k = Math.floor(u / DAY) * DAY;
    if (!days.has(k)) days.set(k, { day: k, C: 0, M: 0, X: 0, maxFlux: 0, points: 0, valid: 0 });
    return days.get(k);
  };
  for (let i = 0; i < t.length; i++) {
    const r = row(t[i]);
    r.points++;
    if (typeof f[i] === "number" && isFinite(f[i])) { r.valid++; if (f[i] > r.maxFlux) r.maxFlux = f[i]; }
  }
  for (const fl of D.flares || []) {
    const c = String(fl.class)[0];
    if (c === "C" || c === "M" || c === "X") row(iso(fl.peak))[c]++;
  }
  return [...days.values()].sort((a, b) => a.day - b.day);
}

/** Flares listed in the payload with their forecast outcome, newest first. */
export function flareRows(D, minClass = "C") {
  const floor = { C: 2, M: 3, X: 4 }[minClass] ?? 2;
  const rank = { A: 0, B: 1, C: 2, M: 3, X: 4 };
  return (D.flares || []).filter((f) => rank[String(f.class)[0]] >= floor).sort((a, b) => iso(b.peak) - iso(a.peak));
}

/** Of the M/X flares that had coverage, how many were warned in time. */
export function warnedShare(D) {
  const mx = (D.flares || []).filter(isMX).filter((f) => f.forecast !== "NO COVERAGE");
  const warned = mx.filter((f) => f.forecast === "WARNED").length;
  return { total: mx.length, warned, share: mx.length ? warned / mx.length : null };
}

/**
 * Summary of the daily forward test rows from /api/forward-log. Only resolved days with an alert decision count; days whose
 * coverage was too low to say whether a flare happened are resolved but may carry flare = null.
 */
export function forwardSummary(rows) {
  const list = Array.isArray(rows) ? rows.filter(Boolean) : [];
  const done = list.filter((r) => r.resolved === true && typeof r.alert === "boolean" && typeof r.flare === "boolean");
  const c = { tp: 0, fp: 0, fn: 0, tn: 0 };
  for (const r of done) c[r.alert ? (r.flare ? "tp" : "fp") : r.flare ? "fn" : "tn"]++;
  return { days: list.length, scored: done.length, pending: list.filter((r) => !r.resolved).length, ...c };
}
