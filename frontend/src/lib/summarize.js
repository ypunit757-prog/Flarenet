import { classOf, iso, isMX } from "./format.js";

/** Numbers for the KPI cards, computed from the exported data.json payload. */
export function summarize(D) {
  const F = D.forecast, n = F.p.length - 1, end = iso(D.period.end);
  const fl = D.series.flux, ts = D.series.t;
  let i = fl.length - 1;
  while (i > 0 && !isFinite(fl[i])) i--;
  const t24 = end - 86400;
  let pk = 0, pkT = null;
  for (let k = 0; k < fl.length; k++) if (ts[k] >= t24 && isFinite(fl[k]) && fl[k] > pk) { pk = fl[k]; pkT = ts[k]; }
  const ev = D.flares.filter((f) => isMX(f) && iso(f.peak) >= t24 && iso(f.peak) <= end);
  const by = { M: 0, X: 0 };
  ev.forEach((f) => by[f.class[0]]++);
  let inAlert = 0;
  for (let k = 0; k < F.t.length; k++) if (F.alert[k] && F.t[k] + 3600 > t24) inAlert++;
  let since = null;
  if (F.alert[n]) { let k = n; while (k > 0 && F.alert[k - 1]) k--; since = F.t[k]; }
  const last = D.flares.filter((f) => isMX(f) && iso(f.peak) <= end).sort((a, b) => iso(b.peak) - iso(a.peak))[0];
  return {
    p: F.p[n], alert: F.alert[n] === 1, thr: D.model.threshold, end, cur: fl[i], gc: D.source.goes_calibrated,
    curClass: classOf(fl[i]), pk, pkT, ev, by, inAlert, since, last, ft: F.t[n],
  };
}
