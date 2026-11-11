// Browser alert rules. They mirror src/flarenet/alerts/engine.py (the server-side rules), so a person sees the same alerts
// in the browser and in the e-mail: refuse stale, simulated or broken data; one alert per episode; a cool-down between
// announcements; an observed flare is announced once; and a predicted risk is never worded as an observed flare.

import { iso } from "./format.js";

export const LEVELS = ["sensitive", "balanced", "strict"];
export const LEVEL_TEXT = {
  sensitive: "Sensitive: more alerts, fewer missed flares, more false alarms",
  balanced: "Balanced: the model's default alert line",
  strict: "Strict: fewer alerts, fewer false alarms, more missed flares",
};
export const OBSERVED_CLASSES = ["C", "M", "X"];
export const MAX_AGE_H = 3;       // data older than this is never used for an alert (same limit as the server)
export const RECENT_H = 3;        // an observed flare older than this is not announced
const RANK = { A: 0, B: 1, C: 2, M: 3, X: 4 };

export const DEFAULTS = Object.freeze({
  enabled: false, level: "balanced", custom: false, threshold: 0.3, cooldownMin: 180, observed: true, observedMin: "M",
});

const clamp = (x, lo, hi) => Math.min(hi, Math.max(lo, x));

/** Settings read from storage or a form: anything missing or out of range falls back to the default. */
export function normalizeSettings(raw) {
  const r = raw && typeof raw === "object" ? raw : {};
  const num = (v, d) => (typeof v === "number" && isFinite(v) ? v : d);
  return {
    enabled: r.enabled === true,
    level: LEVELS.includes(r.level) ? r.level : DEFAULTS.level,
    custom: r.custom === true,
    threshold: clamp(num(r.threshold, DEFAULTS.threshold), 0.01, 0.99),
    cooldownMin: clamp(Math.round(num(r.cooldownMin, DEFAULTS.cooldownMin)), 0, 1440),
    observed: r.observed !== false,
    observedMin: OBSERVED_CLASSES.includes(r.observedMin) ? r.observedMin : DEFAULTS.observedMin,
  };
}

/** The alert line in force: the custom value, else the model's level, else its default threshold. */
export function thresholdFor(D, s) {
  if (s.custom) return { value: s.threshold, name: "custom" };
  const lv = D?.model?.alert_levels?.[s.level];
  if (typeof lv === "number" && lv > 0 && lv < 1) return { value: lv, name: s.level };
  return { value: D?.model?.threshold, name: "default" };
}

/** Why this payload must not raise an alert, or null when it may. `nowS` is the time in seconds. */
export function dataProblem(D, nowS = Date.now() / 1000) {
  if (!D || typeof D !== "object") return "no forecast data";
  if (D.demo) return "the data is the simulated demo";
  const t = D.forecast?.t, p = D.forecast?.p;
  if (!Array.isArray(t) || !Array.isArray(p) || t.length < 2 || t.length !== p.length) return "the forecast is incomplete";
  const last = p[p.length - 1];
  if (typeof last !== "number" || !isFinite(last) || last < 0 || last > 1) return "the newest score is missing or invalid";
  const end = iso(D.period?.end);
  if (!isFinite(end)) return "the end of the data is unknown";
  const ageH = (nowS - end) / 3600;
  if (ageH > MAX_AGE_H) return "the newest data is more than 3 hours old";
  if (ageH < -0.25) return "the newest data is in the future";
  return null;
}

/** Runs of forecast times at or above `thr` (runs less than an hour apart are one episode) with their outcome. */
export function episodes(D, thr, nowS = Date.now() / 1000) {
  const { t, p } = D.forecast, horizon = (D.model?.horizon_min || 360) * 60;
  const peaks = (D.flares || []).filter((f) => RANK[String(f.class)[0]] >= RANK.M).map((f) => [iso(f.peak), f.class]);
  const out = [];
  let cur = null;
  t.forEach((ti, i) => {
    const pi = p[i];
    if (typeof pi !== "number" || !isFinite(pi) || pi < thr) return;
    if (cur && ti - cur.end <= 3600) { cur.end = ti; if (pi > cur.peakP) { cur.peakP = pi; cur.peakT = ti; } }
    else { cur = { start: ti, end: ti, peakP: pi, peakT: ti }; out.push(cur); }
  });
  out.forEach((e) => {
    const close = e.end + horizon;
    const hit = peaks.find(([u]) => u >= e.start && u <= close);
    e.outcome = hit ? "hit" : nowS < close ? "pending" : "false alarm";
    if (hit) e.flare = hit[1];
  });
  return out.reverse();
}

const fresh = () => ({ inAlert: undefined, lastAnnounced: null, seen: [] });

/**
 * One check of a new forecast. `memory` is what this function returned last time (or null). Returns the notifications to
 * show and the memory to keep. The first check ever only records what is already happening, so enabling alerts does not
 * announce an alert or flare the person can already see.
 */
export function decide(D, settings, memory, nowS = Date.now() / 1000) {
  const m = { ...fresh(), ...(memory || {}), seen: [...((memory && memory.seen) || [])] };
  const why = dataProblem(D, nowS);
  if (why) return { notify: [], memory: m, skipped: why };
  const s = normalizeSettings(settings);
  const first = m.inAlert === undefined;
  const { value: thr, name } = thresholdFor(D, s);
  const p = D.forecast.p[D.forecast.p.length - 1];
  const notify = [];
  const inAlert = p >= thr;
  if (inAlert && m.inAlert === false) {
    const cooling = m.lastAnnounced != null && nowS - m.lastAnnounced < s.cooldownMin * 60;
    if (!cooling) {
      m.lastAnnounced = nowS;
      const h = Math.round((D.model.horizon_min || 360) / 60);
      notify.push({
        kind: "predicted_risk", tag: "risk",
        title: "FlareNet: forecast risk is high",
        body: `Forecast, not an observation: the score is ${Math.round(p * 100)}%, above the ${name} alert line of ${Math.round(thr * 100)}%. An M or X flare may start in the next ${h} h.`,
      });
    }
  }
  m.inAlert = inAlert;
  if (s.observed) {
    for (const f of D.flares || []) {
      const cls = String(f.class || ""), id = String(f.peak);
      if (!(cls[0] in RANK) || RANK[cls[0]] < RANK[s.observedMin] || m.seen.includes(id)) continue;
      const age = (nowS - iso(f.peak)) / 3600;
      m.seen.push(id);
      if (!first && age >= 0 && age <= RECENT_H) {
        notify.push({
          kind: "observed_flare", tag: `flare-${id}`,
          title: `FlareNet: ${cls} flare observed`,
          body: `Observed in the GOES X-ray reading: a ${cls} flare peaked at ${new Date(iso(f.peak) * 1000).toISOString().slice(11, 16)} UTC. This is a reading, not an official NOAA record.`,
        });
      }
    }
    m.seen = m.seen.slice(-60);
  }
  return { notify, memory: m, skipped: null };
}
