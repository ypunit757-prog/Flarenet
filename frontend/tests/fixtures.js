// Small forecast payloads for the tests. Times are relative to NOW (or `at`) so that "how old is the data" is exact.
export const NOW = Date.parse("2026-11-10T12:00:00Z") / 1000;
export const iso = (u) => new Date(u * 1000).toISOString().replace(".000Z", "Z");

/** A live payload ending `ageH` hours before NOW, 10-minute steps, probabilities `p` (last value is the newest). */
export function live({ at = NOW, p = [0.1, 0.1, 0.1, 0.1], ageH = 0.25, flares = [], levels = { sensitive: 0.18, balanced: 0.3, strict: 0.45 }, explain } = {}) {
  const end = at - ageH * 3600;
  const t = p.map((_, i) => Math.round(end - (p.length - 1 - i) * 600));
  return {
    generated: iso(NOW), demo: false,
    source: { name: "NOAA SWPC GOES", kind: "swpc_live", goes_calibrated: true, units: "W/m2" },
    model: { threshold: 0.3, alert_levels: levels, calibrated: true, window_min: 90, horizon_min: 360 },
    period: { start: iso(t[0]), end: iso(end), days: 7 },
    series: { t, flux: t.map(() => 3e-6), step_min: 10 },
    forecast: { t, p, alert: p.map((x) => (x >= 0.3 ? 1 : 0)) },
    flare_class_min: "C",
    flares,
    ...(explain ? { explain } : {}),
  };
}
export const flare = (hoursAgo, cls = "M1.5", forecast = "WARNED") => ({ peak: iso(NOW - hoursAgo * 3600), class: cls, forecast, lead_min: forecast === "WARNED" ? 90 : null, max_prob: 0.5 });
export const demo = () => ({ ...live(), demo: true });

export const EXPLAIN = {
  available: true, method: "Shapley values over groups of inputs, compared with a typical window", time: iso(NOW), p: 0.41, p_base: 0.12, consistent: true,
  models: [{ id: "network", label: "Neural network (last 90 minutes)", p: 0.37, weight: 0.62 }, { id: "trees", label: "Tree model (days of history)", p: 0.45, weight: 0.55 }],
  groups: [
    { id: "level_now", label: "Flux level now", effect: 0.62, model: "trees", detail: "C4.1 now" },
    { id: "rise", label: "Rise over the last 30 min", effect: 0.41, model: "network" },
    { id: "since", label: "Time since last M flare", effect: -0.18, model: "trees" },
  ],
};
