// Helpers for the Explainable AI page. The block comes from src/flarenet/explain/live.py (the newest forecast) and
// models/explain_global.json (what the models rely on, from scripts/explain_model.py).

/** The explain block when it holds an explanation, else null. A block with available:false is a reason, not data. */
export function explanation(D) {
  const e = D && D.explain;
  return e && e.available !== false && Array.isArray(e.groups) && e.groups.length ? e : null;
}

/** Why there is no explanation, in a sentence. */
export function whyNone(D) {
  if (!D) return "The forecast has not loaded.";
  if (D.demo) return "The page is showing simulated demo data, which has no model behind it, so there is nothing to explain.";
  const e = D.explain;
  if (!e) return "This forecast was made without an explanation. Re-export it with a current version of scripts/export_web.py.";
  if (e.available === false) return `The explanation could not be made for this forecast${e.reason ? `: ${e.reason}` : "."}`;
  return "The explanation block is empty.";
}

/** Groups split into those that raised and those that lowered the forecast, each biggest first. */
export function split(groups) {
  const up = groups.filter((g) => g.effect > 0).sort((a, b) => b.effect - a.effect);
  const down = groups.filter((g) => g.effect < 0).sort((a, b) => a.effect - b.effect);
  return { up, down };
}

/** Sum of the listed effects and how many of the biggest groups are not listed (the page shows a top list only). */
export const total = (groups) => groups.reduce((a, g) => a + g.effect, 0);

/** "+0.42" / "-0.18" with a real minus sign, so the sign survives without colour. */
export const signed = (x, d = 2) => (x > 0 ? "+" : x < 0 ? "−" : "") + Math.abs(x).toFixed(d);

/** Plain-words size of an effect in log-odds (a rough guide, not a threshold from the model). */
export function strength(x) {
  const a = Math.abs(x);
  return a >= 0.5 ? "strong" : a >= 0.15 ? "moderate" : "small";
}

/** Global importance rows for one model ("trees" or "network"), biggest AUC drop first, or []. */
export function globalRows(g, key) {
  const rows = g && g[key] && Array.isArray(g[key].groups) ? g[key].groups : [];
  return [...rows].filter((r) => typeof r.auc_drop === "number").sort((a, b) => b.auc_drop - a.auc_drop);
}
