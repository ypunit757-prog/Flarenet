// Helpers for the model-comparison section of the Results page. The file is written by scripts/compare_models.py
// (runs/<run>_compare/model_comparison.json) and copied to frontend/public/model_comparison.json.

export const COLUMNS = [
  ["accuracy", "Accuracy", "share of all times classified right (high even for a model that never alerts)"],
  ["precision", "Precision", "share of alerts followed by an M/X flare"],
  ["recall", "Recall", "share of pre-flare times that were alerted"],
  ["F1", "F1", "balance of precision and recall"],
  ["ROC_AUC", "ROC-AUC", "how well the score ranks pre-flare times above quiet ones; 0.5 is chance"],
  ["PR_AUC", "PR-AUC", "average precision; compare with the share of positive windows"],
  ["TSS", "TSS", "hit rate minus false-alarm rate; 0 is no skill"],
  ["Brier", "Brier", "mean squared error of the probability; lower is better"],
];

/** A report is usable when it has at least one dataset with metrics for at least one model. */
export function valid(rep) {
  return Boolean(rep && typeof rep === "object" && rep.datasets && typeof rep.datasets === "object"
    && Object.values(rep.datasets).some((d) => d && d.metrics && Object.keys(d.metrics).length > 0));
}

export const datasetNames = (rep) => (valid(rep) ? Object.keys(rep.datasets).filter((k) => rep.datasets[k]?.metrics) : []);

/** Prefer the held-out dataset on another satellite (the strictest test), else the first one. */
export function defaultDataset(rep) {
  const names = datasetNames(rep);
  return names.find((n) => !/later period of a training source/.test(n)) || names[0] || null;
}

/** A number for a table cell, or an en dash for a missing one (JSON null). */
export const cell = (v, d = 3) => (typeof v === "number" && isFinite(v) ? v.toFixed(d) : "\u2013");

/**
 * Models whose value in a column equals the best one at the precision shown (3 decimals); highest wins, except Brier where
 * lowest wins. Empty when every model ties, since then nothing stands out. A marker, never a verdict.
 */
export function best(models, metrics, key) {
  const lowerIsBetter = key === "Brier";
  const vals = models.map((m) => [m, metrics[m]?.[key]]).filter(([, v]) => typeof v === "number" && isFinite(v)).map(([m, v]) => [m, Number(v.toFixed(3))]);
  if (vals.length < 2) return [];
  const top = vals.reduce((a, [, v]) => (lowerIsBetter ? Math.min(a, v) : Math.max(a, v)), vals[0][1]);
  const winners = vals.filter(([, v]) => v === top).map(([m]) => m);
  return winners.length === vals.length ? [] : winners;
}

/** "+0.012 (0.003 to 0.021)" for one paired row, with the reading of the interval: better, worse or no established difference. */
export function pairedText(row, metric) {
  if (!row || typeof row.diff !== "number" || !isFinite(row.diff)) return { text: "\u2013", reading: "undetermined" };
  const [lo, hi] = row.ci95 || [null, null];
  const sg = (x) => (x > 0 ? "+" : x < 0 ? "\u2212" : "") + Math.abs(x).toFixed(3);
  const text = Number.isFinite(lo) && Number.isFinite(hi) ? `${sg(row.diff)} (${sg(lo)} to ${sg(hi)})` : sg(row.diff);
  if (!Number.isFinite(lo) || !Number.isFinite(hi)) return { text, reading: "undetermined" };
  const higherBetter = metric !== "Brier" && metric !== "FAR";
  const good = higherBetter ? lo > 0 : hi < 0, bad = higherBetter ? hi < 0 : lo > 0;
  return { text, reading: good ? "better" : bad ? "worse" : "no established difference" };
}
