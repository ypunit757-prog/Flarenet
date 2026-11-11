import test from "node:test";
import assert from "node:assert/strict";
import { createElement as h } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { TimeProvider } from "../src/lib/context.jsx";
import { Diverging, Sizes } from "../src/components/Bars.jsx";
import Matrix from "../src/components/Matrix.jsx";
import CurveChart from "../src/components/CurveChart.jsx";
import Explain from "../src/views/Explain.jsx";
import Alerts, { AlertFeed } from "../src/views/Alerts.jsx";
import History from "../src/views/History.jsx";
import Dataset from "../src/views/Dataset.jsx";
import { ComparisonBody } from "../src/views/ModelComparison.jsx";
import { DEFAULTS } from "../src/lib/alerts.js";
import { EXPLAIN, NOW, demo, flare, live } from "./fixtures.js";

const render = (el) => renderToStaticMarkup(h(TimeProvider, null, el));
const store = (data, extra = {}) => ({ data, loading: false, error: null, ...extra });
const alertsFor = (over = {}) => ({ settings: { ...DEFAULTS }, setSettings() {}, permission: "default", enable() {}, test() {}, last: null, problem: null, ...over });

// the real clock is read by Dataset and a few labels; pin it
const realNow = Date.now;
test.before(() => { Date.now = () => NOW * 1000; });
test.after(() => { Date.now = realNow; });

test("Bars: direction is written as a sign, not only shown by colour", () => {
  const html = render(h(Diverging, { label: "x", items: [{ id: "a", label: "Up one", value: 0.5 }, { id: "b", label: "Down one", value: -0.25, detail: "network" }] }));
  assert.match(html, /\+0\.50/);
  assert.match(html, /−0\.25/);
  assert.match(html, /bars-fill up/);
  assert.match(html, /bars-fill down/);
  assert.match(html, /aria-label="x"/);
  assert.match(render(h(Sizes, { label: "s", items: [{ id: "a", label: "A", value: 0.1 }] })), /0\.100/);
});

test("Matrix: shows all four counts and refuses a malformed matrix", () => {
  const html = render(h(Matrix, { cm: [[9000, 1500], [600, 1568]] }));
  for (const t of ["9,000", "1,500", "600", "1,568", "false alarm", "missed", "caught", "correct quiet"]) assert.ok(html.includes(t), t);
  assert.equal(render(h(Matrix, { cm: null })), "");
  assert.equal(render(h(Matrix, { cm: [[1, 2]] })), "");
  assert.equal(render(h(Matrix, { cm: [[1, 2], [3]] })), "");
});

test("CurveChart: one line per model with a legend; chance line for ROC, base rate for PR", () => {
  const curves = [{ name: "A", points: [[0, 0], [0.5, 0.8], [1, 1]] }, { name: "B", points: [[0, 0], [1, 1]] }, { name: "Empty", points: [] }];
  const roc = render(h(CurveChart, { kind: "roc", curves, label: "roc" }));
  assert.equal((roc.match(/<path /g) || []).length, 2);       // the empty curve draws nothing and is not in the legend
  assert.match(roc, />chance</);
  assert.doesNotMatch(roc, />Empty</);
  const pr = render(h(CurveChart, { kind: "pr", curves, base: 0.13, label: "pr" }));
  assert.match(pr, />base rate</);
  assert.match(pr, /Precision/);
});

test("Explain: demo data says there is nothing to explain; live data shows the parts and the typical window", () => {
  const d = render(h(Explain, { store: store(demo()) }));
  assert.match(d, /simulated demo data/);
  assert.doesNotMatch(d, /Flux level now/);
  const none = render(h(Explain, { store: store(live()) }));
  assert.match(none, /without an explanation/);
  const html = render(h(Explain, { store: store(live({ explain: EXPLAIN })) }));
  for (const t of ["Flux level now", "Time since last M flare", "typical window", "41%", "12%", "Not a cause", "tree model"]) assert.ok(html.includes(t), t);
  assert.match(html, /not been added to this site yet/);      // no global block in this payload
  const withGlobal = live({ explain: { ...EXPLAIN, global: { dataset: "goes18_test", windows: 100, flare_windows: 10, trees: { baseline_auc: 0.8, groups: [{ id: "a", label: "Flux level now", auc_drop: 0.1, auc_drop_std: 0.01 }] }, network: { baseline_auc: 0.7, groups: [] } } } });
  const g = render(h(Explain, { store: store(withGlobal) }));
  assert.match(g, /What the models rely on overall/);
  assert.match(g, /goes18_test/);
  assert.match(render(h(Explain, { store: store(null, { loading: true }) })), /Loading/);
  assert.match(render(h(Explain, { store: store(live({ explain: { ...EXPLAIN, consistent: false } })) })), /do not add up/);
});

test("Alerts: states the M-or-stronger limit, the browser limit, and pauses on bad data", () => {
  const D = live({ p: [0.1, 0.5], flares: [flare(1)] });
  const html = render(h(Alerts, { store: store(D), alerts: alertsFor() }));
  for (const t of ["Alerts in this browser", "Turn alerts on", "Send a test notification", "only while a FlareNet tab is open", "no separate forecast for X-class", "Alert history", "Episodes in the forecast on screen", "How alerts are decided", "built without one"]) assert.ok(html.includes(t), t);
  assert.match(html, /balanced.*30%|\(30%\)/);
  const paused = render(h(Alerts, { store: store(demo()), alerts: alertsFor({ problem: "the data is the simulated demo" }) }));
  assert.match(paused, /Alerts are paused: the data is the simulated demo/);
  assert.match(paused, /simulated demo data/);
  const on = render(h(Alerts, { store: store(D), alerts: alertsFor({ settings: { ...DEFAULTS, enabled: true }, permission: "granted" }) }));
  assert.match(on, /Turn alerts off/);
  assert.match(render(h(Alerts, { store: store(D), alerts: alertsFor({ permission: "denied" }) })), /blocked for this site/);
  assert.match(render(h(Alerts, { store: store(D), alerts: alertsFor({ permission: "unsupported" }) })), /does not support notifications/);
  assert.match(render(h(Alerts, { store: store(D, { data: D }), alerts: alertsFor({ settings: { ...DEFAULTS, custom: true, threshold: 0.2 } }) })), /<b>custom<\/b>, 20%/);
});

test("History: flare table, class floor and honest forward-test text", () => {
  const D = live({ p: Array(6).fill(0.1), flares: [flare(1, "M1.5"), flare(2, "X1.0", "MISSED"), flare(3, "M2.0", "NO COVERAGE")] });
  const html = render(h(History, { store: store(D) }));
  for (const t of ["Flares by day", "M1.5", "X1.0", "warned in time", "not warned", "no coverage", "Daily forward test", "built without an API"]) assert.ok(html.includes(t), t);
  assert.match(html, />C and stronger</);                     // flare_class_min is C here, so the C button exists
  const m = { ...D, flare_class_min: "M" };
  const htmlM = render(h(History, { store: store(m) }));
  assert.doesNotMatch(htmlM, />C and stronger</);             // below the floor: no C button and no C column
  assert.match(render(h(History, { store: store(demo()) })), /simulated demo data/);
});

test("Dataset: describes the live data and states the split without invented dates", () => {
  const D = live({ p: Array(12).fill(0.1) });
  const html = render(h(Dataset, { store: store(D) }));
  for (const t of ["NOAA SWPC GOES", "12 of 12 present", "none", "GOES-16", "GOES-18", "never random", "No sunspot"]) assert.ok(html.includes(t), t);
  D.series.flux[3] = null; D.series.flux[4] = null;
  assert.match(render(h(Dataset, { store: store(D) })), /1 gap, the longest 20 min/);
  assert.match(render(h(Dataset, { store: store(demo()) })), /simulated demo data/);
});

test("ModelComparison: scores, curves, confusion matrix and intervals from a report", () => {
  const met = (a) => ({ threshold: 0.3, accuracy: 0.9, precision: null, recall: 0.6, F1: null, ROC_AUC: a, PR_AUC: 0.3, prevalence: 0.13, TSS: 0.4, Brier: 0.09, confusion: [[9000, 1500], [600, 1568]] });
  const rep = {
    horizon_min: 360, train: { windows: 5000 }, skipped: { xgboost: "package not installed" },
    models: { cnn_bilstm: { label: "CNN-BiLSTM (FlareNet network)", family: "deep" }, lstm: { label: "LSTM" } },
    datasets: { goes18: { windows: 12668, flare_windows: 1646, period: { start: "2025-04-01", end: "2026-09-01" }, metrics: { cnn_bilstm: met(0.81), lstm: met(0.79) },
      curves: { cnn_bilstm: { roc: [[0, 0], [1, 1]], pr: [[0, 1], [1, 0.1]] }, lstm: { roc: [], pr: [] } },
      paired_vs_cnn_bilstm: { lstm: { AUC: { diff: -0.02, ci95: [-0.03, -0.01] }, TSS: { diff: 0, ci95: [-0.02, 0.02] }, Brier: { diff: 0.002, ci95: [0.001, 0.003] } } } } },
  };
  const html = render(h(ComparisonBody, { rep }));
  for (const t of ["12,668", "1,646", "CNN-BiLSTM (FlareNet network)", "0.810", "ROC curve", "Precision-recall curve", "9,000", "Difference from the CNN-BiLSTM", "worse", "no established difference", "XGBoost (package not installed)", "not proof of a better model"]) assert.ok(html.includes(t), t);
  assert.match(html, /–/);                                // null precision and F1 are dashes, never "NaN" or 0
  assert.doesNotMatch(html, /NaN|undefined|null/);
});

test("AlertFeed: forecast and observed events are labelled apart; only real misses of an announcement are flagged", () => {
  const ev = [
    { id: "o", kind: "observed_flare", type: "observed", time: "2026-11-10T10:00:00Z", class: "M2.1", forecast: "WARNED", notified: true },
    { id: "s", kind: "predicted_risk", type: "started", time: "2026-11-10T08:00:00Z", p: 0.44, threshold: 0.3, level: "balanced", notified: true },
    { id: "c", kind: "predicted_risk", type: "started", time: "2026-11-09T08:00:00Z", p: 0.35, threshold: 0.3, level: "balanced", notified: false, reason: "within 180 min of the previous alert" },
    { id: "e", kind: "predicted_risk", type: "ended", time: "2026-11-10T09:00:00Z", p: 0.2, notified: false },
    { id: "r", kind: "observed_flare", type: "observed", time: "2026-11-08T08:00:00Z", class: "M1.0", notified: false, reason: "peaked before the notification window; recorded only" },
  ];
  const html = render(h(AlertFeed, { events: ev }));
  assert.equal((html.match(/kind observed_flare/g) || []).length, 2);
  assert.equal((html.match(/kind predicted_risk/g) || []).length, 3);
  assert.match(html, /Risk raised: score 44% against a line of 30% \(balanced\)/);
  assert.match(html, /Risk back below the line \(20%\)/);
  assert.match(html, /within 180 min of the previous alert/);
  assert.match(html, /recorded only/);
  assert.equal((html.match(/Logged, not announced/g) || []).length, 2);   // not for the "ended" record
});
