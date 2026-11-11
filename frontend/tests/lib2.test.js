import test from "node:test";
import assert from "node:assert/strict";
import { explanation, globalRows, signed, split, strength, whyNone } from "../src/lib/explain.js";
import { daily, flareRows, forwardSummary, warnedShare } from "../src/lib/history.js";
import { COLUMNS, best, cell, datasetNames, defaultDataset, pairedText, valid } from "../src/lib/compare.js";
import { apiUrl, getJSON, staticUrl } from "../src/lib/remote.js";
import { EXPLAIN, NOW, demo, flare, live } from "./fixtures.js";

test("explain: only a block with groups counts; every missing case has a reason", () => {
  assert.equal(explanation(live()), null);
  assert.equal(explanation(live({ explain: { available: false, reason: "boom" } })), null);
  assert.equal(explanation(live({ explain: { available: true, groups: [] } })), null);
  assert.ok(explanation(live({ explain: EXPLAIN })));
  assert.match(whyNone(demo()), /simulated demo/);
  assert.match(whyNone(live()), /without an explanation/);
  assert.match(whyNone(live({ explain: { available: false, reason: "boom" } })), /boom/);
  assert.match(whyNone(null), /not loaded/);
});

test("explain: groups split by sign, biggest first; signs are written, with a real minus", () => {
  const { up, down } = split(EXPLAIN.groups);
  assert.deepEqual(up.map((g) => g.id), ["level_now", "rise"]);
  assert.deepEqual(down.map((g) => g.id), ["since"]);
  assert.equal(signed(0.4), "+0.40");
  assert.equal(signed(-0.18), "−0.18");
  assert.equal(signed(0), "0.00");
  assert.equal(strength(0.6), "strong");
  assert.equal(strength(-0.2), "moderate");
  assert.equal(strength(0.05), "small");
});

test("explain: global rows are sorted by AUC lost and tolerate a missing model", () => {
  const g = { trees: { groups: [{ id: "a", auc_drop: 0.01 }, { id: "b", auc_drop: 0.2 }, { id: "c", auc_drop: null }] } };
  assert.deepEqual(globalRows(g, "trees").map((r) => r.id), ["b", "a"]);
  assert.deepEqual(globalRows(g, "network"), []);
  assert.deepEqual(globalRows(null, "trees"), []);
});

test("history: flares counted per UTC day and by class; floor respected", () => {
  const D = live({ p: Array(6).fill(0.1), flares: [flare(1, "M1.5"), flare(2, "X1.0", "MISSED"), flare(3, "C2.0"), flare(4, "M2.0", "NO COVERAGE")] });
  const rows = daily(D);
  assert.ok(rows.length >= 1);
  const total = rows.reduce((a, r) => ({ C: a.C + r.C, M: a.M + r.M, X: a.X + r.X }), { C: 0, M: 0, X: 0 });
  assert.deepEqual(total, { C: 1, M: 2, X: 1 });
  assert.deepEqual(flareRows(D, "M").map((f) => f.class), ["M1.5", "X1.0", "M2.0"]);
  assert.deepEqual(flareRows(D, "X").map((f) => f.class), ["X1.0"]);
  const w = warnedShare(D);                                  // no coverage is left out of the share
  assert.equal(w.total, 2);
  assert.equal(w.warned, 1);
  assert.equal(w.share, 0.5);
  assert.equal(warnedShare(live()).share, null);
});

test("history: forward test counts only resolved days with a decision", () => {
  const rows = [
    { day: "1", resolved: true, alert: true, flare: true }, { day: "2", resolved: true, alert: true, flare: false },
    { day: "3", resolved: true, alert: false, flare: true }, { day: "4", resolved: true, alert: false, flare: false },
    { day: "5", resolved: true, alert: false, flare: null }, { day: "6", resolved: false, alert: true, flare: null }, null,
  ];
  assert.deepEqual(forwardSummary(rows), { days: 6, scored: 4, pending: 1, tp: 1, fp: 1, fn: 1, tn: 1 });
  assert.deepEqual(forwardSummary(undefined), { days: 0, scored: 0, pending: 0, tp: 0, fp: 0, fn: 0, tn: 0 });
});

const REP = { datasets: {
  "goes16 (later period of a training source)": { metrics: { a: { ROC_AUC: 0.8 } } },
  goes18: { metrics: { a: { ROC_AUC: 0.8, Brier: 0.09, F1: null }, b: { ROC_AUC: 0.8004, Brier: 0.08, F1: 0.5 }, c: { ROC_AUC: 0.7, Brier: 0.09, F1: 0.5 } } },
} };

test("comparison: a report is valid only with metrics; the stricter test set is the default", () => {
  assert.equal(valid(null), false);
  assert.equal(valid({}), false);
  assert.equal(valid({ datasets: { x: { metrics: {} } } }), false);
  assert.equal(valid(REP), true);
  assert.deepEqual(datasetNames(REP).length, 2);
  assert.equal(defaultDataset(REP), "goes18");
  assert.equal(defaultDataset({}), null);
  assert.ok(COLUMNS.some(([k]) => k === "ROC_AUC") && COLUMNS.some(([k]) => k === "F1"));
});

test("comparison: missing numbers show a dash; ties are not marked; Brier is lower-is-better", () => {
  assert.equal(cell(null), "–");
  assert.equal(cell(NaN), "–");
  assert.equal(cell(0.12345), "0.123");
  const m = REP.datasets.goes18.metrics, names = Object.keys(m);
  assert.deepEqual(best(names, m, "ROC_AUC"), ["a", "b"]);   // 0.800 and 0.800 at the precision shown: both, not a false winner
  assert.deepEqual(best(names, m, "Brier"), ["b"]);          // lowest
  assert.deepEqual(best(names, m, "F1"), []);                 // the models that have a value tie; a missing one is left out
  assert.deepEqual(best(["a", "c"], { a: { x: 0.5 }, c: { x: 0.5 } }, "x"), []);   // every model ties: nothing stands out
  assert.deepEqual(best(["a"], { a: { x: 0.5 } }, "x"), []);
});

test("comparison: intervals are read honestly", () => {
  assert.equal(pairedText({ diff: 0.02, ci95: [0.01, 0.03] }, "AUC").reading, "better");
  assert.equal(pairedText({ diff: 0.0, ci95: [-0.03, 0.03] }, "TSS").reading, "no established difference");
  assert.equal(pairedText({ diff: -0.02, ci95: [-0.03, -0.01] }, "AUC").reading, "worse");
  assert.equal(pairedText({ diff: -0.004, ci95: [-0.006, -0.002] }, "Brier").reading, "better");   // lower Brier is better
  assert.equal(pairedText({ diff: 0.004, ci95: [0.002, 0.006] }, "Brier").reading, "worse");
  assert.equal(pairedText({ diff: 0.1, ci95: [null, null] }, "AUC").reading, "undetermined");
  assert.equal(pairedText(null, "AUC").text, "–");
  assert.match(pairedText({ diff: -0.002, ci95: [-0.004, 0] }, "Brier").text, /^−0\.002 \(−0\.004 to 0\.000\)$/);
});

test("remote: no API configured means no URL; JSON errors are explained", async () => {
  assert.equal(apiUrl("alerts"), null);                      // tests run without VITE_API_URL
  assert.equal(staticUrl("x.json"), "/x.json");
  const resp = (status, body, type = "application/json") => async () => ({ ok: status < 400, status, headers: { get: () => type }, json: async () => { if (body === undefined) throw new Error("bad"); return body; } });
  assert.deepEqual(await getJSON("u", { fetcher: resp(200, { a: 1 }) }), { a: 1 });
  await assert.rejects(getJSON("u", { fetcher: resp(404, {}) }), /not found/);
  await assert.rejects(getJSON("u", { fetcher: resp(500, {}) }), /HTTP 500/);
  await assert.rejects(getJSON("u", { fetcher: resp(200, "<html>", "text/html") }), /did not return JSON/);
  await assert.rejects(getJSON("u", { fetcher: resp(200, undefined) }), /not valid JSON/);
});
