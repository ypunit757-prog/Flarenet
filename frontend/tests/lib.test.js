import test from "node:test";
import assert from "node:assert/strict";
import { classOf, dur, pct, ago, letterOf, isMX } from "../src/lib/format.js";
import { summarize } from "../src/lib/summarize.js";

const END = "2026-11-08T01:07:00Z";
const endS = Date.parse(END) / 1000;

function payload({ p = 0.4, alert = 1, demo = false, horizon = 360 } = {}) {
  const n = 24, t = Array.from({ length: n }, (_, i) => endS - (n - 1 - i) * 600);
  return {
    demo,
    source: { goes_calibrated: true },
    model: { threshold: 0.18, horizon_min: horizon },
    period: { start: "2026-11-01T01:07:00Z", end: END },
    series: { t, flux: t.map(() => 3.6e-6) },
    forecast: { t, p: t.map((_, i) => (i === n - 1 ? p : 0.05)), alert: t.map((_, i) => (i === n - 1 ? alert : 0)) },
    flares: [{ peak: "2026-11-07T18:40:00Z", class: "M2.0", forecast: "WARNED", lead_min: 347, max_prob: 0.44 }],
  };
}
// summarize reads the clock; pin it to END + h hours
function at(h, fn) {
  const real = Date.now;
  Date.now = () => (endS + h * 3600) * 1000;
  try { return fn(); } finally { Date.now = real; }
}

test("flare class from flux, on the boundaries", () => {
  assert.equal(classOf(1e-4), "X1.0");
  assert.equal(classOf(9.99e-5), "M10.0");
  assert.equal(classOf(1e-5), "M1.0");
  assert.equal(classOf(3.6e-6), "C3.6");
  assert.equal(classOf(1e-7), "B1.0");
  assert.equal(classOf(5e-9), "A0.5");
  assert.equal(classOf(0), "–");
  assert.equal(classOf(NaN), "–");
});

test("durations and ages read as words", () => {
  assert.equal(dur(347), "5 h 47 min");
  assert.equal(dur(120), "2 h");
  assert.equal(dur(45), "45 min");
  assert.equal(ago(0.2), "12 min");
  assert.equal(ago(8.2), "8 h 12 min");
  assert.equal(ago(72), "3 days");
  assert.equal(pct(0.384), "38%");
});

test("class letters and M/X test", () => {
  assert.equal(letterOf("M2.0"), "M");
  assert.equal(letterOf("–"), "A");
  assert.equal(letterOf(undefined), "A");
  assert.ok(isMX({ class: "X1.2" }) && isMX({ class: "M1.0" }) && !isMX({ class: "C9.9" }));
});

test("a fresh forecast above the alert line is an alert", () => {
  const s = at(0.5, () => summarize(payload()));
  assert.equal(s.state, "alert");
  assert.equal(s.expired, false);
  assert.equal(s.stale, false);
});

test("a fresh forecast below the line is watch near it, quiet far from it", () => {
  assert.equal(at(0.5, () => summarize(payload({ p: 0.15, alert: 0 }))).state, "watch");
  assert.equal(at(0.5, () => summarize(payload({ p: 0.04, alert: 0 }))).state, "quiet");
});

test("older than three hours is stale but still live until the horizon passes", () => {
  const s = at(4.5, () => summarize(payload()));
  assert.equal(s.state, "alert");
  assert.equal(s.stale, true);
  assert.equal(s.expired, false);
});

test("older than its own horizon is out of date, never a live alert", () => {
  const s = at(6.5, () => summarize(payload()));
  assert.equal(s.state, "expired");
  assert.equal(s.expired, true);
  assert.equal(s.alert, true, "the last reading is still reported, just not as live");
  assert.equal(at(8.2, () => summarize(payload({ p: 0.04, alert: 0 }))).state, "expired");
});

test("the horizon comes from the model, not a constant", () => {
  assert.equal(at(3.5, () => summarize(payload({ horizon: 180 }))).state, "expired");
  assert.equal(at(3.5, () => summarize(payload({ horizon: 360 }))).state, "alert");
});

test("demo data is always demo, however old", () => {
  assert.equal(at(100, () => summarize(payload({ demo: true }))).state, "demo");
});

test("counts M and X flares in the last 24 hours before the data ends", () => {
  const d = payload();
  d.flares.push({ peak: "2026-11-05T18:33:00Z", class: "M3.6", forecast: "MISSED", lead_min: null, max_prob: 0.04 });
  const s = at(0.5, () => summarize(d));
  assert.equal(s.ev.length, 1);
  assert.equal(s.by.M, 1);
  assert.equal(s.by.X, 0);
});
