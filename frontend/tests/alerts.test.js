import test from "node:test";
import assert from "node:assert/strict";
import { DEFAULTS, dataProblem, decide, episodes, normalizeSettings, thresholdFor } from "../src/lib/alerts.js";
import { NOW, demo, flare, live } from "./fixtures.js";

const S = { ...DEFAULTS, enabled: true };

test("settings: bad or missing values fall back to safe defaults", () => {
  assert.deepEqual(normalizeSettings(null), { ...DEFAULTS });
  const s = normalizeSettings({ enabled: "yes", level: "wild", threshold: 7, cooldownMin: -5, observedMin: "Q", observed: false });
  assert.equal(s.enabled, false);               // only a real true turns alerts on
  assert.equal(s.level, "balanced");
  assert.equal(s.threshold, 0.99);              // clamped
  assert.equal(s.cooldownMin, 0);
  assert.equal(s.observedMin, "M");
  assert.equal(s.observed, false);
  assert.equal(normalizeSettings({ threshold: NaN }).threshold, DEFAULTS.threshold);
});

test("alert line: custom, then the chosen level, then the model default", () => {
  const D = live();
  assert.deepEqual(thresholdFor(D, { ...S, level: "strict" }), { value: 0.45, name: "strict" });
  assert.deepEqual(thresholdFor(D, { ...S, custom: true, threshold: 0.22 }), { value: 0.22, name: "custom" });
  const noLevels = live({ levels: {} });
  assert.deepEqual(thresholdFor(noLevels, S), { value: 0.3, name: "default" });
});

test("no alert from simulated, stale, broken or future data", () => {
  assert.equal(dataProblem(live(), NOW), null);
  assert.match(dataProblem(demo(), NOW), /simulated/);
  assert.match(dataProblem(live({ ageH: 3.5 }), NOW), /3 hours/);
  assert.match(dataProblem(live({ ageH: -1 }), NOW), /future/);
  assert.match(dataProblem(live({ p: [0.1, NaN] }), NOW), /missing or invalid/);
  assert.match(dataProblem(live({ p: [0.1, null] }), NOW), /missing or invalid/);
  assert.match(dataProblem(live({ p: [0.1, 1.4] }), NOW), /missing or invalid/);
  assert.match(dataProblem(live({ p: [0.5] }), NOW), /incomplete/);
  assert.match(dataProblem(null, NOW), /no forecast/);
  const bad = live(); bad.forecast.t.pop();
  assert.match(dataProblem(bad, NOW), /incomplete/);
});

test("the first check only records the state: enabling alerts never announces what is already on screen", () => {
  const r = decide(live({ p: [0.1, 0.5], flares: [flare(1)] }), S, null, NOW);
  assert.deepEqual(r.notify, []);
  assert.equal(r.memory.inAlert, true);
  assert.equal(r.memory.seen.length, 1);        // the recent flare is marked as seen, silently
});

test("one forecast alert per episode, however long it lasts", () => {
  let m = decide(live({ p: [0.1, 0.1] }), S, null, NOW).memory;
  let r = decide(live({ p: [0.1, 0.5] }), S, m, NOW);
  assert.equal(r.notify.length, 1);
  assert.equal(r.notify[0].kind, "predicted_risk");
  m = r.memory;
  for (const p of [0.6, 0.7, 0.55]) { r = decide(live({ p: [0.1, p] }), S, m, NOW + 600); assert.deepEqual(r.notify, []); m = r.memory; }
  r = decide(live({ p: [0.1, 0.1] }), S, m, NOW + 1200);   // back below the line
  assert.equal(r.memory.inAlert, false);
});

test("cool-down: a new episode soon after the last announcement is not announced", () => {
  const D = (p, at = NOW) => live({ p: [0.1, p], at });
  let m = decide(D(0.1), S, null, NOW).memory;
  let r = decide(D(0.5), S, m, NOW); assert.equal(r.notify.length, 1); m = r.memory;
  m = decide(D(0.1), S, m, NOW + 600).memory;
  r = decide(D(0.5), S, m, NOW + 1200);                     // 20 min later, cool-down is 180 min
  assert.deepEqual(r.notify, []);
  m = decide(D(0.1), S, r.memory, NOW + 1800).memory;
  r = decide(D(0.5, NOW + 181 * 60), S, m, NOW + 181 * 60); // after the cool-down (with fresh data)
  assert.equal(r.notify.length, 1);
});

test("a forecast is never worded as an observation, and an observation says it is not an official record", () => {
  let m = decide(live({ p: [0.1, 0.1] }), S, null, NOW).memory;
  const r = decide(live({ p: [0.1, 0.5], flares: [flare(1, "X2.0")] }), S, m, NOW);
  const [risk, obs] = [r.notify.find((n) => n.kind === "predicted_risk"), r.notify.find((n) => n.kind === "observed_flare")];
  assert.match(risk.body, /Forecast, not an observation/);
  assert.doesNotMatch(risk.title, /observed/i);
  assert.match(obs.title, /X2\.0 flare observed/);
  assert.match(obs.body, /not an official NOAA record/);
  assert.notEqual(risk.tag, obs.tag);
});

test("an observed flare is announced once, only when recent, and only at or above the chosen class", () => {
  const base = decide(live({ p: [0.1, 0.1] }), S, null, NOW).memory;
  const D = live({ p: [0.1, 0.1], flares: [flare(1, "M1.5"), flare(5, "M3.0"), flare(0.5, "C5.0")] });
  let r = decide(D, S, base, NOW);
  assert.deepEqual(r.notify.map((n) => n.title), ["FlareNet: M1.5 flare observed"]);   // 5 h old: recorded, not announced; C below M
  r = decide(D, S, r.memory, NOW + 60);
  assert.deepEqual(r.notify, []);                                                        // never twice
  r = decide(D, { ...S, observedMin: "C" }, base, NOW);
  assert.equal(r.notify.length, 2);                                                      // M1.5 and C5.0
  r = decide(D, { ...S, observed: false }, base, NOW);
  assert.deepEqual(r.notify, []);
});

test("bad data raises nothing and leaves the memory alone", () => {
  const m = decide(live({ p: [0.1, 0.1] }), S, null, NOW).memory;
  for (const D of [demo(), live({ ageH: 5, p: [0.1, 0.9] }), live({ p: [0.1, NaN] })]) {
    const r = decide(D, S, m, NOW);
    assert.deepEqual(r.notify, []);
    assert.ok(r.skipped);
    assert.equal(r.memory.inAlert, false);
  }
});

test("episodes: outcome is hit, false alarm or pending", () => {
  const horizon = 6 * 3600;
  const p = [0.5, 0.5, 0.1, 0.1, 0.1, 0.5];                 // 10-minute steps: one episode, a gap under an hour, another
  const D = live({ p, ageH: 0, flares: [] });
  const none = episodes(D, 0.3, NOW);
  assert.equal(none.length, 1);                              // the gap is under an hour, so one episode
  assert.equal(none[0].outcome, "pending");                  // window still open at NOW
  assert.equal(episodes(D, 0.3, NOW + horizon + 3600)[0].outcome, "false alarm");
  const hit = live({ p, ageH: 0, flares: [{ peak: iso2(D.forecast.t[1] + 1800), class: "M2.0" }] });
  const e = episodes(hit, 0.3, NOW + horizon + 3600)[0];
  assert.equal(e.outcome, "hit");
  assert.equal(e.flare, "M2.0");
  const small = live({ p, ageH: 0, flares: [{ peak: iso2(D.forecast.t[1] + 1800), class: "C9.0" }] });
  assert.equal(episodes(small, 0.3, NOW + horizon + 3600)[0].outcome, "false alarm");   // C-class does not count
});
const iso2 = (u) => new Date(u * 1000).toISOString().replace(".000Z", "Z");
