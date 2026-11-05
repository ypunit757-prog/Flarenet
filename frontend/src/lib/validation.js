// Measured results from this project. Edit these when you re-run an evaluation.
// 6-hour model: the network in models/flarenet.pt (run goes_all_chrono: trained on GOES-16, tested on GOES-18) blended with
// the tree model in models/gbm6h_live.pkl (scripts/gbm_6h.py --live-compatible; numbers from its report, docs/IMPROVING_FURTHER.md).
// Daily model: docs/DAILY_VS_NOAA.md, "First real run".
export const VALIDATION = [
  {
    key: "fc", title: "6-hour forecast (neural network + tree model)", where: "GOES-18, April 2025 onward, never seen in training (GOES-16 only). 12,668 forecast times, 1,646 of them followed by an M/X flare within 6 h.",
    big: [["0.83", "AUC"], ["0.49", "TSS"], ["64%", "pre-flare times alerted"]],
    list: [
      "Compared with the neural network alone on the same windows (95% intervals of the difference): AUC +0.022 (+0.011 to +0.032), Brier -0.003 (-0.005 to -0.002), TSS +0.043 (+0.016 to +0.075), share of pre-flare times alerted +6 points (+3 to +9). False alarms among alerts +0.8 points (-1.2 to +2.5): not established.",
      "Default alert level (threshold fixed on a later validation year than the training data): 64% of the forecast times that preceded an M/X flare within 6 h were alerted, and 61% of alerts were false alarms. The network alone: 58% and 60%.",
      "Less certain: on the later part of GOES-16, a weaker check, the AUC gain was +0.009 (0.000 to +0.019) and TSS was lower (-0.049), because the validation-chosen threshold alerts more there. This is one test period on one satellite: read the gain as roughly +0.01 to +0.02 AUC, not as settled.",
      "Probabilities stay calibrated: on GOES-18 the calibration error is 0.014 and the Brier score 0.086 (the first model: 0.32 and 0.201).",
      "The tree model reads the last three days of X-ray history, so the first three days of the 7-day live chart use the network alone. Flares warned in time and the median warning time (network alone: 309 of 393 flares, 5 h 25 min) have not been re-measured for the blend.",
      "Not re-run for this model: the chance test (alerts moved to random times) and the precision against the base rate.",
    ],
  },
  {
    key: "daily", title: "Daily forecast against NOAA (separate daily model)", where: "2019-03 to 2026-09: 2,740 days with both forecasts, 725 with an M flare. A model trained only on earlier days predicts each day; NOAA's own 22:00 UTC forecast is the comparison.", tag: ["level with NOAA, ahead in 2025-26", "warn"],
    table: { head: ["FlareNet minus NOAA", "AUC (higher is better)", "Brier (lower is better)"], rows: [["All days", "+0.016 (+0.006 to +0.026)", "-0.008 (-0.014 to -0.003)"], ["2019-2024", "+0.001 (-0.008 to +0.010)", "0.000 (-0.005 to +0.005)"], ["2025-2026", "+0.066 (+0.039 to +0.096)", "-0.037 (-0.050 to -0.024)"]] },
    list: [
      "Brackets are 95% intervals. Brier is against NOAA recalibrated on earlier days, the stronger baseline.",
      "Through 2024 the two were level. The pooled advantage comes from 2025-26, and inside each year it is not established (AUC +0.010, -0.013 to +0.031).",
      "Not shown to be better than NOAA. A live forward test, in which every forecast is logged before its day starts, is running and decides it.",
    ],
  },
  {
    key: "now", title: "Flare class now (nowcast)", where: "GOES-18, April 2025 onward. Class of the flare in progress.", tag: ["X-class not detected", "bad"],
    big: [["0.97", "accuracy"], ["0.55", "macro F1"], ["0 of 13", "X read as X"]],
    list: [
      "C or stronger: 91% of windows detected, 21% of those readings wrong. M or stronger: 82% detected, 46% of those readings wrong.",
      "One training run. Multi-seed tests of this project (on simulated data) showed a large seed-to-seed spread in nowcast macro F1, so the rise from the previous model (accuracy 0.94, macro F1 0.42) is not shown to be a real improvement.",
      "On GOES data the class is defined by the flux, so a plain flux rule reads X-class exactly; the learned nowcast matters for instruments without calibrated classes.",
    ],
  },
  {
    key: "isro", title: "ISRO Aditya-L1", where: "SoLEXS and HEL1OS.", tag: ["not tested yet", "warn"],
    list: ["Support exists (configs/aditya_eval.yaml). A real test needs several weeks of data and a calibration against GOES for the same days."],
  },
];

export const CHECKS = [
  ["ok", "Real forecast skill on unseen GOES-18 data", "AUC 0.83, TSS 0.49; 64% of pre-flare forecast times alerted, 61% of alerts false. The neural network alone: AUC 0.81, TSS 0.45. The AUC gain is +0.022 (+0.011 to +0.032) on GOES-18 but only +0.009 on a later GOES-16 period"],
  ["ok", "Works across GOES satellites", "Trained on GOES-16, tested on GOES-18"],
  ["ok", "Calibrated probabilities", "On GOES-18: calibration error 0.014, Brier 0.086. Calibrated on 2023-24, tested on 2025-26 and a different satellite"],
  ["warn", "Better than NOAA", "Not shown. The 6-hour model is not comparable to NOAA's daily forecast; a separate daily model is level with NOAA through 2024 and ahead in 2025-26. The forward test decides"],
  ["bad", "X-class nowcast", "Not detected. 0 of 13 labelled X"],
  ["warn", "Aditya-L1 and other agencies", "Not tested yet"],
];
