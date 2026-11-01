// Measured results from this project. Edit these when you re-run an evaluation.
// 6-hour model: the checkpoint in models/flarenet.pt (run goes_all_chrono: trained on GOES-16, tested on GOES-18).
// Daily model: docs/DAILY_VS_NOAA.md, "First real run".
export const VALIDATION = [
  {
    key: "fc", title: "6-hour forecast", where: "GOES-18, April 2025 onward, never seen in training (GOES-16 only). 393 M/X flares.",
    big: [["0.81", "AUC"], ["0.45", "TSS"], ["309 of 393", "flares warned"]],
    list: [
      "Median warning 5 h 25 min before the flare (warnings ranged from 10 min to 6 h). 4 more flares were warned too late and 80 were not warned.",
      "Default alert level (threshold 0.19, fixed on a later validation year than the training data): 58% of the forecast times that preceded an M/X flare within 6 h were alerted, and 60% of alerts were false alarms. A more sensitive level (0.125) alerts 73% of those times with 72% false alarms; a stricter one (0.37) alerts 36% with 44% false alarms.",
      "Probabilities are calibrated: on GOES-18 the calibration error is 0.01 and the Brier score 0.090 (the previous model: 0.32 and 0.201). The ranking skill (AUC, TSS) did not change; what changed is that a score now tracks how often a flare followed it.",
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
  ["ok", "Real forecast skill on unseen GOES-18 data", "AUC 0.81, TSS 0.45, 309 of 393 M/X flares warned in time (60% of alert windows were false alarms)"],
  ["ok", "Works across GOES satellites", "Trained on GOES-16, tested on GOES-18"],
  ["ok", "Calibrated probabilities", "On GOES-18: calibration error 0.01, Brier 0.090. Calibrated on 2023-24, tested on 2025-26 and a different satellite"],
  ["warn", "Better than NOAA", "Not shown. The 6-hour model is not comparable to NOAA's daily forecast; a separate daily model is level with NOAA through 2024 and ahead in 2025-26. The forward test decides"],
  ["bad", "X-class nowcast", "Not detected. 0 of 13 labelled X"],
  ["warn", "Aditya-L1 and other agencies", "Not tested yet"],
];
