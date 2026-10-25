// Measured results from this project. Edit these when you re-run an evaluation.
export const VALIDATION = [
  {
    key: "fc", title: "6-hour forecast", where: "GOES-18, April 2025 onward, never seen in training. 325 M/X flares.",
    big: [["0.80", "AUC"], ["0.41", "TSS"], ["71%", "flares warned"]],
    list: [
      "Median warning 5 h 22 min before the flare.",
      "43% of alert times were followed by an M/X flare within 6 h, against 13% at any time (lift 3.2x).",
      "Alerts moved to random times would warn 23% of flares on average; this beat 99.9% of those shuffles.",
      "57% of alerts were false alarms.",
    ],
  },
  {
    key: "daily", title: "Daily forecast, NOAA's protocol", where: "GOES-18, 446 days (140 with an M/X flare). One forecast per day, 24 h ahead.", tag: ["below NOAA", "bad"],
    table: { head: ["", "FlareNet", "NOAA*"], rows: [["TSS", "0.36", "0.44"], ["HSS", "0.30", "0.47"], ["AUC", "0.76", "0.87"], ["Brier", "0.220", "0.11"]] },
    list: ["*NOAA figures are published for a different period, so this is not a same-days comparison.", "Brier skill is slightly negative: the probabilities are no better than a constant guess."],
  },
  {
    key: "now", title: "Flare class now (nowcast)", where: "GOES-18, April 2025 onward. Class of the flare in progress.", tag: ["X-class weak", "bad"],
    big: [["0.97", "accuracy"], ["0.58", "macro F1"], ["3 of 13", "X read as X"]],
    list: ["M-class flares are read correctly far more often than X-class.", "On October 2024 flares a model with absolute flux inputs got the class right for 64 of 68 flares."],
  },
  {
    key: "isro", title: "ISRO Aditya-L1", where: "SoLEXS and HEL1OS.", tag: ["not tested yet", "warn"],
    list: ["Support exists (configs/aditya_eval.yaml). A real test needs several weeks of data and a calibration against GOES for the same days."],
  },
];

export const CHECKS = [
  ["ok", "Real forecast skill on unseen GOES-18 data", "AUC 0.80, TSS 0.41, 325 flares, chance test p < 0.01"],
  ["ok", "Works across GOES satellites", "Trained on GOES-16, tested on GOES-18"],
  ["bad", "Better than NOAA", "Not shown. On the daily protocol it is below NOAA's published scores"],
  ["bad", "Calibrated probabilities", "No. Brier skill is about zero"],
  ["bad", "X-class nowcast", "Weak. 3 of 13 labelled X"],
  ["warn", "Aditya-L1 and other agencies", "Not tested yet"],
];
