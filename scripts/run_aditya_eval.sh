#!/usr/bin/env bash
# Evaluate a trained checkpoint on real Aditya-L1 data, then forecast. Edit the paths (data is not included).
set -euo pipefail

SOLEXS="/data/AL1_SOLEXS_20260623_SDD2_L1.lc"
HEL1OS_DIRS=(
  "/data/hel1os/HLS_20260623_000007_43181sec_lev1_V111"
  "/data/hel1os/HLS_20260623_121027_42566sec_lev1_V111"
)

flarenet evaluate --checkpoint runs/default/checkpoint.pt --solexs "$SOLEXS" --hel1os "${HEL1OS_DIRS[@]}"
flarenet predict --checkpoint runs/default/checkpoint.pt --solexs "$SOLEXS" --hel1os "${HEL1OS_DIRS[@]}" \
  --source aditya_l1
