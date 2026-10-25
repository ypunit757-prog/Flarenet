#!/usr/bin/env bash
# Train on NASA/NOAA GOES-R data, test on a different satellite and period, then start the live monitor.
# Needs internet and: pip install -e ".[netcdf]"
set -euo pipefail

flarenet run -c configs/nasa_goes_ncei.yaml            # downloads GOES-R files to data/downloads once
less runs/nasa_goes/report.md

# Optional: live forecasts from the NOAA SWPC GOES X-ray feed (Ctrl-C to stop)
flarenet monitor -c configs/live_monitor.yaml --checkpoint runs/nasa_goes/checkpoint.pt
