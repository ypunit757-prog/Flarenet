"""Day-level forecaster: will an M (or X) class flare start during the next UTC day?

This is NOAA's product (one probability per day, issued around 22:00 UTC), so it can be scored on the very
same days against NOAA's own published forecasts. It combines multi-day X-ray history with NOAA's sunspot
region summaries (what forecasters actually look at) in a calibrated gradient-boosting model.
"""
