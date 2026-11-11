"""Alerts: when to warn, when not to, and how to say it.

Two kinds of event are kept apart everywhere (history file, email, browser notification, web page):

* ``predicted_risk``: the model's forecast crossed an alert threshold. Nothing has happened yet; it is a probability.
* ``observed_flare``: the live GOES X-ray reading shows an M- or X-class flare. This is a measurement from the same feed
  NOAA uses, but it is not an official NOAA event record: NOAA SWPC is the authority.
"""
