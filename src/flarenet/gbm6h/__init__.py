"""Hour-by-hour gradient-boosting forecaster for the 6-hour M/X question, to be blended with the neural network.

The network sees 90 minutes of X-ray flux. A flare in the next six hours also depends on what the Sun did over the
last days (how active the region has been, when the last M flare was, what happened a solar rotation ago), which the
daily forecaster already turns into features. This package computes those features at the end of every network window
and stacks a small tree model with the network on a validation period. See docs/IMPROVING_FURTHER.md.
"""

from flarenet.gbm6h.features import set_features, window_features
from flarenet.gbm6h.stack import Stack, fit_stack

__all__ = ["Stack", "fit_stack", "set_features", "window_features"]
