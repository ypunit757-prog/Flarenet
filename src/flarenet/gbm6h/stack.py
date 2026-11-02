"""Combine the network and the tree model: a logistic regression on the logits of the two probabilities.

Fitted on validation windows only (3 numbers from thousands of windows), so the weights are not learned from the
data either model was trained on. The result is calibrated by construction. If either weight comes out
non-positive, that model adds nothing the other lacks (or works against it) and ``fit_stack`` returns None: use the
network alone rather than a blend that reverses part of a model's ranking.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression

EPS = 1e-6


def _logit(p) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


@dataclass
class Stack:
    a_cnn: float
    a_gbm: float
    b: float

    def predict(self, p_cnn, p_gbm) -> np.ndarray:
        z = self.a_cnn * _logit(p_cnn) + self.a_gbm * _logit(p_gbm) + self.b
        return 1.0 / (1.0 + np.exp(-z))

    def share_gbm(self) -> float:
        """Share of the combined logit weight that sits on the tree model (0 = network only, 1 = trees only)."""
        return float(self.a_gbm / (self.a_cnn + self.a_gbm))

    def to_dict(self) -> dict:
        return asdict(self)


def fit_stack(y, p_cnn, p_gbm, min_positives: int = 20) -> Stack | None:
    y = np.asarray(y).astype(int)
    pc, pg = np.asarray(p_cnn, dtype=float), np.asarray(p_gbm, dtype=float)
    ok = np.isfinite(pc) & np.isfinite(pg)
    y, pc, pg = y[ok], pc[ok], pg[ok]
    if y.sum() < min_positives or (1 - y).sum() < min_positives:
        return None
    X = np.column_stack([_logit(pc), _logit(pg)])
    m = LogisticRegression(C=1e4, max_iter=2000).fit(X, y)
    a, g, b = float(m.coef_[0, 0]), float(m.coef_[0, 1]), float(m.intercept_[0])
    if not all(np.isfinite([a, g, b])) or a <= 0 or g <= 0:
        return None
    return Stack(a, g, b)
