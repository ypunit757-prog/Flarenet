"""Calibrated day-level forecaster: gradient-boosting ensemble blended with a regularised logistic regression.

Why this shape: with a few thousand training days (a few hundred flare days), a handful of well-regularised
tabular models beats a deep network, tolerates missing inputs (no sunspot file for a day) and trains in seconds.
Raw scores of such models are not probabilities, so out-of-fold predictions on the training days fit a
calibrator (Platt or isotonic) and the alert threshold; the final model is then refit on all training days.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import QuantileTransformer

from flarenet.pipelines.evaluation.metrics import best_threshold

# (max_depth, learning_rate, max_iter, min_samples_leaf, l2): small, shallow trees, strong regularisation
GBM_CONFIGS = [(2, 0.05, 160, 30, 2.0), (3, 0.04, 200, 30, 2.0), (4, 0.03, 220, 40, 4.0)]


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


@dataclass
class DailyParams:
    blend_gbm: float = 0.5  # weight of the GBM ensemble; the rest goes to logistic regression
    calibration: str = "platt"  # platt | isotonic | none
    oof_blocks: int = 5
    gap_days: int = 3  # days dropped on each side of a held-out block (flare days cluster in time)
    seed: int = 0
    gbm_configs: list = field(default_factory=lambda: list(GBM_CONFIGS))


class _Blend:
    """GBM ensemble + logistic regression on one feature matrix (no calibration)."""

    def __init__(self, p: DailyParams):
        self.p = p
        self.gbms: list = []
        self.lr = None

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> _Blend:
        self.cols = list(X.columns)
        Xa = X.to_numpy(float)
        self.gbms = []
        for k, (depth, lr, it, leaf, l2) in enumerate(self.p.gbm_configs):
            m = HistGradientBoostingClassifier(
                max_depth=depth, learning_rate=lr, max_iter=it, min_samples_leaf=leaf, l2_regularization=l2,
                random_state=self.p.seed + k, early_stopping=False,
            )  # fmt: skip
            self.gbms.append(m.fit(Xa, y))
        self.lr = make_pipeline(
            SimpleImputer(strategy="median"),
            QuantileTransformer(n_quantiles=min(200, max(10, len(y) // 2)), output_distribution="normal", random_state=0),
            LogisticRegression(C=0.05, max_iter=1000),
        ).fit(Xa, y)  # fmt: skip
        return self

    def raw(self, X: pd.DataFrame) -> np.ndarray:
        Xa = X[self.cols].to_numpy(float)
        g = np.mean([m.predict_proba(Xa)[:, 1] for m in self.gbms], axis=0)
        lr = self.lr.predict_proba(Xa)[:, 1]
        w = self.p.blend_gbm
        return w * g + (1 - w) * lr


class DailyModel:
    """Fit on training days; ``predict_proba`` returns calibrated probabilities for new days."""

    def __init__(self, params: DailyParams | None = None):
        self.params = params or DailyParams()
        self.threshold = 0.5
        self.calibrator = None
        self.oof_prob_: np.ndarray | None = None

    # ---- calibration -------------------------------------------------------------------------
    def _fit_calibrator(self, raw: np.ndarray, y: np.ndarray) -> None:
        kind = self.params.calibration
        if kind == "none" or len(np.unique(y)) < 2:
            self.calibrator = None
        elif kind == "isotonic" and y.sum() >= 40:
            self.calibrator = (
                "iso",
                IsotonicRegression(out_of_bounds="clip", y_min=0.001, y_max=0.999).fit(raw, y),
            )
        else:  # Platt scaling: a 1-D logistic regression on the logit of the raw score
            self.calibrator = (
                "platt",
                LogisticRegression(C=100.0, max_iter=1000).fit(_logit(raw)[:, None], y),
            )

    def _calibrate(self, raw: np.ndarray) -> np.ndarray:
        if self.calibrator is None:
            return raw
        kind, c = self.calibrator
        return c.predict(raw) if kind == "iso" else c.predict_proba(_logit(raw)[:, None])[:, 1]

    # ---- fit / predict -------------------------------------------------------------------------
    def fit(self, X: pd.DataFrame, y: np.ndarray, dates: pd.DatetimeIndex) -> DailyModel:
        """Blocked out-of-fold predictions -> calibrator and threshold, then a final fit on every row."""
        p = self.params
        y = np.asarray(y, dtype=int)
        order = np.argsort(dates.values)
        X, y, dates = X.iloc[order], y[order], dates[order]
        n = len(y)
        self.cols = list(X.columns)
        oof = np.full(n, np.nan)
        if n >= 60 and y.sum() >= 10 and (1 - y).sum() >= 10 and p.oof_blocks >= 2:
            edges = np.linspace(0, n, p.oof_blocks + 1).astype(int)
            day = (dates - dates[0]).days.to_numpy()
            for b in range(p.oof_blocks):
                te = np.arange(edges[b], edges[b + 1])
                lo, hi = day[te[0]] - p.gap_days, day[te[-1]] + p.gap_days
                tr = np.where((day < lo) | (day > hi))[0]
                if len(tr) < 40 or y[tr].sum() < 5 or y[te].size == 0:
                    continue
                oof[te] = _Blend(p).fit(X.iloc[tr], y[tr]).raw(X.iloc[te])
        ok = np.isfinite(oof)
        self.oof_prob_ = None
        if ok.sum() >= 60 and len(np.unique(y[ok])) == 2:
            self._fit_calibrator(oof[ok], y[ok])
            cal = self._calibrate(oof[ok])
            self.threshold = best_threshold(y[ok], cal, smooth=2)[0]
            self.oof_prob_ = np.full(n, np.nan)
            self.oof_prob_[ok] = cal
        else:
            self.calibrator, self.threshold = None, 0.5
        self.blend = _Blend(p).fit(X, y)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return np.clip(self._calibrate(self.blend.raw(X[self.cols])), 0.0, 1.0)


def recalibrate_reference(
    p_ref: np.ndarray, y: np.ndarray
) -> tuple[LogisticRegression, float] | None:
    """Platt-scale another forecaster's probabilities on training days and pick its alert threshold.

    This gives a reference forecaster (NOAA) the same treatment FlareNet gets, so a win cannot come from NOAA
    being thresholded at an arbitrary 0.5 or being mis-calibrated for the current solar cycle. Returns None
    when there are too few days or flare days to fit it.
    """
    ok = np.isfinite(p_ref)
    y = np.asarray(y, dtype=int)
    if ok.sum() < 100 or y[ok].sum() < 15 or (1 - y[ok]).sum() < 15:
        return None
    cal = LogisticRegression(C=100.0, max_iter=1000).fit(_logit(p_ref[ok])[:, None], y[ok])
    thr = best_threshold(y[ok], cal.predict_proba(_logit(p_ref[ok])[:, None])[:, 1], smooth=2)[0]
    return cal, float(thr)


def apply_reference(
    ref: tuple[LogisticRegression, float] | None, p_ref: np.ndarray
) -> tuple[np.ndarray, float]:
    """Calibrated probabilities (NaN where the reference forecast is missing) and the threshold."""
    if ref is None:
        return np.full(len(p_ref), np.nan), float("nan")
    cal, thr = ref
    out = np.full(len(p_ref), np.nan)
    ok = np.isfinite(p_ref)
    if ok.any():
        out[ok] = cal.predict_proba(_logit(p_ref[ok])[:, None])[:, 1]
    return out, thr
