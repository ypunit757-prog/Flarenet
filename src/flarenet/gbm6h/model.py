"""The tree model: ``DailyModel`` (three small gradient-boosting models + a regularised logistic regression, Platt
calibration on blocked out-of-fold predictions) fitted on hourly rows instead of daily rows."""

from __future__ import annotations

import numpy as np
import pandas as pd

from flarenet.daily.model import DailyModel, DailyParams


class HourlyGBM:
    def __init__(self, params: DailyParams | None = None):
        self.model = DailyModel(params)
        self.cols: list[str] = []

    def fit(self, table: pd.DataFrame, y: np.ndarray) -> HourlyGBM:
        # a column that is empty everywhere (e.g. no rotation history in a short series) carries nothing
        self.cols = [c for c in table.columns if table[c].notna().any()]
        self.model.fit(table[self.cols], np.asarray(y).astype(int), pd.DatetimeIndex(table.index))
        return self

    def predict_proba(self, table: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(table.reindex(columns=self.cols))
