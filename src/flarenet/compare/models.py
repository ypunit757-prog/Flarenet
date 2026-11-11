"""The models compared with FlareNet's CNN-BiLSTM: a plain LSTM, Random Forest, XGBoost and gradient boosting.

All of them forecast the same target (an M/X flare starts within the horizon) for the same windows:

* ``LSTM`` reads the same 12-channel, 90-minute window as the CNN-BiLSTM, normalised with the same statistics, but has
  no convolutional encoder, no residual blocks and no bidirectional or attention layers: one LSTM layer and a linear
  head. It answers "do the convolutions and the second branch earn their parameters?".
* ``Random Forest``, ``XGBoost`` and the gradient-boosting model read the long-memory feature table of
  ``flarenet.gbm6h.features`` (flux level over hours to days, flares in the last hours, time since the last M flare),
  which the network does not see.

Every model is trained on the training windows only. Probabilities are mapped onto observed frequencies with a Platt curve
fitted on validation windows (``compare.pipeline``), so Brier scores are comparable.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch import nn

from flarenet.core.utils import get_logger
from flarenet.pipelines.evaluation.metrics import safe_auc
from flarenet.pipelines.features.normalise import normalise

log = get_logger(__name__)

# ---------------------------------------------------------------- sequence model


class LSTMClassifier(nn.Module):
    """One (optionally stacked) unidirectional LSTM over the window, the last time step into a linear head."""

    def __init__(self, n_channels: int, hidden: int = 64, layers: int = 1, dropout: float = 0.1):
        super().__init__()
        self.lstm = nn.LSTM(
            n_channels, hidden, num_layers=layers, batch_first=True,
            dropout=dropout if layers > 1 else 0.0,
        )  # fmt: skip
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (batch, channels, time)
        out, _ = self.lstm(x.transpose(1, 2))
        return self.head(out[:, -1]).squeeze(-1)


@dataclass
class FittedLSTM:
    model: LSTMClassifier
    mu: np.ndarray
    sg: np.ndarray
    device: torch.device
    best_epoch: int
    val_auc: float

    @torch.no_grad()
    def predict_proba(self, X: np.ndarray, batch_size: int = 1024) -> np.ndarray:
        self.model.eval()
        Xn = normalise(X, self.mu, self.sg)
        out = []
        for i in range(0, len(Xn), batch_size):
            xb = torch.from_numpy(Xn[i : i + batch_size]).to(self.device)
            out.append(torch.sigmoid(self.model(xb)).cpu().numpy())
        return np.concatenate(out) if out else np.zeros(0)


def fit_lstm(
    X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray,
    mu: np.ndarray, sg: np.ndarray, device, epochs: int = 8, batch_size: int = 256,
    lr: float = 2e-3, hidden: int = 64, seed: int = 0, patience: int = 3,
) -> FittedLSTM:  # fmt: skip
    """Train on ``X_train`` (raw windows, normalised here with ``mu``/``sg``) and keep the epoch with the best
    validation ROC-AUC. Class imbalance is handled with a positive-class weight of sqrt(negatives / positives),
    capped at 6 (the rule the CNN-BiLSTM's class weights use)."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    device = torch.device(device)
    y_train = np.asarray(y_train).astype(np.float32)
    pos = max(float(y_train.sum()), 1.0)
    neg = max(float(len(y_train) - y_train.sum()), 1.0)
    weight = float(min(np.sqrt(neg / pos), 6.0))
    model = LSTMClassifier(X_train.shape[1], hidden=hidden).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(weight, device=device))
    Xn = normalise(X_train, mu, sg)
    best = (-1.0, copy.deepcopy(model.state_dict()), 0)
    stale = 0
    for ep in range(1, epochs + 1):
        model.train()
        order = rng.permutation(len(Xn))
        for i in range(0, len(order), batch_size):
            idx = order[i : i + batch_size]
            xb = torch.from_numpy(Xn[idx]).to(device)
            yb = torch.from_numpy(y_train[idx]).to(device)
            opt.zero_grad()
            loss_fn(model(xb), yb).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        fitted = FittedLSTM(model, mu, sg, device, ep, float("nan"))
        auc = safe_auc(np.asarray(y_val).astype(int), fitted.predict_proba(X_val))
        log.info("LSTM epoch %d/%d: validation AUC %.3f", ep, epochs, auc)
        if np.isfinite(auc) and auc > best[0]:
            best, stale = (auc, copy.deepcopy(model.state_dict()), ep), 0
        else:
            stale += 1
            if stale >= patience:
                break
    model.load_state_dict(best[1])
    return FittedLSTM(model, mu, sg, device, best[2], float(best[0]))


# ---------------------------------------------------------------- tabular models


class TableModel:
    """A scikit-learn style classifier on the feature table; columns that are empty in training are dropped, as in
    ``flarenet.gbm6h.model.HourlyGBM``, and missing values are filled with the training median (trees that cannot
    take NaN need it; trees that can are given the same input so the comparison stays fair)."""

    def __init__(self, estimator, impute: bool = True):
        self.estimator = estimator
        self.impute = impute
        self.cols: list[str] = []
        self.fill: pd.Series | None = None

    def fit(self, table: pd.DataFrame, y) -> TableModel:
        self.cols = [c for c in table.columns if table[c].notna().any()]
        X = table[self.cols]
        self.fill = X.median()
        self.estimator.fit(self._prep(X), np.asarray(y).astype(int))
        return self

    def _prep(self, X: pd.DataFrame) -> np.ndarray:
        X = X.reindex(columns=self.cols)
        if self.impute:
            X = X.fillna(self.fill)
        return X.to_numpy(dtype=float)

    def predict_proba(self, table: pd.DataFrame) -> np.ndarray:
        if not self.cols:
            return np.full(len(table), np.nan)
        return self.estimator.predict_proba(self._prep(table))[:, 1]

    def importances(self) -> dict[str, float]:
        """The estimator's own impurity/gain importance per feature (not a causal statement; see flarenet.explain)."""
        imp = getattr(self.estimator, "feature_importances_", None)
        return {} if imp is None else dict(zip(self.cols, map(float, imp), strict=True))


def random_forest(seed: int = 0, n_jobs: int = -1) -> TableModel:
    from sklearn.ensemble import RandomForestClassifier

    return TableModel(
        RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=20,
            max_features="sqrt",
            class_weight="balanced_subsample",
            n_jobs=n_jobs,
            random_state=seed,
        )  # fmt: skip
    )


def xgboost_model(y_train, seed: int = 0, n_jobs: int = -1) -> TableModel:
    """Raises ImportError when xgboost is not installed (``pip install xgboost``)."""
    from xgboost import XGBClassifier

    y = np.asarray(y_train).astype(int)
    pos = max(int(y.sum()), 1)
    scale = float(min(np.sqrt(max(len(y) - pos, 1) / pos), 6.0))
    return TableModel(
        XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=5,
            scale_pos_weight=scale,
            eval_metric="logloss",
            tree_method="hist",
            n_jobs=n_jobs,
            random_state=seed,
        ),  # fmt: skip
        impute=False,  # xgboost handles missing values natively
    )
