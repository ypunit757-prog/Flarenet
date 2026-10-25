"""Inference and evaluation: nowcast + forecast scores, per-class detection, lead times."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score

from flarenet.core.utils import get_logger
from flarenet.pipelines.evaluation.lead_time import compute_lead_times, summarise_lead_times
from flarenet.pipelines.evaluation.metrics import (
    best_threshold,
    per_class_metrics,
    safe_auc,
    safe_roc,
    score_at_threshold,
)
from flarenet.pipelines.features.normalise import normalise

log = get_logger(__name__)


@torch.no_grad()
def infer(
    model,
    X: np.ndarray,
    mu: np.ndarray,
    sg: np.ndarray,
    device,
    batch_size: int = 512,
    now_bias=None,
):
    """Run the model over raw (unnormalised) windows.

    ``now_bias`` (optional, one value per class) is added to the nowcast logits first.
    Returns (nowcast class ids, nowcast class probabilities, forecast probabilities).
    """
    model.eval()
    Xn = normalise(X, mu, sg)
    bias = (
        None
        if now_bias is None
        else torch.as_tensor(np.asarray(now_bias, dtype=np.float32)).to(device)
    )
    classes, class_probs, probs = [], [], []
    for i in range(0, len(Xn), batch_size):
        xb = torch.from_numpy(Xn[i : i + batch_size]).to(device)
        now_logits, fc_logits, _ = model(xb)
        if bias is not None:
            now_logits = now_logits + bias
        classes.append(now_logits.argmax(1).cpu().numpy())
        class_probs.append(torch.softmax(now_logits, -1).cpu().numpy())
        probs.append(torch.sigmoid(fc_logits).cpu().numpy())
    return np.concatenate(classes), np.concatenate(class_probs), np.concatenate(probs)


@dataclass
class EvalResult:
    """Everything the report and dashboard need for one dataset."""

    name: str
    nowcast_true: np.ndarray
    nowcast_pred: np.ndarray
    forecast_true: np.ndarray
    forecast_prob: np.ndarray
    times: np.ndarray
    accuracy: float
    macro_f1: float
    auc: float
    scores: dict
    oracle_tss: float
    per_class: list
    lead: pd.DataFrame
    roc: tuple = field(default_factory=tuple)
    ci: dict = field(default_factory=dict)  # 95% block-bootstrap intervals
    calibration: dict = field(default_factory=dict)
    baselines: dict = field(default_factory=dict)  # reference forecasters on the same windows
    kind: str = "test"  # test (held-out split of a train source) | external (eval source)

    def summary(self) -> dict:
        return {
            "accuracy": self.accuracy,
            "macro_F1": self.macro_f1,
            "AUC": self.auc,
            "TSS": self.scores["TSS"],
            "HSS": self.scores["HSS"],
            "POD": self.scores["POD"],
            "FAR": self.scores["FAR"],
            "TSS_oracle": self.oracle_tss,
            "perclass": self.per_class,
            "lead_time": summarise_lead_times(self.lead),
            "confidence_95": self.ci,
            "calibration": {k: v for k, v in self.calibration.items() if k in ("brier", "ece")},
            "baselines": self.baselines,
            "kind": self.kind,
            "n_windows": int(len(self.forecast_true)),
            "forecast_positive_frac": float(np.mean(self.forecast_true))
            if len(self.forecast_true)
            else None,
        }


def evaluate_dataset(
    name: str,
    model,
    X,
    y_now,
    y_fc,
    times,
    mu,
    sg,
    threshold: float,
    flare_db,
    device,
    horizon_min: int,
    now_bias=None,
) -> EvalResult:
    """Score a labelled set of windows at the saved alert threshold."""
    pred_cls, _, prob = infer(model, X, mu, sg, device, now_bias=now_bias)
    acc = float((pred_cls == y_now).mean())
    macro_f1 = float(f1_score(y_now, pred_cls, average="macro"))
    auc = safe_auc(y_fc, prob)
    scores = score_at_threshold(y_fc, prob, threshold)
    _, oracle = best_threshold(y_fc, prob)
    per_class = per_class_metrics(y_now, pred_cls, label=name)
    lead = compute_lead_times(
        flare_db, times, prob, threshold, max_lead_min=horizon_min, label=name
    )
    log.info(
        "[%s] nowcast acc=%.4f macro-F1=%.4f | forecast AUC=%.4f TSS=%.4f POD=%.4f FAR=%.4f (oracle TSS %.4f)",
        name,
        acc,
        macro_f1,
        auc,
        scores["TSS"],
        scores["POD"],
        scores["FAR"],
        oracle,
    )
    return EvalResult(
        name,
        np.asarray(y_now),
        pred_cls,
        np.asarray(y_fc),
        prob,
        np.asarray(times),
        acc,
        macro_f1,
        auc,
        scores,
        oracle,
        per_class,
        lead,
        safe_roc(y_fc, prob),
    )
