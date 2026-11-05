"""Use the tree model + stack on a live forecast, and refuse to when it would be wrong.

The stack file written by ``scripts/gbm_6h.py --live-compatible --install`` holds the tree model, the stacking weights
and the alert thresholds. Its weights belong to ONE network checkpoint, so the file records that checkpoint's SHA-256 and
is ignored for any other. It also stores a few feature rows with the probabilities they gave when it was fitted; a
machine whose libraries give different numbers (another scikit-learn, say) is refused. In every refusal the live forecast
falls back to the network alone and says why: it never mixes, and never fails the page.
"""

from __future__ import annotations

import hashlib
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from flarenet.gbm6h.features import prepare, window_features
from flarenet.gbm6h.stack import Stack

KIND = "gbm6h_stack"
VERSION = 1
MIN_HISTORY_H = (
    72.0  # the longest window of a live-compatible feature set (the 168-hour ones are dropped)
)
FORBIDDEN = ("rot_", "168")  # features a 7-day live feed cannot provide


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def package(gbm, stack: Stack | None, thresholds: dict, levels: dict | None, horizon_min: int,
            checkpoint_sha256: str | None, check: dict | None, dropped: list[str]) -> dict:  # fmt: skip
    import sklearn

    return {
        "kind": KIND, "version": VERSION, "gbm": gbm, "stack": stack, "thresholds": thresholds,
        "alert_levels": levels or {}, "horizon_min": int(horizon_min), "min_history_h": MIN_HISTORY_H,
        "checkpoint_sha256": checkpoint_sha256, "selfcheck": check, "features_dropped": dropped,
        "sklearn_version": sklearn.__version__,
    }  # fmt: skip


@dataclass
class LiveStack:
    gbm: object
    stack: Stack
    thresholds: dict
    alert_levels: dict
    horizon_min: int
    min_history_h: float
    sklearn_version: str = ""

    @property
    def threshold(self) -> float:
        return float(self.thresholds["stack"])

    def apply(self, fc: pd.DataFrame, bundle) -> pd.DataFrame:
        """``fc`` (one row per forecast time, ``p_mx_6h`` and ``alert`` from the network) with the stacked probability
        and alert wherever there are ``min_history_h`` hours of data behind the forecast time; earlier rows keep the
        network's values. Adds ``p_network`` and ``stack_used``."""
        out = fc.copy()
        t = out["time_min"].to_numpy(float)
        p_net = out["p_mx_6h"].to_numpy(float)
        rows = np.flatnonzero(t >= self.min_history_h * 60.0)
        p, used = p_net.copy(), np.zeros(len(out), dtype=bool)
        if len(rows):
            lf, ev = prepare(bundle)
            tab = pd.DataFrame([window_features(int(round(t[i])), lf, ev, bundle.t0_unix, self.horizon_min)
                                for i in rows])  # fmt: skip
            p_gbm = self.gbm.predict_proba(tab)
            good = np.isfinite(p_gbm) & np.isfinite(p_net[rows])
            r = rows[good]
            p[r] = self.stack.predict(p_net[r], p_gbm[good])
            used[r] = True
        alert = out["alert"].to_numpy().copy()
        alert[used] = (p[used] >= self.threshold).astype(int)
        out["p_network"], out["p_mx_6h"], out["alert"], out["stack_used"] = p_net, p, alert, used
        return out

    def info(self, fc_stacked: pd.DataFrame) -> dict:
        used = fc_stacked["stack_used"].to_numpy(bool)
        return {
            "used": True, "threshold": round(self.threshold, 3),
            "weights": {"network": round(self.stack.a_cnn, 3), "tree_model": round(self.stack.a_gbm, 3)},
            "alert_levels": {k: round(float(v["threshold"]), 3) for k, v in self.alert_levels.items() if v},
            "history_hours": self.min_history_h, "rows": int(len(used)), "network_only_rows": int((~used).sum()),
        }  # fmt: skip


def load_live_stack(path: str | Path, checkpoint_path: str | Path) -> tuple[LiveStack | None, str]:
    """``(stack, "")`` when the file is usable with this checkpoint, else ``(None, reason)``."""
    path = Path(path)
    if not path.exists():
        return None, f"{path} does not exist"
    try:
        with open(path, "rb") as f:
            pkg = pickle.load(f)  # noqa: S301 - the project's own model file, as for the daily model
    except Exception as e:  # noqa: BLE001 - a pickle from another library version can fail in many ways
        return None, f"{path} could not be read ({type(e).__name__}: {e})"
    if not isinstance(pkg, dict) or pkg.get("kind") != KIND or pkg.get("version") != VERSION:
        return (
            None,
            "it is not a gbm6h stack file of this version; re-create it with scripts/gbm_6h.py --install",
        )
    if pkg.get("stack") is None:
        return (
            None,
            "it holds no stack: the stacking regression gave the tree model no positive weight",
        )
    if not pkg.get("checkpoint_sha256"):
        return None, "it does not record which checkpoint it was fitted with; re-create it"
    if pkg["checkpoint_sha256"] != file_sha256(checkpoint_path):
        return None, "it was fitted with a different network checkpoint than the one in use"
    cols = list(getattr(pkg["gbm"], "cols", []))
    if not cols or any(c.startswith(FORBIDDEN[0]) or FORBIDDEN[1] in c for c in cols):
        return (
            None,
            "it uses features a 7-day live feed cannot provide; re-create it with --live-compatible",
        )
    chk = pkg.get("selfcheck")
    if chk:
        got = pkg["gbm"].predict_proba(chk["X"])
        if not np.allclose(got, np.asarray(chk["p_gbm"]), atol=1e-5):
            return None, (f"its predictions differ on this machine (scikit-learn {_sk()} here, "
                          f"{pkg.get('sklearn_version')} when fitted)")  # fmt: skip
    return LiveStack(pkg["gbm"], pkg["stack"], pkg["thresholds"], pkg.get("alert_levels") or {},
                     int(pkg["horizon_min"]), float(pkg["min_history_h"]), pkg.get("sklearn_version", "")), ""  # fmt: skip


def _sk() -> str:
    import sklearn

    return sklearn.__version__
